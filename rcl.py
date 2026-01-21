import argparse
import importlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml


# -----------------------
# Utils
# -----------------------
def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_ckpt(path: str | Path, payload: dict):
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


# -----------------------
# Eval (task-aware se supportato)
# -----------------------
@torch.no_grad()
def eval_accuracy(model, loader, device: torch.device, task_id: int | None = None) -> float:
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        # task-aware se il modello supporta task_id, altrimenti fallback
        try:
            logits = model(x, task_id=task_id) if task_id is not None else model(x)
        except TypeError:
            logits = model(x)

        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()

    return correct / max(1, total)


def train_supervised(
    model,
    loader,
    device,
    lr: float,
    epochs: int,
    grad_clip: float | None,
    log: bool = False,
):
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(1, epochs + 1):
        total = 0
        correct = 0
        loss_sum = 0.0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()

            with torch.no_grad():
                total += y.numel()
                correct += (logits.argmax(dim=1) == y).sum().item()
                loss_sum += float(loss.item()) * y.size(0)

        if log:
            print(
                f"    [train] epoch {ep:03d}/{epochs} "
                f"loss={loss_sum / max(1,total):.4f} acc={correct / max(1,total):.4f}",
                flush=True,
            )


# -----------------------
# YAML
# -----------------------
def import_from_target(target: str):
    """target: 'package.module:ClassName'"""
    mod_name, cls_name = target.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def instantiate(cfg_block: dict):
    cfg_block = dict(cfg_block)
    target = cfg_block.pop("_target_")
    cls = import_from_target(target)
    return cls(**cfg_block)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------
# Coherence checks
# -----------------------
def validate_action_spec(task_net, controller, cfg: dict):
    spec_task = task_net.action_spec()
    spec_controller = getattr(controller, "action_spec", None)

    if spec_controller is not None and list(spec_controller) != list(spec_task):
        raise ValueError(
            f"Action spec mismatch.\n"
            f"task_net.action_spec()={spec_task}\n"
            f"controller.action_spec={spec_controller}\n"
            f"Fix YAML so they coincide."
        )

    if "action_spec" in cfg.get("controller", {}):
        if list(cfg["controller"]["action_spec"]) != list(spec_task):
            raise ValueError(
                f"YAML controller.action_spec != task_net.action_spec(). "
                f"{cfg['controller']['action_spec']} vs {spec_task}"
            )


# -----------------------
# Value net helper
# -----------------------
# def value_net_scalar(value_net, device: torch.device, t: int):
#     # 1) la tuValueNet vuole device esplicito
#     try:
#         v = value_net(device=device)
#         return v.reshape(())
#     except TypeError:
#         pass

#     # 2) fallback: value_net() senza args
#     try:
#         v = value_net()
#         if not torch.is_tensor(v):
#             v = torch.tensor(float(v), device=device)
#         return v.reshape(())
#     except TypeError:
#         pass

#     # 3) fallback: value_net([[t]])
#     s = torch.tensor([[float(t)]], device=device)
#     v = value_net(s)
#     if not torch.is_tensor(v):
#         v = torch.tensor(float(v), device=device)
#     return v.reshape(())
def value_net_scalar(value_net, device: torch.device, t: int, num_tasks: int):

    denom = max(1, num_tasks - 1)
    s = torch.tensor([[float(t) / float(denom)]], device=device)
    v = value_net(s)
    if not torch.is_tensor(v):
        v = torch.tensor(float(v), device=device)
    return v.reshape(())


# -----------------------
# Freezing: ExpandableMLP-specific
# -----------------------
def register_freeze_masks_expandable_mlp(child, old_hidden_sizes):
    """
    Train only newly added parameters (paper-style).
    """
    if not (hasattr(child, "layers") and hasattr(child, "hidden_sizes")):
        return False
    if len(child.layers) != 3 or len(child.hidden_sizes) < 2:
        return False
    if old_hidden_sizes is None or len(old_hidden_sizes) < 2:
        return False

    old_h1, old_h2 = int(old_hidden_sizes[0]), int(old_hidden_sizes[1])
    new_h1, new_h2 = int(child.hidden_sizes[0]), int(child.hidden_sizes[1])

    # layer0: input -> h1   W: [h1, in]
    l0 = child.layers[0]
    m_w0 = torch.zeros_like(l0.weight)
    m_b0 = torch.zeros_like(l0.bias)
    if new_h1 > old_h1:
        m_w0[old_h1:new_h1, :] = 1.0
        m_b0[old_h1:new_h1] = 1.0
    l0.weight.register_hook(lambda g, m=m_w0: g * m.to(g.device))
    l0.bias.register_hook(lambda g, m=m_b0: g * m.to(g.device))

    # layer1: h1 -> h2   W: [h2, h1]
    l1 = child.layers[1]
    m_w1 = torch.zeros_like(l1.weight)
    m_b1 = torch.zeros_like(l1.bias)
    if new_h2 > old_h2:
        m_w1[old_h2:new_h2, :] = 1.0
        m_b1[old_h2:new_h2] = 1.0
    if new_h1 > old_h1:
        m_w1[:, old_h1:new_h1] = 1.0
    l1.weight.register_hook(lambda g, m=m_w1: g * m.to(g.device))
    l1.bias.register_hook(lambda g, m=m_b1: g * m.to(g.device))

    # layer2: h2 -> classes  W: [C, h2]
    l2 = child.layers[2]
    m_w2 = torch.zeros_like(l2.weight)
    m_b2 = torch.zeros_like(l2.bias)
    if new_h2 > old_h2:
        m_w2[:, old_h2:new_h2] = 1.0
    m_b2[:] = 0.0
    l2.weight.register_hook(lambda g, m=m_w2: g * m.to(g.device))
    l2.bias.register_hook(lambda g, m=m_b2: g * m.to(g.device))

    return True


# -----------------------
# Pretty table
# -----------------------
def print_summary_table(train_stats: list[dict]):
    def _fmt(x, w=8):
        if x is None:
            return " " * w
        if isinstance(x, float):
            return f"{x:>{w}.4f}"
        return f"{str(x):>{w}}"

    header = f"{'task':>4} | {'val':>8} | {'test':>8} | {'best_r':>8} | {'cx':>5} | {'note':>10}"
    print("\n" + header)
    print("-" * len(header))
    for s in train_stats:
        t = s.get("task")
        val = s.get("best_val_acc", s.get("val_acc"))
        test = s.get("best_test_acc", s.get("test_acc"))
        best_r = s.get("best_reward")
        cx = s.get("best_complexity")
        note = s.get("note", "")
        print(f"{t:>4} | {_fmt(val)} | {_fmt(test)} | {_fmt(best_r)} | {_fmt(cx, w=5)} | {note:>10}")
    print("", flush=True)


# -----------------------
# Main training (RCL)
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg.get("seed", 42)))

    out_dir = ensure_dir(cfg["experiment"]["out_dir"])
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    save_json(cfg, out_dir / "config_used.json")

    device = get_device(cfg["training"]["device"])
    print(f"[setup] device={device}", flush=True)
    print(f"[setup] out_dir={out_dir}", flush=True)

    dataset_obj = instantiate(cfg["dataset"])
    task_net = instantiate(cfg["task_net"]).to(device)

    controller_cfg = dict(cfg["controller"])
    controller_target = controller_cfg.pop("_target_")
    ControllerCls = import_from_target(controller_target)

    action_spec = task_net.action_spec()
    if "action_spec" not in controller_cfg:
        controller_cfg["action_spec"] = action_spec

    controller = ControllerCls(**controller_cfg).to(device)
    value_net = instantiate(cfg["value_net"]).to(device)

    validate_action_spec(task_net, controller, cfg)

    train_stats = []
    first_task_curve_online = []
    task0_eval_data = None  # cache test-set task 0

    num_tasks = int(cfg["dataset"]["num_tasks"])
    batch_size = int(cfg["training"]["batch_size"])
    epochs_task = int(cfg["training"]["epochs_task"])
    lr_task = float(cfg["training"]["lr_task"])
    grad_clip = cfg["training"].get("grad_clip", None)

    trials = int(cfg["training"]["controller_trials"])
    alpha = float(cfg["rcl"]["reward_alpha"])

    print_every = int(cfg.get("logging", {}).get("print_every_trials", 25))
    print_first = int(cfg.get("logging", {}).get("print_first_trials", 3))

    for t in range(num_tasks):
        print("\n" + "=" * 80, flush=True)
        print(f"[task {t}] start", flush=True)
        task_data = dataset_obj.get_task(t, batch_size=batch_size)

        if task0_eval_data is None:
            task0_eval_data = dataset_obj.get_task(0, batch_size=batch_size)

        if t == 0:
            print(f"[task {t}] base train: epochs={epochs_task} lr={lr_task}", flush=True)
            train_supervised(
                model=task_net,
                loader=task_data["train"],
                device=device,
                lr=lr_task,
                epochs=epochs_task,
                grad_clip=grad_clip,
                log=True,
            )
            val_acc = eval_accuracy(task_net, task_data["val"], device)
            test_acc = eval_accuracy(task_net, task_data["test"], device)
            print(f"[task {t}] base done: val={val_acc:.4f} test={test_acc:.4f}", flush=True)

            train_stats.append(
                {"task": t, "val_acc": float(val_acc), "test_acc": float(test_acc), "note": "base_train"}
            )

            # timestamp per task-aware inference
            if hasattr(task_net, "register_task_slice"):
                task_net.register_task_slice()

        else:
            opt_c = torch.optim.Adam(controller.parameters(), lr=float(cfg["training"]["lr_controller"]))
            opt_v = torch.optim.Adam(value_net.parameters(), lr=float(cfg["training"]["lr_value"]))

            best_reward = -1e9
            best_model_state = None
            best_actions = None
            best_val_acc = None
            best_test_acc = None

            print(f"[task {t}] controller search: trials={trials} alpha={alpha}", flush=True)
            t0 = time.time()

            for k in range(1, trials + 1):
                actions, logp_sum = controller.sample(device=device)

                old_hidden = getattr(task_net, "hidden_sizes", None)
                child = task_net.expanded_copy(actions).to(device)

                masked = register_freeze_masks_expandable_mlp(child, old_hidden)

                train_supervised(
                    model=child,
                    loader=task_data["train"],
                    device=device,
                    lr=lr_task,
                    epochs=epochs_task,
                    grad_clip=grad_clip,
                    log=True,
                )

                val_acc = eval_accuracy(child, task_data["val"], device)
                complexity = int(sum(actions))
                reward = float(val_acc) - alpha * float(complexity)

                v = value_net_scalar(value_net, device=device, t=t, num_tasks=num_tasks)
                advantage = torch.tensor(reward, device=device) - v

                loss_c = -(advantage.detach() * logp_sum)
                loss_v = advantage.pow(2)

                opt_c.zero_grad()
                loss_c.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(controller.parameters(), float(grad_clip))
                opt_c.step()

                opt_v.zero_grad()
                loss_v.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), float(grad_clip))
                opt_v.step()

                improved = False
                if reward > best_reward:
                    test_acc = eval_accuracy(child, task_data["test"], device)
                    best_reward = float(reward)
                    best_actions = list(actions)
                    best_val_acc = float(val_acc)
                    best_test_acc = float(test_acc)
                    best_model_state = {kk: vv.detach().cpu() for kk, vv in child.state_dict().items()}
                    improved = True

                if improved or (k <= print_first) or (k % print_every == 0) or (k == trials):
                    elapsed = time.time() - t0
                    tag = "BEST" if improved else "info"
                    print(
                        f"[task {t}] trial {k:04d}/{trials} ({tag}) "
                        f"reward={reward:.4f} val={val_acc:.4f} cx={complexity} "
                        f"masked={int(masked)} "
                        f"loss_c={float(loss_c.item()):.4f} loss_v={float(loss_v.item()):.4f} "
                        f"best_reward={best_reward:.4f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    if improved:
                        print(f"[task {t}]   best_actions={best_actions}", flush=True)

            if best_actions is None or best_model_state is None:
                raise RuntimeError(f"[task {t}] No best model selected; check controller.sample().")

            # Promote best child: ricrea architettura espansa + carica pesi
            task_net = task_net.expanded_copy(best_actions).to(device)
            task_net.load_state_dict(best_model_state)

            # timestamp per task-aware inference
            if hasattr(task_net, "register_task_slice"):
                task_net.register_task_slice()

            train_stats.append(
                {
                    "task": t,
                    "best_reward": float(best_reward),
                    "best_actions": best_actions,
                    "best_val_acc": float(best_val_acc),
                    "best_test_acc": float(best_test_acc),
                    "best_complexity": int(sum(best_actions)),
                }
            )

            print(
                f"[task {t}] done: best_reward={best_reward:.4f} "
                f"best_val={best_val_acc:.4f} best_test={best_test_acc:.4f} "
                f"best_cx={int(sum(best_actions))}",
                flush=True,
            )

        # --- Online lightweight metric: A[t][0] ---
        acc0 = float(eval_accuracy(task_net, task0_eval_data["test"], device, task_id=0))
        first_task_curve_online.append(acc0)
        print(f"[task {t}] online first-task acc={acc0:.4f}", flush=True)

        save_ckpt(
            ckpt_dir / f"task_{t:03d}.pt",
            {
                "task": t,
                "task_net_state": task_net.state_dict(),
                "controller_state": controller.state_dict(),
                "value_net_state": value_net.state_dict(),
                "cfg": cfg,
                "task_net_hidden_sizes": getattr(task_net, "hidden_sizes", None),
                "task_net_task_slices": getattr(task_net, "task_slices", None),
            },
        )

        save_json(
            {"train_stats": train_stats, "first_task_curve_online": first_task_curve_online},
            out_dir / "train_stats.json",
        )
        print(f"[task {t}] saved checkpoint + train_stats.json", flush=True)

    print_summary_table(train_stats)
    print("Done. Results saved to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
