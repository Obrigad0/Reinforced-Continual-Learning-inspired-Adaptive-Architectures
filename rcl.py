import argparse
import importlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

# plotting
import matplotlib
matplotlib.use("Agg")  # salva png senza aprire finestre
import matplotlib.pyplot as plt


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


def append_jsonl(obj, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


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
        try:
            logits = model(x, task_id=task_id) if task_id is not None else model(x)
        except TypeError:
            logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return correct / max(1, total)


# -----------------------
# Train supervised
# -----------------------
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
                f" [train] epoch {ep:03d}/{epochs} "
                f"loss={loss_sum / max(1,total):.4f} acc={correct / max(1,total):.4f}",
                flush=True,
            )


# -----------------------
# YAML instantiate
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
def value_net_scalar(value_net, device: torch.device, t: int, num_tasks: int):
    denom = max(1, num_tasks - 1)
    s = torch.tensor([[float(t) / float(denom)]], device=device)
    v = value_net(s)
    if not torch.is_tensor(v):
        v = torch.tensor(float(v), device=device)
    return v.reshape(())


# -----------------------
# Generic hooks (modular)
# -----------------------
def maybe_apply_freeze_policy(child, parent) -> bool:
    if hasattr(child, "apply_freeze_policy"):
        try:
            return bool(child.apply_freeze_policy(parent=parent))
        except TypeError:
            return bool(child.apply_freeze_policy(parent))
    return False


def compute_actions_complexity(task_net, actions) -> float:
    if hasattr(task_net, "actions_complexity"):
        return float(task_net.actions_complexity(actions))
    return float(sum(actions))


# -----------------------
# Plotting helpers
# -----------------------
def _savefig(fig, path: Path):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_first_task_curve(first_task_curve_online: list[float], out_path: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = list(range(len(first_task_curve_online)))
    ax.plot(xs, first_task_curve_online, marker="o")
    ax.set_xlabel("Task index t (after finishing task t)")
    ax.set_ylabel("Test accuracy on task 0")
    ax.set_title("First-task accuracy vs time")
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, out_path)


def plot_avg_accuracy_curve(avg_acc_curve: list[float], out_path: Path):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = list(range(len(avg_acc_curve)))
    ax.plot(xs, avg_acc_curve, marker="o")
    ax.set_xlabel("Task index t (after finishing task t)")
    ax.set_ylabel("Average test accuracy (tasks 0..t)")
    ax.set_title("Average accuracy vs time")
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, out_path)


def plot_acc_heatmap(acc_matrix: list[list[float | None]], out_path: Path):
    import numpy as np

    T = len(acc_matrix)
    A = np.full((T, T), np.nan, dtype=float)
    for t in range(T):
        for j in range(T):
            v = acc_matrix[t][j]
            if v is not None:
                A[t, j] = float(v)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(A, aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Test task j")
    ax.set_ylabel("After training task t")
    ax.set_title("Accuracy matrix A[t,j]")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _savefig(fig, out_path)


def plot_forgetting_bar(acc_matrix: list[list[float | None]], out_path: Path):
    T = len(acc_matrix)
    if T == 0:
        return
    last = acc_matrix[-1]

    forgetting = []
    for j in range(T):
        vals = [acc_matrix[t][j] for t in range(j, T) if acc_matrix[t][j] is not None]
        if not vals or last[j] is None:
            forgetting.append(0.0)
        else:
            forgetting.append(float(max(vals) - float(last[j])))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(list(range(T)), forgetting)
    ax.set_xlabel("Task j")
    ax.set_ylabel("Forgetting F_j")
    ax.set_title("Forgetting per task")
    ax.set_ylim(0.0, max(0.05, max(forgetting) * 1.1))
    _savefig(fig, out_path)


def plot_acc_vs_complexity(train_stats: list[dict], out_path: Path):
    xs = []
    ys = []
    labels = []
    for s in train_stats:
        t = s.get("task")
        cx = s.get("best_complexity", None)
        acc = s.get("best_val_acc", s.get("val_acc", None))
        if cx is None or acc is None:
            continue
        xs.append(float(cx))
        ys.append(float(acc))
        labels.append(int(t))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys)
    for x, y, t in zip(xs, ys, labels):
        ax.annotate(str(t), (x, y))
    ax.set_xlabel("Complexity (proxy)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Accuracy vs complexity (task labels)")
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, out_path)


def make_all_plots(
    out_dir: Path,
    first_task_curve_online: list[float],
    avg_acc_curve: list[float],
    acc_matrix: list[list[float | None]],
    train_stats: list[dict],
):
    plots_dir = ensure_dir(out_dir / "plots")
    plot_first_task_curve(first_task_curve_online, plots_dir / "first_task_acc.png")
    plot_avg_accuracy_curve(avg_acc_curve, plots_dir / "avg_acc.png")
    plot_acc_heatmap(acc_matrix, plots_dir / "acc_matrix_heatmap.png")
    plot_forgetting_bar(acc_matrix, plots_dir / "forgetting_bar.png")
    plot_acc_vs_complexity(train_stats, plots_dir / "acc_vs_complexity.png")


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

    header = f"{'task':>4} | {'val':>8} | {'test':>8} | {'best_r':>8} | {'cx':>8} | {'note':>10}"
    print("\n" + header)
    print("-" * len(header))
    for s in train_stats:
        t = s.get("task")
        val = s.get("best_val_acc", s.get("val_acc"))
        test = s.get("best_test_acc", s.get("test_acc"))
        best_r = s.get("best_reward")
        cx = s.get("best_complexity")
        note = s.get("note", "")
        print(f"{t:>4} | {_fmt(val)} | {_fmt(test)} | {_fmt(best_r)} | {_fmt(cx)} | {note:>10}")
    print("", flush=True)


# -----------------------
# Main
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

    spec = task_net.action_spec()
    if "action_spec" not in controller_cfg:
        controller_cfg["action_spec"] = spec
    controller = ControllerCls(**controller_cfg).to(device)

    value_net = instantiate(cfg["value_net"]).to(device)

    validate_action_spec(task_net, controller, cfg)

    train_stats: list[dict] = []
    first_task_curve_online: list[float] = []
    avg_acc_curve: list[float] = []
    acc_matrix: list[list[float | None]] = []
    task0_eval_data = None

    trial_log_path = out_dir / "trial_log.jsonl"

    num_tasks = int(cfg["dataset"]["num_tasks"])
    batch_size = int(cfg["training"]["batch_size"])
    epochs_task = int(cfg["training"]["epochs_task"])
    lr_task = float(cfg["training"]["lr_task"])
    grad_clip = cfg["training"].get("grad_clip", None)
    trials = int(cfg["training"]["controller_trials"])
    alpha = float(cfg["rcl"]["reward_alpha"])
    print_every = int(cfg.get("logging", {}).get("print_every_trials", 25))
    print_first = int(cfg.get("logging", {}).get("print_first_trials", 3))

    # -----------------------
    # MICRO-OTTIMIZZAZIONE: cache loaders per tutti i task (una sola volta)
    # -----------------------
    tasks_data_cache = [dataset_obj.get_task(j, batch_size=batch_size) for j in range(num_tasks)]

    for t in range(num_tasks):
        print("\n" + "=" * 80, flush=True)
        print(f"[task {t}] start", flush=True)

        task_data = tasks_data_cache[t]
        if task0_eval_data is None:
            task0_eval_data = tasks_data_cache[0]

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

            train_stats.append({"task": t, "val_acc": float(val_acc), "test_acc": float(test_acc), "note": "base_train"})

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
            best_complexity = None

            print(f"[task {t}] controller search: trials={trials} alpha={alpha}", flush=True)
            t0 = time.time()

            for k in range(1, trials + 1):
                actions, logp_sum = controller.sample(device=device)

                child = task_net.expanded_copy(actions).to(device)
                masked = maybe_apply_freeze_policy(child=child, parent=task_net)

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

                complexity = compute_actions_complexity(task_net, actions)
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

                append_jsonl(
                    {
                        "task": int(t),
                        "trial": int(k),
                        "reward": float(reward),
                        "val_acc": float(val_acc),
                        "complexity": float(complexity),
                        "actions": list(map(int, actions)),
                        "masked": int(masked),
                        "loss_c": float(loss_c.item()),
                        "loss_v": float(loss_v.item()),
                        "alpha": float(alpha),
                    },
                    trial_log_path,
                )

                improved = False
                if reward > best_reward:
                    test_acc = eval_accuracy(child, task_data["test"], device)
                    best_reward = float(reward)
                    best_actions = list(actions)
                    best_val_acc = float(val_acc)
                    best_test_acc = float(test_acc)
                    best_complexity = float(complexity)
                    best_model_state = {kk: vv.detach().cpu() for kk, vv in child.state_dict().items()}
                    improved = True

                if improved or (k <= print_first) or (k % print_every == 0) or (k == trials):
                    elapsed = time.time() - t0
                    tag = "BEST" if improved else "info"
                    print(
                        f"[task {t}] trial {k:04d}/{trials} ({tag}) "
                        f"reward={reward:.4f} val={val_acc:.4f} cx={complexity:.2f} "
                        f"masked={int(masked)} "
                        f"loss_c={float(loss_c.item()):.4f} loss_v={float(loss_v.item()):.4f} "
                        f"best_reward={best_reward:.4f} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    if improved:
                        print(f"[task {t}] best_actions={best_actions}", flush=True)

            if best_actions is None or best_model_state is None:
                raise RuntimeError(f"[task {t}] No best model selected; check controller.sample().")

            task_net = task_net.expanded_copy(best_actions).to(device)
            task_net.load_state_dict(best_model_state)

            if hasattr(task_net, "register_task_slice"):
                task_net.register_task_slice()

            train_stats.append(
                {
                    "task": t,
                    "best_reward": float(best_reward),
                    "best_actions": best_actions,
                    "best_val_acc": float(best_val_acc),
                    "best_test_acc": float(best_test_acc),
                    "best_complexity": float(best_complexity),
                }
            )

            print(
                f"[task {t}] done: best_reward={best_reward:.4f} "
                f"best_val={best_val_acc:.4f} best_test={best_test_acc:.4f} "
                f"best_cx={float(best_complexity):.2f}",
                flush=True,
            )

        # -----------------------
        # Evaluate CL metrics after each task t
        # -----------------------
        acc0 = float(eval_accuracy(task_net, task0_eval_data["test"], device, task_id=0))
        first_task_curve_online.append(acc0)
        print(f"[task {t}] online first-task acc={acc0:.4f}", flush=True)

        row: list[float | None] = [None] * num_tasks
        test_accs_seen = []
        for j in range(t + 1):
            aj = float(eval_accuracy(task_net, tasks_data_cache[j]["test"], device, task_id=j))
            row[j] = aj
            test_accs_seen.append(aj)
        acc_matrix.append(row)

        avg_acc = float(sum(test_accs_seen) / max(1, len(test_accs_seen)))
        avg_acc_curve.append(avg_acc)
        print(f"[task {t}] avg test acc (0..t)={avg_acc:.4f}", flush=True)

        save_ckpt(
            ckpt_dir / f"task_{t:03d}.pt",
            {
                "task": t,
                "task_net_state": task_net.state_dict(),
                "controller_state": controller.state_dict(),
                "value_net_state": value_net.state_dict(),
                "cfg": cfg,
                "task_net_task_slices": getattr(task_net, "task_slices", None),
            },
        )

        save_json(
            {
                "train_stats": train_stats,
                "first_task_curve_online": first_task_curve_online,
                "avg_acc_curve": avg_acc_curve,
            },
            out_dir / "train_stats.json",
        )
        save_json({"acc_matrix": acc_matrix}, out_dir / "acc_matrix.json")

        make_all_plots(out_dir, first_task_curve_online, avg_acc_curve, acc_matrix, train_stats)

        print(f"[task {t}] saved checkpoint + logs + plots", flush=True)
        print_summary_table(train_stats)

    make_all_plots(out_dir, first_task_curve_online, avg_acc_curve, acc_matrix, train_stats)
    print("Done. Results saved to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
