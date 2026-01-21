import argparse
import importlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm


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


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def eval_accuracy(model, loader, device: torch.device) -> float:
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return correct / max(1, total)


def train_supervised(model, loader, device, lr: float, epochs: int, grad_clip: float | None):
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()



#          YAML
# -----------------------
def import_from_target(target: str):
    """
    target: "package.module:ClassName"
    """
    mod_name, cls_name = target.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def instantiate(cfg_block: dict):
    """
    cfg_block:
      _target_: "...:Class"
      other_key: other_value
    """
    cfg_block = dict(cfg_block)
    target = cfg_block.pop("_target_")
    cls = import_from_target(target)
    return cls(**cfg_block)


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)



#    Coherence checks
# -----------------------
def validate_action_spec(task_net, controller, cfg: dict):
    spec_task = task_net.action_spec()
    spec_controller = getattr(controller, "action_spec", None)

    # Se il controller non espone spec interna, ok: userà quella passata (noi gliela settiamo).
    # Se la espone (es. definita in init), deve coincidere.
    if spec_controller is not None and list(spec_controller) != list(spec_task):
        raise ValueError(
            f"Action spec mismatch.\n"
            f"task_net.action_spec()={spec_task}\n"
            f"controller.action_spec={spec_controller}\n"
            f"Fix YAML so they coincide."
        )

    # Check config (se user mette controller.action_spec nel YAML)
    if "action_spec" in cfg.get("controller", {}):
        if list(cfg["controller"]["action_spec"]) != list(spec_task):
            raise ValueError(
                f"YAML controller.action_spec != task_net.action_spec(). "
                f"{cfg['controller']['action_spec']} vs {spec_task}"
            )


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

    # Instantiate dataset/scenario
    dataset_obj = instantiate(cfg["dataset"])

    # Instantiate task_net
    task_net = instantiate(cfg["task_net"]).to(device)

    # Instantiate controller + value_net
    # Nota: il controller ha bisogno dell'action_spec (n_i per layer) -> gliela passiamo dalla task net.
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

    num_tasks = int(cfg["dataset"]["num_tasks"])
    for t in range(num_tasks):
        task_data = dataset_obj.get_task(t, batch_size=int(cfg["training"]["batch_size"]))

        if t == 0:
            train_supervised(
                model=task_net,
                loader=task_data["train"],
                device=device,
                lr=float(cfg["training"]["lr_task"]),
                epochs=int(cfg["training"]["epochs_task"]),
                grad_clip=cfg["training"].get("grad_clip", None),
            )
            val_acc = eval_accuracy(task_net, task_data["val"], device)
            test_acc = eval_accuracy(task_net, task_data["test"], device)

            stats = {"task": t, "val_acc": float(val_acc), "test_acc": float(test_acc), "note": "base_train"}
            train_stats.append(stats)

        else:
            # Actor-critic optimizers
            opt_c = torch.optim.Adam(controller.parameters(), lr=float(cfg["training"]["lr_controller"]))
            opt_v = torch.optim.Adam(value_net.parameters(), lr=float(cfg["training"]["lr_value"]))

            best_reward = -1e9
            best_model_state = None
            best_actions = None
            best_val_acc = None
            best_test_acc = None

            trials = int(cfg["training"]["controller_trials"])
            alpha = float(cfg["rcl"]["reward_alpha"])

            for _ in tqdm(range(trials), desc=f"Task {t} trials"):
                actions, logp_sum = controller.sample(device=device)  # list[int], scalar logp

                # Build child network
                child = task_net.expanded_copy(actions).to(device)

                # Train child
                train_supervised(
                    model=child,
                    loader=task_data["train"],
                    device=device,
                    lr=float(cfg["training"]["lr_task"]),
                    epochs=int(cfg["training"]["epochs_task"]),
                    grad_clip=cfg["training"].get("grad_clip", None),
                )

                # Reward: val accuracy - alpha * complexity (sum actions)
                val_acc = eval_accuracy(child, task_data["val"], device)
                complexity = sum(actions)
                reward = float(val_acc) - alpha * float(complexity)

                # Actor-critic update
                v = value_net(device=device)  # scalar
                advantage = torch.tensor(reward, device=device) - v

                loss_c = -(advantage.detach() * logp_sum)
                loss_v = advantage.pow(2)

                opt_c.zero_grad()
                loss_c.backward()
                torch.nn.utils.clip_grad_norm_(controller.parameters(), float(cfg["training"]["grad_clip"]))
                opt_c.step()

                opt_v.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), float(cfg["training"]["grad_clip"]))
                opt_v.step()

                if reward > best_reward:
                    test_acc = eval_accuracy(child, task_data["test"], device)
                    best_reward = reward
                    best_actions = list(actions)
                    best_val_acc = float(val_acc)
                    best_test_acc = float(test_acc)
                    best_model_state = {k: v.detach().cpu() for k, v in child.state_dict().items()}

            # Promote best child
            task_net = instantiate(cfg["task_net"])  # rebuild fresh model
            task_net.load_state_dict(best_model_state)
            task_net = task_net.to(device)

            stats = {
                "task": t,
                "best_reward": float(best_reward),
                "best_actions": best_actions,
                "best_val_acc": float(best_val_acc),
                "best_test_acc": float(best_test_acc),
                "best_complexity": int(sum(best_actions)),
            }
            train_stats.append(stats)

        # Save checkpoint per task
        save_ckpt(ckpt_dir / f"task_{t:03d}.pt", {
            "task": t,
            "task_net_state": task_net.state_dict(),
            "controller_state": controller.state_dict(),
            "value_net_state": value_net.state_dict(),
            "cfg": cfg,
        })
        save_json({"train_stats": train_stats}, out_dir / "train_stats.json")

    print("Done. Results saved to:", out_dir)


if __name__ == "__main__":
    main()
