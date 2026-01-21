import argparse
import importlib
import json
from pathlib import Path

import torch
import yaml


def import_from_target(target: str):
    mod_name, cls_name = target.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def instantiate(cfg_block: dict):
    cfg_block = dict(cfg_block)
    target = cfg_block.pop("_target_")
    cls = import_from_target(target)
    return cls(**cfg_block)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt_dir", type=str, default=None)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["experiment"]["out_dir"])
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir is not None else (out_dir / "checkpoints")

    device = get_device(cfg["evaluation"]["device"])

    dataset_obj = instantiate(cfg["dataset"])

    # task net class
    task_cfg = dict(cfg["task_net"])
    task_target = task_cfg.pop("_target_")
    TaskCls = import_from_target(task_target)

    num_tasks = int(cfg["dataset"]["num_tasks"])
    batch_size = int(cfg["evaluation"]["batch_size"])

    accs = []
    for t in range(num_tasks):
        ckpt = torch.load(ckpt_dir / f"task_{t:03d}.pt", map_location=device)
        model = TaskCls(**{k: v for k, v in task_cfg.items()}).to(device)
        model.load_state_dict(ckpt["task_net_state"])

        task_data = dataset_obj.get_task(t, batch_size=batch_size)
        accs.append(float(eval_accuracy(model, task_data["test"], device)))

    metrics = {
        "acc_by_task": accs,
        "avg_acc": sum(accs) / len(accs),
        "first_task_acc": accs[0],
        "forgetting_first_task": float(accs[0] - accs[-1]),
    }

    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval" / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
