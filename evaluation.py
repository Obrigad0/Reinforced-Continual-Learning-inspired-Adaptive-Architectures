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

    # Precarico tutti i test set una sola volta
    tasks_data = [dataset_obj.get_task(j, batch_size=batch_size) for j in range(num_tasks)]

    # A[t][j]
    A = [[None for _ in range(num_tasks)] for _ in range(num_tasks)]

    for t in range(num_tasks):
        ckpt = torch.load(ckpt_dir / f"task_{t:03d}.pt", map_location=device)

        task_net_hidden_sizes = ckpt.get("task_net_hidden_sizes", task_cfg.get("hidden_sizes"))
        model = TaskCls(
            input_dim=task_cfg["input_dim"],
            hidden_sizes=task_net_hidden_sizes,
            num_classes=task_cfg["num_classes"],
            action_spec=task_cfg["action_spec"],
        ).to(device)

        model.load_state_dict(ckpt["task_net_state"])

        if "task_net_task_slices" in ckpt and ckpt["task_net_task_slices"] is not None:
            model.task_slices = ckpt["task_net_task_slices"]

        # Valuto su tutti i task visti finora (j <= t)
        for j in range(t + 1):
            A[t][j] = float(eval_accuracy(model, tasks_data[j]["test"], device, task_id=j))

    # Metriche derivate
    first_task_curve = [A[t][0] for t in range(num_tasks)]
    acc_by_task_final = [A[num_tasks - 1][j] for j in range(num_tasks)]

    forgetting_by_task = []
    for j in range(num_tasks):
        best = max(A[t][j] for t in range(j, num_tasks) if A[t][j] is not None)
        last = A[num_tasks - 1][j]
        forgetting_by_task.append(float(best - last))

    metrics = {
        "A_matrix": A,  # triangolare: valori solo per j<=t
        "first_task_curve": first_task_curve,
        "acc_by_task_final": acc_by_task_final,
        "avg_acc_final": sum(acc_by_task_final) / len(acc_by_task_final),
        "forgetting_by_task": forgetting_by_task,
        "avg_forgetting": sum(forgetting_by_task) / len(forgetting_by_task),
        "forgetting_first_task": float(forgetting_by_task[0]),
    }

    (out_dir / "eval").mkdir(parents=True, exist_ok=True)
    with (out_dir / "eval" / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(metrics)


if __name__ == "__main__":
    main()
