import argparse
import importlib
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


@torch.no_grad()
def eval_accuracy(model, loader, device, task_id=None):
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if task_id is None:
            logits = model(x)
        else:
            # Se il modello non supporta task_id, fallirà: meglio esplicito
            logits = model(x, task_id=int(task_id))
        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return correct / max(1, total)


def get_device(name: str):
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--device", type=str, default=None, help="Override config evaluation.device")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = get_device(args.device if args.device is not None else cfg["evaluation"]["device"])
    batch_size = args.batch_size if args.batch_size is not None else int(cfg["evaluation"]["batch_size"])

    ckpt = torch.load(args.ckpt, map_location=device)

    # dataset
    dataset_obj = instantiate(cfg["dataset"])
    num_tasks = int(cfg["dataset"]["num_tasks"])
    tasks_data = [dataset_obj.get_task(j, batch_size=batch_size) for j in range(num_tasks)]

    # model class
    task_cfg = dict(cfg["task_net"])
    task_target = task_cfg.pop("_target_")
    TaskCls = import_from_target(task_target)

    hidden_sizes = ckpt.get("task_net_hidden_sizes", task_cfg.get("hidden_sizes"))

    model = TaskCls(
        input_dim=task_cfg["input_dim"],
        hidden_sizes=hidden_sizes,
        num_classes=task_cfg["num_classes"],
        action_spec=task_cfg["action_spec"],
    ).to(device)

    model.load_state_dict(ckpt["task_net_state"])

    if "task_net_task_slices" in ckpt and ckpt["task_net_task_slices"] is not None:
        model.task_slices = ckpt["task_net_task_slices"]

    print(f"Loaded ckpt={args.ckpt}")
    print(f"device={device} batch_size={batch_size} num_tasks={num_tasks}")
    print("-" * 72)
    print(f"{'task':>4} | {'task-aware':>10} | {'full':>10} | {'delta(full-aware)':>16}")
    print("-" * 72)

    for j in range(num_tasks):
        acc_aware = eval_accuracy(model, tasks_data[j]["test"], device, task_id=j)
        acc_full = eval_accuracy(model, tasks_data[j]["test"], device, task_id=None)
        print(f"{j:>4} | {acc_aware:>10.4f} | {acc_full:>10.4f} | {(acc_full-acc_aware):>16.4f}")

    print("-" * 72)
    print("Note: se 'full' è molto peggiore di 'task-aware' sui task vecchi,")
    print("vuol dire che il routing task-aware sta davvero prevenendo semantic drift.")
    print("Se invece full ~= task-aware ovunque, allora l'espansione sta interferendo poco (o per nulla).")


if __name__ == "__main__":
    main()
