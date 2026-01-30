from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
import torch.nn.functional as F
import yaml

from router.router_model import RouterActorCritic


# -----------------------
# YAML instantiate helpers
# -----------------------
def import_from_target(target: str):
    mod_name, cls_name = target.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def instantiate(cfg_block: dict):
    cfg_block = dict(cfg_block)
    cls = import_from_target(cfg_block.pop("_target_"))
    return cls(**cfg_block)


# -----------------------
# Utils
# -----------------------
def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


@torch.no_grad()
def freeze_module(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad = False


def find_last_ckpt(ckpt_dir: Path) -> Path:
    cands = list(ckpt_dir.glob("task_*.pt"))
    if not cands:
        raise FileNotFoundError(f"Nessun checkpoint task_*.pt in {ckpt_dir}")

    def idx(p: Path) -> int:
        m = re.search(r"task_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    cands.sort(key=idx)
    if idx(cands[-1]) < 0:
        raise RuntimeError("Impossibile parsare indice task dai checkpoint.")
    return cands[-1]


def torch_load_safe(path: Path, device: torch.device):
    """
    Compatibile con PyTorch recenti che introducono weights_only e warning.
    Se weights_only non è supportato, fa fallback.
    Vedi doc torch.load. 
    """
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


# -----------------------
# Tasknet load (modulare)
# -----------------------
def load_tasknet_last(cfg: dict, device: torch.device, ckpt_dir: Path):
    ckpt_path = find_last_ckpt(ckpt_dir)
    ckpt = torch_load_safe(ckpt_path, device=device)

    # Costruisci la tasknet dal YAML (qualunque classe sia)
    tasknet = instantiate(cfg["task_net"]).to(device)

    # Carica pesi
    tasknet.load_state_dict(ckpt["task_net_state"])

    # Carica task_slices (serve per task-aware inference)
    task_slices = ckpt.get("task_net_task_slices", None)
    if task_slices is None:
        raise KeyError("Checkpoint non contiene task_net_task_slices (serve per task-aware inference).")
    tasknet.task_slices = task_slices

    return tasknet, ckpt_path


# -----------------------
# Forward helper: task-aware per batch mista
# -----------------------
def grouped_tasknet_forward(tasknet, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
    B = x.size(0)
    out = torch.empty((B, tasknet.num_classes), device=x.device, dtype=torch.float32)
    for t in task_ids.unique().tolist():
        idx = (task_ids == t).nonzero(as_tuple=True)[0]
        out[idx] = tasknet(x[idx], task_id=int(t))
    return out


def iterate_task_batches(tasks_data: List[dict], split: str):
    for t, td in enumerate(tasks_data):
        for x, y in td[split]:
            yield x, y, t


@torch.no_grad()
def eval_end_to_end(router, tasknet, tasks_data, device: torch.device, split: str):
    router.eval()
    tasknet.eval()

    total = 0
    correct_y = 0
    correct_t = 0

    for x, y, t_true in iterate_task_batches(tasks_data, split=split):
        x = x.to(device)
        y = y.to(device)
        t_true_vec = torch.full((x.size(0),), int(t_true), device=device, dtype=torch.long)

        pred_t = router.greedy_task(x)
        correct_t += (pred_t == t_true_vec).sum().item()

        logits = grouped_tasknet_forward(tasknet, x, pred_t)
        pred_y = logits.argmax(dim=-1)

        total += y.numel()
        correct_y += (pred_y == y).sum().item()

    return {
        "acc_end2end": correct_y / max(1, total),
        "acc_router": correct_t / max(1, total),
        "total": int(total),
    }


def get_task_costs(tasknet, device: torch.device) -> Optional[torch.Tensor]:
    """
    Modularità:
    - Se la tasknet implementa task_costs(), usiamo quella.
    - Altrimenti: niente penalità (None).
    """
    if hasattr(tasknet, "task_costs"):
        costs = tasknet.task_costs(device=device)
        if not torch.is_tensor(costs):
            costs = torch.tensor(costs, dtype=torch.float32, device=device)
        return costs.to(device)
    return None


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)

    # override CLI (così non tocchi YAML)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=None)  # default: training.batch_size
    ap.add_argument("--device", type=str, default=None)      # default: evaluation.device
    ap.add_argument("--reward", type=str, default="logprob", choices=["logprob", "acc"])
    ap.add_argument("--beta_cost", type=float, default=0.0)
    ap.add_argument("--entropy_coef", type=float, default=0.01)
    ap.add_argument("--ckpt_dir", type=str, default=None)

    # router arch
    ap.add_argument("--router_hidden", type=int, default=256)
    ap.add_argument("--router_emb", type=int, default=128)

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = ensure_dir(cfg["experiment"]["out_dir"])
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir is not None else (out_dir / "checkpoints")

    device_name = args.device if args.device is not None else cfg.get("evaluation", {}).get("device", "cuda")
    device = get_device(device_name)

    dataset_obj = instantiate(cfg["dataset"])
    num_tasks = int(cfg["dataset"]["num_tasks"])

    batch_size = args.batch_size if args.batch_size is not None else int(cfg["training"]["batch_size"])
    tasks_data = [dataset_obj.get_task(t, batch_size=batch_size) for t in range(num_tasks)]

    tasknet, used_ckpt = load_tasknet_last(cfg, device=device, ckpt_dir=ckpt_dir)
    freeze_module(tasknet)

    # Qui NON leggiamo più cfg["task_net"]["input_dim"] (MLP-only).
    # Usiamo una sezione router nel YAML.
    router_input_dim = int(cfg.get("router", {}).get("input_dim", 784))

    router = RouterActorCritic(
        num_tasks=num_tasks,
        input_dim=router_input_dim,
        hidden=int(args.router_hidden),
        emb=int(args.router_emb),
    ).to(device)

    opt = torch.optim.Adam(router.parameters(), lr=float(args.lr))

    for ep in range(1, args.epochs + 1):
        router.train()
        total_loss = 0.0
        total = 0

        # precompute costs una volta per epoca
        costs = get_task_costs(tasknet, device=device)

        for x, y, _t_true in iterate_task_batches(tasks_data, split="train"):
            x = x.to(device)
            y = y.to(device)

            logits_pi, v = router(x)
            dist = torch.distributions.Categorical(logits=logits_pi)
            a = dist.sample()
            logp = dist.log_prob(a)
            ent = dist.entropy().mean()

            logits = grouped_tasknet_forward(tasknet, x, a)

            if args.reward == "acc":
                r = (logits.argmax(dim=-1) == y).float()
            else:
                r = -F.cross_entropy(logits, y, reduction="none")

            if args.beta_cost > 0.0 and costs is not None:
                r = r - float(args.beta_cost) * costs[a]

            adv = r - v
            loss_actor = -(logp * adv.detach()).mean()
            loss_critic = 0.5 * (adv ** 2).mean()
            loss = loss_actor + loss_critic - float(args.entropy_coef) * ent

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)

        valm = eval_end_to_end(router, tasknet, tasks_data, device, split="val")
        print(
            f"[router] ep {ep:03d}/{args.epochs} loss={total_loss/max(1,total):.4f} "
            f"val_end2end={valm['acc_end2end']:.4f} val_router={valm['acc_router']:.4f}",
            flush=True,
        )

    testm = eval_end_to_end(router, tasknet, tasks_data, device, split="test")
    print(f"[router] test_end2end={testm['acc_end2end']:.4f} test_router={testm['acc_router']:.4f}")

    save_dir = ensure_dir(out_dir / "router")
    torch.save(
        {
            "router_state": router.state_dict(),
            "router_cfg": {"hidden": args.router_hidden, "emb": args.router_emb, "input_dim": router_input_dim},
            "tasknet_ckpt": str(used_ckpt),
            "yaml_cfg": cfg,
        },
        save_dir / "router.pt",
    )
    with (save_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(testm, f, indent=2)

    print(f"[router] saved to {save_dir}", flush=True)


if __name__ == "__main__":
    main()
