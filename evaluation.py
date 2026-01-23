# evaluation.py (print migliorati)
from __future__ import annotations

import argparse
import importlib
import json
import re
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml

from router.router_model import RouterActorCritic


# -----------------------
# Small pretty-print utils
# -----------------------
def _hr(char: str = "=", n: int = 78) -> str:
    return char * n


def _section(title: str):
    print("\n" + _hr())
    print(title)
    print(_hr(), flush=True)


def _kv(d: Dict[str, Any], indent: int = 2):
    pad = " " * indent
    keys = list(d.keys())
    if not keys:
        print(pad + "(empty)", flush=True)
        return
    w = max(len(k) for k in keys)
    for k in keys:
        print(f"{pad}{k:<{w}} : {d[k]}", flush=True)


def _fmt_acc(x: float) -> str:
    return f"{100.0 * float(x):.2f}%"


def _torch_load(path: Path, map_location, *, weights_only_preferred: bool = True):
    """
    Try to avoid the torch.load FutureWarning by using weights_only=True when available.
    If not supported or fails, fall back to standard torch.load with a clear message. [web:13]
    """
    if weights_only_preferred:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            # older PyTorch: no weights_only kwarg
            pass
        except Exception as e:
            print(
                "[WARN] torch.load(weights_only=True) failed; falling back to weights_only=False.\n"
                f"       Reason: {type(e).__name__}: {e}",
                flush=True,
            )

    # Fall back: might show FutureWarning in some PyTorch versions [web:13]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=FutureWarning)
        obj = torch.load(path, map_location=map_location)
        # If a FutureWarning was emitted, show a friendly one-liner
        if any(issubclass(ww.category, FutureWarning) for ww in w):
            print(
                "[INFO] Nota: PyTorch ha mostrato un FutureWarning su torch.load(weights_only=False). "
                "Non è un errore; è un avviso di sicurezza/compatibilità. "
                "Per checkpoint fidati (creati da te) puoi ignorarlo.",
                flush=True,
            )
    return obj


# -----------------------
# Core helpers
# -----------------------
def import_from_target(target: str):
    mod_name, cls_name = target.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def instantiate(cfg_block: dict):
    cfg_block = dict(cfg_block)
    cls = import_from_target(cfg_block.pop("_target_"))
    return cls(**cfg_block)


def get_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_last_ckpt(ckpt_dir: Path) -> Path:
    cands = list(ckpt_dir.glob("task_*.pt"))
    if not cands:
        raise FileNotFoundError(f"Nessun checkpoint task_*.pt in {ckpt_dir}")

    def idx(p: Path) -> int:
        m = re.search(r"task_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    cands.sort(key=idx)
    return cands[-1]


def load_tasknet_last(cfg: dict, device: torch.device, ckpt_dir: Path):
    ckpt_path = find_last_ckpt(ckpt_dir)
    ckpt = _torch_load(ckpt_path, map_location=device, weights_only_preferred=True)

    task_cfg = dict(cfg["task_net"])
    TaskCls = import_from_target(task_cfg.pop("_target_"))

    hidden_sizes = ckpt.get("task_net_hidden_sizes", task_cfg["hidden_sizes"])
    tasknet = TaskCls(
        input_dim=task_cfg["input_dim"],
        hidden_sizes=hidden_sizes,
        num_classes=task_cfg["num_classes"],
        action_spec=task_cfg["action_spec"],
    ).to(device)

    tasknet.load_state_dict(ckpt["task_net_state"])
    tasknet.task_slices = ckpt["task_net_task_slices"]
    return tasknet, ckpt_path


@torch.no_grad()
def grouped_tasknet_forward(tasknet, x: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
    B = x.size(0)
    out = torch.empty((B, tasknet.num_classes), device=x.device, dtype=torch.float32)
    for t in task_ids.unique().tolist():
        idx = (task_ids == t).nonzero(as_tuple=True)[0]
        out[idx] = tasknet(x[idx], task_id=int(t))
    return out


@torch.no_grad()
def eval_oracle(tasknet, tasks_data: List[dict], device: torch.device, split: str):
    tasknet.eval()
    total = 0
    correct = 0
    for t, td in enumerate(tasks_data):
        for x, y in td[split]:
            x = x.to(device)
            y = y.to(device)
            logits = tasknet(x, task_id=int(t))
            pred = logits.argmax(dim=-1)
            total += y.numel()
            correct += (pred == y).sum().item()
    return correct / max(1, total)


@torch.no_grad()
def eval_router(router, tasknet, tasks_data: List[dict], device: torch.device, split: str):
    router.eval()
    tasknet.eval()
    total = 0
    correct_y = 0
    correct_t = 0

    for t, td in enumerate(tasks_data):
        for x, y in td[split]:
            x = x.to(device)
            y = y.to(device)
            t_true = torch.full((x.size(0),), int(t), device=device, dtype=torch.long)

            pred_t = router.greedy_task(x)
            correct_t += (pred_t == t_true).sum().item()

            logits = grouped_tasknet_forward(tasknet, x, pred_t)
            pred_y = logits.argmax(dim=-1)

            total += y.numel()
            correct_y += (pred_y == y).sum().item()

    return {
        "acc_end2end": correct_y / max(1, total),
        "acc_router": correct_t / max(1, total),
        "total": int(total),
    }


@torch.no_grad()
def eval_tryall(tasknet, tasks_data: List[dict], device: torch.device, split: str):
    tasknet.eval()
    T = len(tasks_data)
    total = 0
    correct = 0

    for _t_true, td in enumerate(tasks_data):
        for x, y in td[split]:
            x = x.to(device)
            y = y.to(device)

            losses = []
            for t in range(T):
                logits_t = tasknet(x, task_id=int(t))
                ce = F.cross_entropy(logits_t, y, reduction="none")
                losses.append(ce.unsqueeze(1))
            L = torch.cat(losses, dim=1)  # (B,T)

            best_t = L.argmin(dim=1)
            best_logits = grouped_tasknet_forward(tasknet, x, best_t)
            pred = best_logits.argmax(dim=-1)

            total += y.numel()
            correct += (pred == y).sum().item()

    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--mode", type=str, default="both", choices=["oracle", "router", "both", "tryall"])
    ap.add_argument("--ckpt_dir", type=str, default=None)
    ap.add_argument("--router_ckpt", type=str, default=None)  # default: out_dir/router/router.pt
    args = ap.parse_args()

    t_global0 = time.perf_counter()

    _section("EVALUATION - SETUP")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["experiment"]["out_dir"])
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir is not None else (out_dir / "checkpoints")
    device = get_device(cfg.get("evaluation", {}).get("device", "cuda"))

    num_tasks = int(cfg["dataset"]["num_tasks"])
    batch_size = int(cfg.get("evaluation", {}).get("batch_size", 256))

    _kv(
        {
            "Config": str(Path(args.config).resolve()),
            "Mode": args.mode,
            "Split": args.split,
            "Device": str(device),
            "num_tasks": num_tasks,
            "batch_size": batch_size,
            "out_dir": str(out_dir),
            "ckpt_dir": str(ckpt_dir),
        }
    )

    print("\nLoading dataset...", flush=True)
    dataset_obj = instantiate(cfg["dataset"])
    tasks_data = [dataset_obj.get_task(t, batch_size=batch_size) for t in range(num_tasks)]
    print("Dataset loaded.", flush=True)

    print("\nLoading tasknet checkpoint (latest)...", flush=True)
    tasknet, tasknet_ckpt = load_tasknet_last(cfg, device=device, ckpt_dir=ckpt_dir)
    print(f"Tasknet loaded from: {tasknet_ckpt}", flush=True)

    results: Dict[str, Any] = {
        "mode": args.mode,
        "split": args.split,
        "device": str(device),
        "num_tasks": num_tasks,
        "batch_size": batch_size,
        "tasknet_ckpt": str(tasknet_ckpt),
    }

    # -----------------------
    # Run selected evaluations
    # -----------------------
    _section("EVALUATION - RUN")

    if args.mode in ("oracle", "both"):
        print("[RUN] Oracle (task-id noto): START", flush=True)
        t0 = time.perf_counter()
        oracle_acc = float(eval_oracle(tasknet, tasks_data, device, split=args.split))
        dt = time.perf_counter() - t0
        results["oracle_acc"] = oracle_acc
        results["oracle_time_sec"] = dt
        print(f"[RUN] Oracle: DONE in {dt:.2f}s | oracle_acc={_fmt_acc(oracle_acc)}", flush=True)

    if args.mode in ("router", "both"):
        router_ckpt_path = (
            Path(args.router_ckpt) if args.router_ckpt is not None else (out_dir / "router" / "router.pt")
        )

        print(f"\n[RUN] Router end-to-end: loading router from {router_ckpt_path}", flush=True)
        rckpt = _torch_load(router_ckpt_path, map_location=device, weights_only_preferred=True)

        rcfg = rckpt.get("router_cfg", {"hidden": 256, "emb": 128})
        router = RouterActorCritic(
            num_tasks=num_tasks,
            input_dim=int(cfg["task_net"]["input_dim"]),
            hidden=int(rcfg["hidden"]),
            emb=int(rcfg["emb"]),
        ).to(device)
        router.load_state_dict(rckpt["router_state"])

        print("[RUN] Router end-to-end: START", flush=True)
        t0 = time.perf_counter()
        m = eval_router(router, tasknet, tasks_data, device, split=args.split)
        dt = time.perf_counter() - t0

        results["router_ckpt"] = str(router_ckpt_path)
        results.update(m)
        results["router_time_sec"] = dt

        print(
            f"[RUN] Router end-to-end: DONE in {dt:.2f}s | "
            f"acc_end2end={_fmt_acc(results['acc_end2end'])} | acc_router={_fmt_acc(results['acc_router'])}",
            flush=True,
        )

    if args.mode == "tryall":
        print("[RUN] Try-all baseline (usa y per scegliere il task): START", flush=True)
        t0 = time.perf_counter()
        tryall_acc = float(eval_tryall(tasknet, tasks_data, device, split=args.split))
        dt = time.perf_counter() - t0
        results["tryall_acc"] = tryall_acc
        results["tryall_time_sec"] = dt
        print(f"[RUN] Try-all: DONE in {dt:.2f}s | tryall_acc={_fmt_acc(tryall_acc)}", flush=True)

    # -----------------------
    # Save + final print
    # -----------------------
    results["total_time_sec"] = time.perf_counter() - t_global0

    _section("EVALUATION - RESULTS (SUMMARY)")
    # Stampa “umana”
    human = {
        "oracle_acc": _fmt_acc(results["oracle_acc"]) if "oracle_acc" in results else None,
        "acc_end2end": _fmt_acc(results["acc_end2end"]) if "acc_end2end" in results else None,
        "acc_router": _fmt_acc(results["acc_router"]) if "acc_router" in results else None,
        "tryall_acc": _fmt_acc(results["tryall_acc"]) if "tryall_acc" in results else None,
        "total_time_sec": f"{results['total_time_sec']:.2f}",
    }
    # Rimuove i None dalla vista summary
    human = {k: v for k, v in human.items() if v is not None}
    _kv(human)

    _section("EVALUATION - RESULTS (FULL JSON)")
    print(json.dumps(results, indent=2), flush=True)

    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_path = eval_dir / "metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _section("EVALUATION - SAVED")
    print(f"Saved metrics to: {out_path}", flush=True)
    print("Evaluation finished successfully.", flush=True)


if __name__ == "__main__":
    main()
