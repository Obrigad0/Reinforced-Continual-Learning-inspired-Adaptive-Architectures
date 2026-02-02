import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import yaml

import matplotlib
matplotlib.use("Agg")
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

def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _get_any(d: dict, keys: List[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def _as_list(x) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return list(x)


# -----------------------
# Import/instantiate
# -----------------------
def import_from_target(target: str):
    # supporta "pkg.mod:Class" o "pkg.mod.Class"
    if ":" in target:
        mod_name, cls_name = target.split(":")
    else:
        mod_name, cls_name = target.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)

def instantiate(cfg_block: dict):
    cfg_block = dict(cfg_block)
    t = _get_any(cfg_block, ["_target_", "target"], None)
    if t is None:
        raise KeyError("Missing '_target_' or 'target' in cfg block.")
    cfg_block.pop("_target_", None)
    cfg_block.pop("target", None)
    cls = import_from_target(t)
    return cls(**cfg_block)


# -----------------------
# Repo hooks
# -----------------------
def get_action_spec(task_net) -> List[int]:
    if hasattr(task_net, "actionspec"):
        a = getattr(task_net, "actionspec")
        return _as_list(a() if callable(a) else a)
    if hasattr(task_net, "action_spec"):
        a = getattr(task_net, "action_spec")
        return _as_list(a() if callable(a) else a)
    raise AttributeError("Task net has no actionspec/action_spec.")

def expanded_copy(task_net, actions):
    if hasattr(task_net, "expandedcopy"):
        return task_net.expandedcopy(actions)
    if hasattr(task_net, "expanded_copy"):
        return task_net.expanded_copy(actions)
    raise AttributeError("Task net has no expandedcopy/expanded_copy.")

def maybe_apply_freeze_policy(child, parent) -> bool:
    if hasattr(child, "applyfreezepolicy"):
        fn = getattr(child, "applyfreezepolicy")
        try:
            return bool(fn(parent=parent))
        except TypeError:
            return bool(fn(parent))
    if hasattr(child, "apply_freeze_policy"):
        fn = getattr(child, "apply_freeze_policy")
        try:
            return bool(fn(parent=parent))
        except TypeError:
            return bool(fn(parent))
    return False

def compute_actions_complexity(task_net, actions) -> float:
    if hasattr(task_net, "actionscomplexity"):
        return float(task_net.actionscomplexity(actions))
    if hasattr(task_net, "actions_complexity"):
        return float(task_net.actions_complexity(actions))
    return float(sum(actions))

def dataset_get_task(dataset_obj, t: int, batch_size: int):
    if hasattr(dataset_obj, "gettask"):
        return dataset_obj.gettask(t, batchsize=batch_size)
    if hasattr(dataset_obj, "get_task"):
        return dataset_obj.get_task(t, batch_size=batch_size)
    raise AttributeError("Dataset has no gettask/get_task.")


# -----------------------
# Controller cfg (LSTM only)
# -----------------------
def normalize_lstm_controller_cfg(controller_cfg: dict, spec: List[int]) -> dict:
    """
    LSTMController nel tuo repo: __init__(hidden_size, num_layers, action_spec) [file:51]
    Supporta alias: hiddensize/numlayers/actionspec.
    """
    c = dict(controller_cfg)

    # alias -> canonical
    if "hiddensize" in c and "hidden_size" not in c:
        c["hidden_size"] = c.pop("hiddensize")
    if "numlayers" in c and "num_layers" not in c:
        c["num_layers"] = c.pop("numlayers")
    if "actionspec" in c and "action_spec" not in c:
        c["action_spec"] = c.pop("actionspec")

    # default
    if "action_spec" not in c:
        c["action_spec"] = list(spec)

    # cleanup: non lasciare nomi sbagliati
    c.pop("hiddensize", None)
    c.pop("numlayers", None)
    c.pop("actionspec", None)

    return c


# -----------------------
# Supervised train/eval
# -----------------------
@torch.no_grad()
def eval_accuracy(model, loader, device: torch.device, task_id: Optional[int] = None) -> float:
    model.eval()
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if task_id is not None:
            try:
                logits = model(x, taskid=task_id)
            except TypeError:
                logits = model(x, task_id)
        else:
            logits = model(x)

        pred = logits.argmax(dim=1)
        total += y.numel()
        correct += (pred == y).sum().item()
    return correct / max(1, total)

def train_supervised(model, loader, device, lr: float, epochs: int, grad_clip: Optional[float], log: bool = False):
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
            print(f"  [train] epoch {ep:03d}/{epochs} loss={loss_sum/max(1,total):.4f} acc={correct/max(1,total):.4f}", flush=True)


# -----------------------
# Value net state (1D only)
# -----------------------
def make_state_1d(device: torch.device, t: int, num_tasks: int, complexity: float, cmax: float) -> torch.Tensor:
    denom = max(1, num_tasks - 1)
    t_norm = float(t) / float(denom)
    c_norm = float(complexity) / float(max(1e-8, cmax))
    s = 0.5 * (t_norm + c_norm)
    return torch.tensor([[s]], device=device, dtype=torch.float32)  # [1,1]


# -----------------------
# RCL step (paper-style)
# -----------------------
def rcl_expand_and_train_one_task(
    *,
    t: int,
    num_tasks: int,
    task_net,
    task_data: dict,
    controller,
    value_net,
    device: torch.device,
    trials: int,
    alpha: float,
    cmax: float,
    epochs_task: int,
    lr_task: float,
    grad_clip: Optional[float],
    opt_c,
    opt_v,
    trial_log_path: Path,
) -> Dict[str, Any]:
    """
    Paper-style: sample N architectures, train each child, compute reward,
    update controller with REINFORCE + baseline V(s) and update value net with MSE. [file:1]
    """
    controller.train()
    value_net.train()

    logp_list: List[torch.Tensor] = []
    reward_list: List[float] = []
    complexity_list: List[float] = []

    best_reward = -1e9
    best_actions = None
    best_model_state = None
    best_val_acc = None
    best_test_acc = None
    best_complexity = None

    t0 = time.time()

    for k in range(1, trials + 1):
        actions, logp_sum = controller.sample(device=device)

        child = expanded_copy(task_net, actions).to(device)
        masked = maybe_apply_freeze_policy(child=child, parent=task_net)

        train_supervised(child, task_data["train"], device=device, lr=lr_task, epochs=epochs_task, grad_clip=grad_clip, log=(k <= 2))

        val_acc = eval_accuracy(child, task_data["val"], device)
        complexity = compute_actions_complexity(task_net, actions)
        reward = float(val_acc) - alpha * (float(complexity) / float(max(1e-8, cmax)))

        logp_list.append(logp_sum.view(()))
        reward_list.append(float(reward))
        complexity_list.append(float(complexity))

        append_jsonl({
            "task": int(t),
            "trial": int(k),
            "reward": float(reward),
            "val_acc": float(val_acc),
            "complexity": float(complexity),
            "actions": list(map(int, actions)),
            "masked": int(masked),
            "alpha": float(alpha),
            "logp_sum": float(logp_sum.detach().cpu().item()),
            "algo": "rcl_lstm_reinforce",
        }, trial_log_path)

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

        if improved or k == 1 or k == trials or (k % 5 == 0):
            elapsed = time.time() - t0
            tag = "BEST" if improved else "info"
            print(
                f"[task {t}] trial {k:04d}/{trials} {tag} "
                f"reward={reward:.4f} val={val_acc:.4f} cx={complexity:.2f} "
                f"best_reward={best_reward:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )
            if improved:
                print(f"[task {t}] best_actions={best_actions}", flush=True)

    if best_actions is None or best_model_state is None:
        raise RuntimeError(f"[task {t}] No best model selected.")

    # --- REINFORCE + baseline
    rewards_t = torch.tensor(reward_list, device=device, dtype=torch.float32)  # [N]
    logp_t = torch.stack(logp_list).to(device).view(-1)  # [N]

    states = torch.cat(
        [make_state_1d(device, t, num_tasks, cx, cmax) for cx in complexity_list],
        dim=0,
    )  # [N,1]

    v = value_net(states).view(-1)  # [N]
    adv = rewards_t - v.detach()

    loss_actor = -(logp_t * adv).mean()

    opt_c.zero_grad()
    loss_actor.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(controller.parameters(), float(grad_clip))
    opt_c.step()

    loss_v = 0.5 * (v - rewards_t).pow(2).mean()

    opt_v.zero_grad()
    loss_v.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), float(grad_clip))
    opt_v.step()

    print(f"[task {t}] update: loss_actor={float(loss_actor.item()):.4f} loss_v={float(loss_v.item()):.4f}", flush=True)

    return dict(
        best_reward=best_reward,
        best_actions=best_actions,
        best_val_acc=best_val_acc,
        best_test_acc=best_test_acc,
        best_complexity=best_complexity,
        best_model_state=best_model_state,
        loss_actor=float(loss_actor.detach().cpu().item()),
        loss_v=float(loss_v.detach().cpu().item()),
    )


# -----------------------
# Plotting
# -----------------------
def _savefig(fig, path: Path):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_curves(out_dir: Path, first_task_curve: List[float], avg_acc_curve: List[float]):
    plots_dir = ensure_dir(out_dir / "plots")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(len(first_task_curve))), first_task_curve, marker="o")
    ax.set_xlabel("Task index t (after finishing task t)")
    ax.set_ylabel("Test accuracy on task 0")
    ax.set_title("First-task accuracy vs time")
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, plots_dir / "first_task_acc.png")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(len(avg_acc_curve))), avg_acc_curve, marker="o")
    ax.set_xlabel("Task index t (after finishing task t)")
    ax.set_ylabel("Average test accuracy (tasks 0..t)")
    ax.set_title("Average accuracy vs time")
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, plots_dir / "avg_acc.png")

def plot_all_children_acc_vs_complexity(trial_log_path: Path, out_path: Path):
    xs, ys, labels = [], [], []
    if not trial_log_path.exists():
        return

    with trial_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cx = obj.get("complexity", None)
            va = obj.get("val_acc", None)
            t = obj.get("task", None)
            tr = obj.get("trial", None)
            if cx is None or va is None or t is None or tr is None:
                continue
            xs.append(float(cx))
            ys.append(float(va))
            labels.append(f"{int(t)}-{int(tr)}")  # task-trial

    if not xs:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys, s=18)

    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), fontsize=7, alpha=0.85)

    ax.set_xlabel("Complexity (proxy)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("All children: accuracy vs complexity (labels = task-trial)")
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, out_path)

def plot_task_children_acc_vs_complexity(
    trial_log_path: Path,
    out_path: Path,
    task_id: int,
    best_point: Optional[Dict[str, float]] = None,
):
    xs, ys, labels = [], [], []
    if not trial_log_path.exists():
        return

    with trial_log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if int(obj.get("task", -1)) != int(task_id):
                continue

            cx = obj.get("complexity", None)
            va = obj.get("val_acc", None)
            tr = obj.get("trial", None)
            if cx is None or va is None or tr is None:
                continue

            xs.append(float(cx))
            ys.append(float(va))
            labels.append(int(tr))

    if not xs:
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # tutti i child del task (blu)
    ax.scatter(xs, ys, s=22)
    for x, y, tr in zip(xs, ys, labels):
        ax.annotate(str(tr), (x, y), fontsize=8, alpha=0.9)

    # BEST scelto (rosso)
    if best_point is not None:
        bx = best_point.get("complexity", None)
        by = best_point.get("val_acc", None)
        if bx is not None and by is not None:
            ax.scatter([float(bx)], [float(by)], color="red", s=80, zorder=5)

    ax.set_xlabel("Complexity (proxy)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(f"Task {int(task_id)}: children accuracy vs complexity (labels = trial id)")
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, out_path)


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
    print(f"setup device={device}", flush=True)
    print(f"setup outdir={out_dir}", flush=True)

    dataset_obj = instantiate(cfg["dataset"])
    task_net = instantiate(cfg["task_net"]).to(device)

    # controller (LSTM only)
    controller_cfg = dict(cfg["controller"])
    controller_target = _get_any(controller_cfg, ["_target_", "target"], None)
    if controller_target is None:
        raise KeyError("controller must have 'target' or '_target_'")
    controller_cfg.pop("_target_", None)
    controller_cfg.pop("target", None)
    ControllerCls = import_from_target(controller_target)

    spec = get_action_spec(task_net)
    controller_cfg = normalize_lstm_controller_cfg(controller_cfg, spec)
    controller = ControllerCls(**controller_cfg).to(device)

    # value net (MLPValueNet 1D)
    value_net = instantiate(cfg["value_net"]).to(device)

    num_tasks = int(cfg["dataset"]["num_tasks"])
    batch_size = int(cfg["training"]["batch_size"])
    epochs_task = int(cfg["training"]["epochs_task"])
    lr_task = float(cfg["training"]["lr_task"])
    grad_clip = cfg["training"].get("grad_clip", None)

    trials = int(cfg["training"]["controller_trials"])
    alpha = float(cfg["rcl"]["reward_alpha"])

    opt_c = torch.optim.Adam(controller.parameters(), lr=float(cfg["training"]["lr_controller"]))
    opt_v = torch.optim.Adam(value_net.parameters(), lr=float(cfg["training"]["lr_value"]))

    # cmax coerente con azioni in [0..n-1]
    cmax = float(sum(max(0, int(n) - 1) for n in spec))

    tasks_data_cache = [dataset_get_task(dataset_obj, j, batch_size=batch_size) for j in range(num_tasks)]

    train_stats: List[dict] = []
    first_task_curve_online: List[float] = []
    avg_acc_curve: List[float] = []
    acc_matrix: List[List[Optional[float]]] = []

    trial_log_path = out_dir / "trial_log.jsonl"

    task0_eval_data = None

    for t in range(num_tasks):
        print("\n" + "=" * 80, flush=True)
        print(f"[task {t}] start", flush=True)

        task_data = tasks_data_cache[t]
        if task0_eval_data is None:
            task0_eval_data = tasks_data_cache[0]

        if t == 0:
            print(f"[task {t}] base train: epochs={epochs_task} lr={lr_task}", flush=True)
            train_supervised(task_net, task_data["train"], device=device, lr=lr_task, epochs=epochs_task, grad_clip=grad_clip, log=True)
            val_acc = eval_accuracy(task_net, task_data["val"], device)
            test_acc = eval_accuracy(task_net, task_data["test"], device)
            print(f"[task {t}] base done: val={val_acc:.4f} test={test_acc:.4f}", flush=True)
            train_stats.append({"task": t, "val_acc": float(val_acc), "test_acc": float(test_acc), "note": "base_train"})

            if hasattr(task_net, "registertaskslice"):
                task_net.registertaskslice()
            elif hasattr(task_net, "register_task_slice"):
                task_net.register_task_slice()
        else:
            print(f"[task {t}] RCL search: trials={trials} alpha={alpha}", flush=True)

            res = rcl_expand_and_train_one_task(
                t=t,
                num_tasks=num_tasks,
                task_net=task_net,
                task_data=task_data,
                controller=controller,
                value_net=value_net,
                device=device,
                trials=trials,
                alpha=alpha,
                cmax=cmax,
                epochs_task=epochs_task,
                lr_task=lr_task,
                grad_clip=grad_clip,
                opt_c=opt_c,
                opt_v=opt_v,
                trial_log_path=trial_log_path,
            )

            task_net = expanded_copy(task_net, res["best_actions"]).to(device)
            task_net.load_state_dict(res["best_model_state"])

            if hasattr(task_net, "registertaskslice"):
                task_net.registertaskslice()
            elif hasattr(task_net, "register_task_slice"):
                task_net.register_task_slice()

            train_stats.append({
                "task": int(t),
                "best_reward": float(res["best_reward"]),
                "best_actions": list(map(int, res["best_actions"])),
                "best_val_acc": float(res["best_val_acc"]),
                "best_test_acc": float(res["best_test_acc"]),
                "best_complexity": float(res["best_complexity"]),
                "loss_actor": float(res["loss_actor"]),
                "loss_v": float(res["loss_v"]),
                "note": "rcl_lstm_reinforce",
            })

        acc0 = float(eval_accuracy(task_net, task0_eval_data["test"], device, task_id=0))
        first_task_curve_online.append(acc0)
        print(f"[task {t}] online first-task acc={acc0:.4f}", flush=True)

        row: List[Optional[float]] = [None] * num_tasks
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
                "task": int(t),
                "task_net_state": task_net.state_dict(),
                "controller_state": controller.state_dict(),
                "value_net_state": value_net.state_dict(),
                "cfg": cfg,
                "task_net_task_slices": getattr(task_net, "taskslices", None),
            },
        )

        save_json(
            {"train_stats": train_stats, "first_task_curve_online": first_task_curve_online, "avg_acc_curve": avg_acc_curve},
            out_dir / "train_stats.json",
        )
        save_json({"acc_matrix": acc_matrix}, out_dir / "acc_matrix.json")

        # --- plots base
        plot_curves(out_dir, first_task_curve_online, avg_acc_curve)

        # --- NEW plots richiesti
        plots_dir = ensure_dir(out_dir / "plots")
        plot_all_children_acc_vs_complexity(trial_log_path, plots_dir / "all_children_acc_vs_complexity.png")

        if t > 0:
            best_point = None
            for s in train_stats:
                if int(s.get("task", -999)) == int(t) and "best_val_acc" in s and "best_complexity" in s:
                    best_point = {"val_acc": float(s["best_val_acc"]), "complexity": float(s["best_complexity"])}
                    break

            plot_task_children_acc_vs_complexity(
                trial_log_path,
                plots_dir / f"task_{t:03d}_children_acc_vs_complexity.png",
                task_id=int(t),
                best_point=best_point,
            )

        print(f"[task {t}] saved checkpoint + logs + plots", flush=True)

    print("Done. Results saved to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
