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
    if "_target_" in cfg_block:
        cfg_block.pop("_target_")
    else:
        cfg_block.pop("target")
    cls = import_from_target(t)
    return cls(**cfg_block)

# -----------------------
# Task/model hooks (compat repo)
# -----------------------
def get_action_spec(task_net) -> List[int]:
    # ExpandableVGG nel tuo repo: actionspec()
    if hasattr(task_net, "actionspec"):
        a = getattr(task_net, "actionspec")
        return _as_list(a() if callable(a) else a)
    # fallback: action_spec (metodo/attributo)
    if hasattr(task_net, "action_spec"):
        # print("aaaaaaaaaaa")
        a = getattr(task_net, "action_spec")
        return _as_list(a() if callable(a) else a)
    raise AttributeError("Task net has no actionspec/action_spec.")

def expanded_copy(task_net, actions):
    # ExpandableVGG: expandedcopy(actions)
    if hasattr(task_net, "expandedcopy"):
        return task_net.expandedcopy(actions)
    if hasattr(task_net, "expanded_copy"):
        return task_net.expanded_copy(actions)
    raise AttributeError("Task net has no expandedcopy/expanded_copy.")

def maybe_apply_freeze_policy(child, parent) -> bool:
    # ExpandableVGG: applyfreezepolicy(parent)
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
    # ExpandableVGG: actionscomplexity(actions)
    if hasattr(task_net, "actionscomplexity"):
        return float(task_net.actionscomplexity(actions))
    if hasattr(task_net, "actions_complexity"):
        return float(task_net.actions_complexity(actions))
    return float(sum(actions))

def dataset_get_task(dataset_obj, t: int, batch_size: int):
    # PermutedMNISTDataset nel repo: gettask(taskid, batchsize)
    if hasattr(dataset_obj, "gettask"):
        return dataset_obj.gettask(t, batchsize=batch_size)
    if hasattr(dataset_obj, "get_task"):
        return dataset_obj.get_task(t, batch_size=batch_size)
    raise AttributeError("Dataset has no gettask/get_task.")

# -----------------------
# Controller cfg normalization (IMPORTANT)
# -----------------------
def normalize_controller_cfg(controller_cfg: dict, spec: List[int]) -> dict:
    """
    PPOLSTMController accetta actionspec (non action_spec).
    Questa funzione:
    - rinomina action_spec -> actionspec se presente
    - se manca, inserisce actionspec=spec
    """
    controller_cfg = dict(controller_cfg)

    if "action_spec" in controller_cfg and "actionspec" not in controller_cfg:
        controller_cfg["actionspec"] = controller_cfg.pop("action_spec")

    if "actionspec" not in controller_cfg:
        controller_cfg["actionspec"] = list(spec)

    return controller_cfg

# -----------------------
# Eval / Train supervised
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
                try:
                    logits = model(x, task_id=task_id)
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
# Value nets
# -----------------------
def value_net_scalar_tc(value_net, device: torch.device, t: int, num_tasks: int, complexity: float, cmax: float) -> torch.Tensor:
    denom = max(1, num_tasks - 1)
    t_norm = float(t) / float(denom)
    c_norm = float(complexity) / float(max(1e-8, cmax))
    x2 = torch.tensor([t_norm, c_norm], device=device, dtype=torch.float32)
    v = value_net(x2)
    if not torch.is_tensor(v):
        v = torch.tensor(float(v), device=device)
    return v.reshape(())

# -----------------------
# PPO trainer (critic V(t,c) hard)
# -----------------------
# -----------------------
# PPO trainer (critic V(t,c) hard)
# -----------------------
class ControllerTrainerPPO:
    def __init__(self, device: torch.device):
        self.device = device

    def require(self, controller):
        if not hasattr(controller, "log_prob"):
            raise TypeError("PPO controller must implement log_prob(actions, device).")

    def run(self, **kw) -> Dict[str, Any]:
        device = self.device
        t = kw["t"]
        num_tasks = kw["num_tasks"]
        controller = kw["controller"]
        value_net = kw["value_net"]
        task_net = kw["task_net"]
        task_data = kw["task_data"]
        trials = kw["trials"]
        alpha = kw["alpha"]
        epochs_task = kw["epochs_task"]
        lr_task = kw["lr_task"]
        grad_clip = kw["grad_clip"]
        opt_c = kw["opt_c"]
        opt_v = kw["opt_v"]
        print_every = kw["print_every"]
        print_first = kw["print_first"]
        trial_log_path = kw["trial_log_path"]

        # NEW: path per log PPO scalari
        ppo_log_path = kw.get("ppo_log_path", None)

        ppo_clip_eps = float(kw["ppo_clip_eps"])
        ppo_epochs = int(kw["ppo_epochs"])
        ppo_entropycoef = float(kw["ppo_entropycoef"])
        ppo_advnorm = bool(kw["ppo_advnorm"])

        cmax = float(kw["cmax"])

        use_ema = bool(kw["use_ema"])
        ema_beta = float(kw["ema_beta"])
        ema_mix = float(kw["ema_mix"])
        ema_b = kw["ema_b"]
        ema_init = kw["ema_init"]

        actions_batch: List[List[int]] = []
        logp_old_batch: List[float] = []
        reward_batch: List[float] = []
        baseline_batch: List[float] = []
        complexity_batch: List[float] = []

        best_reward = -1e9
        best_model_state = None
        best_actions = None
        best_val_acc = None
        best_test_acc = None
        best_complexity = None

        t0 = time.time()

        # -----------------------
        # NEW: parent baseline (val acc del modello corrente sul task t)
        # Calcolata una sola volta per task, così il reward misura "miglioramento"
        # -----------------------
        parent_val_acc = eval_accuracy(task_net, task_data["val"], device)  # [file:7]

        for k in range(1, trials + 1):
            actions, logp_old = controller.sample(device=device)

            child = expanded_copy(task_net, actions).to(device)
            masked = maybe_apply_freeze_policy(child=child, parent=task_net)

            train_supervised(
                child,
                task_data["train"],
                device=device,
                lr=lr_task,
                epochs=epochs_task,
                grad_clip=grad_clip,
                log=(k <= print_first),
            )

            val_acc = eval_accuracy(child, task_data["val"], device)
            complexity = compute_actions_complexity(task_net, actions)

            # -----------------------
            # NEW: reward = improvement over parent - complexity penalty
            # -----------------------
            delta = float(val_acc) - float(parent_val_acc)
            reward = float(delta) - alpha * (float(complexity) / float(max(1e-8, cmax)))  # [file:7]

            v = value_net_scalar_tc(
                value_net,
                device=device,
                t=t,
                num_tasks=num_tasks,
                complexity=float(complexity),
                cmax=cmax,
            )
            v_val = float(v.detach().cpu().item())

            baseline_val = v_val
            if use_ema:
                if not ema_init[t]:
                    ema_b[t] = float(reward)
                    ema_init[t] = True
                else:
                    ema_b[t] = (1.0 - ema_beta) * float(ema_b[t]) + ema_beta * float(reward)
                baseline_val = (1.0 - ema_mix) * v_val + ema_mix * float(ema_b[t])

            actions_batch.append(list(actions))
            logp_old_batch.append(float(logp_old.detach().cpu().item()))
            reward_batch.append(float(reward))
            baseline_batch.append(float(baseline_val))
            complexity_batch.append(float(complexity))

            append_jsonl({
                "task": int(t),
                "trial": int(k),

                # NEW: log parent + delta
                "parent_val_acc": float(parent_val_acc),
                "delta": float(delta),

                "reward": float(reward),
                "val_acc": float(val_acc),
                "complexity": float(complexity),
                "actions": list(map(int, actions)),
                "masked": int(masked),
                "alpha": float(alpha),
                "logp_old": float(logp_old_batch[-1]),
                "baseline": float(baseline_batch[-1]),
                "v": float(v_val),
                "ema_used": int(use_ema),
                "ema_mix": float(ema_mix),
                "algo": "ppo",
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

            if improved or (k <= print_first) or (k % print_every == 0) or (k == trials):
                elapsed = time.time() - t0
                tag = "BEST" if improved else "info"
                print(
                    f"[task {t}] trial {k:04d}/{trials} {tag} "
                    f"reward={reward:.4f} (delta={delta:.4f} parent_val={parent_val_acc:.4f}) "
                    f"val={val_acc:.4f} cx={complexity:.2f} masked={int(masked)} "
                    f"best_reward={best_reward:.4f} elapsed={elapsed:.1f}s",
                    flush=True,
                )
                if improved:
                    print(f"[task {t}] best_actions={best_actions}", flush=True)

        rewards_t = torch.tensor(reward_batch, device=device, dtype=torch.float32)
        baselines_t = torch.tensor(baseline_batch, device=device, dtype=torch.float32)
        adv_t = rewards_t - baselines_t
        adv_raw_t = adv_t.clone()
        if ppo_advnorm:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        logp_old_t = torch.tensor(logp_old_batch, device=device, dtype=torch.float32)

        denom = max(1, num_tasks - 1)
        t_norm = float(t) / float(denom)
        c_norms = torch.tensor(
            [cx / float(max(1e-8, cmax)) for cx in complexity_batch],
            device=device,
            dtype=torch.float32,
        ).view(-1, 1)
        t_norms = torch.full_like(c_norms, float(t_norm))
        x_v = torch.cat([t_norms, c_norms], dim=1)

        with torch.no_grad():
            testv = value_net(x_v)
        if not torch.is_tensor(testv):
            raise TypeError("PPO requires value_net(x_v) to return a Tensor.")
        if testv.numel() != rewards_t.numel():
            raise ValueError(f"PPO expects value_net(x_v) to return B scalars, got shape {tuple(testv.shape)}")

        controller.train()
        value_net.train()

        for ep in range(1, ppo_epochs + 1):
            logp_new = torch.stack([controller.log_prob(acts, device) for acts in actions_batch]).view(-1)
            ratio = torch.exp(logp_new - logp_old_t)

            obj1 = ratio * adv_t
            obj2 = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * adv_t
            loss_actor = -torch.min(obj1, obj2).mean()

            ent_mean = None
            if ppo_entropycoef != 0.0 and hasattr(controller, "entropy"):
                ent = torch.stack([controller.entropy(acts, device) for acts in actions_batch]).view(-1)
                ent_mean = float(ent.mean().detach().cpu().item())
                loss_actor = loss_actor - float(ppo_entropycoef) * ent.mean()

            opt_c.zero_grad()
            loss_actor.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(controller.parameters(), float(grad_clip))
            opt_c.step()

            vpred = value_net(x_v).view(-1)
            loss_v = 0.5 * (vpred - rewards_t).pow(2).mean()

            opt_v.zero_grad()
            loss_v.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), float(grad_clip))
            opt_v.step()

            if ppo_log_path is not None:
                rec = {
                    "task": int(t),
                    "ppo_epoch": int(ep),
                    "loss_actor": float(loss_actor.detach().cpu().item()),
                    "loss_v": float(loss_v.detach().cpu().item()),
                    "ratio_mean": float(ratio.mean().detach().cpu().item()),
                    "ratio_min": float(ratio.min().detach().cpu().item()),
                    "ratio_max": float(ratio.max().detach().cpu().item()),
                    "reward_mean": float(rewards_t.mean().detach().cpu().item()),
                    "reward_std": float(rewards_t.std(unbiased=False).detach().cpu().item()),
                    "baseline_mean": float(baselines_t.mean().detach().cpu().item()),
                    "baseline_std": float(baselines_t.std(unbiased=False).detach().cpu().item()),
                    "adv_raw_mean": float(adv_raw_t.mean().detach().cpu().item()),
                    "adv_raw_std": float(adv_raw_t.std(unbiased=False).detach().cpu().item()),
                    "adv_norm_mean": float(adv_t.mean().detach().cpu().item()),
                    "adv_norm_std": float(adv_t.std(unbiased=False).detach().cpu().item()),
                    "entropy_mean": ent_mean,
                    "clip_eps": float(ppo_clip_eps),
                    "entropycoef": float(ppo_entropycoef),
                    "advnorm": int(ppo_advnorm),
                    "algo": "ppo",
                }
                append_jsonl(rec, ppo_log_path)

            if ep == 1 or ep == ppo_epochs:
                print(
                    f"  [ppo] ep {ep:02d}/{ppo_epochs} loss_actor={float(loss_actor.item()):.4f} "
                    f"loss_v={float(loss_v.item()):.4f} ratio_mean={float(ratio.mean().detach().cpu().item()):.3f} critic2d=1",
                    flush=True,
                )

        return dict(
            algo="ppo",
            best_reward=best_reward,
            best_actions=best_actions,
            best_val_acc=best_val_acc,
            best_test_acc=best_test_acc,
            best_complexity=best_complexity,
            best_model_state=best_model_state,
        )



# -----------------------
# Plotting
# -----------------------
def _savefig(fig, path: Path):
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_best_per_task_acc_vs_complexity(train_stats: List[dict], out_path: Path):
    xs, ys, labels = [], [], []
    for s in train_stats:
        t = s.get("task", None)
        cx = s.get("best_complexity", None)
        acc = s.get("best_val_acc", None)
        if t is None or cx is None or acc is None:
            continue
        xs.append(float(cx))
        ys.append(float(acc))
        labels.append(int(t))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xs, ys)
    for x, y, tt in zip(xs, ys, labels):
        ax.annotate(str(tt), (x, y))
    ax.set_xlabel("Complexity (proxy)")
    ax.set_ylabel("Validation accuracy (best child per task)")
    ax.set_title("Best child per task: accuracy vs complexity (task labels)")
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, out_path)

def plot_trials_acc_vs_complexity(
    trial_log_path: Path,
    out_path: Path,
    task_id: Optional[int] = None,
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

            if task_id is not None and int(obj.get("task", -1)) != int(task_id):
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

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # tutti i trial (blu)
    ax.scatter(xs, ys)

    for x, y, tr in zip(xs, ys, labels):
        ax.annotate(str(tr), (x, y))

    # BEST (rosso): child finale scelta per quel task
    if best_point is not None:
        bx = best_point.get("complexity", None)
        by = best_point.get("val_acc", None)
        if bx is not None and by is not None:
            ax.scatter([float(bx)], [float(by)], color="red", s=60, zorder=5)

    ax.set_xlabel("Complexity (proxy)")
    ax.set_ylabel("Validation accuracy (each trial)")
    title = "Trials: accuracy vs complexity" if task_id is None else f"Task {task_id}: trials accuracy vs complexity (trial labels)"
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    _savefig(fig, out_path)

def make_all_plots(
    out_dir: Path,
    first_task_curve_online,
    avg_acc_curve,
    acc_matrix,
    train_stats,
    trial_log_path: Path,
    task_id_for_trials: Optional[int] = None,
):
    plots_dir = ensure_dir(out_dir / "plots")

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(len(first_task_curve_online))), first_task_curve_online, marker="o")
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
    _savefig(fig, plots_dir / "acc_matrix_heatmap.png")

    plot_best_per_task_acc_vs_complexity(train_stats, plots_dir / "best_per_task_acc_vs_complexity.png")

    if task_id_for_trials is not None:
        best_point = None
        for s in train_stats:
            if int(s.get("task", -999)) == int(task_id_for_trials):
                if "best_complexity" in s and "best_val_acc" in s:
                    best_point = {"complexity": float(s["best_complexity"]), "val_acc": float(s["best_val_acc"])}
                break

        plot_trials_acc_vs_complexity(
            trial_log_path,
            plots_dir / f"task_{int(task_id_for_trials):03d}_trials_acc_vs_complexity.png",
            task_id=int(task_id_for_trials),
            best_point=best_point,
        )


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

    controller_cfg = dict(cfg["controller"])
    controller_target = _get_any(controller_cfg, ["_target_", "target"], None)
    if controller_target is None:
        raise KeyError("controller must have 'target' or '_target_'")
    controller_cfg.pop("_target_", None)
    controller_cfg.pop("target", None)
    ControllerCls = import_from_target(controller_target)

    spec = get_action_spec(task_net)
    controller_cfg = normalize_controller_cfg(controller_cfg, spec)
    controller = ControllerCls(**controller_cfg).to(device)

    value_net = instantiate(cfg["valuenet"]).to(device)

    num_tasks = int(cfg["dataset"]["num_tasks"])
    batch_size = int(cfg["training"]["batch_size"])
    epochs_task = int(cfg["training"]["epochs_task"])
    lr_task = float(cfg["training"]["lr_task"])
    grad_clip = cfg["training"].get("grad_clip", None)
    trials = int(cfg["training"]["controller_trials"])
    alpha = float(cfg["rcl"]["reward_alpha"])

    opt_c = torch.optim.Adam(controller.parameters(), lr=float(cfg["training"]["lr_controller"]))
    opt_v = torch.optim.Adam(value_net.parameters(), lr=float(cfg["training"]["lr_value"]))

    ppo_cfg = cfg.get("ppo", {})
    ppo_clip_eps = float(ppo_cfg.get("clip_eps", 0.2))
    ppo_epochs = int(ppo_cfg.get("epochs", 4))
    ppo_entropycoef = float(ppo_cfg.get("entropycoef", 0.01))
    ppo_advnorm = bool(ppo_cfg.get("advnorm", True))

    baseline_cfg = cfg.get("baseline", {})
    use_ema = bool(baseline_cfg.get("use_ema", True))
    ema_beta = float(baseline_cfg.get("ema_beta", 0.05))
    ema_mix = float(baseline_cfg.get("mix", 0.1))
    ema_b = [0.0 for _ in range(num_tasks)]
    ema_init = [False for _ in range(num_tasks)]

    print_every = int(cfg.get("logging", {}).get("print_every_trials", 25))
    print_first = int(cfg.get("logging", {}).get("print_first_trials", 3))

    # cmax coerente con azioni in [0..n-1]
    cmax = float(sum(max(0, int(n) - 1) for n in spec))

    # hard check critic 2D per PPO
    with torch.no_grad():
        test = value_net(torch.tensor([0.0, 0.0], device=device, dtype=torch.float32))
    if not torch.is_tensor(test) or test.numel() != 1:
        raise ValueError("PPO requires 2D valuenet: forward([t_norm,c_norm]) -> scalar.")

    trainer = ControllerTrainerPPO(device=device)
    trainer.require(controller)

    tasks_data_cache = [dataset_get_task(dataset_obj, j, batch_size=batch_size) for j in range(num_tasks)]

    train_stats: List[dict] = []
    first_task_curve_online: List[float] = []
    avg_acc_curve: List[float] = []
    acc_matrix: List[List[Optional[float]]] = []

    trial_log_path = out_dir / "trial_log.jsonl"

    # NEW: PPO logs jsonl
    ppo_log_path = out_dir / "ppo_log.jsonl"

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
            print(f"[task {t}] PPO search: trials={trials} alpha={alpha}", flush=True)

            res = trainer.run(
                t=t,
                num_tasks=num_tasks,
                controller=controller,
                value_net=value_net,
                task_net=task_net,
                task_data=task_data,
                trials=trials,
                alpha=alpha,
                epochs_task=epochs_task,
                lr_task=lr_task,
                grad_clip=grad_clip,
                opt_c=opt_c,
                opt_v=opt_v,
                print_every=print_every,
                print_first=print_first,
                trial_log_path=trial_log_path,

                # NEW: pass ppo log path
                ppo_log_path=ppo_log_path,

                ppo_clip_eps=ppo_clip_eps,
                ppo_epochs=ppo_epochs,
                ppo_entropycoef=ppo_entropycoef,
                ppo_advnorm=ppo_advnorm,
                cmax=cmax,
                use_ema=use_ema,
                ema_beta=ema_beta,
                ema_mix=ema_mix,
                ema_b=ema_b,
                ema_init=ema_init,
            )

            best_actions = res["best_actions"]
            best_model_state = res["best_model_state"]
            if best_actions is None or best_model_state is None:
                raise RuntimeError(f"[task {t}] No best model selected.")

            task_net = expanded_copy(task_net, best_actions).to(device)
            task_net.load_state_dict(best_model_state)

            if hasattr(task_net, "registertaskslice"):
                task_net.registertaskslice()
            elif hasattr(task_net, "register_task_slice"):
                task_net.register_task_slice()

            train_stats.append({
                "task": int(t),
                "best_reward": float(res["best_reward"]),
                "best_actions": list(map(int, best_actions)),
                "best_val_acc": float(res["best_val_acc"]),
                "best_test_acc": float(res["best_test_acc"]),
                "best_complexity": float(res["best_complexity"]),
                "note": str(res["algo"]),
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

        save_json({"train_stats": train_stats, "first_task_curve_online": first_task_curve_online, "avg_acc_curve": avg_acc_curve}, out_dir / "train_stats.json")
        save_json({"acc_matrix": acc_matrix}, out_dir / "acc_matrix.json")

        make_all_plots(
            out_dir,
            first_task_curve_online,
            avg_acc_curve,
            acc_matrix,
            train_stats,
            trial_log_path=trial_log_path,
            task_id_for_trials=t if t > 0 else None,
        )

        print(f"[task {t}] saved checkpoint + logs + plots", flush=True)

    print("Done. Results saved to:", out_dir, flush=True)


if __name__ == "__main__":
    main()
