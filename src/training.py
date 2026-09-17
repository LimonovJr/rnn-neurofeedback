"""Supervised task training: masked cross-entropy on hold/release, plus a penalty on releasing
inside a distractor's reaction-time window."""

import copy
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src import trials


@dataclass
class TrainConfig:
    batch_size: int = 64
    lr: float = 1e-3
    max_updates: int = 5000
    print_every: int = 50
    l2_h: float = 1e-6
    l2_w: float = 1e-6
    grad_clip: float = 10.0
    fa_penalty_weight: float = 0.4
    device: str = "cpu"
    stop_on_no_miss: int = 3      # print steps without a miss before stopping; 0 disables
    rollback_steps: int = 5
    warmup_steps: int = 10


def cross_entropy(logits, targets, mask):
    b, t, c = logits.shape
    loss = F.cross_entropy(logits.reshape(b * t, c), targets.reshape(b * t),
                           reduction="none").reshape(b, t)
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def penalty(h_seq, model, l2_h, l2_w):
    reg = l2_h * h_seq.pow(2).mean()
    for name, p in model.named_parameters():
        if any(k in name for k in ("W_in", "W_rec", "W_out")):
            reg = reg + l2_w * p.pow(2).mean()
    return reg


@torch.no_grad()
def outcomes(actions, targets, lengths, fa_mask):
    """Fractions of correct / abort / miss trials, and the false-alarm rate over the trials that
    carried a distractor."""
    counts = {"correct": 0, "miss": 0, "abort": 0}
    with_distractor, with_fa = 0, 0
    for b in range(actions.shape[0]):
        t = int(lengths[b])
        a = actions[b, :t].cpu().numpy()
        gt = targets[b, :t].cpu().numpy()
        released = np.where(a == 1)[0]
        if len(released) == 0:
            counts["miss"] += 1
        else:
            counts["correct" if gt[int(released[0])] == 1 else "abort"] += 1

        fa = fa_mask[b, :t].cpu().numpy()
        if (fa > 0).any():
            with_distractor += 1
            with_fa += bool(((a == 1) & (fa > 0)).any())

    total = sum(counts.values())
    out = {k: v / total for k, v in counts.items()}
    out["n_distractor_trials"] = with_distractor
    out["fa_rate"] = with_fa / with_distractor if with_distractor else float("nan")
    return out


def train_supervised(model, env_fn, cfg: TrainConfig, on_update=None):
    """Train to the update cap, optionally stopping once no trial in a print step is a miss.

    `on_update(upd, model)` runs after each optimizer step, so a caller can checkpoint inside one
    continuous optimizer run rather than restarting Adam at every save point.
    """
    model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = {k: [] for k in ("loss", "ce", "fa_pen", "p_correct", "p_abort", "p_miss", "fa_rate")}

    early_stop = cfg.stop_on_no_miss > 0
    checkpoints = deque(maxlen=cfg.rollback_steps) if early_stop else None
    no_miss_streak, print_step = 0, 0

    for upd in range(1, cfg.max_updates + 1):
        model.train()
        x, y, mask, fa, _, _, lengths = trials.batch(env_fn, cfg.batch_size, int(model.dt),
                                                     device=cfg.device)
        logits, _, h_seq = model(x, return_hidden=True)

        ce = cross_entropy(logits, y, mask)
        fa_sum = fa.sum()
        if fa_sum > 0:
            fa_pen = (F.softmax(logits, dim=-1)[..., 1] * fa).sum() / fa_sum
        else:
            fa_pen = torch.zeros((), device=logits.device)
        loss = ce + cfg.fa_penalty_weight * fa_pen + penalty(h_seq, model, cfg.l2_h, cfg.l2_w)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if on_update is not None:
            on_update(upd, model)

        stats = outcomes(logits.argmax(dim=-1), y, lengths, fa)
        history["loss"].append(loss.item())
        history["ce"].append(ce.item())
        history["fa_pen"].append(fa_pen.item())
        for k in ("correct", "abort", "miss"):
            history[f"p_{k}"].append(stats[k])
        history["fa_rate"].append(stats["fa_rate"])

        if upd % cfg.print_every == 0 or upd == 1:
            fa_str = f"{stats['fa_rate'] * 100:5.1f}%" if stats["n_distractor_trials"] else "  n/a"
            print(f"upd {upd:5d}/{cfg.max_updates} | loss {loss.item():.4f} | ce {ce.item():.4f} | "
                  f"hit {stats['correct'] * 100:5.1f}% miss {stats['miss'] * 100:5.1f}% "
                  f"abort {stats['abort'] * 100:5.1f}% fa {fa_str}", flush=True)

            if not early_stop:
                continue
            print_step += 1
            checkpoints.append(copy.deepcopy(model.state_dict()))
            if print_step <= cfg.warmup_steps:
                continue

            no_miss_streak = no_miss_streak + 1 if stats["miss"] == 0.0 else 0
            if no_miss_streak >= cfg.stop_on_no_miss:
                model.load_state_dict(checkpoints[0])
                back = len(checkpoints) - 1
                print(f"early stop at upd {upd}: no misses for {no_miss_streak} print steps, "
                      f"restored weights from ~{upd - back * cfg.print_every}", flush=True)
                break

    return history
