"""Closed-loop difficulty from the decoded state, after deBettencourt et al. (2015).

There, attention to the task-relevant category was decoded from the brain in real time and the
task-relevant stimulus faded when attention lapsed, so a lapse made the task harder at once; the
control group received another participant's feedback (yoked). Here the frozen position head
decodes the cued position from the last 300 ms of the delay, and the target that follows is shown
at a strength that falls as that decoding error rises:

    s = 1 - drop * sigmoid((error - tau) / temp),

with tau the starting network's median error, so about half the trials start as lapses. The
network is stepped through time so that the target it sees depends on its own state in the same
trial. The learner is the supervised task loss through the frozen decoder: the strength is a smooth
function of the decoded state, so the task error reaches the delay state through it. (A reward
learner would treat the strength as part of the environment and could not be credited for its
lapses.) A yoked network receives another network's recorded strengths, trial by trial, with no
path from its own state to them.
"""

from collections import defaultdict

import numpy as np
import torch

from src import signals, trials
from src.training import cross_entropy

BATCH = 48
WINDOW_MS = signals.WINDOW_MS


def onsets(delay_mask):
    """Target onset per trial: the step after the delay window the head is read in."""
    last = delay_mask.shape[1] - 1 - torch.flip(delay_mask > 0, [1]).float().argmax(1)
    return last + 1


def rollout(model, batch, tau, drop, temp, forced=None):
    """Step the network through a batch; the target channel from each trial's onset on is scaled by
    that trial's strength, computed from its decoding error at onset, or taken from `forced`."""
    x, _, _, _, xy, delay_mask, _ = batch
    b, t_max, _ = x.shape
    onset = onsets(delay_mask)
    h = model.init_hidden(b, x.device)
    err_sum = torch.zeros(b)
    n = delay_mask.sum(1).clamp_min(1)
    scale = torch.ones(b) if forced is None else forced
    logits = []
    for t in range(t_max):
        x_t = x[:, t, :]
        after = t >= onset
        if after.any():
            if forced is None:
                arriving = t == onset
                if arriving.any():
                    err = err_sum / n
                    s = 1 - drop * torch.sigmoid((err - tau) / temp)
                    scale = torch.where(arriving, s, scale)
            x_t = torch.cat([x_t[:, :6], (x_t[:, 6] * torch.where(after, scale, 1.0))[:, None]], 1)
        h, y_t = model.step(x_t, h)
        logits.append(y_t)
        w = delay_mask[:, t]
        if w.any():
            err_sum = err_sum + w * ((model.decode_xy(h) - xy[:, t]) ** 2).sum(-1)
    return torch.stack(logits, 1), scale, err_sum / n


def calibrate(model, env_fn, n=512):
    """The starting network's median decoding error at target onset: the lapse boundary."""
    batch = trials.batch(env_fn, n, int(model.dt), window_ms=WINDOW_MS)
    model.train()
    with torch.no_grad():
        _, _, err = rollout(model, batch, tau=0.0, drop=0.0, temp=1.0)
    return float(err.median())


def train(model, env_fn, updates, lr, tau, drop=0.4, temp=0.05, yoke=None, eval_every=None,
          hook=None, grad_clip=10.0):
    """Supervised task training under closed-loop difficulty. `yoke` is another run's recorded
    strengths, shape (updates, BATCH); the returned history holds this run's own."""
    frozen = (model.W_aux, model.b_aux, model.W_out, model.b_out)
    for p in frozen:
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    history = defaultdict(list)
    schedule = []
    if hook is not None and eval_every:
        hook(model, 0)
    for upd in range(1, updates + 1):
        model.train()
        batch = trials.batch(env_fn, BATCH, int(model.dt), window_ms=WINDOW_MS)
        forced = None if yoke is None else torch.as_tensor(yoke[upd - 1], dtype=torch.float32)
        logits, scale, err = rollout(model, batch, tau, drop, temp, forced)
        loss = cross_entropy(logits, batch[1], batch[2])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()
        schedule.append(scale.detach().numpy().copy())
        history["loss"].append(float(loss))
        history["strength"].append(float(scale.mean()))
        history["lapse"].append(float((err > tau).float().mean()))
        if upd % 100 == 0 or upd == 1:
            print(f"upd {upd:5d}  loss={history['loss'][-1]:.4f}  strength={history['strength'][-1]:.3f}"
                  f"  lapse={history['lapse'][-1]:.3f}", flush=True)
        if hook is not None and eval_every and upd % eval_every == 0:
            hook(model, upd)
    for p in frozen:
        p.requires_grad_(True)
    history["schedule"] = np.stack(schedule)
    return dict(history)
