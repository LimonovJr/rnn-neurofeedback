"""Where one step of each signal moves the decision, split into criterion and separation.

The release logit is read on the steps where a release is correct and on the steps where it is a
false alarm, before and after one step of fixed length along a signal's gradient. The mean of the
two shifts moves the criterion; their difference moves the separation between the two kinds of
step, which is what discriminability is made of. Block 17 found that on c1 the behavioural
differences between signals are criterion shifts; this asks whether the first step already says
so, and which of the two components each signal, and each angle of an axis target, pushes.

The step length is in the regime where the first-order prediction was shown to hold (block 9), the
gradient is averaged over batches under training conditions, and the held-out logits are read with
the recurrent noise off on the same batches for every signal, so the shifts differ only by the
step.

    python -m experiments.components --net 0
"""

import argparse
import os

import numpy as np
import torch

from src import representation, runs, signals, tasks, trials
from src.model import load

ENV = tasks.sameloc(*tasks.C1)
N_BATCH, N_HELD, BATCH = 4, 8, 128
COSINES = [-1.0, -0.5, 0.0, 0.5, 1.0]


def trainable(model):
    frozen = {id(p) for p in (model.W_aux, model.b_aux, model.W_out, model.b_out)}
    return [p for p in model.parameters() if id(p) not in frozen]


def signal_losses(net):
    """name -> function(model) returning the signal's loss on a fresh training batch."""
    def supervised(step):
        def loss(model):
            batch = trials.batch(ENV, signals.BATCH, int(model.dt), window_ms=signals.WINDOW_MS)
            return step(model, batch)[0]
        return loss

    def reward(model):
        x, target, fa, valid = signals.outcome_batch(ENV, signals.BATCH, "cpu")
        logits = model(x, return_hidden=True)[0]
        return signals.policy_loss(torch.log_softmax(logits, -1), target, fa, valid)[0]

    out = {"control": supervised(signals.task_step(0.0, 1.0)),
           "control_fa": supervised(signals.control_fa_step(signals.FA_PENALTY)),
           "joint": supervised(signals.task_step(1.0, 1.0)),
           "position k1": supervised(signals.expand_step(1.0)),
           "position k2": supervised(signals.expand_step(2.0)),
           "shuffled": supervised(signals.expand_step(1.0, shuffle=True)),
           "reward": reward}
    for c in COSINES:
        out[f"axis {c:+g}"] = ("axis", c)
    return out


def held_out(model):
    return [trials.batch(ENV, BATCH, int(model.dt), window_ms=signals.WINDOW_MS)
            for _ in range(N_HELD)]


def release_logits(model, held):
    """Mean release-minus-hold logit on correct-release steps and on false-alarm steps."""
    model.eval()
    sig, noise = [], []
    with torch.no_grad():
        for x, y, _, fa, _, _, lengths in held:
            logits = model(x, return_hidden=True)[0]
            margin = logits[..., 1] - logits[..., 0]
            valid = torch.arange(x.shape[1])[None, :] < lengths[:, None]
            sig.append(margin[(y == 1) & valid])
            noise.append(margin[(fa > 0) & valid])
    return float(torch.cat(sig).mean()), float(torch.cat(noise).mean())


def step_components(net, name, loss, held, length):
    model = load(net, head="c1")
    if isinstance(loss, tuple):
        target = representation.decision_axis_target(model, ENV, loss[1], seed=net)
        step = signals.axis_step(target)
        loss = lambda m: step(m, trials.batch(ENV, signals.BATCH, int(m.dt),
                                              window_ms=signals.WINDOW_MS))[0]
    params = trainable(model)
    model.train()
    grad = None
    for _ in range(N_BATCH):
        g = torch.autograd.grad(loss(model), params, allow_unused=True)
        g = torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1)
                       for p, gi in zip(params, g)]).detach()
        grad = g if grad is None else grad + g
    grad = grad / N_BATCH
    sig0, noise0 = release_logits(model, held)
    unit, i = -length * grad / grad.norm().clamp_min(1e-12), 0
    with torch.no_grad():
        for p in params:
            p.add_(unit[i:i + p.numel()].view_as(p))
            i += p.numel()
    sig1, noise1 = release_logits(model, held)
    d_sig, d_noise = sig1 - sig0, noise1 - noise0
    return dict(net=net, signal=name, grad_norm=round(float(grad.norm()), 6),
                d_signal=round(d_sig, 6), d_noise=round(d_noise, 6),
                criterion=round((d_sig + d_noise) / 2, 6), separation=round(d_sig - d_noise, 6))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--length", type=float, default=0.003,
                   help="step length in parameter space, inside the first-order regime")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/components_c1_net{a.net}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(7)
    np.random.seed(7)
    held = held_out(load(a.net, head="c1"))
    rows = []
    for name, loss in signal_losses(a.net).items():
        row = step_components(a.net, name, loss, held, a.length)
        rows.append(row)
        print(f"  [net {a.net} {name:<12}] criterion {row['criterion']:+.5f}  "
              f"separation {row['separation']:+.5f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
