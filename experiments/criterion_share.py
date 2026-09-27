"""How much of the reward step on the context task moves the criterion, at the start and at the end.

Clamping the reward learner's criterion can only redirect its steps toward separation if those steps
had a criterion component to lose. On c1 the first reward step is mostly criterion (block 27); on
the context task the payoff is nearly flat in the threshold at d' near 0.2, so the component may be
small from the start. The step split of `experiments.components`, on the context task, at the
pretrained network and at the end of each saved reward and supervised run, for the gradient of both
signals. The reward gradient is noisy, so it is averaged over many batches and the two halves of
those batches are split separately: a component whose sign the halves do not agree on is estimate
noise, not a direction the signal prefers.

    python -m experiments.criterion_share --net 0
"""

import argparse
import glob
import os

import numpy as np
import torch

from experiments.components import release_logits, trainable
from src import runs, signals, tasks, trials
from src.model import CKPT, load

ENV = tasks.context()
N_BATCH, N_HELD, BATCH = 16, 8, 128


def reward_loss(model):
    x, target, fa, valid = signals.outcome_batch(ENV, signals.BATCH, "cpu")
    logits = model(x, return_hidden=True)[0]
    return signals.policy_loss(torch.log_softmax(logits, -1), target, fa, valid)[0]


def control_loss(model):
    batch = trials.batch(ENV, signals.BATCH, int(model.dt), window_ms=signals.WINDOW_MS)
    return signals.task_step(0.0, 1.0)(model, batch)[0]


def restore(net, path=None):
    model = load(net, head="context")
    if path:
        model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
    return model


def gradient(model, loss, n):
    params = trainable(model)
    model.train()
    total = None
    for _ in range(n):
        g = torch.autograd.grad(loss(model), params, allow_unused=True)
        g = torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1)
                       for p, gi in zip(params, g)]).detach()
        total = g if total is None else total + g
    return total / n


def split(model, grad, held, length, base):
    params = trainable(model)
    state = [p.detach().clone() for p in params]
    unit, i = -length * grad / grad.norm().clamp_min(1e-12), 0
    with torch.no_grad():
        for p in params:
            p.add_(unit[i:i + p.numel()].view_as(p))
            i += p.numel()
    sig1, noise1 = release_logits(model, held)
    with torch.no_grad():
        for p, s in zip(params, state):
            p.copy_(s)
    d_sig, d_noise = sig1 - base[0], noise1 - base[1]
    return (d_sig + d_noise) / 2, d_sig - d_noise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--length", type=float, default=0.003)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/criterion_share_context_net{a.net}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(7)
    np.random.seed(7)
    held = [trials.batch(ENV, BATCH, 20, window_ms=signals.WINDOW_MS) for _ in range(N_HELD)]
    states = [("start", None)] + [
        (os.path.basename(f)[:-3], f)
        for cond in ("reward", "control")
        for f in sorted(glob.glob(str(CKPT / "nf_context" / f"{cond}_net{a.net}_r*_u8000.pt")))]
    rows = []
    for state, path in states:
        model = restore(a.net, path)
        base = release_logits(model, held)
        for name, loss in (("reward", reward_loss), ("control", control_loss)):
            halves = [gradient(model, loss, N_BATCH // 2) for _ in range(2)]
            whole = (halves[0] + halves[1]) / 2
            crit, sep = split(model, whole, held, a.length, base)
            (c0, s0), (c1, s1) = (split(model, h, held, a.length, base) for h in halves)
            cos = float(torch.nn.functional.cosine_similarity(halves[0], halves[1], dim=0))
            rows.append(dict(net=a.net, state=state, signal=name, criterion=round(crit, 6),
                             separation=round(sep, 6),
                             criterion_share=round(abs(crit) / max(abs(crit) + abs(sep), 1e-12), 4),
                             halves_criterion=f"{c0:+.5f}/{c1:+.5f}",
                             halves_separation=f"{s0:+.5f}/{s1:+.5f}", halves_cos=round(cos, 4),
                             margin_signal=round(base[0], 4), margin_noise=round(base[1], 4)))
            r = rows[-1]
            print(f"  [net {a.net} {state:<22} {name:<7}] criterion {crit:+.5f} "
                  f"({r['halves_criterion']})  separation {sep:+.5f} ({r['halves_separation']})  "
                  f"share {r['criterion_share']:.2f}  cos {cos:+.2f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
