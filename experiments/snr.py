"""How noisy each signal's gradient is, and what that alone does to an Adam step.

Adam divides every parameter's step by the running root mean square of its gradient, so a parameter
whose gradient changes sign from batch to batch takes a small step whatever the gradient's size. A
signal with a noisier gradient therefore moves the weights less under Adam for a reason that has
nothing to do with where it puts its credit, and "moves the weights less" stops being evidence about
credit. This measures, on the frozen state and under training conditions, the mean gradient of the
reward and of the supervised control over many batches, its batch-to-batch spread, and the step
Adam would take per parameter in expectation, |mean| / sqrt(mean^2 + variance), in units of the
learning rate.

    python -m experiments.snr --net 0
"""

import argparse
import os

import numpy as np
import torch

from src import runs, signals, tasks, trials
from src.model import load

ENV = tasks.sameloc(*tasks.C1)
N_BATCH = 32


def trainable(model):
    frozen = {id(p) for p in (model.W_aux, model.b_aux, model.W_out, model.b_out)}
    return [(name, p) for name, p in model.named_parameters() if id(p) not in frozen]


def loss(model, cond):
    if cond == "control":
        batch = trials.batch(ENV, signals.BATCH, int(model.dt), window_ms=signals.WINDOW_MS)
        return signals.task_step(0.0, 1.0)(model, batch)[0]
    x, target, fa, valid = signals.outcome_batch(ENV, signals.BATCH, "cpu")
    logits = model(x, return_hidden=True)[0]
    return signals.policy_loss(torch.log_softmax(logits, -1), target, fa, valid)[0]


def measure(model, cond):
    model.train()
    named = trainable(model)
    params = [p for _, p in named]
    rec = [i for i, (name, _) in enumerate(named) if name == "W_rec"]
    grads = []
    for _ in range(N_BATCH):
        g = torch.autograd.grad(loss(model, cond), params, allow_unused=True)
        grads.append([torch.zeros_like(p) if gi is None else gi.detach() for p, gi in zip(params, g)])

    def summary(index):
        g = torch.stack([torch.cat([grads[b][i].reshape(-1) for i in index]) for b in range(N_BATCH)])
        mean, var = g.mean(0), g.var(0)
        spread = float(((g - mean) ** 2).sum(1).mean())
        step = mean.abs() / torch.sqrt(mean ** 2 + var).clamp_min(1e-12)
        return dict(norm_mean=float(mean.norm()), snr=float(mean.norm() ** 2) / max(spread, 1e-12),
                    adam_step=float(step.pow(2).mean().sqrt()))

    everything = summary(range(len(params)))
    recurrent = summary(rec)
    return dict(cond=cond, **everything,
                **{f"{k}_rec": v for k, v in recurrent.items()})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/snr_c1_net{a.net}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(11)
    np.random.seed(11)
    rows = []
    for cond in ("reward", "control"):
        row = measure(load(a.net, head="c1"), cond)
        rows.append(dict(net=a.net, **{k: (round(v, 6) if isinstance(v, float) else v)
                                        for k, v in row.items()}))
        print(f"  [net {a.net} {cond}] snr={row['snr']:.4f} adam step={row['adam_step']:.4f} "
              f"(W_rec {row['adam_step_rec']:.4f})", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
