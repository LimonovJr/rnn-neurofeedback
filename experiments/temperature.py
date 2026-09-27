"""The policy temperature that makes an unclamped reward learner start as stochastic as a clamped one.

The clamp moves the decoder's bias to the point where, among the trials that reach the response
window, the hit rate equals one minus the false-alarm rate; the temperature arm leaves the bias
alone and divides the policy's logits instead. For the comparison between the two to be about
where the threshold is and not about how random the decision is at the start, the temperature is
chosen, per network on the starting state, so that the mean entropy of the release policy on the
response-window steps equals the clamped policy's. The hold steps are left out of it: there the
policy should be certain, and their entropy would say how often it aborts, not how it decides.
Both are read the way training reads them: recurrent noise on, releases sampled. The offset does not feed back into the state (a
release ends the trial), so one pass of margins serves every offset and every temperature.

    python -m experiments.temperature --task context --nets 0,1,2,3,4
"""

import argparse

import numpy as np
import torch

from src import runs, signals
from experiments.nf import DEFAULTS
from src.model import load


def margins(model, env_fn, n):
    x, target, fa, valid = signals.outcome_batch(env_fn, n, "cpu")
    model.train()
    with torch.no_grad():
        logits = model(x, return_hidden=True)[0]
    return logits[..., 1] - logits[..., 0], target, fa, valid


def rates(m, target, fa, valid, draws=20):
    """Hit and false-alarm rates among the trials that reached the response window, under sampled
    releases, and the share of trials that released before it."""
    go = target.any(1)
    start = (target | fa).float().argmax(1)
    steps = torch.arange(m.shape[1])[None, :]
    out = []
    for _ in range(draws):
        released = (torch.rand_like(m) < torch.sigmoid(m)) & valid
        first = torch.where(released, steps, m.shape[1]).min(1).values
        at = first.clamp(max=m.shape[1] - 1)
        did = first < m.shape[1]
        reached = first >= start
        hit = (target.gather(1, at[:, None]).squeeze(1) & did).float()
        fa_ = (fa.gather(1, at[:, None]).squeeze(1) & did).float()
        out.append((float(hit[go].sum() / max(int((go & reached).sum()), 1)),
                    float(fa_[~go].sum() / max(int((~go & reached).sum()), 1)),
                    float(1 - reached.float().mean())))
    return np.mean(out, 0)


def entropy(m, window):
    p = torch.sigmoid(m[window])
    return float((-(p * p.clamp_min(1e-12).log() + (1 - p) * (1 - p).clamp_min(1e-12).log())).mean())


def bisect(f, lo, hi, n=40):
    for _ in range(n):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    return (lo + hi) / 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="context")
    p.add_argument("--nets", default="0,1,2,3,4")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--out", default="results/temperature_context.csv")
    a = p.parse_args()
    torch.manual_seed(5)
    np.random.seed(5)
    env_fn = DEFAULTS[a.task]["env"]
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        m, target, fa, valid = margins(load(net, head=a.task), env_fn, a.n)
        window = (target | fa) & valid
        # among the trials that reach the window both rates rise with the offset, so c = 0 is the
        # one place their sum crosses one
        offset = bisect(lambda o: sum(rates(m + o, target, fa, valid, 5)[:2]) - 1.0, -30.0, 30.0,
                        n=25)
        goal = entropy(m + offset, window)
        start = entropy(m, window)
        temp = 1.0 if start >= goal else bisect(lambda t: entropy(m / t, window) - goal, 1.0, 100.0)
        hit, fa_rate, abort = rates(m + offset, target, fa, valid)
        _, _, abort0 = rates(m, target, fa, valid)
        rows.append(dict(net=net, offset_c0=round(offset, 3), hit_c0=round(hit, 3),
                         fa_c0=round(fa_rate, 3), abort_c0=round(abort, 3),
                         abort_start=round(abort0, 3), entropy_start=round(start, 4),
                         entropy_c0=round(goal, 4), temperature=round(temp, 2),
                         entropy_at_t=round(entropy(m / temp, window), 4)))
        print(rows[-1], flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
