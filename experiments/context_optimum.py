"""Where on the context task the reward learner's payoff is best, against where c = 0 is.

Blocks 37, 41 and 42 read the adaptive threshold's collapse as a conflict: a tracker held at c = 0
keeps the reward learner away from its own payoff optimum, so its push on the criterion never
decays. That needs the optimum to be more conservative than c = 0 on this task, which blocks 39
and 43 showed for c1 only. Here both points are found on saved context states, read the way the
learner trains (recurrent noise on, sampled releases): c = 0 as in the tracker (among trials that
reach the response window), and the offset that maximises the learner's own payoff, +1 for a hit
or a correct rejection and -1 for a false alarm, a miss or an abort.

    python -m experiments.context_optimum --nets 0,1,2,3,4
"""

import argparse
import glob
import os

import numpy as np
import torch

from experiments.temperature import margins, rates
from src import runs, tasks
from src.model import CKPT, load

ENV = tasks.context()
GRID = np.arange(-14.0, 6.01, 0.25)


def payoff(m, target, fa, valid, draws=10):
    """Mean outcome under sampled releases, and the share of trials aborted."""
    go = target.any(1)
    start = (target | fa).float().argmax(1)
    steps = torch.arange(m.shape[1])[None, :]
    value, aborts = [], []
    for _ in range(draws):
        released = (torch.rand_like(m) < torch.sigmoid(m)) & valid
        first = torch.where(released, steps, m.shape[1]).min(1).values
        at = first.clamp(max=m.shape[1] - 1)
        did = first < m.shape[1]
        abort = did & (first < start)
        hit = target.gather(1, at[:, None]).squeeze(1) & did
        rejection = ~did & ~go
        value.append(float(torch.where(hit | rejection, 1.0, -1.0).mean()))
        aborts.append(float(abort.float().mean()))
    return np.mean(value), np.mean(aborts)


def states(net):
    yield "start", None
    for upd in (2000, 4000, 8000):
        for path in glob.glob(str(CKPT / "nf_context_b37" /
                                  f"nf_context_reward_lam1_n10000_crit_net{net}_r9_u{upd}.pt")):
            yield f"reward u{upd}", path
    for cond in ("reward", "control"):
        for path in sorted(glob.glob(str(CKPT / "nf_context" / f"{cond}_net{net}_r*_u8000.pt"))):
            yield f"{cond} end ({os.path.basename(path).split('_')[2]})", path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", default="0,1,2,3,4")
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--out", default="results/context_optimum.csv")
    a = p.parse_args()
    torch.manual_seed(17)
    np.random.seed(17)
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        for state, path in states(net):
            model = load(net, head="context")
            if path:
                model.load_state_dict(torch.load(path, weights_only=True)["state_dict"],
                                      strict=False)
            m, target, fa, valid = margins(model, ENV, a.n)
            # once every trial aborts no rate is left and the sum drops to zero, so the crossing is
            # the first one coming from the conservative side, not a bisection over the whole range
            sums = [sum(rates(m + o, target, fa, valid, 5)[:2]) for o in GRID]
            k = next((i for i, v in enumerate(sums) if v >= 1.0), None)
            c0 = float("nan") if not k else float(
                GRID[k - 1] + (1.0 - sums[k - 1]) / (sums[k] - sums[k - 1]) * (GRID[k] - GRID[k - 1]))
            curve = [payoff(m + o, target, fa, valid) for o in GRID]
            best = int(np.argmax([v for v, _ in curve]))
            at_c0 = payoff(m + c0, target, fa, valid)
            at_zero = payoff(m, target, fa, valid)
            rows.append(dict(net=net, state=state, offset_c0=round(c0, 3),
                             offset_payoff=float(GRID[best]),
                             payoff_best=round(curve[best][0], 4),
                             payoff_at_c0=round(at_c0[0], 4), payoff_at_own=round(at_zero[0], 4),
                             abort_best=round(curve[best][1], 4), abort_c0=round(at_c0[1], 4),
                             abort_own=round(at_zero[1], 4)))
            r = rows[-1]
            print(f"  [net {net} {state:<18}] c=0 at {r['offset_c0']:+6.2f}  payoff optimum at "
                  f"{r['offset_payoff']:+6.2f}  payoff: best {r['payoff_best']:+.3f}, at c=0 "
                  f"{r['payoff_at_c0']:+.3f}, at own {r['payoff_at_own']:+.3f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
