"""The BCI-learning experiment inside the model: perturb the frozen decision read-out either within
the intrinsic manifold of the decision-time states or outside it, then let the recurrent weights
learn back the behaviour with the reward signal alone.

Within-manifold: the in-manifold part of the decision axis is rotated by a fixed angle toward a
random direction inside the top-k principal subspace, so the decoder reads a different combination
of directions the network already varies along. Outside-manifold: the same rotation, toward a
random direction in the orthogonal complement, so the decoder reads a direction the network does
not currently vary along. The angle is the same in both kinds and the release threshold is
recalibrated after each change, so the two differ in geometry rather than in how much of the axis
was moved.

`--steps` runs the outside-manifold case as a ladder of intermediate read-outs, each trained for a
share of the budget: the incremental training that makes outside-manifold learning succeed in
monkeys, and the "proximal learning" the project was asked to test.

The angle has to be calibrated first: a decision axis is fragile, and too large a rotation floors
behaviour at once in both kinds, which leaves nothing to compare. `--updates 0` reports the
immediate drop of every candidate and nothing else.

    python -m experiments.manifold --net 0 --kind within --angle 30 --updates 0
    python -m experiments.manifold --net 0 --kind within --angle 30 --candidate 3
    python -m experiments.manifold --net 0 --kind outside --angle 30 --candidate 3 --steps 8
"""

import argparse

import numpy as np
import torch

from src import behaviour, representation, runs, signals, tasks
from src.model import load

ENV = tasks.sameloc(*tasks.C1)
DECISION_WINDOW = (0, 300)             # the frames the decision read-out is actually read on
BASIS_K = 10
CANDIDATES = 20
EVAL_N = 2000


def decision_axis(model):
    """Release wins when (W_out[1] - W_out[0]) h + (b1 - b0) > 0, so only the difference matters."""
    return (model.W_out[1] - model.W_out[0]).detach().clone()


def set_axis(model, w):
    """Move the release row so the decision axis becomes `w`, leaving the hold row alone."""
    with torch.no_grad():
        model.W_out[1].copy_(model.W_out[0] + w)


def perturbations(w, P, kind, angle, n, seed):
    """`n` candidate axes: the in-manifold part of `w` rotated by `angle` toward a random unit
    direction that lies inside span(P) (within) or in its complement (outside), with the part of
    `w` that was already outside the manifold left alone."""
    gen = torch.Generator().manual_seed(seed)
    inside = P @ (P.T @ w)
    rest = w - inside
    out = []
    for _ in range(n):
        v = torch.randn(w.shape[0], generator=gen)
        v = P @ (P.T @ v) if kind == "within" else v - P @ (P.T @ v)
        v = v - inside * (v @ inside) / (inside @ inside)          # orthogonal to what is rotated
        v = v / v.norm() * inside.norm()
        out.append(np.cos(angle) * inside + np.sin(angle) * v + rest)
    return out


def recalibrate(model, states, margin0):
    """Move the release threshold so the mean decision margin on the decision-window states is
    what it was before the axis changed; without this a rotated axis mostly never releases."""
    with torch.no_grad():
        margin = states @ decision_axis(model) + (model.b_out[1] - model.b_out[0])
        model.b_out[1] += margin0 - margin.mean()


def score(model):
    return behaviour.measure(model, ENV, n=EVAL_N)


def run(a):
    model = load(a.net)
    d_intact = score(model)
    basis = representation.delay_basis(model, ENV, k=a.basis_k, window_ms=DECISION_WINDOW)
    P, explained = basis["P"], basis["explained"]
    w0 = decision_axis(model)
    b0 = model.b_out.detach().clone()
    states, _, _ = representation.delay_population(model, ENV, n_trials=600,
                                                   window_ms=DECISION_WINDOW, outcome=None)
    states = torch.tensor(states, dtype=torch.float32)
    margin0 = float((states @ w0 + (b0[1] - b0[0])).mean())
    print(f"[manifold net{a.net} {a.kind}] intact d'={d_intact:+.3f}, top-{a.basis_k} explains "
          f"{explained:.3f}, |w| inside/outside manifold = "
          f"{(P @ (P.T @ w0)).norm():.3f}/{(w0 - P @ (P.T @ w0)).norm():.3f}", flush=True)

    # the immediate drop of every candidate, so the chosen one is reported against its peers
    candidates = perturbations(w0, P, a.kind, np.radians(a.angle), a.candidates, seed=100 + a.net)
    drops = []
    for w in candidates:
        set_axis(model, w)
        recalibrate(model, states, margin0)
        drops.append(score(model))
        set_axis(model, w0)
        with torch.no_grad():
            model.b_out.copy_(b0)
    print(f"  immediate d' over {a.candidates} candidates at {a.angle:g} deg: "
          f"median {np.median(drops):+.3f}, chosen #{a.candidate} {drops[a.candidate]:+.3f}",
          flush=True)
    w_target = candidates[a.candidate]

    torch.manual_seed(a.rep)
    np.random.seed(a.rep)
    upds, curve = [], []
    offset = 0

    def hook(m, upd):
        d = score(m)
        upds.append(offset + upd)
        curve.append(round(d, 4))
        print(f"  upd {offset + upd:5d}  d'={d:+.4f}", flush=True)

    # a ladder of read-outs interpolated toward the target, each held for a share of the budget;
    # the first point of every rung is the drop right after the read-out moved
    n_steps = max(1, a.steps)
    for s in range(1, n_steps + 1):
        t = s / n_steps
        w = (1 - t) * w0 + t * w_target
        set_axis(model, w / w.norm() * w0.norm())
        recalibrate(model, states, margin0)
        if s == 1:
            d_pert = score(model)
            upds.append(0)
            curve.append(round(d_pert, 4))
        budget = a.updates // n_steps
        signals.train(model, a.cond, ENV, budget, lr=a.lr, eval_every=a.eval_every, hook=hook,
                      anchor=a.anchor)
        offset += budget

    drift = float((decision_axis(model) - w_target / w_target.norm() * w0.norm()).abs().max())
    assert drift < 1e-6, f"the read-out moved during relearning: {drift:.2e}"
    runs.write(a.out, dict(net=a.net, kind=a.kind, angle=a.angle, candidate=a.candidate, rep=a.rep,
                           cond=a.cond, steps=n_steps, updates=a.updates, lr=a.lr,
                           anchor=f"{a.anchor:g}",
                           basis_k=a.basis_k, explained=f"{explained:.3f}",
                           d_intact=f"{d_intact:.4f}", d_perturbed=f"{curve[0]:.4f}",
                           candidate_drops=runs.joined(round(d, 3) for d in drops),
                           curve_upd=runs.joined(upds), curve_dprime=runs.joined(curve)))
    print(f"[manifold net{a.net} {a.kind} #{a.candidate}] recovered to {curve[-1]:+.3f} of "
          f"{d_intact:+.3f}; curve={curve}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--kind", choices=["within", "outside"], required=True)
    p.add_argument("--angle", type=float, default=60.0,
                   help="degrees the in-manifold part of the axis is rotated by")
    p.add_argument("--candidate", type=int, default=0, help="which random perturbation to train on")
    p.add_argument("--candidates", type=int, default=CANDIDATES)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--cond", choices=["reward", "control"], default="reward",
                   help="reward is the label-free relearning; control is the supervised reference")
    p.add_argument("--steps", type=int, default=1, help="intermediate read-outs on the way to the target")
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--lr", type=float)
    p.add_argument("--anchor", type=float, default=0.0)
    p.add_argument("--basis-k", type=int, default=BASIS_K)
    p.add_argument("--out")
    a = p.parse_args()
    a.lr = a.lr or signals.LR[a.cond]
    a.out = a.out or (f"results/manifold_{a.kind}_{a.cond}_a{a.angle:g}_s{a.steps}_net{a.net}"
                      f"_c{a.candidate}_r{a.rep}.csv")
    run(a)


if __name__ == "__main__":
    main()
