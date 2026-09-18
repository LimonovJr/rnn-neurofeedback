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

`--auto` closes that loop. The read-out advances one rung only while behaviour on the next rung
holds up, so the schedule is set by the network rather than by hand -- a decoder that follows the
brain toward a mapping the brain does not yet have.

The angle has to be calibrated first: a decision axis is fragile, and too large a rotation floors
behaviour at once in both kinds, which leaves nothing to compare. `--updates 0` reports the
immediate drop of every candidate and nothing else.

    python -m experiments.manifold --net 0 --kind within --angle 30 --updates 0
    python -m experiments.manifold --net 0 --kind within --angle 30 --candidate 3
    python -m experiments.manifold --net 0 --kind outside --angle 30 --candidate 3 --steps 8
"""

import argparse
import os

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


def auto_ladder(model, a, d_intact, move_to, hook, schedule):
    """Advance the read-out toward the target one rung at a time, and only while behaviour holds:
    the next rung is tried, scored on a short evaluation, and kept if that score is at least
    `a.auto_threshold` of the intact d'; otherwise the read-out stays where it is and the network
    trains on. Records (update, position, probe d') at every decision and returns the rungs taken.
    This is how incremental BCI training is paced in practice -- the mapping moves when the
    subject can still perform."""
    t, spent, rungs = 0.0, 0, 0
    rung = a.rung / a.angle
    while spent < a.updates:
        if t < 1.0:
            move_to(min(1.0, t + rung))
            probe = behaviour.measure(model, ENV, n=500)
            if probe >= a.auto_threshold * d_intact:
                t = min(1.0, t + rung)
                rungs += 1
                print(f"  rung {rungs} -> t={t:.2f} at update {spent} (probe d'={probe:+.3f})",
                      flush=True)
            else:
                move_to(t)
            schedule.append((spent, t, probe))
        block = min(a.block, a.updates - spent)
        # the hook's own update-0 call would duplicate the previous block's last point
        signals.train(model, a.cond, ENV, block, lr=a.lr, eval_every=a.eval_every,
                      hook=lambda m, u, base=spent: hook(m, base + u) if u else None,
                      anchor=a.anchor)
        spent += block
    return rungs


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

    def move_to(t):
        w = (1 - t) * w0 + t * w_target
        set_axis(model, w / w.norm() * w0.norm())
        recalibrate(model, states, margin0)

    schedule = []
    if a.auto:
        n_steps = auto_ladder(model, a, d_intact, move_to, hook, schedule)
    else:
        # a ladder of read-outs interpolated toward the target, each held for a share of the
        # budget; the first point of every rung is the drop right after the read-out moved
        n_steps = max(1, a.steps)
        for s in range(1, n_steps + 1):
            move_to(s / n_steps)
            if s == 1:
                upds.append(0)
                curve.append(round(score(model), 4))
            budget = a.updates // n_steps
            signals.train(model, a.cond, ENV, budget, lr=a.lr, eval_every=a.eval_every,
                          hook=hook, anchor=a.anchor)
            offset += budget

    if not a.auto:
        drift = float((decision_axis(model) - w_target / w_target.norm() * w0.norm()).abs().max())
        assert drift < 1e-6, f"the read-out moved during relearning: {drift:.2e}"
    runs.write(a.out, dict(net=a.net, kind=a.kind, angle=a.angle, candidate=a.candidate, rep=a.rep,
                           cond=a.cond, steps=n_steps, auto=int(a.auto), updates=a.updates,
                           lr=a.lr, anchor=f"{a.anchor:g}",
                           schedule=runs.joined(f"{u}:{t:.3f}:{v:.3f}" for u, t, v in schedule),
                           basis_k=a.basis_k, explained=f"{explained:.3f}",
                           d_intact=f"{d_intact:.4f}",
                           d_perturbed=f"{drops[a.candidate] if a.auto else curve[0]:.4f}",
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
    p.add_argument("--auto", action="store_true",
                   help="closed-loop ladder: advance a rung only once the network varies enough "
                        "along the target direction")
    p.add_argument("--rung", type=float, default=11.25, help="degrees per rung in auto mode")
    p.add_argument("--block", type=int, default=250, help="updates between rung decisions in auto mode")
    p.add_argument("--auto-threshold", type=float, default=0.5,
                   help="fraction of the intact d' the next rung has to keep to be taken")
    p.add_argument("--updates", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--lr", type=float)
    p.add_argument("--anchor", type=float, default=0.0)
    p.add_argument("--basis-k", type=int, default=BASIS_K)
    p.add_argument("--out")
    a = p.parse_args()
    a.lr = a.lr or signals.LR[a.cond]
    ladder = "auto" if a.auto else f"s{a.steps}"
    a.out = a.out or (f"results/manifold_{a.kind}_{a.cond}_a{a.angle:g}_{ladder}_net{a.net}"
                      f"_c{a.candidate}_r{a.rep}.csv")
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    run(a)


if __name__ == "__main__":
    main()
