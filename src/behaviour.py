"""Behaviour is scored as d', from the release the network actually makes."""

from collections import Counter

import numpy as np
import torch
from scipy.stats import norm

from src import trials


def dprime(hit, fa):
    clamp = lambda p: min(max(float(p), 0.01), 0.99)
    return float(norm.ppf(clamp(hit)) - norm.ppf(clamp(fa)))


def detection_rates(tr):
    """Hits over all trials, false alarms over the trials that carried a distractor."""
    counts = Counter(t["train_outcome"] for t in tr)
    n_distractor = sum(1 for t in tr if t["has_distractors"])
    return (counts.get("correct", 0) / max(len(tr), 1),
            counts.get("false_alarm", 0) / max(n_distractor, 1))


def detection(tr):
    return dprime(*detection_rates(tr))


def go_nogo(tr):
    """Hits over go trials, false alarms over no-go trials, which are the ones without a response
    window. Returns d' with the rates and counts behind it."""
    go = [t for t in tr if t["target_win_on"] is not None]
    nogo = [t for t in tr if t["target_win_on"] is None]
    hit = sum(1 for t in go if t["train_outcome"] == "correct") / max(len(go), 1)
    fa = sum(1 for t in nogo if t["train_outcome"] == "false_alarm") / max(len(nogo), 1)
    return dprime(hit, fa), hit, fa, len(go), len(nogo)


OUTCOMES = ("correct", "false_alarm", "miss", "abort")
COARSE = np.arange(-8.0, 4.01, 0.5)
FINE = np.arange(-0.5, 0.51, 0.125)


def measure_outcomes(model, env_fn, n=1000, score=detection, device="cpu", sampled=False):
    """d', the hit and false-alarm rates behind it, and the share of trials ending each way."""
    tr = trials.collect(model, env_fn, n_trials=n, device=device, batch_size=256, sampled=sampled)
    if score is go_nogo:
        d, hit, fa, _, _ = go_nogo(tr)
    else:
        hit, fa = detection_rates(tr)
        d = dprime(hit, fa)
    counts = Counter(t["train_outcome"] for t in tr)
    return dict(d=d, hit=hit, fa=fa, **{k: counts.get(k, 0) / max(len(tr), 1) for k in OUTCOMES})


def threshold_sweep(model, env_fn, score, n_select, n_report, sampled=False, coarse=None):
    """The best the frozen decoder can do by moving its release threshold alone: the offset on the
    release logit that maximises d' on one draw, coarse then fine, re-scored on an independent draw
    so the choice does not flatter the number it is reported with. Nothing is trained and the bias
    is put back. Every coarse point is kept, with the share of each outcome, so the threshold that
    is best for any payoff can be read off afterwards.

    `coarse` replaces the default grid of offsets. A learner that ends far on the conservative side
    needs offsets beyond +4 before its curve reaches its peak; on a grid that stops short, the hull
    joins the last point to (1, 1) by a line the network cannot reach and the area is too small."""
    # with a nonlinear head attached the release logit comes from the head's last layer
    release = model.b_out if model.readout_head is None else model.readout_head[-1].bias
    bias = release.detach().clone()
    offsets = COARSE if coarse is None else np.asarray(coarse, dtype=float)

    def at(offset, n):
        with torch.no_grad():
            release.copy_(bias)
            release[1] += offset
        return measure_outcomes(model, env_fn, n=n, score=score, sampled=sampled)

    coarse = [at(offset, n_select) for offset in offsets]
    centre = float(offsets[int(np.argmax([c["d"] for c in coarse]))])
    grid = centre + FINE
    fine = [at(offset, n_select)["d"] for offset in grid]
    best = float(grid[int(np.argmax(fine))])
    d = at(best, n_report)["d"]
    with torch.no_grad():
        release.copy_(bias)
    # releasing too early is an abort, not a hit, so hits fall again at the liberal end and the
    # points are not a curve; the area is taken under their upper convex hull, the best any
    # threshold mixture reaches
    hull = []
    for point in sorted([(0.0, 0.0), (1.0, 1.0)] + [(c["fa"], c["hit"]) for c in coarse]):
        while len(hull) >= 2 and ((hull[-1][0] - hull[-2][0]) * (point[1] - hull[-2][1])
                                  - (hull[-1][1] - hull[-2][1]) * (point[0] - hull[-2][0])) >= 0:
            hull.pop()
        hull.append(point)
    auc = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(hull, hull[1:]))
    roc = "|".join(f"{o:g}:" + ":".join(f"{c[k]:.4f}" for k in ("hit", "fa") + OUTCOMES)
                   for o, c in zip(offsets, coarse))
    return dict(offset=best, d=round(d, 4), edge=int(centre in (offsets[0], offsets[-1])),
                auc=round(auc, 4), roc=roc)


def payoff_optimal(roc, fa_cost=1.0):
    """The offset on a recorded sweep that maximises the expected payoff of a detection trial:
    +1 for a hit, -fa_cost for a false alarm, -1 for a miss or an abort. Detection only: a go/no-go
    sweep does not separate correct rejections from misses."""
    best, value = None, -np.inf
    for point in roc.split("|"):
        offset, _, _, correct, false_alarm, miss, abort = (float(v) for v in point.split(":"))
        payoff = correct - fa_cost * false_alarm - miss - abort
        if payoff > value:
            best, value = offset, payoff
    return best, value


def measure(model, env_fn, n=1000, score=detection, device="cpu"):
    tr = trials.collect(model, env_fn, n_trials=n, device=device, batch_size=256)
    out = score(tr)
    return out[0] if isinstance(out, tuple) else out
