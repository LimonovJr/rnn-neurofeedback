"""Whether a feedback signal reaches the decision through the decoder's own coupling to it.

In a linear network trained by gradient descent on a decoder's output, activity moves along the
noise covariance applied to the decoder, not along the decoder's weights, and the decision variable
follows the decoder's output with the slope it has under the network's own fluctuations: how the
two fluctuate together says how one responds when the other is trained. That is taken literally
here, with nothing fitted to the outcome.

On the starting network the fluctuations of the release margin are regressed on those of the
decoder's output: the same trials with and without the recurrent noise, the steps before the
response window. The trained network's decoder output is then read on the same trials, and its
change, times that slope, is added to the starting network's margin at every step. Releases, the
ROC and the margin's statistics are read off this predicted margin and set beside the trained
network's own.

For the positional targets the change of the delay state's tuning to the cued position is also
set against two directions: the head's weights, and the noise covariance applied to them.

    python -m experiments.response --nets 10,11
"""

import argparse
import os

import numpy as np
import torch

from experiments.access import STATES, restore
from experiments.stepwise import ENV
from src import behaviour, runs, signals, trials
from src.model import CKPT

DOSE = "nf_c1_expand_lam1_n2000_k{kappa}_lr0.0001_net{net}_r6_u1200.pt"
POSITIONAL = {"k0.5": DOSE.format(kappa="0.5", net="{net}"), "k1": DOSE.format(kappa="1", net="{net}"),
              "k1.5": DOSE.format(kappa="1.5", net="{net}"), "k2": STATES["k2"],
              "shuffled": STATES["shuffled"],
              "sector": "nf_c1_expand_lam1_n2000_k2_foc0_lr0.0001_net{net}_r6_u1200.pt"}
NEUTRAL = {"neutral k2": "nf_c1_neutral_lam1_n2000_k2_lr0.0001_net{net}_r6_u1200.pt",
           "neutral k3": "nf_c1_neutral_lam1_n2000_k3_lr0.0001_net{net}_r6_u1200.pt"}
MEASURES = ("level", "sd", "hold_max", "resp_max", "auc", "offset", "best_d", "own_d", "hit", "fa",
            "abort")


def rolled(model, seed, n, decoder, noisy=False):
    """Each trial as (env, targets, logits, decoder output per step, delay-mean state)."""
    model.noise_at_eval = noisy
    out = []
    for env, _, gt, logits, h in trials.rolled(model, trials.seeded(ENV, seed), n):
        a, b = trials.delay_window(int(env.start_ind["target"]), env.dt, signals.WINDOW_MS)
        out.append((env, gt, logits, decoder(h), h[a:b].mean(0)))
    model.noise_at_eval = False
    return out


def hold_steps(gt):
    window = np.where(gt == 1)[0]
    return np.arange(len(gt)) < (window[0] if len(window) else len(gt))


def coupling(model, net, decoder, n=512, repeats=4):
    """Slope of the margin's fluctuations on the decoder output's during the hold, the share of
    the margin's fluctuations it accounts for, and the noise covariance of the delay state."""
    clean = rolled(model, 6000 + net, n, decoder)
    dy, dm, dh = [], [], []
    for k in range(repeats):
        torch.manual_seed(100 * net + k)
        for (_, gt, logits, y, state), (_, _, logits0, y0, state0) in zip(
                rolled(model, 6000 + net, n, decoder, noisy=True), clean):
            hold = hold_steps(gt)
            dy.append((y - y0)[hold])
            dm.append(((logits[:, 1] - logits[:, 0]) - (logits0[:, 1] - logits0[:, 0]))[hold])
            dh.append(state - state0)
    dy, dm = np.concatenate(dy), np.concatenate(dm)
    slope = np.linalg.lstsq(dy, dm, rcond=None)[0]
    explained = 1 - ((dm - dy @ slope) ** 2).sum() / (dm ** 2).sum()
    return slope, float(explained), np.cov(np.array(dh).T)


def margin_statistics(kept):
    """As `stepwise.passage`, on trials already rolled out."""
    level, sd, hold_max, resp_max = [], [], [], []
    for _, gt, logits in kept:
        margin, hold = logits[:, 1] - logits[:, 0], hold_steps(gt)
        if hold.sum() > 1:
            level.append(margin[hold].mean())
            sd.append(margin[hold].std(ddof=1))
            hold_max.append(margin[hold].max())
        if (gt == 1).any():
            resp_max.append(margin[gt == 1].max())
    return dict(level=np.mean(level), sd=np.mean(sd), hold_max=np.mean(hold_max),
                resp_max=np.mean(resp_max))


def read(select, report):
    """Everything read off two samples of trials: the threshold is chosen on the first."""
    first, second = behaviour.roc_points(select), behaviour.roc_points(report)
    best = int(np.argmax([p["d"] for p in first]))
    own = next(p for p in second if p["offset"] == 0.0)
    return dict(margin_statistics(report),
                auc=behaviour.hull_area([(p["fa"], p["hit"]) for p in second]),
                offset=first[best]["offset"], best_d=second[best]["d"], own_d=own["d"],
                hit=own["hit"], fa=own["fa"], abort=own["abort"])


def tuning(sample):
    """How the delay-mean state depends on the cued position: one column per coordinate."""
    xy = np.array([env.trial["cue_pos"] for env, *_ in sample], dtype=float)
    states = np.array([state for *_, state in sample])
    design = np.column_stack([xy, np.ones(len(xy))])
    return np.linalg.lstsq(design, states, rcond=None)[0][:2].T


def inside(change, directions):
    """Share of a change of tuning that lies in the span of these directions."""
    q = np.linalg.qr(directions)[0]
    return float(((q.T @ change) ** 2).sum() / (change ** 2).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--n", type=int, default=3000, help="trials in each of the two seeded samples")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/response_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        start = restore(net, "theta0")
        head = start.W_aux.detach().numpy()
        positional = lambda h, w=head, b=start.b_aux.detach().numpy(): h @ w.T + b
        w, b, _ = signals.neutral_readout(start, ENV, net)
        neutral = lambda h, w=w.numpy(), b=b.numpy(): h @ w.T + b
        for decoder, stems in ((positional, POSITIONAL), (neutral, NEUTRAL)):
            stems = {k: v for k, v in stems.items()
                     if (CKPT / "nf_c1_steps" / v.format(net=net)).exists()}
            if not stems:
                continue
            slope, explained, noise = coupling(start, net, decoder)
            before = [rolled(start, seed + net, a.n, decoder) for seed in (5000, 7000)]
            at_start = read(*([t[:3] for t in sample] for sample in before))
            for state, stem in stems.items():
                end = restore(net, "theta0")
                end.load_state_dict(torch.load(CKPT / "nf_c1_steps" / stem.format(net=net),
                                               weights_only=True)["state_dict"], strict=False)
                after = [rolled(end, seed + net, a.n, decoder) for seed in (5000, 7000)]
                predicted = [[(env, gt, logits + np.column_stack(
                    [np.zeros(len(gt)), (y1 - y0) @ slope]).astype(np.float32))
                    for (env, gt, logits, y0, _), (_, _, _, y1, _) in zip(s0, s1)]
                    for s0, s1 in zip(before, after)]
                at_end, as_predicted = read(*([t[:3] for t in s] for s in after)), read(*predicted)
                # step by step during the hold: does the margin move as the decoder output times
                # the slope says, and by how much more than that
                moved, said = (np.concatenate(
                    [f(t0, t1)[hold_steps(t0[1])] for t0, t1 in zip(before[1], after[1])])
                    for f in (lambda t0, t1: (t1[2][:, 1] - t1[2][:, 0]) - (t0[2][:, 1] - t0[2][:, 0]),
                              lambda t0, t1: (t1[3] - t0[3]) @ slope))
                row = dict(net=net, state=state, slope_x=round(float(slope[0]), 4),
                           slope_y=round(float(slope[1]), 4), explained=round(explained, 4),
                           step_r=round(float(np.corrcoef(moved, said)[0, 1]), 4),
                           step_ratio=round(float(said.std() / moved.std()), 4),
                           output_change=round(float(np.sqrt(np.mean(
                               [((y1 - y0) ** 2).sum(1).mean() for (*_, y0, _), (*_, y1, _)
                                in zip(before[1], after[1])]))), 4))
                for m in MEASURES:
                    row[f"{m}_start"] = round(float(at_start[m]), 4)
                    row[f"{m}_end"] = round(float(at_end[m]), 4)
                    row[f"{m}_pred"] = round(float(as_predicted[m]), 4)
                if decoder is positional:
                    change = tuning(after[1]) - tuning(before[1])
                    row["in_pattern"] = round(inside(change, noise @ head.T), 4)
                    row["in_filter"] = round(inside(change, head.T), 4)
                else:
                    row["in_pattern"] = row["in_filter"] = ""
                rows.append(row)
                print(f"  [net {net} {state:<10}] best offset {at_end['offset'] - at_start['offset']:+.2f} "
                      f"(predicted {as_predicted['offset'] - at_start['offset']:+.2f})  spread "
                      f"{at_end['sd'] - at_start['sd']:+.2f} ({as_predicted['sd'] - at_start['sd']:+.2f})"
                      f"  best d' {at_end['best_d'] - at_start['best_d']:+.2f} "
                      f"({as_predicted['best_d'] - at_start['best_d']:+.2f})", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
