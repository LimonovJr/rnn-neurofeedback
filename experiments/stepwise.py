"""What the saved states of a c1 run say that its end point alone cannot.

Four questions, on the networks a run kept at chosen updates (`nf --save-at`):

- **Before saturation.** Is discriminability the same for reward and supervision before either
  reaches the ceiling, or is "they differ only in the threshold" what any two learners look like
  once both have used up the headroom? The threshold sweep at every kept update.
- **Noise at evaluation.** The networks learn with recurrent noise on and are scored with it off.
  The payoff-optimal threshold at the end, greedy and sampled, with the noise left on.
- **Re-aiming.** How many dimensions the change occupies: the participation ratio of the singular
  values of the change in W_rec, and the share of the change in activity taken by its first
  principal direction.
- **Potent and null.** Where the change in activity goes relative to the frozen decision read-out:
  the share along its axis (output-potent) against the rest (output-null), and along the axis the
  part common to correct-release and false-alarm steps (the criterion) against the part that
  separates them (discriminability).

    python -m experiments.stepwise --net 0
"""

import argparse
import glob
import os

import numpy as np
import torch

from src import behaviour, runs, tasks, trials
from src.model import CKPT, load

ENV = tasks.sameloc(*tasks.C1)
SWEPT = ("reward", "control")


def restore(net, path):
    model = load(net, head="c1")
    model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
    model.eval()
    return model


def sweep(model, n_select, n_report, sampled=False):
    return behaviour.threshold_sweep(model, ENV, behaviour.detection, n_select, n_report,
                                     sampled=sampled)


def passage(model, batch):
    """The release margin during the hold and in the response window, read the way a greedy release
    reads it: a trial releases at the first step where margin plus offset is above zero, so what
    sets a trial's effective criterion is the largest margin it reaches, not the average one."""
    x, y, _, _, _, _, lengths = batch
    with torch.no_grad():
        logits = model(x, return_hidden=True)[0]
    margin = logits[..., 1] - logits[..., 0]
    t = torch.arange(x.shape[1])[None, :]
    valid = t < lengths[:, None]
    window = (y == 1) & valid
    first = torch.where(window.any(1), window.float().argmax(1), lengths)
    hold = valid & (t < first[:, None])
    keep = hold.sum(1) > 1
    m, h, w = margin[keep], hold[keep], window[keep]
    n = h.sum(1).float()
    level = (m * h).sum(1) / n
    sd = (((m - level[:, None]) ** 2 * h).sum(1) / (n - 1)).sqrt()
    low = torch.finfo(m.dtype).min
    hold_max = m.masked_fill(~h, low).max(1).values
    resp_max = m.masked_fill(~w, low).max(1).values[w.any(1)]
    return dict(level=float(level.mean()), sd=float(sd.mean()), hold_max=float(hold_max.mean()),
                resp_max=float(resp_max.mean()), z=float(np.sqrt(2 * np.log(n.mean()))))


def activity_change(start, end, batch):
    """Where the change in the state goes, on one fixed batch read with the noise off."""
    x, y, _, fa, _, _, lengths = batch
    with torch.no_grad():
        h0 = start(x, return_hidden=True)[2]
        h1 = end(x, return_hidden=True)[2]
    valid = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    dh = (h1 - h0)[valid]
    w = (end.W_out_eff()[1] - end.W_out_eff()[0]).detach()
    w_hat = w / w.norm()
    along = dh @ w_hat
    total = float((dh ** 2).sum())
    potent = float((along ** 2).sum())
    sig, noise = ((y == 1) & valid)[valid], ((fa > 0) & valid)[valid]
    s = torch.linalg.svdvals(dh - dh.mean(0))
    dw = (end.W_rec - start.W_rec).detach()
    sv = torch.linalg.svdvals(dw) ** 2
    # a representational target can move the operating point by scaling the state rather than by
    # aiming it: the mean shift of the unnormalised logit is in the units of the swept offset, and
    # the norm ratio says whether the population as a whole got louder or quieter
    gain = float(h1[valid].norm(dim=-1).mean() / h0[valid].norm(dim=-1).mean())
    # the part of the logit shift a uniform rescaling of the old state would have produced
    p0, p1 = passage(start, batch), passage(end, batch)
    # the offset that sits halfway between where the hold and the response window peak moves by
    # minus the mean of the two peaks' shifts; the level and the spread say which one moved them
    return dict(passage_offset=round(-((p1["hold_max"] - p0["hold_max"])
                                       + (p1["resp_max"] - p0["resp_max"])) / 2, 5),
                d_hold_level=round(p1["level"] - p0["level"], 5),
                d_hold_sd=round(p1["sd"] - p0["sd"], 5),
                d_hold_max=round(p1["hold_max"] - p0["hold_max"], 5),
                d_resp_max=round(p1["resp_max"] - p0["resp_max"], 5),
                hold_z=round(p0["z"], 3),
                logit_shift=round(float((dh @ w).mean()), 5),
                norm_ratio=round(gain, 4),
                gain_part=round((gain - 1) * float((h0[valid] @ w).mean()), 5),
                potent_share=round(potent / max(total, 1e-12), 5),
                potent_over_random=round(potent / max(total, 1e-12) * dh.shape[1], 3),
                criterion_along_axis=round(float((along[sig].mean() + along[noise].mean()) / 2), 5),
                separation_along_axis=round(float(along[sig].mean() - along[noise].mean()), 5),
                top_pc_share=round(float(s[0] ** 2 / (s ** 2).sum()), 4),
                dw_rank=round(float(sv.sum() ** 2 / (sv ** 2).sum()), 3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--ckpt-dir", default="nf_c1_steps")
    p.add_argument("--select-n", type=int, default=2000)
    p.add_argument("--report-n", type=int, default=10000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/stepwise_c1_net{a.net}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(11)
    np.random.seed(11)
    batch = trials.batch(ENV, 256, 20)
    rows = []
    for end_path in sorted(glob.glob(str(CKPT / a.ckpt_dir / f"*_net{a.net}_*_u1200.pt"))):
        stem = os.path.basename(end_path)[:-len("_u1200.pt")]
        meta = torch.load(end_path, weights_only=True)
        cond = meta["cond"]
        start = restore(a.net, CKPT / a.ckpt_dir / f"{stem}_u0.pt")
        end = restore(a.net, end_path)
        rows.append(dict(net=a.net, run=stem, cond=cond, kind="activity", upd=1200,
                         **activity_change(start, end, batch)))
        print(f"  [net {a.net} {stem}] {rows[-1]}", flush=True)
        if cond not in SWEPT or "_fa" in stem:
            # the end point only, so every run's threshold shift can be set against its logit shift
            s = sweep(end, a.select_n, a.report_n)
            rows.append(dict(net=a.net, run=stem, cond=cond, kind="sweep", upd=1200,
                             best_d=s["d"], offset=s["offset"], auc=s["auc"]))
            continue
        for path in sorted(glob.glob(str(CKPT / a.ckpt_dir / f"{stem}_u*.pt")),
                           key=lambda q: int(q.rsplit("_u", 1)[1][:-3])):
            upd = int(path.rsplit("_u", 1)[1][:-3])
            s = sweep(restore(a.net, path), a.select_n, a.report_n)
            rows.append(dict(net=a.net, run=stem, cond=cond, kind="sweep", upd=upd,
                             best_d=s["d"], offset=s["offset"], auc=s["auc"]))
            print(f"  [net {a.net} {cond} u{upd}] best d'={s['d']:.3f} auc={s['auc']:.3f}", flush=True)
        noisy = restore(a.net, end_path)
        noisy.noise_at_eval = True
        for sampled in (False, True):
            s = sweep(noisy, a.select_n, a.report_n, sampled=sampled)
            rows.append(dict(net=a.net, run=stem, cond=cond, upd=1200,
                             kind="noise_sampled" if sampled else "noise_greedy",
                             best_d=s["d"], offset=s["offset"], auc=s["auc"],
                             payoff_offset=behaviour.payoff_optimal(s["roc"])[0]))
    runs.write(a.out, rows, ["net", "run", "cond", "kind", "upd", "best_d", "offset", "auc",
                             "payoff_offset", "passage_offset", "d_hold_level", "d_hold_sd", "d_hold_max", "d_resp_max", "hold_z",
                             "logit_shift", "norm_ratio", "gain_part", "potent_share", "potent_over_random",
                             "criterion_along_axis", "separation_along_axis", "top_pc_share",
                             "dw_rank"])


if __name__ == "__main__":
    main()
