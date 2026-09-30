"""How well a representational target was induced, against what behaviour did, run by run.

Decoded-neurofeedback studies rest on a correlation across subjects between how well the target
pattern was induced and how much behaviour changed. The same correlation can be computed here
across networks, for the magnified and shuffled targets, and asked where it runs: through the
decoder's threshold, or through discriminability.

Induction is scored the way the targeted decoder scores it: the relative drop of the run's own
positional loss (its kappa, or the shuffled pairing) from the start to the end, on fixed batches.
The gain and R^2 of the frozen position head, and crossnobis and participation ratio of the delay
code, are recorded beside it. Behaviour is d' at the decoder's own threshold on 10 000 trials, the
best threshold and its d', the area under the ROC, and the first-passage statistics of the release
margin (`experiments.stepwise`). The threshold sweeps of block 34 and 40 are reused where they
exist; the kappa dose (block 45) is swept here.

    python -m experiments.induction --nets 10,11 --conds expand
"""

import argparse
import glob
import os

import numpy as np
import torch

from experiments.stepwise import activity_change, sweep
from src import behaviour, representation, runs, signals, tasks, trials
from src.model import CKPT, load

ENV = tasks.sameloc(*tasks.C1)
N_BATCH, BATCH = 8, 128


def restore(net, path=None):
    model = load(net, head="c1")
    if path:
        model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
    model.eval()
    return model


def kappa_of(stem):
    return next((float(t[1:]) for t in stem.split("_") if t.startswith("k") and t[1:2].isdigit()),
                1.0)


def head_fit(model, batches):
    """Gain and R^2 of the frozen position head on the delay window: slope of decoded on true."""
    dec, true = [], []
    with torch.no_grad():
        for x, _, _, _, xy, mask, _ in batches:
            out = model.decode_xy(model(x, return_hidden=True)[2])
            keep = mask > 0
            dec.append(out[keep])
            true.append(xy[keep])
    d, t = torch.cat(dec).numpy(), torch.cat(true).numpy()
    gain = float(np.mean([np.polyfit(t[:, k], d[:, k], 1)[0] for k in range(2)]))
    r2 = 1 - ((d - t * gain) ** 2).sum() / ((d - d.mean(0)) ** 2).sum()
    return gain, float(r2)


def own_loss(model, batches, kappa, shuffle, seed):
    step = signals.expand_step(kappa, shuffle=shuffle)
    torch.manual_seed(seed)
    with torch.no_grad():
        return float(np.mean([float(step(model, b)[0]) for b in batches]))


def stepwise_sweeps(net):
    """Block 34/40 sweeps of this network, keyed by run stem and update, where they exist."""
    path = f"results/stepwise_c1_net{net}.csv"
    if not os.path.exists(path):
        return {}
    return {(r["run"], int(r["upd"])): r for r in runs.read(path) if r["kind"] == "sweep"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--glob", default="nf_c1_expand_*_net{net}_r6_u1200.pt")
    p.add_argument("--panel-trials", type=int, default=2500)
    p.add_argument("--sweep-n", default="2000,10000", help="trials to choose and to report a sweep")
    p.add_argument("--exclude", help="leave out runs whose name contains this")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/induction_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    n_select, n_report = (int(v) for v in a.sweep_n.split(","))
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        torch.manual_seed(29 + net)
        np.random.seed(29 + net)
        batches = [trials.batch(ENV, BATCH, 20, window_ms=signals.WINDOW_MS) for _ in range(N_BATCH)]
        activity = trials.batch(ENV, 256, 20)
        start = restore(net)
        old = stepwise_sweeps(net)
        zero = [r for (run, upd), r in old.items() if upd == 0]
        s0 = (dict(best_d=np.mean([float(r["best_d"]) for r in zero]),
                   offset=np.mean([float(r["offset"]) for r in zero]),
                   auc=np.mean([float(r["auc"]) for r in zero])) if zero
              else {k: v for k, v in sweep(start, n_select, n_report).items()
                    if k in ("d", "offset", "auc")})
        s0 = dict(best_d=s0.get("best_d", s0.get("d")), offset=s0["offset"], auc=s0["auc"])
        own0 = behaviour.measure(start, ENV, n=10000)
        gain0, r20 = head_fit(start, batches)
        pan0 = representation.panel(start, ENV, n_trials=a.panel_trials, outcome=None)
        for path in sorted(glob.glob(str(CKPT / "nf_c1_steps" / a.glob.format(net=net)))):
            stem = os.path.basename(path)[:-len("_u1200.pt")]
            if a.exclude and a.exclude in stem:
                continue
            kappa, shuffle = kappa_of(stem), "_shuf" in stem
            end = restore(net, path)
            if (stem, 1200) in old:
                r = old[(stem, 1200)]
                s1 = dict(best_d=float(r["best_d"]), offset=float(r["offset"]), auc=float(r["auc"]))
            else:
                s = sweep(end, n_select, n_report)
                s1 = dict(best_d=s["d"], offset=s["offset"], auc=s["auc"])
            loss0 = own_loss(start, batches, kappa, shuffle, seed=net)
            loss1 = own_loss(end, batches, kappa, shuffle, seed=net)
            gain1, r21 = head_fit(end, batches)
            pan1 = representation.panel(end, ENV, n_trials=a.panel_trials, outcome=None)
            act = activity_change(start, end, activity)
            rows.append(dict(
                net=net, run=stem, kappa=kappa, shuffle=int(shuffle),
                induction=round((loss0 - loss1) / loss0, 4), loss_start=round(loss0, 5),
                loss_end=round(loss1, 5), d_gain=round(gain1 - gain0, 4), gain_end=round(gain1, 4),
                d_r2=round(r21 - r20, 4), d_crossnobis=round(pan1["crossnobis"] - pan0["crossnobis"], 5),
                d_pr=round(pan1["pr"] - pan0["pr"], 4),
                d_own=round(behaviour.measure(end, ENV, n=10000) - own0, 4),
                d_best=round(s1["best_d"] - s0["best_d"], 4), d_offset=round(s1["offset"] - s0["offset"], 4),
                d_auc=round(s1["auc"] - s0["auc"], 4),
                **{k: act[k] for k in ("d_hold_level", "d_hold_sd", "d_resp_max", "passage_offset",
                                       "norm_ratio", "logit_shift")}))
            r = rows[-1]
            print(f"  [net {net} k{kappa:g}{' shuf' if shuffle else ''}] induction {r['induction']:+.3f} "
                  f"own {r['d_own']:+.3f} best {r['d_best']:+.3f} offset {r['d_offset']:+.2f}",
                  flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
