"""Supervised training on c1 under closed-loop difficulty (`src/closed_loop.py`), or yoked to
another run's difficulty, with the frozen decoder's threshold swept at the start and at the end
and the delay decoding error read before and after.

    python -m experiments.closed_loop --net 10 --rep 0
    python -m experiments.closed_loop --net 10 --rep 0 --yoke-from results/closed_loop_net11_r0.npy
"""

import argparse
import os

import numpy as np
import torch

from src import behaviour, closed_loop, runs, signals, tasks, trials
from src.model import CKPT, load, save

ENV = tasks.sameloc(*tasks.C1)


def decoding_error(model, n=512, seed=0):
    """Mean decoding error of the cued position over the last 300 ms of the delay, noise off."""
    torch.manual_seed(seed)
    batch = trials.batch(ENV, n, int(model.dt), window_ms=signals.WINDOW_MS)
    model.eval()
    with torch.no_grad():
        _, _, err = closed_loop.rollout(model, batch, tau=0.0, drop=0.0, temp=1.0)
    return float(err.mean())


def swept(model, n_select, n_report):
    s = behaviour.threshold_sweep(model, ENV, behaviour.detection, n_select, n_report)
    own = next(pt for pt in s["roc"].split("|") if float(pt.split(":")[0]) == 0.0)
    hit, fa = (float(v) for v in own.split(":")[1:3])
    return dict(best_d=s["d"], offset=s["offset"], auc=s["auc"], hit=hit, fa=fa,
                own_d=round(behaviour.dprime(hit, fa), 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--updates", type=int, default=1200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--drop", type=float, default=0.4, help="largest fall of the target strength")
    p.add_argument("--temp", type=float, default=0.05, help="width of the lapse boundary")
    p.add_argument("--yoke-from", help="another run's recorded strengths (.npy) to replay")
    p.add_argument("--sweep-n", default="2000,10000")
    p.add_argument("--ckpt-dir", default="closed_loop")
    p.add_argument("--out")
    a = p.parse_args()
    kind = "yoked" if a.yoke_from else "closed"
    stem = f"closed_loop_{kind}_net{a.net}_r{a.rep}"
    a.out = a.out or f"results/{stem}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    n_select, n_report = (int(v) for v in a.sweep_n.split(","))
    model = load(a.net, head="c1")
    torch.manual_seed(a.rep)
    tau = closed_loop.calibrate(model, ENV)
    start = dict(err=decoding_error(model), **swept(model, n_select, n_report))
    yoke = np.load(a.yoke_from) if a.yoke_from else None
    history = closed_loop.train(model, ENV, a.updates, a.lr, tau, drop=a.drop, temp=a.temp,
                                yoke=yoke)
    end = dict(err=decoding_error(model), **swept(model, n_select, n_report))
    np.save(f"results/{stem}.npy", history["schedule"])
    save(model, CKPT / a.ckpt_dir / f"{stem}_u{a.updates}.pt", net=a.net, rep=a.rep, kind=kind)
    late = slice(-len(history["lapse"]) // 4, None)
    runs.write(a.out, dict(net=a.net, rep=a.rep, kind=kind, yoke_from=a.yoke_from or "",
                           tau=round(tau, 5), drop=a.drop, temp=a.temp, updates=a.updates,
                           lapse_late=round(float(np.mean(history["lapse"][late])), 4),
                           strength_late=round(float(np.mean(history["strength"][late])), 4),
                           **{f"{k}_start": v for k, v in start.items()},
                           **{f"{k}_end": v for k, v in end.items()}))
    print(f"[{stem}] decoding error {start['err']:.4f} -> {end['err']:.4f}; best d' "
          f"{start['best_d']:.3f} -> {end['best_d']:.3f}; own d' {start['own_d']:.3f} -> "
          f"{end['own_d']:.3f}", flush=True)


if __name__ == "__main__":
    main()
