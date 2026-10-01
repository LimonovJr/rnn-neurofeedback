"""After training under closed-loop or yoked difficulty: is the positional code lost, or has it
moved out of the directions the frozen head reads?

Block 52 found the frozen head's decoding error nine times larger after yoked training, with
behaviour unchanged. The head was fitted on the starting state, so a larger error through it says
the code left its directions, not that the network holds less about the cued position. Three
readings on the start and on both end states separate the two:

- a decoder refitted on that state the way the head was fitted (ridge in the top-32 principal
  directions of the delay states), scored on held-out trials with the noise off, beside the frozen
  head scored the same way;
- crossnobis and participation ratio of the delay code;
- the share of the refitted decoder's weights that lies inside the frozen head's two directions.

Whether behaviour still uses location in each end state is asked as in `experiments.location`:
the drop of best-threshold d' when the distractors are moved to the cued position.

    python -m experiments.relocated --nets 10,11
"""

import argparse
import os

import numpy as np
import torch

from experiments.fit_head import ALPHAS, delay_states, fit, score
from experiments.location import VARIANTS
from src import behaviour, representation, runs, tasks
from src.model import CKPT, load

ENV = tasks.sameloc(*tasks.C1)
RANK = 32


def refit(model, net, n_fit, n_eval):
    """The frozen head and a decoder fitted on this state, on the same held-out trials."""
    Xf, Yf, Tf = delay_states(model, ENV, n_fit, True, seed=4000 + net)
    Xe, Ye, Te = delay_states(model, ENV, n_eval, False, seed=9000 + net)
    ids = np.unique(Tf)
    held = set(np.random.default_rng(0).choice(ids, size=max(1, len(ids) // 5), replace=False).tolist())
    inner = np.array([t in held for t in Tf])
    alpha = min(ALPHAS, key=lambda al: score(Xf[inner], Yf[inner], Tf[inner],
                                             *fit(Xf[~inner], Yf[~inner], RANK, al))[0])
    W, b = fit(Xf, Yf, RANK, alpha)
    W0, b0 = model.W_aux.detach().numpy(), model.b_aux.detach().numpy()
    basis = np.linalg.qr(W0.T)[0]
    inside = float(((W @ basis) ** 2).sum() / (W ** 2).sum())
    rmse, acc = score(Xe, Ye, Te, W, b)
    rmse0, acc0 = score(Xe, Ye, Te, W0, b0)
    return dict(refit_rmse=round(rmse, 4), refit_acc=round(acc, 4), frozen_rmse=round(rmse0, 4),
                frozen_acc=round(acc0, 4), inside_frozen=round(inside, 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--fit-n", type=int, default=1200)
    p.add_argument("--eval-n", type=int, default=800)
    p.add_argument("--panel-trials", type=int, default=2500)
    p.add_argument("--sweep-n", default="1000,4000")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/relocated_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    n_select, n_report = (int(v) for v in a.sweep_n.split(","))
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        for state in ("theta0", "closed", "yoked"):
            model = load(net, head="c1")
            if state != "theta0":
                path = CKPT / "closed_loop" / f"closed_loop_{state}_net{net}_r0_u1200.pt"
                model.load_state_dict(torch.load(path, weights_only=True)["state_dict"],
                                      strict=False)
            row = dict(net=net, state=state, **refit(model, net, a.fit_n, a.eval_n))
            model.eval()
            torch.manual_seed(43 + net)
            np.random.seed(43 + net)
            pan = representation.panel(model, ENV, n_trials=a.panel_trials, outcome=None)
            row.update(crossnobis=round(pan["crossnobis"], 5), pr=round(pan["pr"], 3))
            # the start was swept on these variants in block 46; the end states are swept here
            for name in ("task", "at_cue"):
                row[f"best_{name}"] = "" if state == "theta0" else behaviour.threshold_sweep(
                    model, VARIANTS[name], behaviour.detection, n_select, n_report)["d"]
            rows.append(row)
            print(f"  [net {net} {state:<6}] frozen acc {row['frozen_acc']:.3f} refit acc "
                  f"{row['refit_acc']:.3f} crossnobis {row['crossnobis']:.4f} inside "
                  f"{row['inside_frozen']:.3f} at_cue {row.get('best_at_cue', '')}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
