"""Whether the discriminability a representational target costs is lost from the network or from
its read-out.

After training on the magnified or the shuffled positional target, best-threshold d' through the
frozen decision read-out is lower than at the start, and moving the threshold has already been
allowed for. What is left can be a loss of access: the network still separates target from
distractor, in directions the frozen read-out does not use. Or it can be a loss of computation: it
separates them less, whatever reads it. The two are told apart the way the headroom instrument's
read-out arm does it. With the recurrent weights frozen, a fresh decision read-out is fitted on
each end state by read-out-only supervised training, and the threshold is swept again.

The same is done on the starting state and on the reward and supervised end states, so that what
refitting adds by itself is known. The fit is then continued for as many updates again and swept
once more, which says whether the first fit had converged.

    python -m experiments.access --nets 10,11
"""

import argparse
import os

import numpy as np
import torch

from experiments.nf import adapt_decoder, refit_decoder
from experiments.stepwise import ENV, sweep
from src import behaviour, runs
from src.model import CKPT, load

STATES = {"control": "nf_c1_control_lam1_n10000_lr0.0001_net{net}_r6_u1200.pt",
          "reward": "nf_c1_reward_lam1_n10000_net{net}_r6_u1200.pt",
          "k2": "nf_c1_expand_lam1_n10000_k2_lr0.0001_net{net}_r6_u1200.pt",
          "shuffled": "nf_c1_expand_lam1_n10000_k1_shuf_lr0.0001_net{net}_r6_u1200.pt"}


def restore(net, state):
    model = load(net, head="c1")
    if state != "theta0":
        path = CKPT / "nf_c1_steps" / STATES[state].format(net=net)
        model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
    model.eval()
    return model


def decision_axis(model):
    w = model.W_out_eff().detach()
    return w[1] - w[0]


def read(model, n_select, n_report, tag):
    s = sweep(model, n_select, n_report)
    own = next(pt for pt in s["roc"].split("|") if float(pt.split(":")[0]) == 0.0)
    hit, fa = (float(v) for v in own.split(":")[1:3])
    return {f"{tag}_best_d": s["d"], f"{tag}_offset": s["offset"], f"{tag}_auc": s["auc"],
            f"{tag}_edge": s["edge"], f"{tag}_own_d": round(behaviour.dprime(hit, fa), 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--states", default="theta0,control,reward,k2,shuffled")
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--select-n", type=int, default=2000)
    p.add_argument("--report-n", type=int, default=10000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/access_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        for state in a.states.split(","):
            # nothing but the state differs between a network's rows; the trials of each sweep
            # are drawn afresh
            torch.manual_seed(71 + net)
            np.random.seed(71 + net)
            model = restore(net, state)
            row = dict(net=net, state=state, updates=a.updates)
            row.update(read(model, a.select_n, a.report_n, "frozen"))
            old = decision_axis(model)
            refit_decoder(model, ENV, net, a.updates)
            new = decision_axis(model)
            row["cosine"] = round(float(torch.cosine_similarity(old, new, dim=0)), 4)
            row.update(read(model, a.select_n, a.report_n, "refit"))
            adapt_decoder(model, ENV, a.updates, 1e-3)
            model.eval()
            row.update(read(model, a.select_n, a.report_n, "longer"))
            rows.append(row)
            print(f"  [net {net} {state:<8}] best d' frozen {row['frozen_best_d']:.3f}  refit "
                  f"{row['refit_best_d']:.3f}  twice the updates {row['longer_best_d']:.3f}  "
                  f"cos {row['cosine']:+.2f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
