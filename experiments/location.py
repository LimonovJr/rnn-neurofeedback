"""How much of the network's discriminability on c1 passes through where things are.

On c1 the target appears at the cued position and distractors, stronger than the target, anywhere
on the same disk, during the delay. A release can be held back from a distractor because it is in
the wrong place, or because it comes at the wrong time. Which the network relies on decides whether
improving its positional code could ever help behaviour. The frozen network is swept on the task as
trained and on three variants that change only where the distractors are, or whether there is a
cue at all:

- `at_cue`: every distractor at the cued position, so location no longer tells them apart;
- `opposite`: every distractor at the edge of the disk opposite the cue, so location tells them
  apart as well as it can;
- `no_cue`: the cue is not shown (strength 0), so there is no position to hold. This one is also
  an input the network never saw after pretraining, which the other two are not.

The drop of best-threshold d' from the task to `at_cue` is the part of discriminability carried by
location.

    python -m experiments.location --nets 0,1,2
"""

import argparse
import math
import os

import numpy as np
import torch

from src import behaviour, runs, tasks
from src.model import CKPT, load

CUE, TARGET, DISTRACTOR = tasks.C1


class CueLocked(tasks.SameLocation):
    def positions(self):
        pos, theta = self.disk_point()
        self._cue = (pos, theta)
        return pos, pos, theta, self.quadrant(theta)

    def distractor_position(self, cue_theta):
        return self._cue


class Opposite(CueLocked):
    def distractor_position(self, cue_theta):
        theta = cue_theta + math.pi
        r = self.target_radius
        return (r * math.cos(theta), r * math.sin(theta)), theta


def variant(cls, cue=CUE):
    return lambda: cls(dt=20, cue_strength=cue, target_strength=TARGET, p_distractor_trial=0.6,
                       distractor_strength=DISTRACTOR)


VARIANTS = {"task": variant(tasks.SameLocation), "at_cue": variant(CueLocked),
            "opposite": variant(Opposite), "no_cue": variant(tasks.SameLocation, cue=0.0)}


def states(net, conds):
    yield "theta0", None
    names = {"control": f"nf_c1_control_lam1_n10000_lr0.0001_net{net}_r6_u1200.pt",
             "reward": f"nf_c1_reward_lam1_n10000_net{net}_r6_u1200.pt"}
    for cond in conds:
        path = CKPT / "nf_c1_steps" / names[cond]
        if path.exists():
            yield cond, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nets", required=True)
    p.add_argument("--conds", default="control", help="trained end states to read besides theta0")
    p.add_argument("--select-n", type=int, default=1000)
    p.add_argument("--report-n", type=int, default=4000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/location_c1_{a.nets.replace(',', '_')}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    torch.manual_seed(19)
    np.random.seed(19)
    rows = []
    for net in (int(n) for n in a.nets.split(",")):
        for state, path in states(net, a.conds.split(",")):
            model = load(net, head="c1")
            if path:
                model.load_state_dict(torch.load(path, weights_only=True)["state_dict"],
                                      strict=False)
            model.eval()
            for name, env_fn in VARIANTS.items():
                s = behaviour.threshold_sweep(model, env_fn, behaviour.detection, a.select_n,
                                              a.report_n)
                own = next(pt for pt in s["roc"].split("|") if float(pt.split(":")[0]) == 0.0)
                hit, fa = (float(v) for v in own.split(":")[1:3])
                rows.append(dict(net=net, state=state, variant=name, best_d=s["d"],
                                 offset=s["offset"], auc=s["auc"], edge=s["edge"],
                                 own_d=round(behaviour.dprime(hit, fa), 4), hit=hit, fa=fa))
                print(f"  [net {net} {state:<8} {name:<9}] best d'={s['d']:.3f} auc={s['auc']:.3f} "
                      f"own d'={rows[-1]['own_d']:.3f}", flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
