"""What the threshold sweep can tell apart: the same frozen network swept again and again.

The best-threshold d' and the area under the ROC hull are measured repeatedly on one network with
nothing changed but the trials drawn. Two standard deviations over the repeats is the resolution
of each, and it is what the equivalence tests use as their margin, so the margin is fixed by the
instrument rather than chosen after the differences were seen.

    python -m experiments.criterion_resolution --task c1 --net 0 --repeats 10
"""

import argparse
import os

import numpy as np
import torch

from src import behaviour, runs, tasks
from src.model import load

TASKS = {"c1": (tasks.sameloc(*tasks.C1), behaviour.detection),
         "context": (tasks.context(), behaviour.go_nogo)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), default="c1")
    p.add_argument("--net", type=int, default=0)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--select-n", type=int, default=2000)
    p.add_argument("--report-n", type=int, default=10000)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/criterion_resolution_{a.task}_net{a.net}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    env_fn, score = TASKS[a.task]
    rows = []
    for rep in range(a.repeats):
        torch.manual_seed(rep)
        np.random.seed(rep)
        sweep = behaviour.threshold_sweep(load(a.net, head=a.task), env_fn, score, a.select_n,
                                          a.report_n)
        rows.append(dict(task=a.task, net=a.net, rep=rep, offset=sweep["offset"], best_d=sweep["d"],
                         auc=sweep["auc"]))
        print(f"  [{a.task} net {a.net} rep {rep}] best d'={sweep['d']:.4f} auc={sweep['auc']:.4f}",
              flush=True)
    runs.write(a.out, rows)


if __name__ == "__main__":
    main()
