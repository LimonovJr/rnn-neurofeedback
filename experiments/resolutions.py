"""What the instrument can tell apart, measured rather than assumed.

One repeat is one full measurement call on a frozen network with no training: behaviour, the decode
battery and the representation panel. The resolution of each quantity is two standard deviations
over repeats, and the paired behavioural evaluations give the evaluation noise at the reporting
sample size directly.

    python -m experiments.resolutions --task c1 --rep 0
"""

import argparse
import os

from src import behaviour, representation, runs, tasks
from src.model import load

REEVAL = 10000
PANEL_N = 2500
TASKS = {"c1": dict(env=tasks.sameloc(*tasks.C1), score=behaviour.detection, bin_by="target"),
         "context": dict(env=tasks.context(), score=behaviour.go_nogo, bin_by="cue")}


def run(task, net, rep, reeval, panel_n, out):
    cfg = TASKS[task]
    model = load(net, head=task)
    d = [behaviour.measure(model, cfg["env"], n=reeval, score=cfg["score"]) for _ in range(2)]
    panel = representation.panel(model, cfg["env"], n_trials=panel_n, bin_by=cfg["bin_by"])

    row = dict(task=task, net=net, rep=rep, dprime=round(d[0], 4), dprime_paired=round(d[1], 4),
               crossnobis=round(panel["crossnobis"], 4), pr=round(panel["pr"], 4))
    if task == "context":
        row.update({k: round(v, 4) for k, v in
                    representation.decode_battery(model, cfg["env"]).items()})
    runs.write(out, row)
    print(f"[resolutions {task} rep {rep}] d'={d[0]:+.4f}/{d[1]:+.4f} "
          f"crossnobis={panel['crossnobis']:.4f} pr={panel['pr']:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--net", type=int, default=0)
    p.add_argument("--rep", type=int, required=True)
    p.add_argument("--reeval", type=int, default=REEVAL)
    p.add_argument("--panel-trials", type=int, default=PANEL_N)
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/resolution_{a.task}_net{a.net}_r{a.rep}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    run(a.task, a.net, a.rep, a.reeval, a.panel_trials, a.out)


if __name__ == "__main__":
    main()
