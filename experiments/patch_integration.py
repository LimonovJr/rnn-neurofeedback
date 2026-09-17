"""Does the network average the stimulus over the window, and what is that worth?

Holding the blur fixed and varying only whether the stimulus window contains independent samples
(T_avg 20 ms redraws every frame, 100 ms holds one draw across the whole window) prices the
averaging directly. Pure evaluation, no training.

    python -m experiments.patch_integration --net 0
"""

import argparse

from src import behaviour, runs, tasks
from src.model import load
from src.tasks import PatchStimulus

REEVAL = 10000
JITTERS = [0.30, 0.50, 0.70, 0.90, 1.20]
T_AVGS = [20.0, 100.0]
R_RATIO = 1.2                                     # r_patch as a multiple of the jitter, as published


def patch_env(jitter, t_avg):
    cue, tgt, dstr = tasks.C1
    return lambda: PatchStimulus(dt=20, cue_strength=cue, target_strength=tgt,
                                 p_distractor_trial=0.6, distractor_strength=dstr,
                                 r_patch=R_RATIO * jitter, patch_jitter=jitter, T_avg=t_avg)


def run(net, reeval, out):
    model = load(net)
    model.rec_noise = False
    model.eval()

    plain = behaviour.measure(model, tasks.sameloc(*tasks.C1), n=reeval)
    rows = [dict(net=net, kind="no_patch", jitter=0.0, t_avg=0.0, r_patch=0.0,
                 dprime=round(plain, 4))]
    print(f"[patch net{net}] no patch: d'={plain:+.4f}", flush=True)

    for t_avg in T_AVGS:
        for jitter in JITTERS:
            d = behaviour.measure(model, patch_env(jitter, t_avg), n=reeval)
            rows.append(dict(net=net, kind="patch", jitter=jitter, t_avg=t_avg,
                             r_patch=round(R_RATIO * jitter, 3), dprime=round(d, 4)))
            print(f"[patch net{net}] jitter={jitter:.2f} T_avg={t_avg:.0f} d'={d:+.4f} "
                  f"({d - plain:+.4f} against no patch)", flush=True)

    runs.write(out, rows, ["net", "kind", "jitter", "t_avg", "r_patch", "dprime"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--reeval", type=int, default=REEVAL)
    p.add_argument("--out")
    a = p.parse_args()
    run(a.net, a.reeval, a.out or f"results/patch_net{a.net}.csv")


if __name__ == "__main__":
    main()
