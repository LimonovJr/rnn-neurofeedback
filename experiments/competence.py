"""Does the representation form before behaviour leaves the floor?

Training an untrained network on c1 and measuring both quantities every few updates. A d' of exactly
zero is a clamp rather than a measurement: a network that never releases has a hit rate and a false
alarm rate both floored, so behaviour has no resolution below the release threshold. Crossnobis has
no such floor, and is pooled over all trials here, because the correct-only pool is empty exactly
where the question is.

    python -m experiments.competence --net 0 --rep 0
"""

import argparse

from src import behaviour, representation, runs, tasks
from src.model import build
from src.training import TrainConfig, train_supervised

MARKS = list(range(10, 201, 10))
BEHAVIOUR_N, PANEL_N = 4000, 2500
ENV = tasks.sameloc(*tasks.C1)


def measure(model, upd, behaviour_n, panel_n):
    d = behaviour.measure(model, ENV, n=behaviour_n)
    panel = representation.panel(model, ENV, n_trials=panel_n, outcome=None)
    model.train()
    print(f"  upd {upd:4d}  d'={d:+.4f}  crossnobis={panel['crossnobis']:.4f} pr={panel['pr']:.3f}",
          flush=True)
    return dict(upd=upd, dprime=round(d, 4), crossnobis=round(panel["crossnobis"], 4),
                pr=round(panel["pr"], 3))


def run(net, rep, marks, behaviour_n, panel_n, out):
    model = build(net)
    model.rec_noise = True
    rows = [measure(model, 0, behaviour_n, panel_n)]
    seen = set()

    def on_update(upd, m):
        if upd in marks and upd not in seen:
            seen.add(upd)
            rows.append(measure(m, upd, behaviour_n, panel_n))

    cfg = TrainConfig(batch_size=64, lr=1e-3, max_updates=marks[-1], print_every=10_000,
                      fa_penalty_weight=0.4, stop_on_no_miss=0)
    train_supervised(model, ENV, cfg, on_update=on_update)
    for r in rows:
        r.update(net=net, rep=rep)
    runs.write(out, rows, ["net", "rep", "upd", "dprime", "crossnobis", "pr"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--rep", type=int, default=0, help="repeat index; the transition is not stable "
                                                      "between repeats, which is the finding")
    p.add_argument("--marks", default=",".join(str(m) for m in MARKS))
    p.add_argument("--trials", type=int, default=BEHAVIOUR_N)
    p.add_argument("--panel-trials", type=int, default=PANEL_N)
    p.add_argument("--out")
    a = p.parse_args()
    run(a.net, a.rep, [int(m) for m in a.marks.split(",")], a.trials, a.panel_trials,
        a.out or f"results/competence_net{a.net}_r{a.rep}.csv")


if __name__ == "__main__":
    main()
