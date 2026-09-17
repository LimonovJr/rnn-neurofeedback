"""The positive control: a task where a linear read-out of frozen dynamics provably cannot succeed,
so the instrument must find headroom there or it is not measuring anything.

Go is the exclusive-or of "the probe is at the cued location" and the context, and the context
arrives as a pulse that has to be held across the delay. Before trusting the arms, the decode
battery checks that context and match are linearly available at decision time while the
exclusive-or is not: if the frozen dynamics have already linearised it, the task is mis-encoded
rather than the instrument being blunt.

This uses the earlier protocol -- best checkpoint, unequal restart counts between arms -- because
that is how the number in the report was produced. It is read for its order of magnitude.

    python -m experiments.positive_control --net 0
"""

import argparse
import copy

import numpy as np
import torch

from src import behaviour, ceiling, representation, runs, tasks
from src.model import build, load
from src.training import TrainConfig, train_supervised

CHUNK = 500
REEVAL = 10000
ARM = {"A": dict(lr=1e-3, cap=1500, early_stop=2, restarts=4),
       "B": dict(lr=5e-4, cap=3000, early_stop=3, restarts=3)}
ENV = tasks.context()


def score(model, n):
    return behaviour.measure(model, ENV, n=n, score=behaviour.go_nogo)


def train_to_best(model, arm, lr, cap, early_stop, margin=0.05):
    """Chunked training that keeps the best checkpoint on a held-out evaluation."""
    for p in model.parameters():
        p.requires_grad_(arm == "B")
    if arm == "A":
        model.W_out.requires_grad_(True)
        model.b_out.requires_grad_(True)

    wrec0 = model.W_rec.detach().clone()
    cfg = TrainConfig(batch_size=64, lr=lr, max_updates=CHUNK, print_every=CHUNK + 1,
                      fa_penalty_weight=0.4, stop_on_no_miss=0)
    best, best_state, curve, stalled = -1e9, None, [], 0
    for _ in range(max(1, cap // CHUNK)):
        train_supervised(model, ENV, cfg)
        d = score(model, 1200)
        curve.append(round(d, 2))
        if d > best + margin:
            best, best_state, stalled = d, copy.deepcopy(model.state_dict()), 0
        else:
            if d > best:
                best, best_state = d, copy.deepcopy(model.state_dict())
            stalled += 1
            if early_stop and stalled >= early_stop:
                break

    drift = float((model.W_rec.detach() - wrec0).abs().max())
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, curve, drift


def arm_restarts(net, state, convex, arm, reeval):
    cfg = ARM[arm]
    rows = []
    for i, init in enumerate(["incumbent", "convex"] + ["fresh"] * max(0, cfg["restarts"] - 2)):
        model = build(net)
        model.load_state_dict(state)
        model.rec_noise = True
        if init == "fresh":
            torch.manual_seed(500 + i)
            np.random.seed(500 + i)
        ceiling.set_readout(model, init, convex)
        best, curve, drift = train_to_best(model, arm, cfg["lr"], cfg["cap"], cfg["early_stop"])
        d = score(model, reeval)
        rows.append(dict(arm=arm, init=init, train_best=round(best, 3), reeval=round(d, 3),
                         wrec_drift=round(drift, 4)))
        print(f"  [{arm} {init:9s}] train={best:+.2f} reeval={d:+.2f} drift={drift:.1e} "
              f"curve={curve}", flush=True)
    return rows


def run(net, decode_only, reeval, out):
    model = load(net)
    theta0 = score(model, reeval)
    battery = representation.decode_battery(model, ENV)
    print(f"[control net{net}] theta0 d'={theta0:+.3f}  decode: context={battery['context']:.2f} "
          f"match={battery['match']:.2f} xor={battery['xor']:.2f}", flush=True)
    valid = battery["context"] > 0.7 and battery["match"] > 0.7 and battery["xor"] < 0.6
    print(f"  the control is {'valid' if valid else 'NOT valid: the task is mis-encoded'}", flush=True)
    if decode_only:
        return

    state = model.state_dict()
    convex = ceiling.convex_readout(model, ENV)
    rows = arm_restarts(net, state, convex, "A", reeval) + arm_restarts(net, state, convex, "B", reeval)
    for r in rows:
        r.update(net=net, theta0_d=f"{theta0:.3f}")
    runs.write(out, rows, ["net", "theta0_d", "arm", "init", "train_best", "reeval", "wrec_drift"])

    by_arm = {a: [r["reeval"] for r in rows if r["arm"] == a] for a in ("A", "B")}
    print(f"[control net{net}] gap by max = {max(by_arm['B']) - max(by_arm['A']):+.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--decode-only", action="store_true", help="run the validity check and stop")
    p.add_argument("--reeval", type=int, default=REEVAL)
    p.add_argument("--out")
    a = p.parse_args()
    run(a.net, a.decode_only, a.reeval, a.out or f"results/control_net{a.net}.csv")


if __name__ == "__main__":
    main()
