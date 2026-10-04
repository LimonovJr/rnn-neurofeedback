"""One cell of the headroom measurement: network x arm x restart.

Arm A freezes everything but the decision read-out, arm B trains all weights, and arm A+ attaches a
32-unit nonlinear head to the frozen dynamics. Arm A+ is a third arm rather than a variant of A: the
head has sixteen times the parameters of the linear read-out, so it bounds the information present
in the code rather than what a read-out of the same class could use.

    python -m experiments.headroom --setting c1 --arm A --init fresh_a --net 0
"""

import argparse

import numpy as np
import torch

from src import behaviour, ceiling, runs, tasks
from src.model import load, mlp_readout

SETTINGS = {
    "c1": dict(env=tasks.sameloc(*tasks.C1), sigma_in=0.0, cap=4000),
    "c1r1": dict(env=tasks.sameloc(*tasks.C1), sigma_in=0.18, cap=4000),
    "c1r3": dict(env=tasks.sameloc(*tasks.C1), sigma_in=0.34, cap=4000),
    "patch": dict(env=tasks.patch(), sigma_in=0.0, cap=6000),
    "context": dict(env=tasks.context(), sigma_in=0.0, cap=6000, score=behaviour.go_nogo),
}


def run(setting, arm, init, net, cap, chunk, hidden, reeval, out):
    cfg = SETTINGS[setting]
    sigma_in = cfg["sigma_in"]
    env_fn = cfg["env"]
    score = cfg.get("score", behaviour.detection)
    lr = ceiling.LR[arm]

    m0 = load(net)
    # on a degraded rung the input noise is the degradation, so theta_0 and the convex fit are
    # scored on the network that is actually being measured
    m0.sigma_in = sigma_in
    m0.rec_noise = sigma_in == 0
    m0.noise_at_eval = sigma_in > 0
    m0.eval()

    state = m0.state_dict()
    d0 = behaviour.measure(m0, env_fn, n=reeval, score=score)
    convex = ceiling.convex_readout(m0, env_fn)
    assert convex[2] < 1e-4, f"folded read-out does not reproduce the fit: {convex[2]:.2e}"
    print(f"[{setting} {arm} net{net} {init}] theta0 d'={d0:+.3f} lr={lr} cap={cap} "
          f"sigma_in={sigma_in}", flush=True)

    head = None
    if arm == "Aplus":
        head = mlp_readout(hidden, ceiling.HEAD_SEEDS[init])
    elif init in ceiling.FRESH_SEEDS:
        torch.manual_seed(ceiling.FRESH_SEEDS[init])
        np.random.seed(ceiling.FRESH_SEEDS[init])

    def chunk_done(i, d):
        print(f"  chunk {i:3d} upd {(i + 1) * chunk:6d} d'(1k)={d:+.2f}", flush=True)

    C, traj, reevals, drift = ceiling.ceiling(net, state, env_fn, convex, init, arm, lr, cap,
                                              head=head, sigma_in=sigma_in, score=score,
                                              chunk=chunk, reeval=reeval, on_chunk=chunk_done)
    sd, se = ceiling.spread(reevals)
    if arm != "B":
        assert drift < 1e-6, f"arm {arm} moved the dynamics: {drift:.2e}"

    runs.write(out, dict(net=net, setting=setting, arm=arm, init=init, cap=cap, lr=lr,
                         sigma_in=sigma_in, theta0_d=f"{d0:.3f}", C_mean=f"{C:.3f}",
                         sd_K=f"{sd:.3f}", se_C=f"{se:.3f}", wrec_drift=f"{drift:.4f}",
                         reevals=runs.joined(reevals), curve=runs.joined(traj)))
    print(f"[{setting} {arm} net{net} {init}] C={C:+.3f} SD_K={sd:.3f} drift={drift:.1e}\n"
          f"  reevals={reevals}\n  curve={traj}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--setting", choices=list(SETTINGS), required=True)
    p.add_argument("--arm", choices=["A", "B", "Aplus"], required=True)
    p.add_argument("--init", choices=list(ceiling.RESTARTS), required=True)
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--cap", type=int)
    p.add_argument("--chunk", type=int, default=ceiling.CHUNK,
                   help="updates between checkpoints")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--reeval", type=int, default=ceiling.REEVAL,
                   help="trials in the independent re-evaluation of each checkpoint")
    p.add_argument("--out")
    a = p.parse_args()

    cap = a.cap or SETTINGS[a.setting]["cap"]
    out = a.out or f"results/headroom_{a.setting}_{a.arm}_net{a.net}_{a.init}.csv"
    run(a.setting, a.arm, a.init, a.net, cap, a.chunk, a.hidden, a.reeval, out)


if __name__ == "__main__":
    main()
