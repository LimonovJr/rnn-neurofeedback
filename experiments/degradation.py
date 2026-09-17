"""The sub-optimality axis: how much of its own performance a network loses to noise, with no
training at all, so degradation is separated from failure to adapt.

Recurrent noise is not usable as a knob. The level is fixed in a buffer when the network is built,
so raising it after loading only switches the noise on, and the measured d' is flat across the grid.
Input noise scales directly and is graded, which is why the ladder rungs are input-noise levels.

    python -m experiments.degradation --net 0
"""

import argparse

from src import behaviour, runs, tasks
from src.model import build, load

REEVAL = 10000
SIGMAS_IN = [0.05, 0.10, 0.15, 0.18, 0.20, 0.22, 0.26, 0.30, 0.34, 0.50]
ENV = tasks.sameloc(*tasks.C1)


def measure(model, sigma, reeval=REEVAL):
    model.sigma_in = float(sigma)
    model.noise_at_eval = sigma > 0.0             # the degradation has to be present when scoring
    model.rec_noise = False
    model.eval()
    return behaviour.measure(model, ENV, n=reeval)


def run(net, sigmas, reeval, out):
    rows = []
    intact = measure(load(net), 0.0, reeval)
    rows.append(dict(net=net, knob="input", sigma=0.0, dprime=round(intact, 4), suboptimality=0.0))
    print(f"[degradation net{net}] intact d'={intact:+.3f}", flush=True)

    naive = measure(build(net), 0.0, reeval)
    rows.append(dict(net=net, knob="untrained", sigma="", dprime=round(naive, 4), suboptimality=""))
    print(f"[degradation net{net}] untrained network d'={naive:+.3f} "
          f"(a clamp, not a point on the axis: it never releases)", flush=True)

    for sigma in sigmas:
        d = measure(load(net), sigma, reeval)
        rows.append(dict(net=net, knob="input", sigma=sigma, dprime=round(d, 4),
                         suboptimality=round(1 - d / intact, 4)))
        print(f"[degradation net{net}] sigma_in={sigma:<5} d'={d:+.3f}  s={1 - d / intact:+.3f}",
              flush=True)

    runs.write(out, rows, ["net", "knob", "sigma", "dprime", "suboptimality"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--sigmas", default=",".join(str(s) for s in SIGMAS_IN))
    p.add_argument("--reeval", type=int, default=REEVAL)
    p.add_argument("--out")
    a = p.parse_args()
    run(a.net, [float(s) for s in a.sigmas.split(",")], a.reeval,
        a.out or f"results/degradation_net{a.net}.csv")


if __name__ == "__main__":
    main()
