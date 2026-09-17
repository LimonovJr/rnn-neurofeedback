"""Train one network through the three-stage curriculum: target only, then add the cue, then add
distractors. Each stage is saved, so a run can resume and the intermediate states stay available.

    python -m experiments.pretrain --net 0
"""

import argparse

import numpy as np
import torch

from src import tasks
from src.model import CKPT, build, load, save
from src.training import TrainConfig, train_supervised

UPDATES = (1000, 1000, 12000)


def run(net, updates):
    model = build(net)
    torch.manual_seed(net)
    np.random.seed(net)

    for k, cap in enumerate(updates):
        path = CKPT / f"net{net}" / f"stage{k}.pt"
        if path.exists():
            model = load(net, stage=k)
            print(f"[net {net}] reusing {path.name}", flush=True)
            continue
        print(f"[net {net}] stage {k}: {cap} updates", flush=True)
        # only the distractor stage runs long enough for early stopping to mean anything
        last = k == len(updates) - 1
        cfg = TrainConfig(batch_size=64, lr=1e-3, max_updates=cap, fa_penalty_weight=0.0,
                          print_every=50 if last else 100, stop_on_no_miss=3 if last else 0)
        train_supervised(model, tasks.stage(k), cfg)
        save(model, path, net=net, stage=k, updates=cap)
        print(f"[net {net}] saved {path}", flush=True)

    model.eval()
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--updates", default=",".join(str(u) for u in UPDATES))
    a = p.parse_args()
    run(a.net, [int(u) for u in a.updates.split(",")])


if __name__ == "__main__":
    main()
