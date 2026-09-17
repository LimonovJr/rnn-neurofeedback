"""What the code contains at the end of a context-task neurofeedback run.

A separate pass over the checkpoints the run kept, so the representation can be measured without
retraining, and so the behavioural curve it is read against was produced by weights that no
evaluation draw sits between. The seed is fixed per checkpoint position, which makes the pass
repeatable.

    python -m experiments.representation --cond joint --net 0 --rep 0
"""

import argparse
import glob

import numpy as np
import torch

from src import representation, runs, tasks
from src.model import CKPT, load

PANEL_N = 2500
ENV = tasks.context()
FIELDS = ["cond", "net", "rep", "upd", "crossnobis", "pr", "context_acc", "match_acc", "xor_acc"]


def measure(model, panel_n=PANEL_N):
    model.rec_noise = True                # the resolutions were measured with the noise on
    model.eval()
    panel = representation.panel(model, ENV, n_trials=panel_n, bin_by="cue")
    decode = representation.decode_battery(model, ENV)
    return dict(crossnobis=panel["crossnobis"], pr=panel["pr"], context_acc=decode["context"],
                match_acc=decode["match"], xor_acc=decode["xor"])


def rows_for(cond, net, rep, ckpt_dir, repeats, panel_n):
    if cond == "theta0":
        for i in range(repeats):
            torch.manual_seed(1000 + i)
            np.random.seed(1000 + i)
            yield dict(upd="", **measure(load(net, head="context"), panel_n))
        return

    pattern = str(CKPT / ckpt_dir / f"{cond}_net{net}_r{rep}_u*.pt")
    files = sorted(glob.glob(pattern), key=lambda p: int(p.rsplit("_u", 1)[1].split(".")[0]))
    assert files, f"no checkpoints matched {pattern}"
    for i, path in enumerate(files):
        checkpoint = torch.load(path, weights_only=True)
        model = load(net, head="context")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        torch.manual_seed(1000 + i)
        np.random.seed(1000 + i)
        yield dict(upd=int(checkpoint["upd"]), **measure(model, panel_n))


def run(cond, net, rep, ckpt_dir, repeats, panel_n, out):
    rows = []
    for r in rows_for(cond, net, rep, ckpt_dir, repeats, panel_n):
        print(f"  [{cond} net{net} r{rep} u{r['upd']}] crossnobis={r['crossnobis']:+.4f} "
              f"pr={r['pr']:.3f} xor={r['xor_acc']:.4f}", flush=True)
        rows.append(dict(cond=cond, net=net, rep=rep,
                         **{k: (v if k == "upd" else round(v, 4)) for k, v in r.items()}))
    runs.write(out, rows, FIELDS)
    print(f"  crossnobis mean={np.mean([r['crossnobis'] for r in rows]):+.4f}  "
          f"xor mean={np.mean([r['xor_acc'] for r in rows]):.4f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", choices=["theta0", "control", "position", "joint", "reward"],
                   required=True)
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--repeats", type=int, default=6, help="only for theta0, which has no checkpoints")
    p.add_argument("--ckpt-dir", default="nf_context")
    p.add_argument("--panel-trials", type=int, default=PANEL_N)
    p.add_argument("--out")
    a = p.parse_args()
    run(a.cond, a.net, a.rep, a.ckpt_dir, a.repeats, a.panel_trials,
        a.out or f"results/repr_context_{a.cond}_net{a.net}_r{a.rep}.csv")


if __name__ == "__main__":
    main()
