"""One neurofeedback run: task x condition x network x repeat.

Every condition trains the recurrent weights only and is scored through the read-outs it started
with, so the comparison is between signals rather than between a trained and an untrained network.
The control receives the same budget of ordinary supervised training through the same frozen
decoder, which is what makes the contrast a statement about the signal.

The run writes its raw curve; the level at the budget, the coupling and the contrasts are derived
in the aggregation pass.

    python -m experiments.nf --task context --cond joint --net 0 --rep 0
"""

import argparse
import copy
from collections import deque

import numpy as np
import torch

from src import behaviour, representation, runs, signals, tasks
from src.model import CKPT, load, save

K = 6                              # checkpoints averaged into the reported level
DEFAULTS = {
    "c1": dict(env=tasks.sameloc(*tasks.C1), score=behaviour.detection, bin_by="target",
               updates=1200, eval_every=150, eval_n=1000),
    "context": dict(env=tasks.context(), score=behaviour.go_nogo, bin_by="cue",
                    updates=8000, eval_every=500, eval_n=10000),
}


def run(task, cond, net, rep, lam, updates, eval_every, eval_n, panel, panel_n,
        train_decoder, ckpt_dir, out):
    cfg = DEFAULTS[task]
    env_fn = cfg["env"]
    model = load(net, head=task)
    w_out0 = model.W_out.detach().clone()
    w_aux0 = model.W_aux.detach().clone()
    torch.manual_seed(rep)
    np.random.seed(rep)

    start = {}
    if panel:
        p = representation.panel(model, env_fn, n_trials=panel_n, bin_by=cfg["bin_by"])
        start = dict(crossnobis_start=round(p["crossnobis"], 4), pr_start=round(p["pr"], 4))
        print(f"  start panel: crossnobis={p['crossnobis']:.4f} pr={p['pr']:.3f}", flush=True)

    upds, curve, kept = [], [], deque(maxlen=K)

    def hook(m, upd):
        d = behaviour.measure(m, env_fn, n=eval_n, score=cfg["score"])
        upds.append(upd)
        curve.append(round(d, 4))
        kept.append(copy.deepcopy(m.state_dict()))
        print(f"  [{cond} net{net} r{rep}] upd {upd:5d}  d'={d:+.4f}", flush=True)

    print(f"[nf {task} {cond} net{net} r{rep}] {updates} updates, lr={signals.LR[cond]}, lam={lam:g}",
          flush=True)
    signals.train(model, cond, env_fn, updates, lam=lam, train_decoder=train_decoder,
                  eval_every=eval_every, hook=hook)

    d_out = float((model.W_out.detach() - w_out0).abs().max())
    d_aux = float((model.W_aux.detach() - w_aux0).abs().max())
    if not train_decoder:
        assert d_out == 0.0, f"the decision read-out moved: {d_out:.3e}"
    assert d_aux == 0.0, f"the position head moved: {d_aux:.3e}"

    end = {}
    if panel:
        p = representation.panel(model, env_fn, n_trials=panel_n, bin_by=cfg["bin_by"])
        end = dict(crossnobis=round(p["crossnobis"], 4), pr=round(p["pr"], 4))

    # saved after training and outside the hook, so no evaluation draws sit between the reported
    # curve and the weights the representation pass will read
    if ckpt_dir:
        for upd, state in zip(upds[-K:], kept):
            save(state, CKPT / ckpt_dir / f"{cond}_net{net}_r{rep}_u{upd}.pt",
                 cond=cond, net=net, rep=rep, upd=int(upd))
        print(f"  wrote {len(kept)} checkpoints to {CKPT / ckpt_dir}", flush=True)

    runs.write(out, dict(task=task, cond=cond, lam=f"{lam:g}", net=net, rep=rep, updates=updates,
                         lr=signals.LR[cond], eval_n=eval_n, train_decoder=int(train_decoder),
                         d0=curve[0] if curve else "", dW_out=f"{d_out:.3e}", dW_aux=f"{d_aux:.3e}",
                         curve_upd=runs.joined(upds), curve_dprime=runs.joined(curve),
                         **start, **end),
               ["task", "cond", "lam", "net", "rep", "updates", "lr", "eval_n", "train_decoder",
                "d0", "crossnobis", "pr", "crossnobis_start", "pr_start", "dW_out", "dW_aux",
                "curve_upd", "curve_dprime"])
    print(f"[nf {task} {cond} net{net} r{rep}] curve={curve}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(DEFAULTS), required=True)
    p.add_argument("--cond", choices=["control", "position", "joint", "reward"], required=True)
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--lam", type=float, default=1.0, help="weight on the positional term of joint")
    p.add_argument("--updates", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--eval-n", type=int)
    p.add_argument("--panel", action="store_true", help="also measure the representation")
    p.add_argument("--panel-trials", type=int, default=2500)
    p.add_argument("--train-decoder", action="store_true",
                   help="leave the decision read-out trainable under the position signal")
    p.add_argument("--ckpt-dir", help="keep the last checkpoints, for the representation pass")
    p.add_argument("--out")
    a = p.parse_args()

    cfg = DEFAULTS[a.task]
    updates = a.updates or cfg["updates"]
    eval_n = a.eval_n or cfg["eval_n"]
    out = a.out or (f"results/nf_{a.task}_{a.cond}_lam{a.lam:g}_n{eval_n}_net{a.net}_r{a.rep}.csv")
    run(a.task, a.cond, a.net, a.rep, a.lam, updates, a.eval_every or cfg["eval_every"], eval_n,
        a.panel, a.panel_trials, a.train_decoder, a.ckpt_dir, out)


if __name__ == "__main__":
    main()
