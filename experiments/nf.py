"""One neurofeedback run: task x condition x network x repeat.

Every condition trains the recurrent weights only and is scored through the read-outs it started
with, so the comparison is between signals rather than between a trained and an untrained network.
The control receives the same budget of ordinary supervised training through the same frozen
decoder, which is what makes the contrast a statement about the signal. `reward_aux` against
`reward` is the label-free version of the same question.

The run writes its raw curve; the level at the budget, the coupling and the contrasts are derived
in the aggregation pass.

    python -m experiments.nf --task context --cond joint --net 0 --rep 0
    python -m experiments.nf --task context --cond reward_aux --net 0 --target cue --lam 0.5
    python -m experiments.nf --task c1 --cond control --net 0 --adapt-decoder 300
"""

import argparse
import os
import copy
from collections import deque

import numpy as np
import torch

from src import behaviour, local, representation, runs, signals, tasks
from src.model import CKPT, load, save
from src.training import TrainConfig, train_supervised

K = 6                              # checkpoints averaged into the reported level
CONDS = ["control", "position", "joint", "reward", "reward_aux", "expand"]
DEFAULTS = {
    "c1": dict(env=tasks.sameloc(*tasks.C1), score=behaviour.detection, bin_by="target",
               updates=1200, eval_every=150, eval_n=1000),
    "context": dict(env=tasks.context(), score=behaviour.go_nogo, bin_by="cue",
                    updates=8000, eval_every=500, eval_n=10000),
}


def adapt_decoder(model, env_fn, steps, lr):
    """A slow decoder update between neurofeedback blocks: the read-out alone is trained for a few
    supervised steps, the way a clinical decoder is periodically recalibrated."""
    trainable = {id(p): p.requires_grad for p in model.parameters()}
    for p in model.parameters():
        p.requires_grad_(False)
    model.W_out.requires_grad_(True)
    model.b_out.requires_grad_(True)
    train_supervised(model, env_fn, TrainConfig(batch_size=64, lr=lr, max_updates=steps,
                                                print_every=steps + 1, fa_penalty_weight=0.4,
                                                stop_on_no_miss=0))
    for p in model.parameters():
        p.requires_grad_(trainable[id(p)])


def run(a):
    cfg = DEFAULTS[a.task]
    env_fn = cfg["env"]
    model = load(a.net, head=a.task)
    w_rec0 = model.W_rec.detach().clone()
    w_out0 = model.W_out.detach().clone()
    w_aux0 = model.W_aux.detach().clone()
    torch.manual_seed(a.rep)
    np.random.seed(a.rep)
    outcome = None if a.all_trials else "correct"

    start = {}
    if a.panel:
        p = representation.panel(model, env_fn, n_trials=a.panel_trials, bin_by=cfg["bin_by"],
                                 outcome=outcome)
        start = dict(crossnobis_start=round(p["crossnobis"], 4), pr_start=round(p["pr"], 4))
        print(f"  start panel: crossnobis={p['crossnobis']:.4f} pr={p['pr']:.3f}", flush=True)

    upds, curve, kept = [], [], deque(maxlen=K)

    def hook(m, upd):
        d = behaviour.measure(m, env_fn, n=a.eval_n, score=cfg["score"])
        upds.append(upd)
        curve.append(round(d, 4))
        kept.append(copy.deepcopy(m.state_dict()))
        print(f"  [{a.cond} net{a.net} r{a.rep}] upd {upd:5d}  d'={d:+.4f}", flush=True)
        if a.adapt_decoder and upd and upd % a.adapt_decoder == 0:
            adapt_decoder(m, env_fn, a.adapt_steps, a.adapt_lr)

    print(f"[nf {a.task} {a.cond} net{a.net} r{a.rep}] {a.updates} updates, lr={a.lr}, "
          f"lam={a.lam:g}, target={a.target}, anchor={a.anchor:g}, rule={a.rule}", flush=True)
    if a.rule == "local":
        assert a.cond in ("reward", "reward_aux"), "the local rule learns from reward only"
        local.train(model, env_fn, a.updates, lr=a.lr, explore=a.explore,
                    aux_weight=a.lam if a.cond == "reward_aux" else 0.0,
                    eval_every=a.eval_every, hook=hook, xy_from=a.target)
    else:
        signals.train(model, a.cond, env_fn, a.updates, lr=a.lr, lam=a.lam,
                      train_decoder=a.train_decoder, eval_every=a.eval_every, hook=hook,
                      xy_from=a.target, anchor=a.anchor, kappa=a.kappa)

    # how far the run moved the weights at all, so that a signal that costs nothing can be told
    # from one that asked for nothing
    d_rec = float((model.W_rec.detach() - w_rec0).norm() / w_rec0.norm())
    d_out = float((model.W_out.detach() - w_out0).abs().max())
    d_aux = float((model.W_aux.detach() - w_aux0).abs().max())
    if not a.train_decoder and not a.adapt_decoder:
        assert d_out == 0.0, f"the decision read-out moved: {d_out:.3e}"
    assert d_aux == 0.0, f"the position head moved: {d_aux:.3e}"

    end = {}
    if a.panel:
        p = representation.panel(model, env_fn, n_trials=a.panel_trials, bin_by=cfg["bin_by"],
                                 outcome=outcome)
        end = dict(crossnobis=round(p["crossnobis"], 4), pr=round(p["pr"], 4))

    # saved after training and outside the hook, so no evaluation draws sit between the reported
    # curve and the weights the representation pass will read
    if a.ckpt_dir:
        for upd, state in zip(upds[-K:], kept):
            save(state, CKPT / a.ckpt_dir / f"{a.cond}_net{a.net}_r{a.rep}_u{upd}.pt",
                 cond=a.cond, net=a.net, rep=a.rep, upd=int(upd))
        print(f"  wrote {len(kept)} checkpoints to {CKPT / a.ckpt_dir}", flush=True)

    runs.write(a.out, dict(task=a.task, cond=a.cond, lam=f"{a.lam:g}", net=a.net, rep=a.rep,
                           updates=a.updates, lr=a.lr, eval_n=a.eval_n,
                           train_decoder=int(a.train_decoder), target=a.target,
                           anchor=f"{a.anchor:g}", kappa=f"{a.kappa:g}",
                           adapt_decoder=a.adapt_decoder or 0,
                           rule=a.rule, explore=f"{a.explore:g}",
                           all_trials=int(a.all_trials), d0=curve[0] if curve else "",
                           dW_rec=f"{d_rec:.4f}", dW_out=f"{d_out:.3e}", dW_aux=f"{d_aux:.3e}",
                           curve_upd=runs.joined(upds), curve_dprime=runs.joined(curve),
                           **start, **end),
               ["task", "cond", "lam", "net", "rep", "updates", "lr", "eval_n", "train_decoder",
                "target", "anchor", "kappa", "adapt_decoder", "rule", "explore", "all_trials", "d0",
                "crossnobis", "pr",
                "crossnobis_start", "pr_start", "dW_rec", "dW_out", "dW_aux",
                "curve_upd", "curve_dprime"])
    print(f"[nf {a.task} {a.cond} net{a.net} r{a.rep}] curve={curve}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(DEFAULTS), required=True)
    p.add_argument("--cond", choices=CONDS, required=True)
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--lam", type=float, default=1.0,
                   help="weight on the positional term (joint, reward_aux) or on leaving the "
                        "manifold (expand)")
    p.add_argument("--lr", type=float)
    p.add_argument("--updates", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--eval-n", type=int)
    p.add_argument("--target", choices=["target", "cue"], default="target",
                   help="what the positional term regresses toward; the published context runs "
                        "used the stimulus position, which is the probe there")
    p.add_argument("--anchor", type=float, default=0.0,
                   help="pull toward the starting recurrent weights (the lazy-regime knob)")
    p.add_argument("--adapt-decoder", type=int, default=0,
                   help="every this many updates, let the decision read-out take a few supervised "
                        "steps (co-adaptation); 0 keeps it frozen")
    p.add_argument("--adapt-steps", type=int, default=20)
    p.add_argument("--adapt-lr", type=float, default=1e-3)
    p.add_argument("--rule", choices=["bptt", "local"], default="bptt",
                   help="local = reward-modulated node perturbation on the recurrent weights, "
                        "with the network's own noise as the exploration")
    p.add_argument("--explore", type=float, default=1.0,
                   help="scale on the recurrent noise while the local rule learns")
    p.add_argument("--kappa", type=float, default=1.5,
                   help="gain the expand signal asks the frozen position head for")
    p.add_argument("--panel", action="store_true", help="also measure the representation")
    p.add_argument("--panel-trials", type=int, default=2500)
    p.add_argument("--all-trials", action="store_true",
                   help="pool every trial in the panel, not only the correct ones")
    p.add_argument("--train-decoder", action="store_true",
                   help="leave the decision read-out trainable under the position signal")
    p.add_argument("--ckpt-dir", help="keep the last checkpoints, for the representation pass")
    p.add_argument("--out")
    a = p.parse_args()

    cfg = DEFAULTS[a.task]
    a.updates = a.updates or cfg["updates"]
    a.eval_every = a.eval_every or cfg["eval_every"]
    a.eval_n = a.eval_n or cfg["eval_n"]
    a.lr = a.lr or (1e-4 if a.rule == "local" else signals.LR[a.cond])
    # every knob that changes what a run measures goes into its name, or a new condition would
    # collide with an old result and be skipped as already done
    tag = (("" if a.target == "target" else "_cue") + ("_local" if a.rule == "local" else "")
           + (f"_anchor{a.anchor:g}" if a.anchor else "")
           + (f"_k{a.kappa:g}" if a.cond == "expand" else "")
           + (f"_adapt{a.adapt_decoder}" if a.adapt_decoder else "")
           + (f"_u{a.updates}" if a.updates != DEFAULTS[a.task]["updates"] else "")
           + (f"_lr{a.lr:g}" if a.lr != signals.LR[a.cond] and a.rule != "local" else ""))
    a.out = a.out or (f"results/nf_{a.task}_{a.cond}_lam{a.lam:g}_n{a.eval_n}{tag}"
                      f"_net{a.net}_r{a.rep}.csv")
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    run(a)


if __name__ == "__main__":
    main()
