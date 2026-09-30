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

from src import behaviour, local, representation, runs, signals, tasks, trials
from src.model import CKPT, load, save
from src.training import TrainConfig, train_supervised

K = 6                              # checkpoints averaged into the reported level
CONDS = ["control", "position", "joint", "reward", "reward_aux", "expand", "gated", "control_fa",
         "operant", "axis"]
# what the network could already do before the run, scored through the same decision read-out
RETENTION = {"pretrain": tasks.stage(2), "patch": tasks.patch()}
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


def policy_shift(start, end, env_fn, n=256):
    """How far the run moved the network's policy on the task it trained on: the KL from the
    starting release probabilities to the final ones, per real time step. On-policy reinforcement
    is expected to find the solution closest to where it started (Shenfeld et al. 2025)."""
    x, _, _, _, _, _, lengths = trials.batch(env_fn, n, int(start.dt), window_ms=signals.WINDOW_MS)
    with torch.no_grad():
        p = torch.log_softmax(start(x, return_hidden=True)[0], -1)
        q = torch.log_softmax(end(x, return_hidden=True)[0], -1)
    kl = (p.exp() * (p - q)).sum(-1)
    valid = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    return float(kl[valid].mean())


def retention(model, n):
    return {name: round(behaviour.measure(model, env, n=n), 4) for name, env in RETENTION.items()}


def run(a):
    cfg = DEFAULTS[a.task]
    env_fn = cfg["env"]
    model = load(a.net, head=a.task)
    if a.refit_decoder:
        # the decoder a neurofeedback or BCI protocol uses is fitted afterwards, in a calibration
        # session, not trained together with the network: a fresh read-out, fitted on the frozen
        # state by read-out-only supervised training (the instrument's arm A), then frozen
        torch.manual_seed(1000 + a.net)
        with torch.no_grad():
            model.W_out.normal_(0.0, 1.0 / np.sqrt(model.W_out.shape[1]))
            model.W_out.mul_(model.out_mask)
            model.b_out.zero_()
        adapt_decoder(model, env_fn, a.refit_decoder, 1e-3)
        model.eval()
    if a.rec_noise_scale != 1.0:
        # internal noise as a degradation: raised in the buffer the network actually draws from,
        # and left on at evaluation, so behaviour is scored on the degraded network
        model.sigma_eff.mul_(a.rec_noise_scale)
        model.noise_at_eval = True
    w_rec0 = model.W_rec.detach().clone()
    w_out0 = model.W_out.detach().clone()
    start_model = copy.deepcopy(model) if a.retention else None
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

    crit = {}
    if a.criterion:
        sweep = behaviour.threshold_sweep(model, env_fn, cfg["score"], a.criterion_n, a.eval_n)
        crit.update(crit_offset_start=sweep["offset"], crit_d_start=sweep["d"],
                    crit_edge_start=sweep["edge"], auc_start=sweep["auc"], roc_start=sweep["roc"])
        if a.sampled_sweep:
            crit.update(roc_start_sampled=behaviour.threshold_sweep(
                model, env_fn, cfg["score"], a.criterion_n, a.eval_n, sampled=True)["roc"])
        print(f"  best threshold at the start: offset {sweep['offset']:+.3f}, d'={sweep['d']:+.4f}"
              + (" (on the edge of the grid)" if sweep["edge"] else ""), flush=True)

    target = None
    if a.cond == "axis":
        target = representation.decision_axis_target(model, env_fn, a.cosine, seed=a.net)
        crit.update(cosine=f"{a.cosine:g}", axis_in_manifold=round(target["in_manifold"], 4),
                    axis_auc_start=round(target["start_auc"], 4),
                    axis_m_start=round(target["start_m"], 4),
                    axis_increment=f"{a.axis_increment:g}" if a.axis_increment is not None else "")
        print(f"  target direction at cosine {a.cosine:+g} to the decision axis "
              f"({target['in_manifold']:.2f} of the axis lies in the manifold); separation along it "
              f"at the start {target['start_auc']:.3f}", flush=True)

    kept_before = retention(model, a.eval_n) if a.retention else {}
    if kept_before:
        print(f"  retention at the start: {kept_before}", flush=True)

    upds, curve, kept = [], [], deque(maxlen=K)
    snapshots = {}

    def hook(m, upd):
        d = behaviour.measure(m, env_fn, n=a.eval_n, score=cfg["score"])
        upds.append(upd)
        curve.append(round(d, 4))
        kept.append(copy.deepcopy(m.state_dict()))
        if upd in a.save_at:
            snapshots[upd] = copy.deepcopy(m.state_dict())
        print(f"  [{a.cond} net{a.net} r{a.rep}] upd {upd:5d}  d'={d:+.4f}", flush=True)
        if a.adapt_decoder and upd and upd % a.adapt_decoder == 0:
            adapt_decoder(m, env_fn, a.adapt_steps, a.adapt_lr)

    tracker = None
    if a.track:
        assert a.cond == "reward", "the threshold is tracked for the reward learner"
        tracker = dict(mode=a.track, rate=a.track_rate, every=a.track_every, step=a.track_step)
        if a.track == "yoke":
            tracker["path"] = runs.split(runs.read(a.yoke)[0]["offset_traj"])

    print(f"[nf {a.task} {a.cond} net{a.net} r{a.rep}] {a.updates} updates, lr={a.lr}, "
          f"lam={a.lam:g}, target={a.target}, anchor={a.anchor:g}, rule={a.rule}", flush=True)
    clipped = ""
    if a.rule == "local":
        assert a.cond in ("reward", "reward_aux"), "the local rule learns from reward only"
        local.train(model, env_fn, a.updates, lr=a.lr, explore=a.explore, explore_k=a.explore_k,
                    aux_weight=a.lam if a.cond == "reward_aux" else 0.0,
                    eval_every=a.eval_every, hook=hook, xy_from=a.target)
    else:
        history = signals.train(model, a.cond, env_fn, a.updates, lr=a.lr, lam=a.lam,
                      train_decoder=a.train_decoder, eval_every=a.eval_every, hook=hook,
                      xy_from=a.target, anchor=a.anchor, kappa=a.kappa, shuffle=a.shuffle,
                      optimizer=a.optimizer, fa_cost=a.fa_cost, fa_penalty=a.fa_penalty,
                      operant_reward=a.operant_reward, axis=target,
                      axis_increment=a.axis_increment, entropy=a.entropy, tracker=tracker,
                      baseline=a.baseline, batch=a.batch, temperature=a.temperature,
                      focus=(np.radians(a.focus_theta), np.radians(a.focus_width))
                      if a.focus_theta is not None else None)
        clipped = round(float(np.mean(history.get("clipped") or [0.0])), 3)
        if tracker:
            late = history.get("imbalance", [])[-len(history.get("imbalance", [])) // 4:]
            crit.update(offset_traj=runs.joined(history.get("offset", [])),
                        imbalance_traj=runs.joined(history.get("imbalance", [])),
                        hit_rate_traj=runs.joined(history.get("hit_rate", [])),
                        abort_traj=runs.joined(history.get("abort_rate", [])),
                        imbalance_late=round(float(np.mean(np.abs(late))), 4))

    # how far the run moved the weights at all, so that a signal that costs nothing can be told
    # from one that asked for nothing
    d_rec = float((model.W_rec.detach() - w_rec0).norm() / w_rec0.norm())
    d_out = float((model.W_out.detach() - w_out0).abs().max())
    d_aux = float((model.W_aux.detach() - w_aux0).abs().max())
    if not a.train_decoder and not a.adapt_decoder:
        assert d_out == 0.0, f"the decision read-out moved: {d_out:.3e}"
    assert d_aux == 0.0, f"the position head moved: {d_aux:.3e}"

    if a.criterion:
        model.eval()
        sweep = behaviour.threshold_sweep(model, env_fn, cfg["score"], a.criterion_n, a.eval_n)
        crit.update(crit_offset_end=sweep["offset"], crit_d_end=sweep["d"],
                    crit_edge_end=sweep["edge"], auc_end=sweep["auc"], roc_end=sweep["roc"])
        if a.sampled_sweep:
            crit.update(roc_end_sampled=behaviour.threshold_sweep(
                model, env_fn, cfg["score"], a.criterion_n, a.eval_n, sampled=True)["roc"])
        print(f"  best threshold at the end: offset {sweep['offset']:+.3f}, d'={sweep['d']:+.4f}"
              + (" (on the edge of the grid)" if sweep["edge"] else ""), flush=True)
    if target is not None:
        auc_u, m_u = representation.axis_separation(model, env_fn, target["u"])
        crit.update(axis_auc_end=round(auc_u, 4), axis_m_end=round(m_u, 4))

    kept_after = {}
    if a.retention:
        model.eval()
        kept_after = dict(kl_task=round(policy_shift(start_model, model, env_fn), 6),
                          **{f"{k}_start": v for k, v in kept_before.items()},
                          **{f"{k}_end": v for k, v in retention(model, a.eval_n).items()})
        print(f"  retention at the end: {kept_after}", flush=True)

    end = {}
    if a.panel:
        p = representation.panel(model, env_fn, n_trials=a.panel_trials, bin_by=cfg["bin_by"],
                                 outcome=outcome)
        end = dict(crossnobis=round(p["crossnobis"], 4), pr=round(p["pr"], 4))

    # saved after training and outside the hook, so no evaluation draws sit between the reported
    # curve and the weights the representation pass will read
    if snapshots:
        stem = os.path.splitext(os.path.basename(a.out))[0]
        for upd, state in snapshots.items():
            save(state, CKPT / a.ckpt_dir / f"{stem}_u{upd}.pt", cond=a.cond, net=a.net,
                 rep=a.rep, upd=int(upd), run=stem)
        print(f"  wrote {len(snapshots)} checkpoints at {sorted(snapshots)} to "
              f"{CKPT / a.ckpt_dir}", flush=True)
    elif a.ckpt_dir:
        for upd, state in zip(upds[-K:], kept):
            save(state, CKPT / a.ckpt_dir / f"{a.cond}_net{a.net}_r{a.rep}_u{upd}.pt",
                 cond=a.cond, net=a.net, rep=a.rep, upd=int(upd))
        print(f"  wrote {len(kept)} checkpoints to {CKPT / a.ckpt_dir}", flush=True)

    runs.write(a.out, dict(task=a.task, cond=a.cond, lam=f"{a.lam:g}", net=a.net, rep=a.rep,
                           updates=a.updates, lr=a.lr, eval_n=a.eval_n,
                           train_decoder=int(a.train_decoder), target=a.target,
                           anchor=f"{a.anchor:g}", kappa=f"{a.kappa:g}",
                           shuffle=int(a.shuffle),
                           adapt_decoder=a.adapt_decoder or 0,
                           rule=a.rule, explore=f"{a.explore:g}", explore_k=a.explore_k,
                           rec_noise_scale=f"{a.rec_noise_scale:g}", optimizer=a.optimizer,
                           refit_decoder=a.refit_decoder, entropy=f"{a.entropy:g}",
                           track=a.track or "", track_rate=f"{a.track_rate:g}" if a.track == "rate"
                           else "", yoke_from=os.path.basename(a.yoke) if a.yoke else "",
                           baseline=a.baseline, batch=a.batch, temperature=f"{a.temperature:g}",
                           focus_theta=f"{a.focus_theta:g}" if a.focus_theta is not None else "",
                           focus_width=f"{a.focus_width:g}" if a.focus_theta is not None else "",
                           baseline_decay=f"{a.baseline_decay:g}" if a.baseline == "running" else "",
                           track_every=a.track_every if a.track else "",
                           track_step=f"{a.track_step:g}" if a.track else "",
                           save_at=runs.joined(sorted(a.save_at)),
                           fa_cost=f"{a.fa_cost:g}", fa_penalty=f"{a.fa_penalty:g}"
                           if a.fa_penalty is not None else "", operant_reward=a.operant_reward,
                           clipped=clipped,
                           all_trials=int(a.all_trials), d0=curve[0] if curve else "",
                           dW_rec=f"{d_rec:.4f}", dW_out=f"{d_out:.3e}", dW_aux=f"{d_aux:.3e}",
                           curve_upd=runs.joined(upds), curve_dprime=runs.joined(curve),
                           **start, **end, **kept_after, **crit),
               ["task", "cond", "lam", "net", "rep", "updates", "lr", "eval_n", "train_decoder",
                "target", "anchor", "kappa", "shuffle", "adapt_decoder", "rule", "explore",
                "explore_k", "rec_noise_scale", "optimizer", "clipped", "refit_decoder", "entropy",
                "track", "track_rate", "track_every", "track_step", "yoke_from", "baseline",
                "baseline_decay", "batch", "temperature", "focus_theta", "focus_width",
                "imbalance_late",
                "offset_traj", "imbalance_traj", "hit_rate_traj", "abort_traj",
                "save_at", "all_trials", "d0",
                "kl_task", "pretrain_start", "pretrain_end", "patch_start", "patch_end",
                "fa_cost", "fa_penalty", "operant_reward", "cosine", "axis_increment",
                "axis_in_manifold", "axis_auc_start", "axis_auc_end", "axis_m_start", "axis_m_end",
                "crit_offset_start", "crit_d_start", "crit_edge_start", "auc_start",
                "crit_offset_end", "crit_d_end", "crit_edge_end", "auc_end", "roc_start", "roc_end",
                "roc_start_sampled", "roc_end_sampled",
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
    p.add_argument("--explore-k", type=int, default=0,
                   help="confine the local rule's exploration to the top-k activity directions")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam",
                   help="sgd removes Adam's per-parameter normalisation, which shrinks the steps "
                        "of a noisy gradient whatever its size")
    p.add_argument("--rec-noise-scale", type=float, default=1.0,
                   help="multiply the recurrent noise and keep it on at evaluation")
    p.add_argument("--fa-cost", type=float, default=1.0,
                   help="what a false alarm costs the reward signal; hits pay 1, misses cost 1")
    p.add_argument("--fa-penalty", type=float,
                   help="weight of the false-alarm penalty in control_fa (default: the ceiling arms')")
    p.add_argument("--operant-reward", choices=["position", "task"], default="position",
                   help="what the operant learner is paid: the position decoder, or the task outcome "
                        "(its positive control)")
    p.add_argument("--cosine", type=float, default=0.0,
                   help="cosine of the axis target's direction to the decision axis")
    p.add_argument("--save-at", default="",
                   help="comma-separated updates at which to keep the whole network, into --ckpt-dir")
    p.add_argument("--refit-decoder", type=int, default=0,
                   help="replace the decision read-out by a fresh one fitted on the frozen state for "
                        "this many read-out-only updates, as a calibration session would")
    p.add_argument("--track", choices=["balance", "rate", "yoke"],
                   help="move the frozen decoder's release bias during a reward run, as an "
                        "adaptive-threshold protocol would: to c = 0, to a hit rate, or along "
                        "another run's trajectory (--yoke)")
    p.add_argument("--track-rate", type=float, default=0.7, help="the hit rate --track rate holds")
    p.add_argument("--track-every", type=int, default=10,
                   help="updates of sampled training trials pooled per threshold move")
    p.add_argument("--track-step", type=float, default=2.0,
                   help="offset moved per unit of rate error")
    p.add_argument("--yoke", help="CSV of a tracked run whose offset trajectory to replay")
    p.add_argument("--baseline", choices=["batch", "type", "running"], default="batch",
                   help="the reward's baseline: the batch mean, each trial type's own mean, or a "
                        "slow average over past batches")
    p.add_argument("--baseline-decay", type=float, default=0.99,
                   help="how slowly the running baseline follows the batches")
    p.add_argument("--batch", type=int, default=signals.BATCH, help="trials per reward update")
    p.add_argument("--focus-theta", type=float,
                   help="expand: ask for kappa only on trials cued within --focus-width degrees of "
                        "this angle (degrees), the present code elsewhere")
    p.add_argument("--focus-width", type=float, default=45.0)
    p.add_argument("--temperature", type=float, default=1.0,
                   help="divide the reward policy's logits by this while training")
    p.add_argument("--entropy", type=float, default=signals.ENTROPY,
                   help="weight of the reward signal's entropy bonus")
    p.add_argument("--axis-increment", type=float,
                   help="ask the axis target for this rise in standardised separation along u, the "
                        "same at every angle, instead of the classifier's loss")
    p.add_argument("--sampled-sweep", action="store_true",
                   help="also sweep the threshold with sampled releases, the policy reward trains")
    p.add_argument("--criterion", action="store_true",
                   help="also find the release threshold that is best for the frozen decoder, at "
                        "the start and at the end, and score it on an independent draw")
    p.add_argument("--criterion-n", type=int, default=2000,
                   help="trials per offset when choosing the threshold")
    p.add_argument("--retention", action="store_true",
                   help="also score the pretraining task and the patch cell before and after the "
                        "run, and the policy shift on the trained task")
    p.add_argument("--kappa", type=float, default=1.5,
                   help="gain the expand signal asks the frozen position head for")
    p.add_argument("--shuffle", action="store_true",
                   help="give the expand signal another trial's position: the same loss asking for "
                        "a code that carries nothing")
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
    a.save_at = {int(u) for u in a.save_at.split(",") if u}
    if a.save_at and not a.ckpt_dir:
        p.error("--save-at needs --ckpt-dir")
    if (a.track == "yoke") != bool(a.yoke):
        p.error("--track yoke and --yoke go together")
    # every knob that changes what a run measures goes into its name, or a new condition would
    # collide with an old result and be skipped as already done
    tag = (("" if a.target == "target" else "_cue") + ("_local" if a.rule == "local" else "")
           + (f"_anchor{a.anchor:g}" if a.anchor else "")
           + (f"_k{a.kappa:g}" if a.cond == "expand" else "") + ("_shuf" if a.shuffle else "")
           + (f"_npk{a.explore_k}" if a.explore_k else "") + ("_ret" if a.retention else "")
           + ("_crit" if a.criterion else "")
           + (f"_rn{a.rec_noise_scale:g}" if a.rec_noise_scale != 1.0 else "")
           + ("_sgd" if a.optimizer == "sgd" else "")
           + (f"_fac{a.fa_cost:g}" if a.fa_cost != 1.0 else "")
           + (f"_fap{a.fa_penalty:g}" if a.fa_penalty is not None else "")
           + ("_task" if a.operant_reward == "task" else "")
           + (f"_cos{a.cosine:g}" if a.cond == "axis" else "")
           + (f"_inc{a.axis_increment:g}" if a.axis_increment is not None else "")
           + ("_samp" if a.sampled_sweep else "")
           + (f"_refit{a.refit_decoder}" if a.refit_decoder else "")
           + (f"_ent{a.entropy:g}" if a.entropy != signals.ENTROPY else "")
           + (f"_track{a.track_rate:g}" if a.track == "rate" else "")
           + ("_balance" if a.track == "balance" else "")
           + (f"_yoke{runs.read(a.yoke)[0]['net']}" if a.yoke else "")
           + ("_typebase" if a.baseline == "type" else "")
           + ("_runbase" if a.baseline == "running" else "")
           # block 37's tracker (every 25, step 1) names nothing more, so its files keep their names
           + (f"_every{a.track_every}_step{a.track_step:g}"
              if a.track and (a.track_every, a.track_step) != (25, 1.0) else "")
           + (f"_b{a.batch}" if a.batch != signals.BATCH else "")
           + (f"_T{a.temperature:g}" if a.temperature != 1.0 else "")
           + (f"_foc{a.focus_theta:g}" if a.focus_theta is not None else "")
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
