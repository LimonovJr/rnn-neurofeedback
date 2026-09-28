"""The neurofeedback signals, and the equal-budget supervised control they are compared against.

All conditions train the recurrent weights only: both read-outs are frozen, so whatever the signal
builds has to reach behaviour through a decoder that did not change.

    control     task cross-entropy through the frozen decision read-out
    position    delay-window error of the frozen position read-out, plus the task term at 0.2
    joint       positional error at weight lam, plus the task term at full weight
    reward      reinforcement on the trial outcome, with the release logit as the policy
    reward_aux  the same reinforcement plus the positional error at weight lam: the label-free
                regime, where a dense representational signal can shape what a sparse reward alone
                learns slowly
    expand      no task signal at all: make the frozen position head read the same positions
                kappa times further apart -- a within-manifold target
    gated       joint, with the positional term switched on only on the updates where its
                gradient points the same way as the task's (Du et al. 2018)
    control_fa  the control with the false-alarm penalty the ceiling arms train with, so the
                supervised learner is not left releasing too readily
    operant     neurofeedback proper: a scalar reward computed by the frozen position decoder from
                the delay state, learned by reinforcement on the network's own noise, with no task
                signal and no gradient through the decoder; with the task's outcome as the reward
                instead, the same learner's positive control
    axis        decoded neurofeedback along a fixed direction at a chosen angle to the decision
                axis: a frozen one-dimensional classifier on the state, separating response steps
                from false-alarm steps, trained through, with no task term

`anchor` adds a pull toward the starting recurrent weights, which is how the learning regime is
moved from rich toward lazy without touching the learning rate.
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from src import local, trials
from src.training import cross_entropy

LR = {"control": 3e-4, "position": 3e-4, "joint": 3e-4, "reward": 1e-4, "reward_aux": 1e-4,
      "expand": 3e-4, "gated": 3e-4, "control_fa": 3e-4, "operant": 1e-4, "axis": 3e-4}
FA_PENALTY = 0.4                                 # the weight the ceiling arms train with
OPTIMISERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
BATCH = 48
WINDOW_MS = (-300, 0)
TASK_WEIGHT = 0.2                                # position's task term, a fifth of the control's
ENTROPY = 0.05


def train(model, cond, env_fn, updates, lr=None, lam=1.0, train_decoder=False, eval_every=None,
          hook=None, device="cpu", xy_from="target", anchor=0.0, kappa=1.5, shuffle=False,
          optimizer="adam", fa_cost=1.0, fa_penalty=None, operant_reward="position", axis=None,
          axis_increment=None, entropy=ENTROPY, tracker=None, baseline="batch", batch=BATCH,
          baseline_decay=0.99,
          temperature=1.0):
    """`train_decoder` only applies to the position signal, whose task term back-propagates through
    the decision read-out unless it is frozen. `kappa` is the gain the expand signal asks the
    frozen position head for."""
    lr = LR[cond] if lr is None else lr
    if cond == "reward":
        return reinforce(model, env_fn, updates, lr, eval_every, hook, device, anchor=anchor,
                         optimizer=optimizer, fa_cost=fa_cost, entropy=entropy, tracker=tracker,
                         baseline=baseline, batch=batch, temperature=temperature,
                         baseline_decay=baseline_decay)

    freeze_decision = cond != "position" or not train_decoder
    if cond == "expand":
        step = expand_step(kappa, shuffle)
    elif cond == "gated":
        step = gated_step(lam)
    elif cond == "control_fa":
        step = control_fa_step(FA_PENALTY if fa_penalty is None else fa_penalty)
    elif cond == "operant":
        step = operant_step(shuffle, operant_reward)
    elif cond == "axis":
        step = axis_step(axis, axis_increment)
    elif cond == "reward_aux":
        step = reward_aux_step(lam)
    else:
        pos_weight = {"control": 0.0, "position": 1.0, "joint": lam}[cond]
        step = task_step(pos_weight, TASK_WEIGHT if cond == "position" else 1.0)
    clip = 5.0 if cond == "reward_aux" else 10.0
    return loop(model, env_fn, step, updates, lr, freeze_decision, eval_every, hook, device,
                xy_from=xy_from, anchor=anchor, grad_clip=clip, optimizer=optimizer)


def positional_error(model, h, xy_true, delay_mask):
    err = ((model.decode_xy(h) - xy_true) ** 2).sum(-1)
    per_trial = (err * delay_mask).sum(1) / delay_mask.sum(1).clamp_min(1)
    return per_trial.sum() / per_trial.shape[0]


def task_step(pos_weight, task_weight):
    def step(model, batch):
        x, y, mask, _, xy_true, delay_mask, _ = batch
        logits, _, h = model(x, return_hidden=True)
        ce = cross_entropy(logits, y, mask)
        if pos_weight == 0.0:
            return task_weight * ce, {"ce": ce.detach()}
        pos_err = positional_error(model, h, xy_true, delay_mask)
        return pos_weight * pos_err + task_weight * ce, {"pos_err": pos_err.detach(),
                                                         "ce": ce.detach()}
    return step


def reward_aux_step(lam):
    def step(model, batch):
        x, y, _, fa, xy_true, delay_mask, lengths = batch
        logits, _, h = model(x, return_hidden=True)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        pg, stats = policy_loss(torch.log_softmax(logits, -1), y == 1, fa > 0, valid)
        pos_err = positional_error(model, h, xy_true, delay_mask)
        return pg + lam * pos_err, {"reward": stats["reward"], "pos_err": pos_err.detach()}
    return step


def expand_step(kappa, shuffle=False):
    """A within-manifold target: ask the frozen position head to read the same positions, further
    apart. The delay state is pushed until the head decodes kappa times the true position, so the
    code is stretched along the directions the head already uses and nothing else is asked for.
    At kappa = 1 this is the plain positional signal, which is what makes the gain the variable.

    `shuffle` pairs each trial with another trial's position. The loss has the same form, the same
    scale and the same reach into the weights, and the code it asks for carries no information, so
    it separates the cost of improving the representation from the cost of training against a
    representational loss at all."""
    def step(model, batch):
        x, _, _, _, xy_true, delay_mask, _ = batch
        _, _, h = model(x, return_hidden=True)
        if shuffle:
            xy_true = xy_true[torch.randperm(xy_true.shape[0], device=xy_true.device)]
        err = positional_error(model, h, kappa * xy_true, delay_mask)
        return err, {"expand_err": err.detach()}
    return step


def control_fa_step(weight):
    def step(model, batch):
        x, y, mask, fa, _, _, _ = batch
        logits = model(x, return_hidden=True)[0]
        ce = cross_entropy(logits, y, mask)
        fa_pen = (torch.softmax(logits, -1)[..., 1] * fa).sum() / fa.sum().clamp_min(1)
        return ce + weight * fa_pen, {"ce": ce.detach(), "fa_pen": fa_pen.detach()}
    return step


def axis_step(target, increment=None):
    """The frozen classifier's loss, or -- with `increment` -- a demand for the same rise in
    separation along u at every angle: the standardised gap between the two classes along u is
    pulled to its starting value plus `increment`. The classifier's loss asks more of a direction
    the further it starts from being satisfied, which ties the dose to the starting distance; the
    increment asks the same of every direction, so the angle is left as the only dose."""
    u, centre, scale = target["u"], target["centre"], target["scale"]

    def step(model, batch):
        x, y, _, fa, _, _, lengths = batch
        _, _, h = model(x, return_hidden=True)
        valid = torch.arange(x.shape[1], device=x.device)[None, :] < lengths[:, None]
        sig, noise = (y == 1) & valid, (fa > 0) & valid
        if increment is not None:
            p_sig, p_noise = h[sig] @ u, h[noise] @ u
            gap = (p_sig.mean() - p_noise.mean()) / torch.sqrt((p_sig.var() + p_noise.var()) / 2)
            loss = (gap - (target["start_m"] + increment)) ** 2
            return loss, {"axis_gap": gap.detach()}
        logit = scale * (h @ u - centre)
        loss = (F.softplus(-logit)[sig].mean() + F.softplus(logit)[noise].mean()) / 2
        return loss, {"axis_loss": loss.detach()}
    return step


def operant_step(shuffle=False, reward_from="position"):
    """Reinforcement on the neural state rather than on an action. The network's own recurrent
    noise is the exploration; the reward of a trial is minus the delay-window error of the frozen
    position decoder on the states the network actually visited; the score function of the noise
    in the delay window carries the credit. With the visited states held fixed, as the score
    function requires, no credit flows back through time: the estimator is node perturbation's,
    here on every parameter and with the optimiser the other conditions use. That is the whole of
    what a scalar computed from the state can teach without differentiating the decoder, which
    would make it the supervised condition. `shuffle` scores each trial against another trial's
    position, the null that asks for nothing. `reward_from="task"` pays the trial's outcome under
    the greedy release instead, over the whole trial: the positive control, which says whether this
    learner can learn anything from a scalar at this scale."""
    def step(model, batch):
        x, y, _, fa, xy_true, delay_mask, lengths = batch
        model.noise_log, model.preact_log = [], []
        logits, _, h = model(x, return_hidden=True)
        xi = torch.stack(model.noise_log, 1).detach()
        mean = torch.stack(model.preact_log, 1)
        model.noise_log = model.preact_log = None
        if shuffle:
            xy_true = xy_true[torch.randperm(xy_true.shape[0], device=xy_true.device)]
        with torch.no_grad():
            err = ((model.decode_xy(h) - xy_true) ** 2).sum(-1)
            pos_err = (err * delay_mask).sum(1) / delay_mask.sum(1).clamp_min(1)
            if reward_from == "task":
                reward, _ = local.outcomes(logits.argmax(-1), y, fa, lengths)
                delay_mask = (torch.arange(x.shape[1], device=x.device)[None, :]
                              < lengths[:, None]).float()
            else:
                reward = -pos_err
            advantage = (reward - reward.mean()) / (reward.std() + 1e-6)
        # d/dtheta of log N(mean + xi | mean, sigma^2) is (xi / sigma^2) d mean/dtheta
        score = ((xi / model.sigma_eff ** 2) * mean).sum(-1)
        score = (score * delay_mask).sum(1) / delay_mask.sum(1).clamp_min(1)
        loss = -(advantage * score).mean()
        return loss, {"pos_err": pos_err.mean(), "reward": reward.mean()}
    return step


def gated_step(lam):
    """The joint signal with the positional term admitted only while it helps. On each batch the
    two gradients are taken separately; the positional one is added when its cosine with the task
    gradient is positive and dropped otherwise. The loss handed back is a surrogate whose gradient
    is exactly that sum, so the optimiser, the clipping and the anchor treat it like any other."""
    def step(model, batch):
        x, y, mask, _, xy_true, delay_mask, _ = batch
        logits, _, h = model(x, return_hidden=True)
        ce = cross_entropy(logits, y, mask)
        pos_err = positional_error(model, h, xy_true, delay_mask)
        params = [p for p in model.parameters() if p.requires_grad]
        filled = lambda gs: [torch.zeros_like(p) if g is None else g for p, g in zip(params, gs)]
        g_task = filled(torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True))
        g_aux = filled(torch.autograd.grad(pos_err, params, allow_unused=True))
        dot = sum((a * b).sum() for a, b in zip(g_task, g_aux))
        norm = lambda gs: torch.sqrt(sum((g ** 2).sum() for g in gs))
        cos = dot / (norm(g_task) * norm(g_aux)).clamp_min(1e-12)
        gate = float(cos > 0)
        surrogate = sum((p * (a + gate * lam * b)).sum() for p, a, b in zip(params, g_task, g_aux))
        return surrogate, {"ce": ce.detach(), "pos_err": pos_err.detach(), "cos": cos.detach(),
                           "gate": torch.tensor(gate)}
    return step


def loop(model, env_fn, step, updates, lr, freeze_decision, eval_every, hook, device,
         xy_from="target", anchor=0.0, grad_clip=10.0, log_every=50, optimizer="adam"):
    frozen = [model.W_aux, model.b_aux] + ([model.W_out, model.b_out] if freeze_decision else [])
    was = [p.requires_grad for p in frozen]
    for p in frozen:
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = OPTIMISERS[optimizer](params, lr=lr)
    w_rec0 = model.W_rec.detach().clone()

    history = defaultdict(list)
    if hook is not None and eval_every:
        hook(model, 0)
    for upd in range(1, updates + 1):
        model.train()
        batch = trials.batch(env_fn, BATCH, int(model.dt), device=device, window_ms=WINDOW_MS,
                             xy_from=xy_from)
        loss, logs = step(model, batch)
        if anchor:
            loss = loss + anchor * ((model.W_rec - w_rec0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        # under plain gradient descent a clipped step is a normalised one, so how often the clip
        # acts is part of what a run did
        history["clipped"].append(float(torch.nn.utils.clip_grad_norm_(params, grad_clip) > grad_clip))
        opt.step()

        for k, v in logs.items():
            history[k].append(float(v))
        if upd % log_every == 0 or upd == 1:
            print(f"upd {upd:5d}  " + "  ".join(f"{k}={history[k][-1]:.4f}" for k in logs), flush=True)
        if hook is not None and eval_every and upd % eval_every == 0:
            hook(model, upd)

    for p, r in zip(frozen, was):
        p.requires_grad_(r)
    return dict(history)


def policy_loss(logp, target, fa, valid, ent_coef=ENTROPY, fa_cost=1.0, groups=None,
                running=None):
    """REINFORCE on a sampled release time. Hit and correct rejection are +1, a false alarm
    -fa_cost, miss and abort -1, with the advantage normalised within the batch. With `groups`
    (one label per trial) the baseline is each trial type's own mean reward instead of the batch's,
    which takes out the variance that only says which kind of trial it was. With `running` (a dict
    holding a decay) the baseline and the scale are slow averages over past batches: when every
    trial of a batch ends the same way the batch's own baseline leaves no advantage at all, and a
    lagging one still does."""
    logp_hold, logp_release = logp[..., 0], logp[..., 1]
    b, t = logp_release.shape
    device = logp.device
    steps = torch.arange(t, device=device)[None, :]

    with torch.no_grad():
        released = (torch.rand_like(logp_release) < logp_release.exp()) & valid
        idx = torch.where(released, steps.expand(b, t),
                          torch.full((b, t), t, device=device, dtype=torch.long))
        first = idx.min(1).values
        did_release = first < t
        at = first.clamp(max=t - 1)
        hit = target.gather(1, at[:, None]).squeeze(1) & did_release
        false_alarm = fa.gather(1, at[:, None]).squeeze(1) & did_release
        has_target = target.any(1)
        correct_rejection = (~did_release) & (~has_target)
        correct = hit | correct_rejection
        reward = torch.where(correct, torch.ones(b, device=device),
                             torch.where(false_alarm, -fa_cost * torch.ones(b, device=device),
                                         -torch.ones(b, device=device)))
        if running is not None:
            if running.get("mean") is None:
                running["mean"], running["sd"] = float(reward.mean()), float(reward.std())
            adv = (reward - running["mean"]) / (running["sd"] + 1e-6)
            keep = running["decay"]
            running["mean"] = keep * running["mean"] + (1 - keep) * float(reward.mean())
            running["sd"] = keep * running["sd"] + (1 - keep) * float(reward.std())
        elif groups is None:
            adv = (reward - reward.mean()) / (reward.std() + 1e-6)
        else:
            centred = reward.clone()
            for g in groups.unique():
                centred[groups == g] -= reward[groups == g].mean()
            adv = centred / (centred.std() + 1e-6)

    before = (steps < first[:, None]) & valid
    logpi = (logp_hold * before).sum(1)
    logpi = logpi + torch.where(did_release, logp_release.gather(1, at[:, None]).squeeze(1),
                                torch.zeros(b, device=device))
    entropy = (-(logp.exp() * logp).sum(-1))[valid].mean()
    loss = -(adv * logpi).mean() - ent_coef * entropy

    stats = {"reward": reward.mean(), "frac_correct": correct.float().mean(),
             "hit": hit.float().mean(), "false_alarm": false_alarm.float().mean(),
             "miss": ((~did_release) & has_target).float().mean(),
             "correct_rejection": correct_rejection.float().mean(),
             **window_counts(first, target, fa, hit, false_alarm)}
    return loss, stats


def window_counts(first, target, fa, hit, false_alarm):
    """Go and no-go trials that reached the response window without releasing first. Over all
    trials, hits and false alarms both fall at the liberal end, where releases turn into aborts, so
    hit rate = 1 - false-alarm rate is crossed twice or not at all; among the trials that reached
    the window both rates rise with the offset and the crossing is unique."""
    window = target | fa
    start = window.float().argmax(1)
    reached = window.any(1) & (first >= start)
    go = target.any(1)
    return {"n_go": (go & reached).float().sum(), "n_hit": hit.float().sum(),
            "n_nogo": (~go & reached).float().sum(), "n_fa": false_alarm.float().sum(),
            "n_abort": (window.any(1) & ~reached).float().sum(),
            "n_trials": torch.tensor(float(len(first)))}


def track(model, tracker, bias0, counts, upd):
    """Move the frozen decoder's release bias the way an adaptive-threshold protocol does, from the
    learner's own sampled trials of the last few updates, with the recurrent noise on as in
    training: toward hit rate = 1 - false-alarm rate among the trials that reached the response
    window ("balance", c = 0, defined at any d'), toward a target hit rate among them ("rate"), or
    along another run's recorded trajectory ("yoke"), which moves the threshold without it
    following this network."""
    hit = counts["n_hit"] / max(counts["n_go"], 1.0)
    fa = counts["n_fa"] / max(counts["n_nogo"], 1.0)
    offset = float(model.b_out[1]) - bias0
    if tracker["mode"] == "yoke":
        path = tracker["path"]
        offset = path[min(upd // tracker["every"], len(path)) - 1]
    elif counts["n_go"] + counts["n_nogo"] == 0:
        pass                                     # nothing reached the window: no rate to follow
    elif tracker["mode"] == "balance":
        offset -= tracker["step"] * (hit + fa - 1.0)
    else:
        offset -= tracker["step"] * (hit - tracker["rate"])
    with torch.no_grad():
        model.b_out[1] = bias0 + offset
    return offset, hit, fa


def reinforce(model, env_fn, updates, lr, eval_every, hook, device, anchor=0.0, grad_clip=5.0,
              log_every=50, optimizer="adam", fa_cost=1.0, entropy=ENTROPY, tracker=None,
              baseline="batch", batch=BATCH, temperature=1.0, baseline_decay=0.99):
    """`temperature` divides the logits of the policy that is sampled and credited, not of the
    decoder that is scored: it keeps releases stochastic, and so the gradient alive, without
    touching where the greedy threshold sits."""
    for p in (model.W_aux, model.b_aux, model.W_out, model.b_out):
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = OPTIMISERS[optimizer](params, lr=lr)
    w_rec0 = model.W_rec.detach().clone()
    bias0 = float(model.b_out[1])
    counts = defaultdict(float)
    running = {"decay": baseline_decay, "mean": None} if baseline == "running" else None

    history = defaultdict(list)
    if hook is not None and eval_every:
        hook(model, 0)
    for upd in range(1, updates + 1):
        model.train()
        x, target, fa, valid, kinds = outcome_batch(env_fn, batch, device, kinds=True)
        logits, _, _ = model(x, return_hidden=True)
        if temperature != 1.0:
            logits = logits / temperature
        loss, stats = policy_loss(torch.log_softmax(logits, -1), target, fa, valid,
                                  ent_coef=entropy, fa_cost=fa_cost,
                                  groups=kinds if baseline == "type" else None,
                                  running=running)
        if anchor:
            loss = loss + anchor * ((model.W_rec - w_rec0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        # under plain gradient descent a clipped step is a normalised one, so how often the clip
        # acts is part of what a run did
        history["clipped"].append(float(torch.nn.utils.clip_grad_norm_(params, grad_clip) > grad_clip))
        opt.step()

        for k, v in stats.items():
            if k.startswith("n_"):
                counts[k] += v.item()
            else:
                history[k].append(v.item())
        if tracker and upd % tracker["every"] == 0:
            offset, hit, fa_rate = track(model, tracker, bias0, counts, upd)
            history["offset"].append(round(offset, 4))
            history["imbalance"].append(round(hit + fa_rate - 1.0, 4))
            history["hit_rate"].append(round(hit, 4))
            history["abort_rate"].append(round(counts["n_abort"] / max(counts["n_trials"], 1.0), 4))
            counts.clear()
        if upd % log_every == 0 or upd == 1:
            print(f"upd {upd:5d}  reward={history['reward'][-1]:+.3f}  "
                  f"hit={history['hit'][-1]:.3f} fa={history['false_alarm'][-1]:.3f} "
                  f"miss={history['miss'][-1]:.3f} cr={history['correct_rejection'][-1]:.3f}",
                  flush=True)
        if hook is not None and eval_every and upd % eval_every == 0:
            hook(model, upd)

    for p in (model.W_aux, model.b_aux, model.W_out, model.b_out):
        p.requires_grad_(True)
    return dict(history)


def outcome_batch(env_fn, batch_size, device, kinds=False):
    """Padded observations plus the masks that turn a chosen release time into an outcome, and with
    `kinds` each trial's type: context and match where the task has them, otherwise whether it
    carried distractors."""
    xs, targets, fas, lens, types = [], [], [], [], []
    for _ in range(batch_size):
        env = env_fn()
        env.reset()
        gt = env.gt
        t = env.trial
        types.append(2 * int(t.get("context", 0)) + int(t.get("is_match", t.get("has_distractors", 0))))
        fa = np.zeros(len(gt), dtype=bool)
        for e in env._distractor_events:
            fa[e["fa_start"]:e["fa_end"]] = True
        xs.append(env.ob.astype(np.float32))
        targets.append(gt == 1)
        fas.append(fa)
        lens.append(len(gt))

    t_max = max(lens)

    def pad(arrays, fill, dtype):
        out = np.full((len(arrays), t_max) + arrays[0].shape[1:], fill, dtype=dtype)
        for i, a in enumerate(arrays):
            out[i, :len(a)] = a
        return torch.tensor(out, device=device)

    valid = torch.zeros(batch_size, t_max, dtype=torch.bool, device=device)
    for i, n in enumerate(lens):
        valid[i, :n] = True
    out = pad(xs, 0.0, np.float32), pad(targets, False, bool), pad(fas, False, bool), valid
    return out + (torch.tensor(types, device=device),) if kinds else out
