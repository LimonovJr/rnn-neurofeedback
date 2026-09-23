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
                signal and no gradient through the decoder

`anchor` adds a pull toward the starting recurrent weights, which is how the learning regime is
moved from rich toward lazy without touching the learning rate.
"""

from collections import defaultdict

import numpy as np
import torch

from src import trials
from src.training import cross_entropy

LR = {"control": 3e-4, "position": 3e-4, "joint": 3e-4, "reward": 1e-4, "reward_aux": 1e-4,
      "expand": 3e-4, "gated": 3e-4, "control_fa": 3e-4, "operant": 1e-4}
FA_PENALTY = 0.4                                 # the weight the ceiling arms train with
OPTIMISERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
BATCH = 48
WINDOW_MS = (-300, 0)
TASK_WEIGHT = 0.2                                # position's task term, a fifth of the control's
ENTROPY = 0.05


def train(model, cond, env_fn, updates, lr=None, lam=1.0, train_decoder=False, eval_every=None,
          hook=None, device="cpu", xy_from="target", anchor=0.0, kappa=1.5, shuffle=False,
          optimizer="adam"):
    """`train_decoder` only applies to the position signal, whose task term back-propagates through
    the decision read-out unless it is frozen. `kappa` is the gain the expand signal asks the
    frozen position head for."""
    lr = LR[cond] if lr is None else lr
    if cond == "reward":
        return reinforce(model, env_fn, updates, lr, eval_every, hook, device, anchor=anchor,
                         optimizer=optimizer)

    freeze_decision = cond != "position" or not train_decoder
    if cond == "expand":
        step = expand_step(kappa, shuffle)
    elif cond == "gated":
        step = gated_step(lam)
    elif cond == "control_fa":
        step = control_fa_step()
    elif cond == "operant":
        step = operant_step(shuffle)
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


def control_fa_step():
    def step(model, batch):
        x, y, mask, fa, _, _, _ = batch
        logits = model(x, return_hidden=True)[0]
        ce = cross_entropy(logits, y, mask)
        fa_pen = (torch.softmax(logits, -1)[..., 1] * fa).sum() / fa.sum().clamp_min(1)
        return ce + FA_PENALTY * fa_pen, {"ce": ce.detach(), "fa_pen": fa_pen.detach()}
    return step


def operant_step(shuffle=False):
    """Reinforcement on the neural state rather than on an action. The network's own recurrent
    noise is the exploration; the reward of a trial is minus the delay-window error of the frozen
    position decoder on the states the network actually visited; the score function of the noise
    in the delay window carries the credit. With the visited states held fixed, as the score
    function requires, no credit flows back through time: the estimator is node perturbation's,
    here on every parameter and with the optimiser the other conditions use. That is the whole of
    what a scalar computed from the state can teach without differentiating the decoder, which
    would make it the supervised condition. `shuffle` scores each trial against another trial's
    position, the null that asks for nothing."""
    def step(model, batch):
        x, _, _, _, xy_true, delay_mask, _ = batch
        model.noise_log, model.preact_log = [], []
        _, _, h = model(x, return_hidden=True)
        xi = torch.stack(model.noise_log, 1).detach()
        mean = torch.stack(model.preact_log, 1)
        model.noise_log = model.preact_log = None
        if shuffle:
            xy_true = xy_true[torch.randperm(xy_true.shape[0], device=xy_true.device)]
        with torch.no_grad():
            err = ((model.decode_xy(h) - xy_true) ** 2).sum(-1)
            pos_err = (err * delay_mask).sum(1) / delay_mask.sum(1).clamp_min(1)
            advantage = -(pos_err - pos_err.mean()) / (pos_err.std() + 1e-6)
        # d/dtheta of log N(mean + xi | mean, sigma^2) is (xi / sigma^2) d mean/dtheta
        score = ((xi / model.sigma_eff ** 2) * mean).sum(-1)
        score = (score * delay_mask).sum(1) / delay_mask.sum(1).clamp_min(1)
        loss = -(advantage * score).mean()
        return loss, {"pos_err": pos_err.mean()}
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


def policy_loss(logp, target, fa, valid, ent_coef=ENTROPY):
    """REINFORCE on a sampled release time. Hit and correct rejection are +1, false alarm, miss and
    abort -1, with the advantage normalised within the batch."""
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
        reward = torch.where(correct, torch.ones(b, device=device), -torch.ones(b, device=device))
        adv = (reward - reward.mean()) / (reward.std() + 1e-6)

    before = (steps < first[:, None]) & valid
    logpi = (logp_hold * before).sum(1)
    logpi = logpi + torch.where(did_release, logp_release.gather(1, at[:, None]).squeeze(1),
                                torch.zeros(b, device=device))
    entropy = (-(logp.exp() * logp).sum(-1))[valid].mean()
    loss = -(adv * logpi).mean() - ent_coef * entropy

    stats = {"reward": reward.mean(), "frac_correct": correct.float().mean(),
             "hit": hit.float().mean(), "false_alarm": false_alarm.float().mean(),
             "miss": ((~did_release) & has_target).float().mean(),
             "correct_rejection": correct_rejection.float().mean()}
    return loss, stats


def reinforce(model, env_fn, updates, lr, eval_every, hook, device, anchor=0.0, grad_clip=5.0,
              log_every=50, optimizer="adam"):
    for p in (model.W_aux, model.b_aux, model.W_out, model.b_out):
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = OPTIMISERS[optimizer](params, lr=lr)
    w_rec0 = model.W_rec.detach().clone()

    history = defaultdict(list)
    if hook is not None and eval_every:
        hook(model, 0)
    for upd in range(1, updates + 1):
        model.train()
        x, target, fa, valid = outcome_batch(env_fn, BATCH, device)
        logits, _, _ = model(x, return_hidden=True)
        loss, stats = policy_loss(torch.log_softmax(logits, -1), target, fa, valid)
        if anchor:
            loss = loss + anchor * ((model.W_rec - w_rec0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        # under plain gradient descent a clipped step is a normalised one, so how often the clip
        # acts is part of what a run did
        history["clipped"].append(float(torch.nn.utils.clip_grad_norm_(params, grad_clip) > grad_clip))
        opt.step()

        for k, v in stats.items():
            history[k].append(v.item())
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


def outcome_batch(env_fn, batch_size, device):
    """Padded observations plus the masks that turn a chosen release time into an outcome."""
    xs, targets, fas, lens = [], [], [], []
    for _ in range(batch_size):
        env = env_fn()
        env.reset()
        gt = env.gt
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
    return pad(xs, 0.0, np.float32), pad(targets, False, bool), pad(fas, False, bool), valid
