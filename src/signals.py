"""The neurofeedback signals, and the equal-budget supervised control they are compared against.

All four train the recurrent weights only: both read-outs are frozen, so whatever the signal builds
has to reach behaviour through a decoder that did not change.

    control   task cross-entropy through the frozen decision read-out
    position  delay-window error of the frozen position read-out, plus the task term at 0.2
    joint     positional error at weight lam, plus the task term at full weight
    reward    reinforcement on the trial outcome, with the release logit as the policy
"""

from collections import defaultdict

import torch

from src import trials
from src.training import cross_entropy

LR = {"control": 3e-4, "position": 3e-4, "joint": 3e-4, "reward": 1e-4}
BATCH = 48
WINDOW_MS = (-300, 0)
TASK_WEIGHT = 0.2                                # position's task term, a fifth of the control's


def train(model, cond, env_fn, updates, lr=None, lam=1.0, train_decoder=False, eval_every=None,
          hook=None, device="cpu"):
    """`train_decoder` only applies to the position signal, whose task term back-propagates through
    the decision read-out unless it is frozen."""
    lr = LR[cond] if lr is None else lr
    if cond == "reward":
        return reinforce(model, env_fn, updates, lr, eval_every, hook, device)

    freeze_decision = cond != "position" or not train_decoder
    return loop(model, env_fn, step_fn(cond, lam), updates, lr, freeze_decision, eval_every, hook,
                device)


def step_fn(cond, lam):
    pos_weight = {"control": 0.0, "position": 1.0, "joint": lam}[cond]
    task_weight = TASK_WEIGHT if cond == "position" else 1.0

    def step(model, batch):
        x, y, mask, _, xy_true, delay_mask, _ = batch
        logits, _, h = model(x, return_hidden=True)
        ce = cross_entropy(logits, y, mask)
        if pos_weight == 0.0:
            return task_weight * ce, {"ce": ce.detach()}

        err = ((model.decode_xy(h) - xy_true) ** 2).sum(-1)
        per_trial = (err * delay_mask).sum(1) / delay_mask.sum(1).clamp_min(1)
        pos_err = per_trial.sum() / per_trial.shape[0]
        return pos_weight * pos_err + task_weight * ce, {"pos_err": pos_err.detach(),
                                                         "ce": ce.detach()}
    return step


def loop(model, env_fn, step, updates, lr, freeze_decision, eval_every, hook, device,
         grad_clip=10.0, log_every=50):
    frozen = [model.W_aux, model.b_aux] + ([model.W_out, model.b_out] if freeze_decision else [])
    was = [p.requires_grad for p in frozen]
    for p in frozen:
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    history = defaultdict(list)
    if hook is not None and eval_every:
        hook(model, 0)
    for upd in range(1, updates + 1):
        model.train()
        batch = trials.batch(env_fn, BATCH, int(model.dt), device=device, window_ms=WINDOW_MS)
        loss, logs = step(model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
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


def reinforce(model, env_fn, updates, lr, eval_every, hook, device, grad_clip=5.0, ent_coef=0.05,
              log_every=50):
    """Sample a release time from the policy, score the trial outcome, and reinforce with an
    advantage against the batch. Hit and correct rejection are +1, false alarm, miss and abort -1."""
    for p in (model.W_aux, model.b_aux, model.W_out, model.b_out):
        p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    history = defaultdict(list)
    if hook is not None and eval_every:
        hook(model, 0)
    for upd in range(1, updates + 1):
        model.train()
        x, target, fa, valid = outcome_batch(env_fn, BATCH, device)
        logits, _, _ = model(x, return_hidden=True)
        logp = torch.log_softmax(logits, -1)
        logp_hold, logp_release = logp[..., 0], logp[..., 1]
        b, t = logp_release.shape
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
        logpi = logpi + torch.where(did_release,
                                    logp_release.gather(1, at[:, None]).squeeze(1),
                                    torch.zeros(b, device=device))
        entropy = (-(logp.exp() * logp).sum(-1))[valid].mean()
        loss = -(adv * logpi).mean() - ent_coef * entropy

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        opt.step()

        rates = (hit.float().mean().item(), false_alarm.float().mean().item(),
                 ((~did_release) & has_target).float().mean().item(),
                 correct_rejection.float().mean().item())
        history["reward"].append(reward.mean().item())
        history["frac_correct"].append(float(correct.float().mean()))
        for k, v in zip(("hit", "false_alarm", "miss", "correct_rejection"), rates):
            history[k].append(v)
        if upd % log_every == 0 or upd == 1:
            print(f"upd {upd:5d}  reward={history['reward'][-1]:+.3f}  "
                  f"hit={rates[0]:.3f} fa={rates[1]:.3f} miss={rates[2]:.3f} cr={rates[3]:.3f}",
                  flush=True)
        if hook is not None and eval_every and upd % eval_every == 0:
            hook(model, upd)

    for p in (model.W_aux, model.b_aux, model.W_out, model.b_out):
        p.requires_grad_(True)
    return dict(history)


def outcome_batch(env_fn, batch_size, device):
    """Padded observations plus the masks that turn a chosen release time into an outcome."""
    import numpy as np

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
