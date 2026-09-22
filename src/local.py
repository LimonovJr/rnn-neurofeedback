"""A local learning rule for the recurrent weights: reward-modulated node perturbation.

The network's own recurrent noise is the exploration. Each trial is run with the noise on, scored
by its outcome, and every recurrent synapse is changed by how much the noise its postsynaptic unit
received covaried with its presynaptic input, times how much better than usual the trial went:

    dW_ij = lr * (R - R_bar) * (1/T) sum_t xi_i(t) h_j(t-1)

No gradient flows through anything. Both read-outs stay frozen, as in every other condition. The
positional signal enters as a second, dense reward term, so the question the rule lets us ask is
whether a representational signal helps a learner that estimates its gradient from noise -- where
it should reduce the variance of that estimate rather than divert a budget.

This is the rule of Legenstein et al. (2010) and Miconi (2017), in its simplest form. With
`explore_k` the noise is confined to the top principal directions of the network's own activity,
which is what makes a noise-based estimate affordable in a large network (Kang, Richards and
Sabatini 2026): isotropic noise has to cover every direction a unit can move in, most of which the
dynamics never visit.
"""

from collections import defaultdict

import torch

from src import trials

BATCH = 48
WINDOW_MS = (-300, 0)
BASELINE_DECAY = 0.9


def outcomes(actions, y, fa, lengths):
    """Per-trial outcome under the greedy policy: +1 for a hit or a correct rejection, -1 for a
    false alarm, a miss or an abort. Returns the reward and the rates behind it."""
    b, t = actions.shape
    steps = torch.arange(t, device=actions.device)[None, :]
    valid = steps < lengths[:, None]
    released = (actions == 1) & valid
    idx = torch.where(released, steps.expand(b, t), torch.full((b, t), t, device=actions.device))
    first = idx.min(1).values
    did_release = first < t
    at = first.clamp(max=t - 1)
    hit = (y.gather(1, at[:, None]).squeeze(1) == 1) & did_release
    false_alarm = (fa.gather(1, at[:, None]).squeeze(1) > 0) & did_release
    has_target = (y == 1).any(1)
    correct = hit | ((~did_release) & (~has_target))
    reward = correct.float() * 2 - 1
    rates = dict(hit=hit.float().mean(), false_alarm=false_alarm.float().mean(),
                 miss=((~did_release) & has_target).float().mean(),
                 correct_rejection=((~did_release) & (~has_target)).float().mean())
    return reward, rates


def activity_basis(model, env_fn, k, n=256, device="cpu"):
    """The top-k principal directions of the hidden state over every real time step of a batch."""
    x, _, _, _, _, _, lengths = trials.batch(env_fn, n, int(model.dt), device=device,
                                             window_ms=WINDOW_MS)
    with torch.no_grad():
        _, _, h = model(x, return_hidden=True)
    valid = torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]
    states = h[valid]
    _, s, vt = torch.linalg.svd(states - states.mean(0), full_matrices=False)
    explained = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return vt[:k].T.contiguous(), explained


def train(model, env_fn, updates, lr=1e-4, aux_weight=0.0, explore=1.0, explore_k=0,
          eval_every=None, hook=None, xy_from="target", device="cpu", log_every=50):
    """`explore` scales the recurrent noise during learning only; `explore_k` confines it to the
    top-k directions of the network's activity; `aux_weight` adds the standardised negative
    positional error to the trial reward."""
    for p in model.parameters():
        p.requires_grad_(False)
    if explore_k:
        model.noise_basis, explained = activity_basis(model, env_fn, explore_k, device=device)
        print(f"  exploration confined to {explore_k} directions, {explained:.3f} of the "
              f"activity variance", flush=True)
    sigma = model.sigma_eff.clone()
    model.sigma_eff.mul_(explore)
    baseline = 0.0
    history = defaultdict(list)

    if hook is not None and eval_every:
        hook(model, 0)
    for upd in range(1, updates + 1):
        model.train()
        x, y, _, fa, xy_true, delay_mask, lengths = trials.batch(
            env_fn, BATCH, int(model.dt), device=device, window_ms=WINDOW_MS, xy_from=xy_from)
        model.noise_log = []
        with torch.no_grad():
            logits, _, h = model(x, return_hidden=True)
        xi = torch.stack(model.noise_log, dim=1)                       # [B, T, H]
        model.noise_log = None

        with torch.no_grad():
            reward, rates = outcomes(logits.argmax(-1), y, fa, lengths)
            # measured whether or not it is fed back, so that a run without the term is the
            # manipulation check for a run with it
            err = ((model.decode_xy(h) - xy_true) ** 2).sum(-1)
            pos_err = (err * delay_mask).sum(1) / delay_mask.sum(1).clamp_min(1)
            rates["pos_err"] = pos_err.mean()
            if aux_weight:
                reward = reward - aux_weight * (pos_err - pos_err.mean()) / (pos_err.std() + 1e-6)
            advantage = reward - baseline
            baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * reward.mean().item()

            # presynaptic input to step t is h_{t-1}, with h_{-1} the resting state
            pre = torch.cat([model.h0.expand(x.shape[0], 1, -1), h[:, :-1]], dim=1)
            valid = (torch.arange(x.shape[1], device=device)[None, :] < lengths[:, None]).float()
            elig = torch.einsum("bti,btj->bij", xi * valid[:, :, None], pre) / lengths[:, None, None]
            step = (advantage[:, None, None] * elig).mean(0)
            model.W_rec.add_(lr * step * model.rec_mask)

        history["reward"].append(reward.mean().item())
        for k, v in rates.items():
            history[k].append(v.item())
        if upd % log_every == 0 or upd == 1:
            print(f"upd {upd:5d}  reward={history['reward'][-1]:+.3f}  hit={rates['hit']:.3f} "
                  f"fa={rates['false_alarm']:.3f} miss={rates['miss']:.3f} "
                  f"pos_err={rates['pos_err']:.4f} "
                  f"|dW|={float((lr * step).abs().max()):.2e}", flush=True)
        if hook is not None and eval_every and upd % eval_every == 0:
            hook(model, upd)

    model.sigma_eff.copy_(sigma)
    model.noise_basis = None
    for p in model.parameters():
        p.requires_grad_(True)
    return dict(history)
