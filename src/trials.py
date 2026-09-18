"""Rolling trials out of a task and turning them into records or training batches."""

import numpy as np
import torch


def collect(model, env_fn, n_trials=200, device="cpu", batch_size=None):
    """Roll out trials under a greedy policy. `batch_size` pads several trials into one forward pass,
    which is 20-60x faster; the network is causal so padding does not change a trial's states, but
    the noise draws differ, so batched and per-trial collection match in distribution only."""
    model.eval()
    dev = torch.device(device)

    if not batch_size:
        out = []
        for _ in range(n_trials):
            env = env_fn()
            env.reset()
            x = env.ob.astype(np.float32).copy()
            gt = env.gt.astype(np.int64).copy()
            with torch.no_grad():
                logits, _, h = model(torch.tensor(x, device=dev).unsqueeze(0), return_hidden=True)
            out.append(record(env, x, gt, logits[0].cpu().numpy(), h[0].cpu().numpy()))
        return out

    out, done = [], 0
    while done < n_trials:
        b = min(int(batch_size), n_trials - done)
        envs, xs, lens = [], [], []
        for _ in range(b):
            env = env_fn()
            env.reset()
            envs.append(env)
            xs.append(env.ob.astype(np.float32))
            lens.append(len(env.gt))
        xb = np.zeros((b, max(lens), xs[0].shape[1]), dtype=np.float32)
        for i, x in enumerate(xs):
            xb[i, :len(x)] = x
        with torch.no_grad():
            logits, _, h = model(torch.tensor(xb, device=dev), return_hidden=True)
        lg, hh = logits.cpu().numpy(), h.cpu().numpy()
        for i, env in enumerate(envs):
            T = lens[i]
            out.append(record(env, xs[i], env.gt.astype(np.int64), lg[i, :T], hh[i, :T]))
        done += b
    return out


def record(env, x_seq, gt_seq, logits, h):
    """One trial dict, with the outcome reconstructed from the env's own event windows."""
    actions = np.argmax(logits, axis=-1)
    win = np.where(gt_seq == 1)[0]
    released = np.where(actions == 1)[0]
    resp_on = int(released[0]) if len(released) else None
    target_onset = int(env.start_ind["target"])

    if resp_on is None:
        outcome, rt_ms = "miss", None
    elif gt_seq[resp_on] == 1:
        outcome, rt_ms = "correct", int((resp_on - target_onset) * env.dt)
    else:
        fa = next((e for e in env._distractor_events if e["fa_start"] <= resp_on < e["fa_end"]), None)
        if fa is None:
            outcome, rt_ms = "abort", None
        else:
            outcome, rt_ms = "false_alarm", int((resp_on - fa["onset_step"]) * env.dt)

    return {
        "x": x_seq, "gt": gt_seq, "h": h, "logits": logits, "a": actions,
        "cue_onset": int(env.start_ind["cue"]), "target_onset": target_onset,
        "target_win_on": int(win[0]) if len(win) else None,
        "target_win_off": int(win[-1] + 1) if len(win) else None,
        "model_resp_on": resp_on,
        "cue_pos": env.trial.get("cue_pos"), "target_pos": env.trial.get("target_pos"),
        "cue_theta": env.trial.get("cue_theta"), "target_theta": env.trial.get("target_theta"),
        "ctoa_ms": env.trial.get("ctoa_ms"), "ctoa_bin": env.trial.get("ctoa_bin"),
        "has_distractors": env.trial.get("has_distractors", False),
        "n_distractors": env.trial.get("n_distractors", 0),
        "distractor_positions": env.trial.get("distractor_positions", []),
        "context": env.trial.get("context"), "is_match": env.trial.get("is_match"),
        "is_go": env.trial.get("is_go"),
        "fa_windows": [(int(e["fa_start"]), int(e["fa_end"])) for e in env._distractor_events],
        "train_outcome": outcome, "rt_ms": rt_ms,
    }


def targets_and_mask(gt_seq, response_weight=1.0, baseline_weight=1.0):
    targets = gt_seq.astype(np.int64).copy()
    mask = np.full(len(gt_seq), baseline_weight, dtype=np.float32)
    release = np.where(gt_seq == 1)[0]
    if len(release):
        mask[int(release[0]):int(release[-1]) + 1] = response_weight
    return targets, mask


def batch(env_fn, batch_size, dt, device="cpu", window_ms=None, xy_from="target"):
    """One padded training batch: inputs, go/no-go targets, loss mask, false-alarm mask, the trial's
    (x, y) and the frames on which a positional loss applies, and trial lengths.

    `window_ms` (relative to target onset) selects a delay window instead of the default, which is
    from target onset to the end of the trial. `xy_from` picks the regression target: the stimulus
    position, or the cue position, which is the one that is knowable during the delay.
    """
    xs, ys, masks, fas, xys, xy_masks, lens = [], [], [], [], [], [], []
    for _ in range(batch_size):
        env = env_fn()
        env.reset()
        gt = env.gt.astype(np.int64)
        y, m = targets_and_mask(gt)
        T = len(gt)

        xy_mask = np.zeros(T, dtype=np.float32)
        if window_ms is None:
            xy_mask[int(env.start_ind["target"]):] = 1.0
        else:
            a, b = delay_window(int(env.start_ind["target"]), dt, window_ms)
            xy_mask[a:min(b, T)] = 1.0

        tx, ty = env.trial["cue_pos" if xy_from == "cue" else "target_pos"]
        xy = np.zeros((T, 2), dtype=np.float32)
        xy[:, 0], xy[:, 1] = tx, ty

        xs.append(env.ob.astype(np.float32).copy())
        ys.append(y)
        masks.append(m)
        fas.append(env._fa_window.astype(np.float32).copy())
        xys.append(xy)
        xy_masks.append(xy_mask)
        lens.append(T)

    def pad(arrays, dtype):
        out = np.zeros((len(arrays), max(lens)) + arrays[0].shape[1:], dtype=dtype)
        for i, a in enumerate(arrays):
            out[i, :len(a)] = a
        return torch.tensor(out, device=device)

    return (pad(xs, np.float32), pad(ys, np.int64), pad(masks, np.float32), pad(fas, np.float32),
            pad(xys, np.float32), pad(xy_masks, np.float32),
            torch.tensor(np.array(lens), dtype=torch.long, device=device))


def delay_window(target_onset, dt, window_ms):
    a = max(0, target_onset + int(round(window_ms[0] / dt)))
    return a, target_onset + int(round(window_ms[1] / dt))


def quiet_delay_mask(env):
    """Delay frames with no cue and no stimulus on screen, so a head fitted there must read a held
    representation rather than the current input."""
    d0, d1 = int(env.start_ind["delay"]), int(env.start_ind["target"])
    mask = np.zeros(len(env.gt), dtype=np.float32)
    if d1 > d0:
        mask[d0:d1] = ((env.ob[d0:d1, 3] == 0.0) & (env.ob[d0:d1, 6] == 0.0)).astype(np.float32)
    return mask
