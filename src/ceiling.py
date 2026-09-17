"""The headroom instrument: from one frozen state, how far behaviour gets when only the read-out
may change (arm A) against when the dynamics may change too (arm B). H = C_B - C_A.

Both arms share the state, the restart set, the budget and the ceiling statistic, so H is the
difference between two continuations rather than a race between two separately trained networks.
"""

import copy
import math
from collections import deque

import numpy as np
import torch

from src import behaviour, trials
from src.model import build
from src.training import TrainConfig, train_supervised

CHUNK = 500
K = 6                                       # checkpoints averaged into one ceiling
REEVAL = 10000                              # independent trials the kept weights are scored on
LR = {"A": 1e-3, "B": 5e-4, "Aplus": 1e-3}
RESTARTS = ("incumbent", "fresh_a", "fresh_b", "fresh_c")
FRESH_SEEDS = {"fresh_a": 1001, "fresh_b": 1002, "fresh_c": 1003}
HEAD_SEEDS = {"incumbent": 3000, "fresh_a": 3001, "fresh_b": 3002, "fresh_c": 3003}


def convex_readout(model, env_fn, n=1500, device="cpu"):
    """Logistic read-out fitted on the frozen states and folded into one (w, b) row, so a linear
    read-out reproduces it exactly. Returns (w, b, fold_error).

    The raw fold has ||w|| ~ 700 because 1/sigma explodes on near-constant units, which saturates
    the softmax and makes the initialisation behaviourally dead; it is rescaled to a sane logit
    spread, which leaves the decision boundary where it is.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = [], []
    for t in trials.collect(model, env_fn, n_trials=n, device=device, batch_size=256):
        on, off = t["target_win_on"], t["target_win_off"]
        h, cue = t["h"], int(t["cue_onset"]) + 2
        if on is None or off is None:
            continue
        X.append(h[on:off])
        y.append(np.ones(off - on))
        if on > cue:
            X.append(h[cue:on])
            y.append(np.zeros(on - cue))
        for a, b in t["fa_windows"]:
            a, b = max(a, 0), min(b, len(t["gt"]))
            if b > a:
                X.append(h[a:b])
                y.append(np.zeros(b - a))
    X, y = np.concatenate(X), np.concatenate(y)

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)).fit(X, y)
    scaler, logreg = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    w, b = logreg.coef_[0], float(logreg.intercept_[0])
    w = w / scaler.scale_
    b = b - float(np.sum(logreg.coef_[0] * scaler.mean_ / scaler.scale_))
    fold_error = float(np.max(np.abs(clf.decision_function(X[:512]) - (X[:512] @ w + b))))

    scale = 2.5 / (float((X @ w + b).std()) + 1e-9)
    return w * scale, b * scale, fold_error


def set_readout(model, init, convex):
    """'incumbent' keeps the network's own read-out; the other restarts replace it."""
    if init == "incumbent":
        return
    with torch.no_grad():
        model.W_out.zero_()
        model.b_out.zero_()
        if init == "convex":
            w, b, _ = convex
            model.W_out[1].copy_(torch.as_tensor(w, dtype=model.W_out.dtype))
            model.b_out[1] = float(b)
        else:
            model.W_out.normal_(0.0, 1.0 / math.sqrt(model.hidden_size))
            model.W_out.mul_(model.out_mask)


def ceiling(net, state, env_fn, convex, init, arm, lr, cap, head=None, sigma_in=0.0,
            score=behaviour.detection, chunk=CHUNK, reeval=REEVAL, on_chunk=None,
            device="cpu"):
    """Train to the cap, then average the re-evaluated d' of the last K chunk checkpoints.

    Averaging rather than taking the best checkpoint: the all-weights trajectory swings by 0.5-0.9
    in d' between chunks, which is instability in the weights and not evaluation noise, so a
    maximum reliably catches a lucky peak. Selection and scoring also use different samples -- the
    per-chunk d' picks nothing, and the reported value comes from an independent draw of REEVAL
    trials with recurrent noise off. Returns (C, trajectory, re-evaluations, W_rec drift).
    """
    m = build(net)
    m.load_state_dict(state)
    m.rec_noise = True
    m.sigma_in = float(sigma_in)
    m.noise_at_eval = sigma_in > 0        # on a degraded rung the noise IS the thing being measured
    set_readout(m, init, convex)

    if head is not None:
        m.readout_head = head()
        for p in m.parameters():
            p.requires_grad_(False)
        for p in m.readout_head.parameters():
            p.requires_grad_(True)
    elif arm == "A":
        for p in m.parameters():
            p.requires_grad_(False)
        m.W_out.requires_grad_(True)
        m.b_out.requires_grad_(True)
    else:
        for p in m.parameters():
            p.requires_grad_(True)

    wrec0 = m.W_rec.detach().clone()
    cfg = TrainConfig(batch_size=64, lr=lr, max_updates=chunk, print_every=chunk + 1,
                      device=device, fa_penalty_weight=0.4, stop_on_no_miss=0)
    kept, traj = deque(maxlen=K), []
    for i in range(max(1, cap // chunk)):
        train_supervised(m, env_fn, cfg)
        traj.append(round(behaviour.measure(m, env_fn, n=1000, score=score, device=device), 2))
        kept.append(copy.deepcopy(m.state_dict()))
        if on_chunk is not None:
            on_chunk(i, traj[-1])
    drift = float((m.W_rec.detach() - wrec0).abs().max())

    reevals = []
    for state_k in kept:
        m.load_state_dict(state_k)
        m.rec_noise = False
        reevals.append(round(behaviour.measure(m, env_fn, n=reeval, score=score, device=device), 3))
        m.rec_noise = True
    return float(np.mean(reevals)), traj, reevals, drift


def spread(reevals):
    sd = float(np.std(reevals, ddof=1)) if len(reevals) > 1 else 0.0
    return sd, sd / np.sqrt(len(reevals))
