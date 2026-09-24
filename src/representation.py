"""What the population code contains during the delay, measured on the same trials for every metric.

Crossnobis is the primary measure: it is cross-validated and noise-normalised, so it is unbiased
under the null and has no accuracy-style ceiling. Participation ratio is secondary and exists to
catch a collapse of the code onto too few dimensions.
"""

import numpy as np
import torch

from src import trials


def angle_bin(pos, n_bins=8):
    theta = np.arctan2(pos[1], pos[0]) % (2 * np.pi)
    return int(theta / (2 * np.pi) * n_bins) % n_bins


def delay_population(model, env_fn, n_bins=8, n_trials=2500, window_ms=(-300, 0), device="cpu",
                     noisy=True, outcome="correct", bin_by="target"):
    """Mean delay-window activity per trial, with the position it is scored against.

    `outcome=None` keeps every trial: below the release threshold a network has no correct trials,
    so the correct-only pool is empty and every metric is undefined exactly where the question is.
    `bin_by="cue"` is required on the context task, where the probe coincides with the cue on only
    half the trials and target position is not knowable during the delay on the others.
    """
    was_noisy = model.noise_at_eval
    model.noise_at_eval = noisy
    tr = trials.collect(model, env_fn, n_trials=n_trials, device=device)
    model.noise_at_eval = was_noisy
    if outcome is not None:
        tr = [t for t in tr if t["train_outcome"] == outcome]

    X, Y, B = [], [], []
    for t in tr:
        a, b = trials.delay_window(int(t["target_onset"]), int(model.dt), window_ms)
        pos = t["cue_pos"] if bin_by == "cue" else t["target_pos"]
        if b <= a or pos is None:
            continue
        X.append(t["h"][a:b].mean(0))
        Y.append(pos)
        B.append(angle_bin(pos, n_bins))
    return np.asarray(X), np.asarray(Y), np.asarray(B)


def crossnobis(X, B, n_bins=8, n_splits=5, seed=0):
    """Cross-validated Mahalanobis distances between position bins: bin differences from training
    trials are dotted with those from held-out trials in the whitened space."""
    from sklearn.covariance import LedoitWolf
    from sklearn.model_selection import StratifiedKFold

    bins = np.arange(n_bins)
    folds = []
    for train_i, test_i in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, B):
        Xtr, Btr, Xte, Bte = X[train_i], B[train_i], X[test_i], B[test_i]
        residual = np.concatenate([Xtr[Btr == k] - Xtr[Btr == k].mean(0) for k in bins
                                   if (Btr == k).any()])
        ev, U = np.linalg.eigh(LedoitWolf().fit(residual).covariance_)
        whiten = U @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-8))) @ U.T
        mu_tr = {k: Xtr[Btr == k].mean(0) @ whiten for k in bins if (Btr == k).any()}
        mu_te = {k: Xte[Bte == k].mean(0) @ whiten for k in bins if (Bte == k).any()}

        rdm = np.full((n_bins, n_bins), np.nan)
        for i in bins:
            for j in bins:
                if i in mu_tr and j in mu_tr and i in mu_te and j in mu_te:
                    rdm[i, j] = float((mu_tr[i] - mu_tr[j]) @ (mu_te[i] - mu_te[j])) / X.shape[1]
        folds.append(rdm)
    return np.nanmean(folds, axis=0)


def participation_ratio(X):
    """(sum lambda)^2 / sum lambda^2: 1 means a line, n_features an isotropic cloud."""
    X = np.asarray(X, dtype=float)
    lam = np.linalg.svd(X - X.mean(axis=0, keepdims=True), compute_uv=False) ** 2
    denom = (lam ** 2).sum()
    return float(lam.sum() ** 2 / denom) if denom > 0 else 1.0


def panel(model, env_fn, n_trials=2500, n_bins=8, n_splits=5, seed=0, noisy=True,
          outcome="correct", bin_by="target", device="cpu"):
    """Crossnobis and participation ratio from one pass of trials."""
    X, _, B = delay_population(model, env_fn, n_bins, n_trials, device=device, noisy=noisy,
                               outcome=outcome, bin_by=bin_by)
    if len(B) < max(2 * n_splits, n_bins + 1):
        return {"n": int(len(B)), "crossnobis": np.nan, "pr": np.nan}

    rdm = crossnobis(X, B, n_bins, n_splits, seed)
    off_diagonal = rdm[np.triu_indices(n_bins, k=1)]
    return {"n": int(len(B)), "crossnobis": float(np.nanmean(off_diagonal)),
            "pr": participation_ratio(X)}


def decode_battery(model, env_fn, n_trials=2000, device="cpu"):
    """Linear decoders of context, match and their exclusive-or from the decision-time state.

    The positive control is only valid if context and match are linearly available while the
    exclusive-or is not: if the frozen dynamics have already linearised it, the task is mis-encoded
    rather than the instrument being blunt.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, labels = [], {"context": [], "match": [], "xor": []}
    for t in trials.collect(model, env_fn, n_trials=n_trials, device=device, batch_size=256):
        if t["context"] is None:
            continue
        t0 = int(t["target_onset"])
        h = t["h"][t0:t0 + 6]
        if len(h) == 0:
            continue
        X.append(h.mean(0))
        labels["context"].append(int(t["context"]))
        labels["match"].append(int(t["is_match"]))
        labels["xor"].append(int(t["is_go"]))

    X = np.array(X)
    out = {}
    for name, y in labels.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        out[name] = float(cross_val_score(clf, X, np.array(y), cv=5, scoring="accuracy").mean())
    return out


def _separation(sig, noise):
    """Probability that a response step projects above a false-alarm step (Mann-Whitney)."""
    both = torch.cat([sig, noise])
    ranks = both.argsort().argsort().float() + 1
    n1, n2 = len(sig), len(noise)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n2))


def _steps(model, env_fn, n, window_ms):
    x, y, _, fa, _, _, lengths = trials.batch(env_fn, n, int(model.dt), window_ms=window_ms)
    model.eval()
    with torch.no_grad():
        _, _, h = model(x, return_hidden=True)
    valid = torch.arange(x.shape[1])[None, :] < lengths[:, None]
    return h, (y == 1) & valid, (fa > 0) & valid, valid


def decision_axis_target(model, env_fn, cosine, k=10, n=256, seed=0, window_ms=(-300, 0)):
    """A read-out direction at an exact cosine to the decision axis, and the standardisation of the
    state's projection on it that separates response steps from false-alarm steps at the start.

    The decision axis is the release-minus-hold row of the frozen read-out. The rest of u is a
    direction inside the top-k principal directions of the activity, orthogonal to the decision
    axis and to the difference between the two classes' mean states, fixed by the seed: a random
    direction there would carry a separation of its own (the two kinds of step differ in many ways,
    time in the trial among them), and the dose would no longer be the angle. So at the start the
    separation along u is whatever its decision-axis part carries, and at cosine 0 it is none."""
    h, sig, noise, valid = _steps(model, env_fn, n, window_ms)
    states = h[valid]
    _, _, vt = torch.linalg.svd(states - states.mean(0), full_matrices=False)
    basis = vt[:k].T
    w = (model.W_out_eff()[1] - model.W_out_eff()[0]).detach()
    w_hat = w / w.norm()
    gap = h[sig].mean(0) - h[noise].mean(0)
    # in the manifold's own coordinates: z orthogonal to the axis and to the class gap there, so
    # r = basis z stays in the manifold and is exactly orthogonal to both
    z = torch.randn(k, generator=torch.Generator().manual_seed(seed))
    done = []
    for v in (basis.T @ w_hat, basis.T @ gap):
        for d in done:
            v = v - (v @ d) * d
        v = v / v.norm()
        z = z - (z @ v) * v
        done.append(v)
    r = basis @ z
    u = cosine * w_hat + float(np.sqrt(max(0.0, 1.0 - cosine ** 2))) * r / r.norm()
    p_sig, p_noise = h[sig] @ u, h[noise] @ u
    w_in = basis @ (basis.T @ w)
    return dict(u=u, centre=float((p_sig.mean() + p_noise.mean()) / 2),
                scale=float(1.0 / torch.sqrt((p_sig.var() + p_noise.var()) / 2)),
                start_auc=_separation(p_sig, p_noise), in_manifold=float(w_in.norm() / w.norm()))


def axis_separation(model, env_fn, u, n=256, window_ms=(-300, 0)):
    h, sig, noise, _ = _steps(model, env_fn, n, window_ms)
    return _separation(h[sig] @ u, h[noise] @ u)


def delay_basis(model, env_fn, k=10, n_trials=1500, window_ms=(-300, 0), device="cpu"):
    """The intrinsic manifold of the delay: mean state, the top-k principal directions of the
    per-trial delay-mean states, and the mean radius of the states inside that subspace."""
    X, _, _ = delay_population(model, env_fn, n_trials=n_trials, window_ms=window_ms,
                               device=device, noisy=True, outcome=None)
    mu = X.mean(0)
    _, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:k].T.copy()
    radius = float(np.linalg.norm((X - mu) @ P, axis=1).mean())
    explained = float((s[:k] ** 2).sum() / (s ** 2).sum())
    return dict(mu=torch.tensor(mu, dtype=torch.float32, device=device),
                P=torch.tensor(P, dtype=torch.float32, device=device),
                radius=radius, explained=explained)
