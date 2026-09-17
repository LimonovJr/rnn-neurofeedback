"""Fit the position read-out on frozen dynamics, so the positional neurofeedback signals have a
real decoder to be scored against.

The pretrained checkpoints ship with an untrained head -- b_aux is exactly zero and W_aux is still
at its random-init scale -- so a signal that reinforces decodability through it would be reinforcing
agreement with an arbitrary projection. The head is fitted on pre-collected states, so the recurrent
weights never enter an autograd graph, and the freeze is then verified tensor by tensor.

Supervision uses the stimulus-free frames of the delay and cue position as the target: the head has
to read a held representation rather than the current input, and on the context task the probe is at
the cued location on only half the trials.

    python -m experiments.fit_head --task c1 --net 0
"""

import argparse
import copy

import numpy as np
import torch

from src import runs, tasks, trials
from src.model import CKPT, load, save

N_FIT, N_EVAL = 1200, 800
N_BINS = 8
RANKS = [8, 16, 32, 64, 128]
ALPHAS = [1e-2, 1.0, 100.0]
SLACK = 0.02                   # a near-equal fit is never bought with a much larger read-out
DEFAULT_RANK = {"c1": 32, "context": 16}
ENV = {"c1": tasks.sameloc(*tasks.C1), "context": tasks.context()}


def delay_states(model, env_fn, n_trials, rec_noise, seed):
    """Per-frame delay states and the trial's cue position, one group per trial."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.rec_noise = rec_noise
    model.eval()

    X, Y, trial_id = [], [], []
    with torch.no_grad():
        for k in range(n_trials):
            env = env_fn()
            env.reset()
            keep = trials.quiet_delay_mask(env) > 0
            if not keep.any():
                continue
            x = torch.tensor(env.ob.astype(np.float32))[None]
            _, _, h = model(x, return_hidden=True)
            states = h[0].numpy()[keep]
            X.append(states)
            target = np.array(env.trial["cue_pos"], dtype=np.float32)
            Y.append(np.repeat(target[None], len(states), axis=0))
            trial_id.append(np.full(len(states), k))
    model.rec_noise = True
    return np.concatenate(X), np.concatenate(Y), np.concatenate(trial_id)


def fit(X, Y, rank, alpha):
    """Ridge in a rank-truncated PCA basis of the delay states, folded back to full units.

    Fitting all 180 units directly is ill-posed -- many delay units are near-constant across trials,
    so the coefficients are not identifiable and ridge alone lets the norm run away. Truncating
    removes that subspace instead of penalising it. Returns (W, b) ready for W_aux/b_aux.
    """
    mu = X.mean(0)
    Xc = X - mu
    V = np.linalg.svd(Xc, full_matrices=False)[2][:rank].T
    Z = Xc @ V
    Wz = np.linalg.solve(Z.T @ Z + alpha * np.eye(rank), Z.T @ (Y - Y.mean(0)))
    W = (V @ Wz).T
    return W, Y.mean(0) - W @ mu


def score(X, Y, trial_id, W, b):
    """Delay-mean decode per trial: distance error and 8-bin angular accuracy."""
    pred = X @ W.T + b
    ids = np.unique(trial_id)
    P = np.array([pred[trial_id == i].mean(0) for i in ids])
    T = np.array([Y[trial_id == i][0] for i in ids])
    angle = lambda V: (np.floor((np.arctan2(V[:, 1], V[:, 0]) % (2 * np.pi))
                                / (2 * np.pi) * N_BINS).astype(int) % N_BINS)
    return float(np.linalg.norm(P - T, axis=1).mean()), float((angle(P) == angle(T)).mean())


def run(task, net, rank, out):
    env_fn = ENV[task]
    model = load(net)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    W0, b0 = model.W_aux.detach().numpy().copy(), model.b_aux.detach().numpy().copy()

    Xf, Yf, Tf = delay_states(model, env_fn, N_FIT, True, seed=4000 + net)
    Xe, Ye, Te = delay_states(model, env_fn, N_EVAL, False, seed=9000 + net)
    print(f"[head {task} net{net}] fit frames={len(Xf)} eval frames={len(Xe)}", flush=True)

    rmse0, acc0 = score(Xe, Ye, Te, W0, b0)
    print(f"  random head: rmse={rmse0:.4f} acc={acc0:.4f} ||W||={np.linalg.norm(W0):.3f}", flush=True)

    ids = np.unique(Tf)
    held = set(np.random.default_rng(0).choice(ids, size=max(1, len(ids) // 5), replace=False).tolist())
    inner = np.array([t in held for t in Tf])

    path = []
    for r in ([rank] if rank else RANKS):
        for alpha in ALPHAS:
            W, b = fit(Xf[~inner], Yf[~inner], r, alpha)
            rmse, acc = score(Xf[inner], Yf[inner], Tf[inner], W, b)
            path.append((r, alpha, rmse, acc, float(np.linalg.norm(W))))
            print(f"  rank={r:<4d} alpha={alpha:<8g} rmse={rmse:.4f} acc={acc:.4f} "
                  f"||W||={np.linalg.norm(W):8.3f}", flush=True)

    best = min(p[2] for p in path)
    rank, alpha = min([p for p in path if p[2] <= best + SLACK], key=lambda p: p[4])[:2]
    W, b = fit(Xf, Yf, rank, alpha)
    rmse, acc = score(Xe, Ye, Te, W, b)
    print(f"  chosen rank={rank} alpha={alpha:g}: rmse={rmse:.4f} acc={acc:.4f} "
          f"||W||={np.linalg.norm(W):.3f}", flush=True)

    with torch.no_grad():
        model.W_aux.copy_(torch.tensor(W, dtype=model.W_aux.dtype))
        model.b_aux.copy_(torch.tensor(b, dtype=model.b_aux.dtype))
    moved = {k: float((model.state_dict()[k] - before[k]).abs().max()) for k in before}
    moved = {k: v for k, v in moved.items() if v > 0 and k not in ("W_aux", "b_aux")}
    assert not moved, f"tensors other than the head moved: {moved}"

    path = CKPT / f"net{net}" / f"head_{task}.pt"
    assert not path.exists(), f"{path} already exists; everything downstream was fitted against it"
    save(model, path, task=task, net=net, rank=rank, alpha=alpha)
    runs.write(out, dict(task=task, net=net, rank=rank, alpha=alpha, fit_frames=len(Xf),
                         eval_frames=len(Xe), rmse_before=f"{rmse0:.4f}", acc_before=f"{acc0:.4f}",
                         rmse=f"{rmse:.4f}", acc=f"{acc:.4f}",
                         wnorm_before=f"{np.linalg.norm(W0):.4f}", wnorm=f"{np.linalg.norm(W):.4f}"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(ENV), required=True)
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--rank", type=int)
    p.add_argument("--out")
    a = p.parse_args()
    rank = a.rank or DEFAULT_RANK[a.task]
    run(a.task, a.net, rank, a.out or f"results/head_{a.task}_net{a.net}.csv")


if __name__ == "__main__":
    main()
