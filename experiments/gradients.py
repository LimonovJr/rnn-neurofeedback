"""How the task gradient and the positional gradient sit relative to each other.

The joint signal adds a positional term to the same task loss the control carries, so the
behavioural deficit should be visible in the geometry: the angle between the two gradients, and the
size of the positional gradient relative to the task one. Both are measured on the parameters the
conditions actually optimise, under training conditions rather than evaluation conditions.

    python -m experiments.gradients --cond joint --net 0 --rep 0
"""

import argparse
import os
import glob

import numpy as np
import torch

from src import runs, signals, tasks, trials
from src.model import CKPT, load
from src.training import cross_entropy

BATCH, N_BATCH = 128, 4
ENT_COEF = 0.05
ENV = tasks.context()


def trainable(model):
    frozen = {id(p) for p in (model.W_aux, model.b_aux, model.W_out, model.b_out)}
    return [p for p in model.parameters() if id(p) not in frozen]


def losses(model, batch):
    """The task and positional terms, formed exactly as the joint signal forms them."""
    x, y, mask, _, xy_true, delay_mask, _ = batch
    logits, _, h = model(x, return_hidden=True)
    task = cross_entropy(logits, y, mask)
    err = ((model.decode_xy(h) - xy_true) ** 2).sum(-1)
    per_trial = (err * delay_mask).sum(1) / delay_mask.sum(1).clamp_min(1)
    aux = per_trial.sum() / per_trial.shape[0]
    return task, aux


def geometry(model):
    model.train()
    model.rec_noise = True
    params = trainable(model)
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])

    acc_task = acc_aux = None
    per_batch = []
    for _ in range(N_BATCH):
        batch = trials.batch(ENV, BATCH, int(model.dt), window_ms=signals.WINDOW_MS)
        task, aux = losses(model, batch)
        g_task = flat(torch.autograd.grad(task, params, retain_graph=True))
        g_aux = flat(torch.autograd.grad(aux, params))
        per_batch.append(float(torch.dot(g_task, g_aux)
                               / (g_task.norm() * g_aux.norm()).clamp_min(1e-12)))
        acc_task = g_task if acc_task is None else acc_task + g_task
        acc_aux = g_aux if acc_aux is None else acc_aux + g_aux

    acc_task, acc_aux = acc_task / N_BATCH, acc_aux / N_BATCH
    n_task, n_aux = float(acc_task.norm()), float(acc_aux.norm())
    cos = float(torch.dot(acc_task, acc_aux) / max(n_task * n_aux, 1e-12))
    return dict(g_task=round(n_task, 6), g_aux=round(n_aux, 6),
                rho=round(n_aux / max(n_task, 1e-12), 4), cos=round(cos, 4),
                cos_sd=round(float(np.std(per_batch, ddof=1)), 4))


def entropy(model):
    """The entropy bonus the reward signal carries, measured on the same conditions."""
    model.train()
    model.rec_noise = True
    values = []
    for _ in range(N_BATCH):
        x, _, _, valid = signals.outcome_batch(ENV, signals.BATCH, "cpu")
        with torch.no_grad():
            logits, _, _ = model(x, return_hidden=True)
            logp = torch.log_softmax(logits, -1)
            values.append(float((-(logp.exp() * logp).sum(-1))[valid].mean()))
    return float(np.mean(values))


def step_probe(model, length, n_held=8):
    """Does behaviour follow the projection? Step a fixed distance in parameter space along the
    task gradient and along the positional one, and compare the realised change in the task loss
    with the first-order prediction -length <g_task, direction>. Fixing the distance rather than
    the learning rate keeps both steps in the regime where the prediction is defined at all: scaled
    by a learning rate the task step is far longer than the positional one and overshoots. The
    gradients are taken under training conditions; the held-out loss is read with the recurrent
    noise off, so the same batches give the same number before and after the step and the
    difference is the step rather than a fresh noise draw."""
    model.train()
    params = trainable(model)
    saved = [p.detach().clone() for p in params]
    flat = lambda gs: torch.cat([g.reshape(-1) for g in gs])

    model.rec_noise = True
    g_task = g_aux = None
    for _ in range(N_BATCH):
        task, aux = losses(model, trials.batch(ENV, BATCH, int(model.dt),
                                               window_ms=signals.WINDOW_MS))
        gt = flat(torch.autograd.grad(task, params, retain_graph=True)).detach()
        ga = flat(torch.autograd.grad(aux, params)).detach()
        g_task = gt if g_task is None else g_task + gt
        g_aux = ga if g_aux is None else g_aux + ga
    g_task, g_aux = g_task / N_BATCH, g_aux / N_BATCH

    model.rec_noise = False
    held = [trials.batch(ENV, BATCH, int(model.dt), window_ms=signals.WINDOW_MS)
            for _ in range(n_held)]

    def task_loss():
        with torch.no_grad():
            return float(np.mean([losses(model, b)[0].item() for b in held]))

    def unflatten(v):
        out, i = [], 0
        for p in params:
            out.append(v[i:i + p.numel()].view_as(p))
            i += p.numel()
        return out

    before = task_loss()
    out = {"task_loss": round(before, 6)}
    for name, g in (("task", g_task), ("aux", g_aux)):
        unit = g / g.norm().clamp_min(1e-12)
        with torch.no_grad():
            for p, d in zip(params, unflatten(unit)):
                p.sub_(length * d)
        realised = task_loss() - before
        with torch.no_grad():
            for p, s0 in zip(params, saved):
                p.copy_(s0)
        out[f"d_task_{name}_pred"] = round(-length * float(torch.dot(g_task, unit)), 6)
        out[f"d_task_{name}_real"] = round(realised, 6)
    return out


def checkpoints(cond, net, rep, ckpt_dir):
    if cond == "theta0":
        return [(None, load(net, head="context"))]
    pattern = str(CKPT / ckpt_dir / f"{cond}_net{net}_r{rep}_u*.pt")
    files = sorted(glob.glob(pattern), key=lambda p: int(p.rsplit("_u", 1)[1].split(".")[0]))
    assert files, f"no checkpoints matched {pattern}"
    out = []
    for path in files[-1:]:                       # the end of the run is what the contrast is about
        checkpoint = torch.load(path, weights_only=True)
        model = load(net, head="context")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        out.append((int(checkpoint["upd"]), model))
    return out


def run(cond, net, rep, ckpt_dir, length, out):
    torch.manual_seed(7)
    np.random.seed(7)
    rows = []
    for upd, model in checkpoints(cond, net, rep, ckpt_dir):
        row = dict(cond=cond, net=net, rep=rep, upd=upd if upd is not None else "", **geometry(model))
        row["entropy"] = round(entropy(model), 4) if cond == "reward" else ""
        if length:
            row.update(step_probe(model, length))
        print(f"  [{cond} net{net} r{rep} u{row['upd']}] cos={row['cos']:+.4f} rho={row['rho']:.2f} "
              f"|g_task|={row['g_task']:.3f} |g_aux|={row['g_aux']:.3f}", flush=True)
        rows.append(row)
    fields = ["cond", "net", "rep", "upd", "g_task", "g_aux", "rho", "cos", "cos_sd", "entropy"]
    if length:
        fields += ["task_loss", "d_task_task_pred", "d_task_task_real",
                   "d_task_aux_pred", "d_task_aux_real"]
    runs.write(out, rows, fields)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cond", choices=["theta0", "control", "position", "joint", "reward"],
                   required=True)
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--ckpt-dir", default="nf_context")
    p.add_argument("--step", type=float, default=0.0,
                   help="also take one step of this length in parameter space along each gradient "
                        "and record the realised against the predicted change in the task loss")
    p.add_argument("--out")
    a = p.parse_args()
    a.out = a.out or f"results/grad_context_{a.cond}_net{a.net}_r{a.rep}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    run(a.cond, a.net, a.rep, a.ckpt_dir, a.step, a.out)


if __name__ == "__main__":
    main()
