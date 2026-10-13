"""How much of a feedback signal helps before it starts to hurt.

A network at its starting state is near the optimum of the task for its frozen decoder. A step
along another objective's gradient changes the task loss by a first-order term, which helps if the
two gradients point the same way, and by a curvature term, which hurts and grows with the square of
the distance. If that is why every representational target ended lower than it started, the dose
has an inverted U: a short stretch of such training helps and a long one harms, and the stretch is
longer where the network is further from the task's optimum. Every result so far was read at one
long dose.

Each signal is trained from the starting state and read after 0, 1, 2, 5, 10, 25, 50, 100, 150 and
300 updates, with the distance moved in parameter space recorded at each. The states of one
network are all read on the same trials: two seeded samples, one to choose the dose and the
threshold on, one to report them on, and fixed held-out batches for the task loss with the noise
off. The ROC of each state comes from one pass over the trials (`behaviour.roc_once`).

The task loss itself falls when the criterion moves, and early in training that is most of what
moves. So it is also read after the decoder's release margin has been recalibrated by a shift and
a scale fitted on separate trials. What is left of the loss then is how well the margin tells the
steps apart, whatever its level and gain: a continuous measure of discriminability, read on every
step and not on a release.

`--first-step` only takes one step of that length along the signal's gradient and sets the realised
change of both losses against the first-order prediction. For the recalibrated loss the prediction
uses the gradient with the shift and scale held at their fitted values, which is the gradient of
the loss already minimised over them.

    python -m experiments.dose --task context --arm k1 --net 10
"""

import argparse
import os

import numpy as np
import torch

import torch.nn.functional as F

from src import behaviour, representation, runs, signals, tasks, trials
from src.model import CKPT, load
from src.training import cross_entropy

TASKS = {"c1": dict(env=tasks.sameloc(*tasks.C1), score=behaviour.detection, xy_from="target",
                    arms=("axis", "k2", "neutral", "control")),
         "context": dict(env=tasks.context(), score=behaviour.go_nogo, xy_from="cue",
                         arms=("k1", "k2", "neutral", "template", "control"))}
DOSES = (0, 1, 2, 5, 10, 25, 50, 100, 150, 300)
SUPERVISED = "nf_context_b60/nf_context_control_lam1_n2000_net{net}_r12_u8000.pt"


def decision_states(model, env_fn, n):
    """Decision-time state and go label of each trial, as the decode battery takes them."""
    X, go = [], []
    for env, _, _, _, h in trials.rolled(model, env_fn, n):
        onset = int(env.start_ind["target"])
        X.append(h[onset:onset + 6].mean(0))
        go.append(int(env.trial["is_go"]))
    return np.array(X), np.array(go)


def fit_template(net, n=2000):
    """A linear decoder of go against no-go on the supervised end state of the same network, folded
    into one weight vector and a bias."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    model = load(net, head="context")
    path = CKPT / SUPERVISED.format(net=net)
    model.load_state_dict(torch.load(path, weights_only=True)["state_dict"], strict=False)
    X, go = decision_states(model, trials.seeded(TASKS["context"]["env"], 300 + net), n)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X), go)
    w = clf.coef_[0] / scaler.scale_
    b = float(clf.intercept_[0]) - float(np.sum(clf.coef_[0] * scaler.mean_ / scaler.scale_))
    return torch.tensor(w, dtype=torch.float32), b


def signal(task, arm, model, net):
    env_fn = TASKS[task]["env"]
    if arm == "control":
        return signals.task_step(0.0, 1.0)
    if arm in ("k1", "k2"):
        return signals.expand_step(float(arm[1:]))
    if arm == "neutral":
        return signals.neutral_step(2.0, signals.neutral_readout(model, env_fn, net))
    if arm == "axis":
        return signals.axis_step(representation.decision_axis_target(model, env_fn, 1.0, seed=net))
    return signals.template_step(*fit_template(net))


def trained(model):
    frozen = {id(p) for p in (model.W_aux, model.b_aux, model.W_out, model.b_out)}
    return [p for p in model.parameters() if id(p) not in frozen]


def flat(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def held_batches(task, net, seed=900, n=8):
    cfg = TASKS[task]
    source = trials.seeded(cfg["env"], seed + net)
    return [trials.batch(source, 128, 20, window_ms=signals.WINDOW_MS, xy_from=cfg["xy_from"])
            for _ in range(n)]


def recalibrated(margin, y, weight, scale, shift):
    """The task loss of a release margin after a scale and a shift of it."""
    z = scale * margin + shift
    return (torch.where(y == 1, F.softplus(-z), F.softplus(z)) * weight).sum() / weight.sum()


def margins(model, batches):
    model.eval()
    with torch.no_grad():
        logits = [model(x, return_hidden=True)[0] for x, *_ in batches]
    return (torch.cat([(l[..., 1] - l[..., 0]).reshape(-1) for l in logits]),
            torch.cat([b[1].reshape(-1) for b in batches]),
            torch.cat([b[2].reshape(-1) for b in batches]))


def calibrate(margin, y, weight):
    """The scale and shift of the margin that minimise the task loss on these steps: all that
    recalibrating the decoder's output could do, the criterion included."""
    scale = torch.ones((), requires_grad=True)
    shift = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([scale, shift], max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = recalibrated(margin, y, weight, scale, shift)
        loss.backward()
        return loss
    opt.step(closure)
    return float(scale), float(shift)


def resolved_loss(model, fit_on, held):
    """Held-out task loss after recalibration on separate trials, with the fitted scale and shift."""
    scale, shift = calibrate(*margins(model, fit_on))
    return float(recalibrated(*margins(model, held), scale, shift)), scale, shift


def task_loss(model, held):
    model.eval()
    with torch.no_grad():
        return float(np.mean([cross_entropy(model(x, return_hidden=True)[0], y, mask).item()
                              for x, y, mask, *_ in held]))


def read(model, task, net, n, template=None):
    """One state on the two seeded samples: area under the ROC on each, the threshold chosen on the
    first and scored on the second, and the decoder's own threshold on the second."""
    cfg = TASKS[task]
    model.eval()
    first, second = (behaviour.roc_once(model, trials.seeded(cfg["env"], seed + net), n, cfg["score"])
                     for seed in (5000, 7000))
    best = int(np.argmax([p["d"] for p in first]))
    own = next(p for p in second if p["offset"] == 0.0)
    out = dict(auc_select=behaviour.hull_area([(p["fa"], p["hit"]) for p in first]),
               auc=behaviour.hull_area([(p["fa"], p["hit"]) for p in second]),
               offset=first[best]["offset"], best_d=second[best]["d"], own_d=own["d"],
               hit=own["hit"], fa=own["fa"])
    if task == "context":
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, go = decision_states(model, trials.seeded(cfg["env"], 8000 + net), 1200)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        out["conjunction"] = float(cross_val_score(clf, X, go, cv=5, scoring="accuracy").mean())
        if template is not None:
            w, b = template
            out["template_acc"] = float((((X @ w.numpy() + b) > 0) == (go == 1)).mean())
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def first_step(model, task, step, length, fit_on, held):
    """One step of fixed length along the signal's gradient: its cosine to the task gradient and to
    the gradient of the recalibrated loss, and the realised change of each held-out loss against
    -length <gradient, direction>."""
    cfg = TASKS[task]
    params = trained(model)
    for p in model.parameters():
        p.requires_grad_(any(p is q for q in params))
    before, (resolved, scale, shift) = task_loss(model, held), resolved_loss(model, fit_on, held)
    model.train()
    model.rec_noise = True
    g_task = g_resolved = g_signal = 0.0
    for _ in range(4):
        batch = trials.batch(cfg["env"], 128, int(model.dt), window_ms=signals.WINDOW_MS,
                             xy_from=cfg["xy_from"])
        logits = model(batch[0], return_hidden=True)[0]
        raw = cross_entropy(logits, batch[1], batch[2])
        fixed = recalibrated(logits[..., 1] - logits[..., 0], batch[1], batch[2], scale, shift)
        g_task = g_task + flat(torch.autograd.grad(raw, params, retain_graph=True)).detach()
        g_resolved = g_resolved + flat(torch.autograd.grad(fixed, params)).detach()
        g_signal = g_signal + flat(torch.autograd.grad(step(model, batch)[0], params)).detach()
    unit = g_signal / g_signal.norm().clamp_min(1e-12)
    with torch.no_grad():
        i = 0
        for p in params:
            p.sub_(length * unit[i:i + p.numel()].view_as(p))
            i += p.numel()
    cosine = lambda g: round(float(torch.dot(g, g_signal) / (g.norm() * g_signal.norm())), 4)
    predicted = lambda g: round(-length * float(torch.dot(g, unit)) / 4, 6)
    return dict(cos=cosine(g_task), rho=round(float(g_signal.norm() / g_task.norm()), 4),
                predicted=predicted(g_task), realised=round(task_loss(model, held) - before, 6),
                task_loss=round(before, 5), cos_resolved=cosine(g_resolved),
                predicted_resolved=predicted(g_resolved),
                realised_resolved=round(resolved_loss(model, fit_on, held)[0] - resolved, 6),
                resolved_loss=round(resolved, 5), scale=round(scale, 4), shift=round(shift, 4))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=list(TASKS), required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--net", type=int, required=True)
    p.add_argument("--rep", type=int, default=13)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--doses", default=",".join(str(d) for d in DOSES))
    p.add_argument("--n", type=int, default=4000, help="trials in each of the two seeded samples")
    p.add_argument("--first-step", type=float, default=0.0)
    p.add_argument("--xy-from", choices=["target", "cue"],
                   help="what the positional targets regress toward, if not the task's own: on the "
                        "context task `target` is the probe, which the first gradient analysis used")
    p.add_argument("--out")
    a = p.parse_args()
    assert a.arm in TASKS[a.task]["arms"], f"{a.task} has arms {TASKS[a.task]['arms']}"
    kind = "first_step" if a.first_step else "dose"
    tag = ("" if a.lr == 1e-4 else f"_lr{a.lr:g}") + (f"_{a.xy_from}" if a.xy_from else "")
    if a.xy_from:
        TASKS[a.task]["xy_from"] = a.xy_from
    a.out = a.out or f"results/{kind}_{a.task}_{a.arm}{tag}_net{a.net}.csv"
    if os.path.exists(a.out):
        print(f"[skip] {a.out} exists", flush=True)
        return
    cfg = TASKS[a.task]
    model = load(a.net, head=a.task)
    template = fit_template(a.net) if a.task == "context" and a.arm == "template" else None
    step = signal(a.task, a.arm, model, a.net)
    held, fit_on = held_batches(a.task, a.net), held_batches(a.task, a.net, seed=950)
    # the dose is chosen on one set of held-out trials and judged on another
    chosen_on = held_batches(a.task, a.net, seed=980)
    identity = dict(task=a.task, arm=a.arm, net=a.net, lr=a.lr)
    if a.first_step:
        runs.write(a.out, dict(identity, length=a.first_step,
                               **first_step(model, a.task, step, a.first_step, fit_on, held)))
        return

    doses = sorted(int(d) for d in a.doses.split(","))
    start = [q.detach().clone() for q in trained(model)]
    w_rec0 = model.W_rec.detach().clone()
    rows = []

    def hook(m, upd):
        if upd not in doses:
            return
        moved = float(flat([q.detach() - q0 for q, q0 in zip(trained(m), start)]).norm())
        resolved, scale, shift = resolved_loss(m, fit_on, held)
        row = dict(identity, dose=upd, moved=round(moved, 5),
                   moved_rel=round(float((m.W_rec.detach() - w_rec0).norm() / w_rec0.norm()), 5),
                   loss=round(task_loss(m, held), 5), loss_resolved=round(resolved, 5),
                   loss_resolved_select=round(resolved_loss(m, fit_on, chosen_on)[0], 5),
                   scale=round(scale, 4), shift=round(shift, 4),
                   **read(m, a.task, a.net, a.n, template))
        rows.append(row)
        # a long run that dies keeps what it had read; the name without .part marks a finished one
        runs.write(a.out + ".part", rows)
        print(f"  [{a.task} {a.arm} net{a.net}] dose {upd:>4}  moved {moved:.4f}  loss "
              f"{row['loss']:.4f}  recalibrated {resolved:.4f}  AUC {row['auc']:.4f}  best d' "
              f"{row['best_d']:+.3f}  own d' {row['own_d']:+.3f}", flush=True)

    torch.manual_seed(a.rep)
    np.random.seed(a.rep)
    signals.loop(model, cfg["env"], step, doses[-1], a.lr, True, 1, hook, "cpu",
                 xy_from=cfg["xy_from"], log_every=10 ** 9)
    runs.write(a.out, rows)
    os.remove(a.out + ".part")


if __name__ == "__main__":
    main()
