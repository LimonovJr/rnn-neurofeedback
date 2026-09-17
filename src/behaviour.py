"""Behaviour is scored as d', from the release the network actually makes."""

from collections import Counter

from scipy.stats import norm

from src import trials


def dprime(hit, fa):
    clamp = lambda p: min(max(float(p), 0.01), 0.99)
    return float(norm.ppf(clamp(hit)) - norm.ppf(clamp(fa)))


def detection(tr):
    """Hits over all trials, false alarms over the trials that carried a distractor."""
    counts = Counter(t["train_outcome"] for t in tr)
    n_distractor = sum(1 for t in tr if t["has_distractors"])
    return dprime(counts.get("correct", 0) / max(len(tr), 1),
                  counts.get("false_alarm", 0) / max(n_distractor, 1))


def go_nogo(tr):
    """Hits over go trials, false alarms over no-go trials, which are the ones without a response
    window. Returns d' with the rates and counts behind it."""
    go = [t for t in tr if t["target_win_on"] is not None]
    nogo = [t for t in tr if t["target_win_on"] is None]
    hit = sum(1 for t in go if t["train_outcome"] == "correct") / max(len(go), 1)
    fa = sum(1 for t in nogo if t["train_outcome"] == "false_alarm") / max(len(nogo), 1)
    return dprime(hit, fa), hit, fa, len(go), len(nogo)


def measure(model, env_fn, n=1000, score=detection, device="cpu"):
    tr = trials.collect(model, env_fn, n_trials=n, device=device, batch_size=256)
    out = score(tr)
    return out[0] if isinstance(out, tuple) else out
