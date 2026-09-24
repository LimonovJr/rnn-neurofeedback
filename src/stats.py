"""Reading rules. An effect has to clear a paired two-sided test at p < 0.05 and exceed the
measured resolution of the instrument; anything else is reported as below the resolution, which is
not the same claim as zero."""

import numpy as np
from scipy import stats


def sem(values):
    """nan below two values, where a spread is not defined -- partial result sets are normal."""
    a = np.asarray(values, dtype=float)
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")


def one_sample(values):
    a = np.asarray(values, dtype=float)
    if len(a) < 2:
        return float(a.mean()) if len(a) else float("nan"), float("nan"), float("nan")
    return float(a.mean()), sem(a), float(stats.ttest_1samp(a, 0.0).pvalue)


def paired(a, b):
    """Difference a - b, paired within network."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2:
        return float((a - b).mean()) if len(a) else float("nan"), float("nan"), float("nan")
    return float((a - b).mean()), sem(a - b), float(stats.ttest_rel(a, b).pvalue)


def equivalence(a, b, margin):
    """Two one-sided tests that the paired difference a - b lies inside +-margin. The p-value is
    the larger of the two, so p < 0.05 is a positive claim that the difference is smaller than the
    margin, not merely a failure to find one."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n, mean, se = len(d), float(d.mean()), sem(d)
    lower = 1.0 - stats.t.cdf((mean + margin) / se, n - 1)
    upper = stats.t.cdf((mean - margin) / se, n - 1)
    return mean, se, float(max(lower, upper))


def resolution(repeats):
    """Two standard deviations over repeat measurements of one frozen model."""
    return 2.0 * float(np.std(np.asarray(repeats, dtype=float), ddof=1))


def verdict(mean, p, res):
    if p < 0.05 and abs(mean) > res:
        return "effect"
    if p >= 0.05 and abs(mean) <= res:
        return "below resolution"
    return "below resolution" if abs(mean) <= res else "not significant"


def level(reevals, k=6):
    """The reported level: the mean of the last k re-evaluations, with its spread and the residual
    bias between that mean and its own second half."""
    tail = list(reevals)[-k:]
    mean = float(np.mean(tail))
    sd = float(np.std(tail, ddof=1))
    return mean, sd, sd / np.sqrt(len(tail)), mean - float(np.mean(tail[len(tail) // 2:]))


def first_within(updates, values, target, res):
    """First checkpoint within one resolution of `target`."""
    return next((int(u) for u, v in zip(updates, values) if abs(v - target) <= res), None)


def plateau(updates, values, target, res):
    """First checkpoint from which every later value stays within one resolution of `target`."""
    for i, u in enumerate(updates):
        if all(abs(v - target) <= res for v in values[i:]):
            return int(u)
    return None


def probit(accuracy):
    """Accuracy to a d'-like scale, so a decoding accuracy and a behavioural d' can be compared."""
    return 2.0 * float(stats.norm.ppf(min(max(accuracy, 0.01), 0.99)))
