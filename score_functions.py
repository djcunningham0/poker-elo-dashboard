import numpy as np


def linear_score_function(n):
    return np.array([(n - p) / (n * (n - 1) / 2) for p in range(1, n + 1)])


def _exponential_score_template(n, base):
    if base < 1:
        raise ValueError("base must be >= 1")
    if base == 1:
        return linear_score_function(n)  # it converges to this at base = 1

    out = np.array([base ** (n - p) for p in range(1, n + 1)])
    out = out - min(out)
    return out / sum(out)


def create_exponential_score_function(base):
    return lambda n: _exponential_score_template(n, base)
