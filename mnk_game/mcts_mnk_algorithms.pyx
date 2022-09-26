# cython: infer_types=True
from libc.math cimport sqrt, log, INFINITY, e


def ucb(float w, float n, float c, float t):
    return c * sqrt(log(t)/n) + w / n if n != 0 else INFINITY


def score(float w, float n):
    return w / n if n != 0 else -INFINITY
