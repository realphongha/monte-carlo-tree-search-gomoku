# cython: infer_types=True
cimport cython
from libc.math cimport sqrt, log, INFINITY, e


@cython.boundscheck(False)  # Deactivate bounds checking
def ucb(float w, float n, float c, float t):
    return c * sqrt(log(t)/n) + w / n if n != 0 else INFINITY


@cython.boundscheck(False)  # Deactivate bounds checking
def score(float w, float n):
    return w / n if n != 0 else -INFINITY
