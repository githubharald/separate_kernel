from dataclasses import dataclass
from typing import Tuple, Callable

import numpy as np
from scipy.optimize import least_squares


def _extract_vectors(x: np.ndarray, a_dim: int, symmetric_kernel: bool) -> Tuple[np.ndarray, np.ndarray]:
    a = x[:a_dim][..., None]
    b = x[:a_dim][None, ...] if symmetric_kernel else x[a_dim:][None, ...]
    return a, b


def _create_cost_function(M: np.ndarray, a_dim: int, symmetric_kernel: bool) -> Callable:
    def cost_function(x: np.ndarray) -> np.ndarray:
        a, b = _extract_vectors(x, a_dim, symmetric_kernel)
        r = a @ b - M  # the residuals to be optimized
        return r.flatten()

    return cost_function


@dataclass
class SeparatedKernel:
    col_vec: np.ndarray
    row_vec: np.ndarray
    error: float


def separate_kernel(M: np.ndarray, symmetric_kernel: bool = False) -> SeparatedKernel:
    """
    Separate 2D kernel M into two 1D kernels C (column vector) and R (row vector).

    Args:
        M: The 2D kernel to be separated.
        symmetric_kernel: True if kernel is symmetric, which means C == R.T.

    Returns:
        An object of type SeparatedKernel containing the column and row vectors C and R, and an error measure.
    """
    assert M.ndim == 2
    a_dim = M.shape[0]
    b_dim = M.shape[1]
    if symmetric_kernel and a_dim != b_dim:
        raise ValueError(f'Symmetric kernels must have shape[0]==shape[1], but got {a_dim}!={b_dim}')

    x0 = np.random.rand(a_dim) if symmetric_kernel else np.random.rand(a_dim + b_dim)
    res = least_squares(_create_cost_function(M, a_dim, symmetric_kernel), x0)
    a, b = _extract_vectors(res.x, a_dim, symmetric_kernel)
    sse = np.sum((M - a @ b) ** 2)

    return SeparatedKernel(a, b, sse)
