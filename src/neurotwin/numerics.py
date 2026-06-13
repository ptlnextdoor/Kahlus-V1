"""Numerical hygiene helpers shared by the synthetic v2/v3/EM scaffolds."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import numpy as np


@contextmanager
def ignore_spurious_matmul_warnings() -> Iterator[None]:
    """Suppress the known-spurious BLAS matmul FP warnings.

    NumPy 2.x on Apple Accelerate emits false-positive ``RuntimeWarning`` messages
    ("divide by zero / overflow / invalid value encountered in matmul") from the SIMD
    matmul padding lanes even when every produced value is finite. This only changes
    warning emission, not numerical results; callers still validate finiteness explicitly,
    so a genuine non-finite value remains detectable downstream.
    """

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        yield
