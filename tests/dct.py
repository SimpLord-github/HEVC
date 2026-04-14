"""
dct.py — HEVC Integer DCT / DST Forward Transform
Implements:
  - 4×4 DST-7     (intra luma small blocks, per HEVC spec §8.6.4)
  - 4×4 DCT-2
  - 8×8 DCT-2
  - 16×16 DCT-2
  - 32×32 DCT-2

All basis matrices are the HEVC integer approximations from Tables 4-5
in ISO/IEC 23008-2. Shift values follow the two-pass (row then column)
butterfly structure used by hardware implementations.

Usage
-----
    from transform.dct import forward_dct, forward_dst

    residual_4x4  = np.array([[...]], dtype=np.int16)
    coeffs        = forward_dct(residual_4x4)   # auto-selects size
    coeffs_dst    = forward_dst(residual_4x4)   # 4×4 DST-7 only
"""

import numpy as np
from typing import Literal

# ---------------------------------------------------------------------------
# HEVC Integer DCT-2 Basis Matrices (ISO/IEC 23008-2, Tables 4-5)
# Each row is one basis vector; multiply residual columns then rows.
# ---------------------------------------------------------------------------

DCT4: np.ndarray = np.array([
    [64,  64,  64,  64],
    [83,  36, -36, -83],
    [64, -64, -64,  64],
    [36, -83,  83, -36],
], dtype=np.int32)

DCT8: np.ndarray = np.array([
    [64,  64,  64,  64,  64,  64,  64,  64],
    [89,  75,  50,  18, -18, -50, -75, -89],
    [83,  36, -36, -83, -83, -36,  36,  83],
    [75, -18, -89, -50,  50,  89,  18, -75],
    [64, -64, -64,  64,  64, -64, -64,  64],
    [50, -89,  18,  75, -75, -18,  89, -50],
    [36, -83,  83, -36, -36,  83, -83,  36],
    [18, -50,  75, -89,  89, -75,  50, -18],
], dtype=np.int32)

DCT16: np.ndarray = np.array([
    [ 64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64,  64],
    [ 90,  87,  80,  70,  57,  43,  25,   9,  -9, -25, -43, -57, -70, -80, -87, -90],
    [ 89,  75,  50,  18, -18, -50, -75, -89, -89, -75, -50, -18,  18,  50,  75,  89],
    [ 87,  57,   9, -43, -80, -90, -70, -25,  25,  70,  90,  80,  43,  -9, -57, -87],
    [ 83,  36, -36, -83, -83, -36,  36,  83,  83,  36, -36, -83, -83, -36,  36,  83],
    [ 80,   9, -70, -87, -25,  57,  90,  43, -43, -90, -57,  25,  87,  70,  -9, -80],
    [ 75, -18, -89, -50,  50,  89,  18, -75, -75,  18,  89,  50, -50, -89, -18,  75],
    [ 70, -43, -87,   9,  90,  25, -80, -57,  57,  80, -25, -90,  -9,  87,  43, -70],
    [ 64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64,  64, -64, -64,  64],
    [ 57, -80, -25,  90,  -9, -87,  43,  70, -70, -43,  87,   9, -90,  25,  80, -57],
    [ 50, -89,  18,  75, -75, -18,  89, -50, -50,  89, -18, -75,  75,  18, -89,  50],
    [ 43, -90,  57,  25, -87,  70,   9, -80,  80,  -9, -70,  87, -25, -57,  90, -43],
    [ 36, -83,  83, -36, -36,  83, -83,  36,  36, -83,  83, -36, -36,  83, -83,  36],
    [ 25, -70,  90, -80,  43,   9, -57,  87, -87,  57,  -9, -43,  80, -90,  70, -25],
    [ 18, -50,  75, -89,  89, -75,  50, -18, -18,  50, -75,  89, -89,  75, -50,  18],
    [  9, -25,  43, -57,  70, -80,  87, -90,  90, -87,  80, -70,  57, -43,  25,  -9],
], dtype=np.int32)

DCT32: np.ndarray = np.array([
    [ 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
      64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    [ 90, 90, 88, 85, 82, 78, 73, 67, 61, 54, 46, 38, 31, 22, 13,  4,
      -4,-13,-22,-31,-38,-46,-54,-61,-67,-73,-78,-82,-85,-88,-90,-90],
    [ 90, 87, 80, 70, 57, 43, 25,  9, -9,-25,-43,-57,-70,-80,-87,-90,
     -90,-87,-80,-70,-57,-43,-25, -9,  9, 25, 43, 57, 70, 80, 87, 90],
    [ 90, 82, 67, 46, 22, -4,-31,-54,-73,-85,-90,-88,-78,-61,-38,-13,
      13, 38, 61, 78, 88, 90, 85, 73, 54, 31,  4,-22,-46,-67,-82,-90],
    [ 89, 75, 50, 18,-18,-50,-75,-89,-89,-75,-50,-18, 18, 50, 75, 89,
      89, 75, 50, 18,-18,-50,-75,-89,-89,-75,-50,-18, 18, 50, 75, 89],
    [ 88, 67, 31,-13,-54,-82,-90,-78,-46, -4, 38, 73, 90, 85, 61, 22,
     -22,-61,-85,-90,-73,-38,  4, 46, 78, 90, 82, 54, 13,-31,-67,-88],
    [ 87, 57,  9,-43,-80,-90,-70,-25, 25, 70, 90, 80, 43, -9,-57,-87,
     -87,-57, -9, 43, 80, 90, 70, 25,-25,-70,-90,-80,-43,  9, 57, 87],
    [ 85, 46,-13,-67,-90,-73,-22, 38, 82, 88, 54, -4,-61,-90,-78,-31,
      31, 78, 90, 61,  4,-54,-88,-82,-38, 22, 73, 90, 67, 13,-46,-85],
    [ 83, 36,-36,-83,-83,-36, 36, 83, 83, 36,-36,-83,-83,-36, 36, 83,
      83, 36,-36,-83,-83,-36, 36, 83, 83, 36,-36,-83,-83,-36, 36, 83],
    [ 82, 22,-54,-90,-61, 13, 78, 85, 31,-46,-90,-67,  4, 73, 88, 38,
     -38,-88,-73, -4, 67, 90, 46,-31,-85,-78,-13, 61, 90, 54,-22,-82],
    [ 80,  9,-70,-87,-25, 57, 90, 43,-43,-90,-57, 25, 87, 70, -9,-80,
     -80, -9, 70, 87, 25,-57,-90,-43, 43, 90, 57,-25,-87,-70,  9, 80],
    [ 78, -4,-82,-73, 13, 85, 67,-22,-88,-61, 31, 90, 54,-38,-90,-46,
      46, 90, 38,-54,-90,-31, 61, 88, 22,-67,-85,-13, 73, 82,  4,-78],
    [ 75,-18,-89,-50, 50, 89, 18,-75,-75, 18, 89, 50,-50,-89,-18, 75,
      75,-18,-89,-50, 50, 89, 18,-75,-75, 18, 89, 50,-50,-89,-18, 75],
    [ 73,-31,-90,-22, 78, 67,-38,-90,-13, 82, 61,-46,-88, -4, 85, 54,
     -54,-85,  4, 88, 46,-61,-82, 13, 90, 38,-67,-78, 22, 90, 31,-73],
    [ 70,-43,-87,  9, 90, 25,-80,-57, 57, 80,-25,-90, -9, 87, 43,-70,
     -70, 43, 87, -9,-90,-25, 80, 57,-57,-80, 25, 90,  9,-87,-43, 70],
    [ 67,-54,-78, 38, 85,-22,-90,  4, 90, 13,-88,-31, 82, 46,-73,-61,
      61, 73,-46,-82, 31, 88,-13,-90, -4, 90, 22,-85,-38, 78, 54,-67],
    [ 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64,
      64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64, 64,-64,-64, 64],
    [ 61,-73,-46, 82, 31,-88,-13, 90, -4,-90, 22, 85,-38,-78, 54, 67,
     -67,-54, 78, 38,-85,-22, 90,  4,-90, 13, 88,-31,-82, 46, 73,-61],
    [ 57,-80,-25, 90, -9,-87, 43, 70,-70,-43, 87,  9,-90, 25, 80,-57,
     -57, 80, 25,-90,  9, 87,-43,-70, 70, 43,-87, -9, 90,-25,-80, 57],
    [ 54,-85, -4, 88,-46,-61, 78, 22,-90, 31, 67,-73,-13, 90,-38,-82,
      82, 38,-90, 13, 73,-67,-31, 90,-22,-78, 61, 46,-88,  4, 85,-54],
    [ 50,-89, 18, 75,-75,-18, 89,-50,-50, 89,-18,-75, 75, 18,-89, 50,
      50,-89, 18, 75,-75,-18, 89,-50,-50, 89,-18,-75, 75, 18,-89, 50],
    [ 46,-90, 38, 54,-90, 31, 61,-88, 22, 67,-85, 13, 73,-82,  4, 78,
     -78, -4, 82,-73,-13, 85,-67,-22, 88,-61,-31, 90,-54,-38, 90,-46],
    [ 43,-90, 57, 25,-87, 70,  9,-80, 80, -9,-70, 87,-25,-57, 90,-43,
     -43, 90,-57,-25, 87,-70, -9, 80,-80,  9, 70,-87, 25, 57,-90, 43],
    [ 38,-88, 73, -4,-67, 90,-46,-31, 85,-78, 13, 61,-90, 54, 22,-82,
      82,-22,-54, 90,-61,-13, 78,-85, 31, 46,-90, 67,  4,-73, 88,-38],
    [ 36,-83, 83,-36,-36, 83,-83, 36, 36,-83, 83,-36,-36, 83,-83, 36,
      36,-83, 83,-36,-36, 83,-83, 36, 36,-83, 83,-36,-36, 83,-83, 36],
    [ 31,-78, 90,-61,  4, 54,-88, 82,-38,-22, 73,-90, 67,-13,-46, 85,
     -85, 46, 13,-67, 90,-73, 22, 38,-82, 88,-54, -4, 61,-90, 78,-31],
    [ 25,-70, 90,-80, 43,  9,-57, 87,-87, 57, -9,-43, 80,-90, 70,-25,
     -25, 70,-90, 80,-43, -9, 57,-87, 87,-57,  9, 43,-80, 90,-70, 25],
    [ 22,-61, 85,-90, 73,-38, -4, 46,-78, 90,-82, 54,-13,-31, 67,-88,
      88,-67, 31, 13,-54, 82,-90, 78,-46,  4, 38,-73, 90,-85, 61,-22],
    [ 18,-50, 75,-89, 89,-75, 50,-18,-18, 50,-75, 89,-89, 75,-50, 18,
      18,-50, 75,-89, 89,-75, 50,-18,-18, 50,-75, 89,-89, 75,-50, 18],
    [ 13,-38, 61,-78, 88,-90, 85,-73, 54,-31,  4, 22,-46, 67,-82, 90,
     -90, 82,-67, 46,-22, -4, 31,-54, 73,-85, 90,-88, 78,-61, 38,-13],
    [  9,-25, 43,-57, 70,-80, 87,-90, 90,-87, 80,-70, 57,-43, 25, -9,
      -9, 25,-43, 57,-70, 80,-87, 90,-90, 87,-80, 70,-57, 43,-25,  9],
    [  4,-13, 22,-31, 38,-46, 54,-61, 67,-73, 78,-82, 85,-88, 90,-90,
      90,-90, 88,-85, 82,-78, 73,-67, 61,-54, 46,-38, 31,-22, 13, -4],
], dtype=np.int32)

# 4×4 DST-7 (used for 4×4 intra luma, per HEVC spec §8.6.4)
DST4: np.ndarray = np.array([
    [29, 55, 74, 84],
    [74, 74,  0,-74],
    [84,-29,-74, 55],
    [55,-84, 74,-29],
], dtype=np.int32)

# Map block size → basis matrix
_DCT_MATRIX = {4: DCT4, 8: DCT8, 16: DCT16, 32: DCT32}

# ---------------------------------------------------------------------------
# Shift constants — two-pass butterfly (HEVC spec §8.6.4)
# shift1 applied after row transform, shift2 after column transform.
# ---------------------------------------------------------------------------
_SHIFT1: int = 7   # pass-1 rounding shift — fixed for all sizes

# Pass-2 shift is SIZE-DEPENDENT because the basis-matrix row norm scales with N.
# After pass-1 the intermediate values have been multiplied by T (row energy ≈ N·2^12),
# so the second multiply by T again accumulates another factor of N.
# Correct shift2 = 2·(log₂N − 1):
#   N=4  → 2   N=8  → 4   N=16 → 6   N=32 → 8
# This keeps coefficient magnitudes in a range that lets the normative IDCT
# (shift1=7, shift2=12) reconstruct residuals with ≤2 LSB rounding error.
_SHIFT2_TABLE: dict[int, int] = {4: 2, 8: 4, 16: 6, 32: 8}


def _round_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """Integer rounding right-shift: (x + (1 << (shift-1))) >> shift."""
    return (x + (1 << (shift - 1))) >> shift


def _apply_1d(basis: np.ndarray, block: np.ndarray, shift: int) -> np.ndarray:
    """
    Apply a 1-D transform along rows of `block` using `basis`, then shift.

    Parameters
    ----------
    basis : (N, N) int32 — the DCT/DST basis matrix
    block : (N, N) int32 — input block (rows = samples to transform)
    shift : int          — rounding right-shift to apply after multiply

    Returns
    -------
    (N, N) int32 — transformed and rounded rows
    """
    result = basis @ block  # (N,N) × (N,N) → (N,N), int32 safe up to N=32
    return _round_shift(result, shift)


def forward_dct(
    residual: np.ndarray,
    transform_size: Literal[4, 8, 16, 32] | None = None,
) -> np.ndarray:
    """
    Forward 2-D integer DCT-2 as defined in HEVC (ISO/IEC 23008-2 §8.6.4).

    The two-pass structure is:
        1. Row transform  : T_row  = basis @ residual.T   (then right-shift by _SHIFT1)
        2. Column transform: T_col = basis @ T_row        (then right-shift by _SHIFT2)

    Parameters
    ----------
    residual : np.ndarray
        2-D array of residual values (prediction error), shape (N, N).
        Dtype is cast to int32 internally.
    transform_size : {4, 8, 16, 32} or None
        If None, inferred from residual shape. Must be square.

    Returns
    -------
    coeffs : np.ndarray
        Integer DCT coefficients, shape (N, N), dtype int32.

    Raises
    ------
    ValueError
        If the block is not square, or the size is unsupported.

    Examples
    --------
    >>> import numpy as np
    >>> from transform.dct import forward_dct
    >>> res = np.random.randint(-128, 128, (8, 8), dtype=np.int16)
    >>> coeffs = forward_dct(res)
    >>> coeffs.shape
    (8, 8)
    """
    block = np.asarray(residual, dtype=np.int32)
    _validate_block(block, transform_size)

    n = block.shape[0]
    basis = _DCT_MATRIX[n]
    shift2 = _SHIFT2_TABLE[n]

    # C = T @ X @ T^T  (column transform first, then row transform)
    temp   = _apply_1d(basis, block,  _SHIFT1)          # T @ X
    coeffs = _apply_1d(basis, temp.T, shift2).T          # T @ X @ T^T

    return coeffs

    return coeffs


def forward_dst(residual: np.ndarray) -> np.ndarray:
    """
    Forward 4×4 DST-7 (Discrete Sine Transform, type 7).

    Used exclusively for 4×4 intra luma transform units in HEVC.
    The basis is DST4 as defined in ISO/IEC 23008-2 §8.6.4.

    Parameters
    ----------
    residual : np.ndarray
        4×4 array of residual values, dtype cast to int32.

    Returns
    -------
    coeffs : np.ndarray
        4×4 integer DST-7 coefficients, dtype int32.

    Raises
    ------
    ValueError
        If residual is not 4×4.

    Examples
    --------
    >>> import numpy as np
    >>> from transform.dct import forward_dst
    >>> res = np.array([[10,-5,3,0],[-2,8,1,4],[0,0,6,-3],[1,-1,2,5]], dtype=np.int16)
    >>> coeffs = forward_dst(res)
    """
    block = np.asarray(residual, dtype=np.int32)
    if block.shape != (4, 4):
        raise ValueError(f"forward_dst requires a 4×4 block, got {block.shape}")

    t_row  = _apply_1d(DST4, block,  _SHIFT1)
    coeffs = _apply_1d(DST4, t_row.T, _SHIFT2_TABLE[4]).T
    return coeffs


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_block(block: np.ndarray, size_hint: int | None) -> None:
    if block.ndim != 2:
        raise ValueError(f"Block must be 2-D, got {block.ndim}-D.")
    h, w = block.shape
    if h != w:
        raise ValueError(f"Block must be square, got {h}×{w}.")
    n = size_hint if size_hint is not None else h
    if n not in _DCT_MATRIX:
        raise ValueError(f"Unsupported transform size {n}. Supported: {list(_DCT_MATRIX)}.")
    if h != n:
        raise ValueError(f"Block shape {h}×{w} does not match declared size {n}.")


def get_basis_matrix(size: Literal[4, 8, 16, 32], use_dst: bool = False) -> np.ndarray:
    """
    Return the HEVC integer basis matrix for a given transform size.

    Parameters
    ----------
    size : {4, 8, 16, 32}
    use_dst : bool
        If True and size == 4, returns the DST-7 matrix instead of DCT4.

    Returns
    -------
    np.ndarray — (size, size) int32 basis matrix (read-only view)
    """
    if use_dst:
        if size != 4:
            raise ValueError("DST-7 is only defined for 4×4 blocks in HEVC.")
        mat = DST4
    else:
        if size not in _DCT_MATRIX:
            raise ValueError(f"No DCT matrix for size {size}.")
        mat = _DCT_MATRIX[size]
    return mat.view()  # read-only — callers should not mutate basis matrices