"""
idct.py — HEVC Integer Inverse DCT / DST Transform
Implements inverse transform as defined in ISO/IEC 23008-2 §8.6.4.

Pipeline position
-----------------
    quantizer.py  →  [idct.py]  →  loop_filters / reconstruction

Hardware Reality
----------------
The IDCT is the normative anchor of HEVC — only the decoder is standardised.
These shifts are FIXED by the spec and cannot be made size-dependent.
Any roundtrip errors from a pure DCT→IDCT test (≤ 2 LSB) are expected:
the integer basis matrices are approximations — H^T × H ≠ I exactly.
The codec absorbs this non-orthogonality inside the Q/IQ scaling stages.

Two-pass structure (ISO/IEC 23008-2 §8.6.4)
--------------------------------------------
    Pass 1 — vertical (column) IDCT:
        G = round_shift( H^T @ D,  shift1 )    clip to int16

    Pass 2 — horizontal (row) IDCT:
        R = round_shift( G  @ H,  shift2 )    clip to int16

Where:
    shift1 = 7  (constant for all block sizes and 8-bit depth)
    shift2 = 20 − bit_depth  (= 12 for standard 8-bit video)

Relationship to dct.py
-----------------------
    dct.py   forward:   C = round_shift(H @ X,       7)  →  round_shift(H @ temp^T, shift2_fwd)^T
    idct.py  inverse:   G = round_shift(H^T @ C,     7)  →  round_shift(G @ H,      12)

The forward shift2_fwd is SIZE-DEPENDENT (2·(log₂N − 1)) to keep coefficient
magnitudes compatible with this fixed-shift inverse. See dct.py for details.
"""

import numpy as np
from typing import Literal

from dct import get_basis_matrix

# ---------------------------------------------------------------------------
# Normative Inverse Shift Constants — ISO/IEC 23008-2 §8.6.4
# These are FIXED for all block sizes. Do not make them size-dependent.
# ---------------------------------------------------------------------------
_INVERSE_SHIFT1_BASE: int = 7   # column pass shift
_INVERSE_SHIFT2_BASE: int = 20  # row pass base; actual = 20 − bit_depth


def _round_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """Integer rounding right-shift: (x + (1 << (shift-1))) >> shift."""
    return (x + (1 << (shift - 1))) >> shift


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def inverse_dct(
    coeffs: np.ndarray,
    transform_size: Literal[4, 8, 16, 32] | None = None,
    bit_depth: int = 8,
) -> np.ndarray:
    """
    2-D integer Inverse DCT-2 as defined in HEVC (ISO/IEC 23008-2 §8.6.4).

    Reconstructs a spatial-domain residual block from dequantised coefficients.
    This is the bit-exact integer inverse of dct.forward_dct().

    Two-pass butterfly:
        Pass 1 (vertical)  : G = round_shift( H^T @ D, shift1 )  → clip int16
        Pass 2 (horizontal): R = round_shift( G  @ H,  shift2 )  → clip int16

    Parameters
    ----------
    coeffs : np.ndarray
        2-D dequantised coefficient block from quantizer.dequantize().
        Shape (N, N), cast to int32 internally.
    transform_size : {4, 8, 16, 32} or None
        If None, inferred from coeffs.shape. When provided, must match shape.
    bit_depth : int
        Pixel bit depth (default 8 for standard SDR video).
        shift2 = 20 − bit_depth; for HDR 10-bit, pass bit_depth=10.

    Returns
    -------
    residual : np.ndarray
        Reconstructed residual block, shape (N, N), dtype int16.
        Clipped to [−32768, 32767] — matches the 16-bit accumulation register
        used by hardware before adding to the prediction signal.

    Raises
    ------
    ValueError
        If the block is not a 2-D square matrix, or the size is not in
        {4, 8, 16, 32}, or shape mismatches the declared transform_size.

    Examples
    --------
    >>> import numpy as np
    >>> from transform.dct import forward_dct
    >>> from transform.quantizer import quantize, dequantize
    >>> from transform.idct import inverse_dct
    >>>
    >>> residual_orig  = np.random.randint(-128, 128, (8, 8), dtype=np.int16)
    >>> coeffs         = forward_dct(residual_orig)
    >>> levels         = quantize(coeffs, qp=28)
    >>> recon_coeffs   = dequantize(levels, qp=28)
    >>> residual_recon = inverse_dct(recon_coeffs)
    """
    block = _validate_and_cast(coeffs, transform_size)
    basis = get_basis_matrix(block.shape[0], use_dst=False)

    shift1 = _INVERSE_SHIFT1_BASE + (bit_depth - 8)
    shift2 = _INVERSE_SHIFT2_BASE - bit_depth  # = 12 for 8-bit

    # Pass 1 — vertical (column) IDCT: G = H^T @ D
    pass1 = _round_shift(basis.T.astype(np.int64) @ block.astype(np.int64), shift1)
    # Intermediate clip: hardware stores this in a 16-bit signed register
    pass1 = np.clip(pass1, -32768, 32767).astype(np.int32)

    # Pass 2 — horizontal (row) IDCT: R = G @ H
    pass2    = _round_shift(pass1.astype(np.int64) @ basis.astype(np.int64), shift2)
    residual = np.clip(pass2, -32768, 32767).astype(np.int16)

    return residual


def inverse_dst(coeffs: np.ndarray, bit_depth: int = 8) -> np.ndarray:
    """
    Inverse 4×4 DST-7 for intra luma residual blocks.

    Exact inverse of dct.forward_dst(). Same two-pass structure and shift
    values as inverse_dct(), using the DST-7 basis matrix instead of DCT-2.

    Parameters
    ----------
    coeffs : np.ndarray
        4×4 dequantised DST-7 coefficient block, cast to int32 internally.
    bit_depth : int
        Pixel bit depth (default 8).

    Returns
    -------
    residual : np.ndarray
        4×4 reconstructed residual, dtype int16, clipped to [−32768, 32767].

    Raises
    ------
    ValueError
        If coeffs is not exactly 4×4.
    """
    block = np.asarray(coeffs, dtype=np.int32)
    if block.shape != (4, 4):
        raise ValueError(f"inverse_dst requires a 4x4 block, got {block.shape}")

    basis = get_basis_matrix(4, use_dst=True)  # DST-7 basis

    shift1 = _INVERSE_SHIFT1_BASE + (bit_depth - 8)
    shift2 = _INVERSE_SHIFT2_BASE - bit_depth

    # Pass 1 — vertical
    pass1 = _round_shift(basis.T.astype(np.int64) @ block.astype(np.int64), shift1)
    pass1 = np.clip(pass1, -32768, 32767).astype(np.int32)

    # Pass 2 — horizontal
    pass2    = _round_shift(pass1.astype(np.int64) @ basis.astype(np.int64), shift2)
    residual = np.clip(pass2, -32768, 32767).astype(np.int16)

    return residual


def full_roundtrip(
    residual: np.ndarray,
    qp: int,
    is_intra: bool = True,
    use_dst: bool = False,
    bit_depth: int = 8,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run the complete forward + inverse transform chain and report distortion.

    Chain: residual → [DCT or DST] → quantize → dequantize → [IDCT or IDST]

    This is the exact sequence executed by forward_tq + inverse_tq in the
    real encoder. Use for integration testing and QP tuning before wiring
    the full pipeline.

    Parameters
    ----------
    residual  : np.ndarray — original NxN residual block (int16 or int32)
    qp        : int        — quantisation parameter [0, 51]
    is_intra  : bool       — intra dead-zone if True, inter if False
    use_dst   : bool       — use DST-7 instead of DCT-2 (4×4 intra luma only)
    bit_depth : int        — pixel bit depth (default 8)

    Returns
    -------
    (recon, error, psnr_proxy) : tuple
        recon       — reconstructed residual (int16)
        error       — per-sample absolute error (int32)
        psnr_proxy  — 10·log₁₀(255² / MSE), or inf if MSE == 0
    """
    import math
    from dct import forward_dct, forward_dst
    from quantizer import quantize, dequantize

    block   = np.asarray(residual, dtype=np.int32)
    coeffs  = forward_dst(block) if use_dst else forward_dct(block)
    levels  = quantize(coeffs, qp=qp, is_intra=is_intra)
    recoeff = dequantize(levels, qp=qp)
    recon   = inverse_dst(recoeff, bit_depth) if use_dst else inverse_dct(recoeff, bit_depth=bit_depth)

    error = np.abs(block.astype(np.int32) - recon.astype(np.int32))
    mse   = float(np.mean(error.astype(np.float64) ** 2))
    psnr  = 10.0 * math.log10(255.0 ** 2 / mse) if mse > 0.0 else float("inf")

    return recon, error, psnr


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_cast(block: np.ndarray, size_hint: int | None) -> np.ndarray:
    """Validate 2-D square shape and cast to int32."""
    if block.ndim != 2:
        raise ValueError(f"Block must be 2-D, got {block.ndim}-D.")
    h, w = block.shape
    if h != w:
        raise ValueError(f"Block must be a square matrix, got {h}×{w}.")
    n = size_hint if size_hint is not None else h
    valid = {4, 8, 16, 32}
    if n not in valid:
        raise ValueError(f"Unsupported transform size {n}. Supported: {sorted(valid)}.")
    if h != n:
        raise ValueError(f"Block shape {h}×{w} does not match transform_size {n}.")
    return block.astype(np.int32)