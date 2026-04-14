"""
quantizer.py — HEVC Scaling & Quantization / Inverse Scaling
Implements forward and inverse quantization as defined in
ISO/IEC 23008-2 (HEVC) §8.6.3.

Pipeline position
-----------------
    dct.py  →  [quantizer.py]  →  cabac_writer.py
    (coeffs)   (quant levels)     (entropy coding)

    cabac_writer.py  →  [quantizer.py inverse]  →  inverse_tq.py
    (quant levels)       (scaled coeffs)            (IDCT → residual)

Quantization model
------------------
HEVC quantisation is a uniform scalar quantiser with a step size that
doubles every 6 QP values (QP range 0–51).  The implementation avoids
division by encoding the reciprocal scale as a multiply-then-shift:

    level = (|coeff| * MF + offset) >> (shift + log2(N) - 1)

where
    MF       — multiplication factor, depends on (QP % 6)
    shift    — depends on (QP // 6) and bit-depth
    offset   — dead-zone control: larger dead-zone keeps more coeffs at 0

Inverse scaling reconstructs coefficients for the IDCT:

    scaled = level * IQ_SCALE[qp % 6] << (qp // 6)   (with shift correction)

All arithmetic uses int32/int64 to match hardware integer pipelines.

Public API
----------
    quantize(coeffs, qp, is_intra, transform_size) -> np.ndarray  (levels)
    dequantize(levels, qp, transform_size)          -> np.ndarray  (scaled coeffs)
    qp_to_step_size(qp)                             -> float       (informational)
"""

import numpy as np
from typing import Literal

# ---------------------------------------------------------------------------
# QP → Multiplication Factor and Inverse Scale tables
# ISO/IEC 23008-2 §8.6.3, Tables 9 and 10.
# Index = QP % 6, giving the 6 base scale factors per doubling period.
# ---------------------------------------------------------------------------

# Forward quantisation: multiply-factors (MF) for QP % 6 in [0..5]
# The full step size at QP q is:  Qstep = MF[q%6] >> (15 - q//6)
_MF: np.ndarray = np.array([26214, 23302, 20560, 18396, 16384, 14564], dtype=np.int64)

# Inverse quantisation scale factors for QP % 6 in [0..5]
# Scaled coefficients = level * _IQ[qp%6] << (qp//6)   (with bit-depth shift)
_IQ: np.ndarray = np.array([40, 45, 51, 57, 64, 72], dtype=np.int32)

# Supported transform sizes and their log2 values
_VALID_SIZES = {4: 2, 8: 3, 16: 4, 32: 5}

# ---------------------------------------------------------------------------
# Shift constants
# HEVC §8.6.3: forward shift base = 29, minus (QP//6), adjusted by log2(N)-1
# The constant 14 encodes: bit_depth(8) + log2(basis_scale=64) - 1 = 14
# so total shift = 14 + log2(N) - 1  =  13 + log2(N)
# ---------------------------------------------------------------------------
_QUANT_SHIFT   = 14   # HEVC §8.6.3: base forward-quantisation shift
_INVERSE_SHIFT = 6    # right-shift after inverse scale to normalise output

# Kept for backward compatibility (used in docstring / older callers)
_FORWARD_SHIFT_BASE = _QUANT_SHIFT


# ---------------------------------------------------------------------------
# Public: forward quantisation
# ---------------------------------------------------------------------------

def quantize(
    coeffs: np.ndarray,
    qp: int,
    is_intra: bool = True,
    transform_size: Literal[4, 8, 16, 32] | None = None,
) -> np.ndarray:
    """
    Forward quantisation — map DCT coefficients to quantisation levels.

    Uses dead-zone scalar quantisation:
        level = (|coeff| * MF + dead_zone_offset) >> shift
        sign  is restored after threshold

    Dead-zone offsets (HEVC convention):
        Intra:  offset = (1 << shift) * 2/3   — narrower dead-zone keeps detail
        Inter:  offset = (1 << shift) * 1/2   — wider dead-zone, smoother motion

    Parameters
    ----------
    coeffs : np.ndarray
        2-D integer DCT coefficient block from dct.py, shape (N, N).
    qp : int
        Quantisation parameter, range [0, 51].
    is_intra : bool
        True for intra-coded TUs, False for inter. Affects dead-zone width.
    transform_size : {4, 8, 16, 32} or None
        If None, inferred from coeffs.shape.

    Returns
    -------
    levels : np.ndarray
        Quantised transform coefficient levels, same shape as coeffs,
        dtype int32. Zero coefficients are already dead-zoned.

    Raises
    ------
    ValueError
        If qp is out of range, block is non-square, or size is unsupported.

    Examples
    --------
    >>> import numpy as np
    >>> from transform.dct import forward_dct
    >>> from transform.quantizer import quantize
    >>> residual = np.random.randint(-128, 128, (8, 8), dtype=np.int16)
    >>> coeffs   = forward_dct(residual)
    >>> levels   = quantize(coeffs, qp=28)
    """
    _validate_qp(qp)
    coeffs_i64 = _validate_and_cast(coeffs, transform_size)

    mf = int(_MF[qp % 6])

    # HEVC §8.6.3: single forward shift = QUANT_SHIFT(14) + qp//6
    # The step size doubles every 6 QP via the increasing shift, not a separate
    # right-shift on top. Combining them into one shift avoids precision loss.
    shift = _QUANT_SHIFT + qp // 6

    # Dead-zone offset: fraction of one quantisation step
    # Intra = 2/3 step → preserve more high-frequency detail
    # Inter = 1/2 step → more aggressive zeroing for motion-coded blocks
    if is_intra:
        offset = (1 << shift) * 2 // 3
    else:
        offset = (1 << shift) >> 1

    abs_coeffs = np.abs(coeffs_i64)
    levels     = ((abs_coeffs * mf + offset) >> shift).astype(np.int32)

    # Restore sign
    levels = np.where(coeffs_i64 < 0, -levels, levels)
    return levels


# ---------------------------------------------------------------------------
# Public: inverse quantisation (dequantisation)
# ---------------------------------------------------------------------------

def dequantize(
    levels: np.ndarray,
    qp: int,
    transform_size: Literal[4, 8, 16, 32] | None = None,
) -> np.ndarray:
    """
    Inverse quantisation — reconstruct scaled DCT coefficients from levels.

    Reconstruction formula (HEVC §8.6.3):
        coeff_scaled = level * IQ[qp % 6] * (1 << (qp // 6))
        coeff        = coeff_scaled >> INVERSE_SHIFT

    The result feeds directly into inverse_tq.py for the IDCT.

    Parameters
    ----------
    levels : np.ndarray
        2-D quantised coefficient levels from quantize(), shape (N, N).
    qp : int
        Quantisation parameter used during encoding, range [0, 51].
    transform_size : {4, 8, 16, 32} or None
        If None, inferred from levels.shape.

    Returns
    -------
    coeffs : np.ndarray
        Reconstructed (approximately dequantised) coefficients, int32.
        These are the inputs expected by the IDCT in inverse_tq.py.

    Examples
    --------
    >>> levels = quantize(coeffs, qp=28)
    >>> recon  = dequantize(levels, qp=28)
    """
    _validate_qp(qp)
    levels_i32 = _validate_and_cast(levels, transform_size)

    iq       = int(_IQ[qp % 6])
    qp_shift = qp // 6

    # Upscale: multiply by IQ factor then shift up by QP period
    # Use int64 to avoid overflow before the final right-shift
    scaled = levels_i32.astype(np.int64) * iq
    scaled = scaled << qp_shift

    # Normalise back to coefficient domain
    coeffs = _round_shift(scaled, _INVERSE_SHIFT).astype(np.int32)
    return coeffs


# ---------------------------------------------------------------------------
# Public: informational helpers
# ---------------------------------------------------------------------------

def qp_to_step_size(qp: int) -> float:
    """
    Return the theoretical (floating-point) quantisation step size for a given QP.

    HEVC step-size formula: Qstep = 2^((qp - 4) / 6)
    This is useful for rate-control calculations in rate_control.py and
    for debugging quantisation error magnitude.

    Parameters
    ----------
    qp : int — quantisation parameter [0, 51]

    Returns
    -------
    float — Qstep in units of DCT coefficient amplitude
    """
    _validate_qp(qp)
    return 2.0 ** ((qp - 4) / 6.0)


def estimate_rdcost(
    coeffs: np.ndarray,
    qp: int,
    lambda_: float,
    is_intra: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Quantize and return levels + a rough rate-distortion cost estimate.

    RD cost = distortion + lambda * rate_proxy
        distortion  = sum of squared (coeff - dequant(quant(coeff)))
        rate_proxy  = count of non-zero levels (proxy for bit cost)

    This is intentionally a fast approximation — a proper CABAC-aware
    RD optimisation belongs in mode_decision.py. Use this for quick
    CU-level decisions or unit-testing the Q/IQ roundtrip error.

    Parameters
    ----------
    coeffs  : np.ndarray — DCT coefficient block
    qp      : int        — quantisation parameter
    lambda_ : float      — Lagrange multiplier (from rate_control.py)
    is_intra: bool

    Returns
    -------
    (levels, rd_cost) : tuple[np.ndarray, float]
    """
    levels  = quantize(coeffs, qp, is_intra)
    recon   = dequantize(levels, qp)
    dist    = float(np.sum((coeffs.astype(np.int64) - recon.astype(np.int64)) ** 2))
    rate    = float(np.count_nonzero(levels))
    rd_cost = dist + lambda_ * rate
    return levels, rd_cost


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _round_shift(x: np.ndarray, shift: int) -> np.ndarray:
    """Rounding right-shift: (x + (1 << (shift-1))) >> shift."""
    return (x + (1 << (shift - 1))) >> shift


def _validate_qp(qp: int) -> None:
    if not (0 <= qp <= 51):
        raise ValueError(f"QP must be in [0, 51], got {qp}.")


def _validate_and_cast(block: np.ndarray, size_hint: int | None) -> np.ndarray:
    """Validate shape and cast to int64 for safe intermediate arithmetic.

    size_hint — if provided, only validates it is a legal HEVC transform size.
    The actual block shape is always used for computation; this allows callers
    to pass sub-blocks or synthetic test arrays with an explicit size context.
    """
    if block.ndim != 2:
        raise ValueError(f"Block must be 2-D, got {block.ndim}-D.")
    h, w = block.shape
    if h != w:
        raise ValueError(f"Block must be square, got {h}×{w}.")
    # Validate the hint if given; otherwise validate the actual block size
    n = size_hint if size_hint is not None else h
    if n not in _VALID_SIZES:
        raise ValueError(
            f"Unsupported transform size {n}. Supported: {sorted(_VALID_SIZES)}."
        )
    return block.astype(np.int64)