"""
cabac.py — HEVC CABAC Entropy Coding Engine
Context-Adaptive Binary Arithmetic Coding for the HEVC bitstream.

Pipeline position
-----------------
    mode_decision.py  →  tu_split.py  →  [cabac.py]  →  nal_packager.py
    (CU mode, residual   (TU levels)     (entropy bits)   (NAL units)
     quantised levels)

What CABAC does
----------------
CABAC converts structured syntax elements (flags, indices, levels) into
a compact binary bitstream. It achieves better compression than VLC by
adapting the coding context to local statistics.

CABAC pipeline for each syntax element
----------------------------------------
    1. BINARISE   — convert symbol value to a bin string (sequence of 0/1)
    2. SELECT CTX — choose context model based on element type and neighbours
    3. ENCODE BIN — arithmetic-encode each bin using the context probability
    4. UPDATE CTX — update context model based on the bin just coded

Binary Arithmetic Coding (BAC)
--------------------------------
The arithmetic coder maintains an interval [low, low+range) that is
subdivided for each bin based on the current context probability pState:

    MPS (Most Probable Symbol): the more likely bin value
    LPS (Least Probable Symbol): the less likely bin value

    pLPS = probability of LPS ≈ 0.5 * 0.9^pState   (64 states, spec Table 9-3)
    range_LPS = range * pLPS (from LPS_range_table)

Context models
--------------
Each context stores a 6-bit state (0..63) representing the probability
of the LPS. State 0 = pLPS≈0.5 (maximum uncertainty), state 63 = pLPS≈0.
The state transitions after each coded bin:

    bin == MPS: state → transIdx_MPS[state]   (state increases toward 63)
    bin == LPS: state → transIdx_LPS[state]   (state decreases toward 0)

HEVC spec tables (ISO/IEC 23008-2 Table 9-3):
    - 64 probability states, initial values depend on QP and slice type
    - LPS range table: tabulated for 4 range scale values × 64 states

Key syntax elements coded by CABAC
------------------------------------
    split_cu_flag      — whether this CTU/CU is further split (1 bit)
    skip_flag          — CU is SKIP mode (1 bit)
    pred_mode_flag     — INTRA(1) or INTER(0) (1 bit)
    intra_luma_pred_mode — chosen from 35 modes (binarised as fixed-length)
    cbf_luma/chroma    — Coded Block Flag: TU has non-zero coefficients
    last_sig_coeff_x/y — position of last non-zero coefficient
    sig_coeff_flag     — significant coefficient map
    coeff_abs_level    — absolute coefficient level (0-based)
    coeff_sign_flag    — coefficient sign (bypass coded)
    mv_diff_x/y        — MV delta from predictor (EG0 + bypass)

For the golden model: we implement accurate context selection and state
machine, but output to an in-memory bitstream. We also provide a
bit-counting mode for RDO (computes fractional bit estimates without
writing bits).

Public API
----------
    CABACEncoder        — stateful encoder object
    enc.encode_split_flag(flag, depth)
    enc.encode_skip_flag(flag, ctx_idx)
    enc.encode_pred_mode_flag(is_intra)
    enc.encode_intra_luma_mode(mode, mpm_list)
    enc.encode_cbf(flag, is_luma, tu_depth)
    enc.encode_last_sig_pos(last_x, last_y, log2_tu_size)
    enc.encode_sig_coeff_map(sig_map, log2_tu_size)
    enc.encode_coeff_levels(levels, sig_map, log2_tu_size, is_luma)
    enc.encode_mv_diff(mvd_x, mvd_y)
    enc.get_bits()      -> bytes   — retrieve encoded bitstream
    enc.get_bit_count() -> int     — bits written so far
    enc.reset()                    — clear state for new slice
    estimate_bits(...)             — bit-cost estimator for RDO
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from typing import Literal
import numpy as np

# ---------------------------------------------------------------------------
# HEVC CABAC probability state tables (ISO/IEC 23008-2 §9.3.4.2)
# ---------------------------------------------------------------------------

# LPS range table: lps_range[pState][qRangeIdx]
# qRangeIdx = (range >> 6) & 3 (0..3)
# Values represent the range interval assigned to LPS.
_LPS_RANGE_TABLE: list[list[int]] = [
    [128, 176, 208, 240], [128, 167, 197, 227], [128, 158, 187, 216],
    [123, 150, 178, 205], [116, 142, 169, 195], [111, 135, 160, 185],
    [105, 128, 152, 175], [100, 122, 144, 166], [ 95, 116, 137, 158],
    [ 90, 110, 130, 150], [ 85, 104, 123, 142], [ 81,  99, 117, 135],
    [ 77,  94, 111, 128], [ 73,  89, 105, 122], [ 69,  85, 100, 116],
    [ 66,  80,  95, 110], [ 62,  76,  90, 104], [ 59,  72,  86,  99],
    [ 56,  69,  81,  94], [ 53,  65,  77,  89], [ 51,  62,  73,  85],
    [ 48,  59,  69,  80], [ 46,  56,  66,  76], [ 43,  53,  63,  72],
    [ 41,  50,  59,  69], [ 39,  48,  56,  65], [ 37,  45,  54,  62],
    [ 35,  43,  51,  59], [ 33,  41,  48,  56], [ 32,  39,  46,  53],
    [ 30,  37,  43,  50], [ 29,  35,  41,  48], [ 27,  33,  39,  45],
    [ 26,  31,  37,  43], [ 24,  30,  35,  41], [ 23,  28,  33,  39],
    [ 22,  27,  32,  37], [ 21,  26,  30,  35], [ 20,  24,  29,  33],
    [ 19,  23,  27,  31], [ 18,  22,  26,  30], [ 17,  21,  25,  28],
    [ 16,  20,  23,  27], [ 15,  19,  22,  25], [ 14,  18,  21,  24],
    [ 14,  17,  20,  23], [ 13,  16,  19,  22], [ 12,  15,  18,  21],
    [ 12,  14,  17,  20], [ 11,  14,  16,  19], [ 11,  13,  15,  18],
    [ 10,  12,  15,  17], [ 10,  12,  14,  16], [  9,  11,  13,  15],
    [  9,  11,  12,  14], [  8,  10,  12,  14], [  8,   9,  11,  13],
    [  7,   9,  11,  12], [  7,   9,  10,  12], [  7,   8,  10,  11],
    [  6,   8,   9,  11], [  6,   7,   9,  10], [  6,   7,   8,   9],
    [  2,   2,   2,   2],
]

# MPS transition table: next state when MPS was coded
_TRANS_MPS: list[int] = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
    49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 62, 63,
]

# LPS transition table: next state when LPS was coded
_TRANS_LPS: list[int] = [
     0,  0,  1,  2,  2,  4,  4,  5,  6,  7,  8,  9,  9, 11, 11, 12,
    13, 13, 15, 15, 16, 16, 18, 18, 19, 19, 21, 21, 22, 22, 23, 24,
    24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 30, 30, 31, 32, 32, 33,
    33, 33, 34, 34, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 63,
]

# Initial pState for context models, computed as:
#   state = clip(((init_val >> 3) - 4) * scale + offset, 0, 63)
# We use pre-computed typical initial states for QP=28.
# In a full implementation, these would be derived per-slice.

# Number of context models for each syntax element
_NUM_CTX_SPLIT_FLAG:      int = 3    # indexed by depth 0..2
_NUM_CTX_SKIP_FLAG:       int = 3    # spatial neighbours
_NUM_CTX_PRED_MODE:       int = 1
_NUM_CTX_INTRA_MODE_MPM:  int = 1
_NUM_CTX_CBF_LUMA:        int = 2    # tu_depth 0 and 1+
_NUM_CTX_CBF_CHROMA:      int = 4
_NUM_CTX_LAST_X:          int = 18   # grouped by log2_size
_NUM_CTX_LAST_Y:          int = 18
_NUM_CTX_SIG_COEFF:       int = 42   # grouped by scan position
_NUM_CTX_COEFF_GREATER1:  int = 24
_NUM_CTX_COEFF_GREATER2:  int = 6

# Total context count
_TOTAL_CTX = (
    _NUM_CTX_SPLIT_FLAG + _NUM_CTX_SKIP_FLAG + _NUM_CTX_PRED_MODE +
    _NUM_CTX_INTRA_MODE_MPM + _NUM_CTX_CBF_LUMA + _NUM_CTX_CBF_CHROMA +
    _NUM_CTX_LAST_X + _NUM_CTX_LAST_Y + _NUM_CTX_SIG_COEFF +
    _NUM_CTX_COEFF_GREATER1 + _NUM_CTX_COEFF_GREATER2
)

# Context offset map (start index of each syntax element's contexts)
_CTX_SPLIT_FLAG      = 0
_CTX_SKIP_FLAG       = _CTX_SPLIT_FLAG     + _NUM_CTX_SPLIT_FLAG
_CTX_PRED_MODE       = _CTX_SKIP_FLAG      + _NUM_CTX_SKIP_FLAG
_CTX_INTRA_MPM       = _CTX_PRED_MODE      + _NUM_CTX_PRED_MODE
_CTX_CBF_LUMA        = _CTX_INTRA_MPM      + _NUM_CTX_INTRA_MODE_MPM
_CTX_CBF_CHROMA      = _CTX_CBF_LUMA       + _NUM_CTX_CBF_LUMA
_CTX_LAST_X          = _CTX_CBF_CHROMA     + _NUM_CTX_CBF_CHROMA
_CTX_LAST_Y          = _CTX_LAST_X         + _NUM_CTX_LAST_X
_CTX_SIG_COEFF       = _CTX_LAST_Y         + _NUM_CTX_LAST_Y
_CTX_GREATER1        = _CTX_SIG_COEFF      + _NUM_CTX_SIG_COEFF
_CTX_GREATER2        = _CTX_GREATER1       + _NUM_CTX_COEFF_GREATER1


# ---------------------------------------------------------------------------
# Binarisation schemes
# ---------------------------------------------------------------------------

def _truncated_unary(val: int, c_max: int) -> list[int]:
    """Truncated unary binarisation (TU). val in [0, c_max]."""
    if val == 0:
        return [0]
    bins = [1] * val
    if val < c_max:
        bins.append(0)
    return bins


def _exp_golomb_k0(val: int) -> list[int]:
    """
    Exponential-Golomb order-0 (EG0) binarisation.
    HEVC: m leading 0s + 1 + m-bit suffix.
        val=0 → [1]
        val=1 → [0, 1, 0]
        val=2 → [0, 1, 1]
        val=3 → [0, 0, 1, 0, 0]
    """
    if val == 0:
        return [1]
    m = int(math.floor(math.log2(val + 1)))
    prefix = [0] * m + [1]                                    # m zeros then 1
    suffix_val = val + 1 - (1 << m)
    suffix = [(suffix_val >> (m - 1 - i)) & 1 for i in range(m)]
    return prefix + suffix


def _fixed_length(val: int, num_bits: int) -> list[int]:
    """Fixed-length binarisation, MSB first."""
    return [(val >> (num_bits - 1 - i)) & 1 for i in range(num_bits)]


# ---------------------------------------------------------------------------
# Context model
# ---------------------------------------------------------------------------

@dataclass
class ContextModel:
    """One CABAC context: pState (0..63) and MPS value (0 or 1)."""
    p_state: int = 0   # 0..63, higher = more confident in MPS
    mps:     int = 1   # 0 or 1 — Most Probable Symbol value

    def update(self, bin_val: int) -> None:
        """Update state after encoding one bin."""
        if bin_val == self.mps:
            self.p_state = _TRANS_MPS[self.p_state]
        else:
            if self.p_state == 0:
                self.mps = 1 - self.mps   # flip MPS
            self.p_state = _TRANS_LPS[self.p_state]

    def lps_range(self, range_val: int) -> int:
        """Range allocated to LPS symbol given current range."""
        q_idx = (range_val >> 6) & 3
        return _LPS_RANGE_TABLE[self.p_state][q_idx]

    def bit_cost(self, bin_val: int) -> float:
        """Fractional bit cost of coding bin_val (for RDO estimation)."""
        # Approximate: -log2(p(bin))
        q_idx = 1   # use mid-range for estimation
        lps_r = _LPS_RANGE_TABLE[self.p_state][q_idx]
        p_lps = lps_r / 256.0
        p_mps = 1.0 - p_lps
        if bin_val == self.mps:
            return -math.log2(max(p_mps, 1e-10))
        else:
            return -math.log2(max(p_lps, 1e-10))

    def copy(self) -> ContextModel:
        return ContextModel(self.p_state, self.mps)


# ---------------------------------------------------------------------------
# Bit buffer
# ---------------------------------------------------------------------------

class _BitWriter:
    """Write individual bits to a growing byte buffer."""

    def __init__(self) -> None:
        self._buf:   bytearray = bytearray()
        self._bits:  int = 0     # pending partial byte
        self._count: int = 0     # bits in _bits
        self._total: int = 0     # total bits written

    def write_bit(self, bit: int) -> None:
        self._bits = (self._bits << 1) | (bit & 1)
        self._count += 1
        self._total += 1
        if self._count == 8:
            self._buf.append(self._bits)
            self._bits  = 0
            self._count = 0

    def flush(self) -> None:
        """Flush remaining partial byte, padding with 1+zeros (RBSP)."""
        if self._count > 0:
            self._bits = (self._bits << 1) | 1   # RBSP stop bit
            self._count += 1
            while self._count < 8:
                self._bits <<= 1
                self._count += 1
            self._buf.append(self._bits)
            self._bits  = 0
            self._count = 0

    def get_bytes(self) -> bytes:
        return bytes(self._buf)

    @property
    def total_bits(self) -> int:
        return self._total


# ---------------------------------------------------------------------------
# CABACEncoder
# ---------------------------------------------------------------------------

class CABACEncoder:
    """
    Stateful HEVC CABAC encoder.

    Maintains the arithmetic coding interval, context model array, and a
    bit writer. One encoder instance is used per slice; call reset() at
    each new slice boundary.

    The encoder also tracks statistics for RDO feedback:
        bits_per_syntax[name] — accumulated fractional bit cost per element

    Usage
    -----
        enc = CABACEncoder(qp=28)
        enc.encode_split_flag(True,  depth=0)
        enc.encode_pred_mode_flag(is_intra=True)
        enc.encode_cbf(True, is_luma=True, tu_depth=0)
        enc.encode_coeff_levels(levels, sig_map, log2_size=3, is_luma=True)
        bitstream = enc.get_bits()
        total_bits = enc.get_bit_count()
    """

    def __init__(self, qp: int = 28) -> None:
        self.qp = max(0, min(51, qp))
        self._ctx: list[ContextModel] = self._init_contexts(self.qp)
        self._writer = _BitWriter()
        self._range:  int = 510   # initial range
        self._low:    int = 0     # arithmetic coding low bound
        self._bits_outstanding: int = 0
        self.stats: dict[str, float] = {}   # bits_per_syntax_element

    # ── Slice-level control ──────────────────────────────────────────────

    def reset(self, qp: int | None = None) -> None:
        """Reset encoder state for a new slice."""
        if qp is not None:
            self.qp = max(0, min(51, qp))
        self._ctx     = self._init_contexts(self.qp)
        self._writer  = _BitWriter()
        self._range   = 510
        self._low     = 0
        self._bits_outstanding = 0
        self.stats    = {}

    def flush(self) -> None:
        """Flush remaining bits with RBSP termination."""
        # Encode terminating bit (CABAC flush)
        self._encode_bypass(1)
        self._write_bits_from_low()
        self._writer.flush()

    def get_bits(self) -> bytes:
        """Return the encoded bitstream bytes."""
        return self._writer.get_bytes()

    def get_bit_count(self) -> int:
        """Total bits written so far (includes partial byte)."""
        return self._writer.total_bits

    # ── Syntax element encoding ──────────────────────────────────────────

    def encode_split_flag(self, flag: bool, depth: int = 0) -> None:
        """
        Encode split_cu_flag (§9.3.3.1).
        depth 0,1,2 → different context models.
        """
        ctx_idx = _CTX_SPLIT_FLAG + min(depth, _NUM_CTX_SPLIT_FLAG - 1)
        self._encode_bin(int(flag), ctx_idx, "split_flag")

    def encode_skip_flag(self, flag: bool, neighbour_skip: int = 0) -> None:
        """
        Encode merge_flag / cu_skip_flag.
        neighbour_skip: number of spatial neighbours that are also SKIP (0..2).
        """
        ctx_idx = _CTX_SKIP_FLAG + min(neighbour_skip, _NUM_CTX_SKIP_FLAG - 1)
        self._encode_bin(int(flag), ctx_idx, "skip_flag")

    def encode_pred_mode_flag(self, is_intra: bool) -> None:
        """
        Encode pred_mode_flag: 1 = INTRA, 0 = INTER.
        Single context (§9.3.3.2).
        """
        self._encode_bin(int(is_intra), _CTX_PRED_MODE, "pred_mode_flag")

    def encode_intra_luma_mode(self, mode: int, mpm_list: list[int]) -> None:
        """
        Encode intra_luma_pred_mode (§9.3.3.5).

        If mode is in the MPM list (Most Probable Mode), encode:
            prev_intra_luma_pred_flag = 1  (1 bit)
            mpm_idx in [0..2]             (TU, 2 bits)
        Else:
            prev_intra_luma_pred_flag = 0  (1 bit)
            remaining_mode_idx            (5 bits fixed-length)
        """
        if mode in mpm_list:
            mpm_idx = mpm_list.index(mode)
            self._encode_bin(1, _CTX_INTRA_MPM, "prev_intra_luma_pred_flag")
            bins = _truncated_unary(mpm_idx, 2)
            for b in bins:
                self._encode_bypass(b, "mpm_idx")
        else:
            self._encode_bin(0, _CTX_INTRA_MPM, "prev_intra_luma_pred_flag")
            # Compute remaining mode index (modes with higher index adjusted)
            sorted_mpm = sorted(mpm_list)
            rem = mode
            for m in sorted_mpm:
                if m <= rem:
                    rem -= 1
            for b in _fixed_length(rem, 5):
                self._encode_bypass(b, "rem_intra_luma_mode")

    def encode_cbf(self, flag: bool, is_luma: bool, tu_depth: int = 0) -> None:
        """
        Encode coded_block_flag (CBF) — 1 if TU has non-zero coefficients.

        Luma contexts: tu_depth=0 uses ctx 0, depth≥1 uses ctx 1.
        Chroma contexts: 4 contexts indexed by tu_depth.
        """
        if is_luma:
            ctx_idx = _CTX_CBF_LUMA + (0 if tu_depth == 0 else 1)
        else:
            ctx_idx = _CTX_CBF_CHROMA + min(tu_depth, _NUM_CTX_CBF_CHROMA - 1)
        self._encode_bin(int(flag), ctx_idx, "cbf")

    def encode_last_sig_pos(
        self,
        last_x: int,
        last_y: int,
        log2_tu_size: int,
    ) -> None:
        """
        Encode last_sig_coeff_x_prefix/suffix and _y_prefix/suffix.

        The last non-zero coefficient position (in the scan order) is
        signalled first, then the significant coefficient map is coded
        backwards from that position.

        Prefix: truncated unary, context-coded.
        Suffix: fixed-length, bypass-coded (for large TUs).
        """
        tu_size  = 1 << log2_tu_size
        ctx_base = min(log2_tu_size - 2, 3) * 3  # 3 contexts per size group

        for coord, label in ((last_x, "last_x"), (last_y, "last_y")):
            ctx_off  = _CTX_LAST_X if label == "last_x" else _CTX_LAST_Y
            prefix   = _last_coord_prefix(coord)
            # Encode prefix (TU-binarised, context-coded)
            max_pref = log2_tu_size * 2 - 1
            for k in range(prefix):
                self._encode_bin(1, ctx_off + ctx_base + min(k, 2), label + "_prefix")
            if prefix < max_pref:
                self._encode_bin(0, ctx_off + ctx_base + min(prefix, 2), label + "_prefix")
            # Encode suffix (bypass, for prefix ≥ 3)
            suffix_bits = (prefix - 1) - 1
            if suffix_bits > 0:
                suffix_val = coord - _prefix_to_min_coord(prefix)
                for b in _fixed_length(suffix_val, suffix_bits):
                    self._encode_bypass(b, label + "_suffix")

    def encode_sig_coeff_map(
        self,
        sig_map:     np.ndarray,
        log2_tu_size: int,
    ) -> None:
        """
        Encode significant_coeff_flag for each scan position.

        Coefficients are scanned in diagonal (up-right) order, and
        sig_coeff_flag signals whether each position has a non-zero coeff.
        Coded backwards from last significant position to DC.

        sig_map : (N, N) bool/int array, 1 = non-zero coefficient
        """
        n     = 1 << log2_tu_size
        scan  = _diagonal_scan(n)
        # Find last significant position
        last  = 0
        for i, (r, c) in enumerate(scan):
            if sig_map[r, c]:
                last = i

        for i in range(last, -1, -1):
            r, c  = scan[i]
            ctx   = _sig_coeff_ctx(i, log2_tu_size)
            flag  = int(bool(sig_map[r, c]))
            self._encode_bin(flag, _CTX_SIG_COEFF + ctx, "sig_coeff_flag")

    def encode_coeff_levels(
        self,
        levels:       np.ndarray,
        sig_map:      np.ndarray,
        log2_tu_size: int,
        is_luma:      bool = True,
    ) -> None:
        """
        Encode coeff_abs_level_greater1_flag, greater2_flag, and level_prefix.

        For each non-zero coefficient (in scan order, starting from last):
            1. greater1_flag: is |level| > 1?  (context coded)
            2. greater2_flag: is |level| > 2?  (if greater1, context coded)
            3. level_remainder: |level| - 3 coded as EG0 (if > 2, bypass)
            4. sign_flag: sign of level (bypass)
        """
        n     = 1 << log2_tu_size
        scan  = _diagonal_scan(n)
        # Collect non-zero coefficients from last to DC
        last  = 0
        for i, (r, c) in enumerate(scan):
            if sig_map[r, c]:
                last = i
        nzc   = [(i, scan[i]) for i in range(last, -1, -1) if sig_map[scan[i][0], scan[i][1]]]

        c1_ctx_set = 0   # context set for greater1
        greater1_count = 0

        for k, (i, (r, c)) in enumerate(nzc):
            lvl   = int(abs(levels[r, c]))
            g1_ctx = _CTX_GREATER1 + min(c1_ctx_set * 4 + min(greater1_count, 3), _NUM_CTX_COEFF_GREATER1 - 1)
            g1    = int(lvl > 1)
            self._encode_bin(g1, g1_ctx, "coeff_greater1")

            if g1:
                greater1_count += 1
                g2_ctx = _CTX_GREATER2 + min(c1_ctx_set, _NUM_CTX_COEFF_GREATER2 - 1)
                g2 = int(lvl > 2)
                self._encode_bin(g2, g2_ctx, "coeff_greater2")
                if g2:
                    # EG0 for level_prefix = |level| - 3
                    for b in _exp_golomb_k0(lvl - 3):
                        self._encode_bypass(b, "coeff_level_prefix")
            else:
                greater1_count = 0

        # Sign flags (bypass, all at once)
        for _, (r, c) in nzc:
            sign = int(levels[r, c] < 0)
            self._encode_bypass(sign, "coeff_sign")

    def encode_mv_diff(self, mvd_x: int, mvd_y: int) -> None:
        """
        Encode MVD (Motion Vector Difference) components.

        Each component:
            abs_mvd_greater0_flag  — is |mvd| > 0? (context coded)
            abs_mvd_greater1_flag  — is |mvd| > 1? (context coded)
            abs_mvd_minus2         — |mvd| - 2 coded as EG0 (bypass)
            mvd_sign_flag          — sign (bypass)
        """
        for mvd, label in ((mvd_x, "mvd_x"), (mvd_y, "mvd_y")):
            abs_mvd = abs(mvd)
            g0 = int(abs_mvd > 0)
            self._encode_bypass(g0, label + "_g0")
            if g0:
                g1 = int(abs_mvd > 1)
                self._encode_bypass(g1, label + "_g1")
                if g1:
                    for b in _exp_golomb_k0(abs_mvd - 2):
                        self._encode_bypass(b, label + "_suffix")
                sign = int(mvd < 0)
                self._encode_bypass(sign, label + "_sign")

    # ── Bit cost estimation (for RDO) ─────────────────────────────────────

    def estimate_split_flag_bits(self, flag: bool, depth: int = 0) -> float:
        ctx = self._ctx[_CTX_SPLIT_FLAG + min(depth, 2)]
        return ctx.bit_cost(int(flag))

    def estimate_skip_flag_bits(self, flag: bool) -> float:
        ctx = self._ctx[_CTX_SKIP_FLAG]
        return ctx.bit_cost(int(flag))

    def estimate_cbf_bits(self, flag: bool, is_luma: bool, tu_depth: int = 0) -> float:
        if is_luma:
            ctx_idx = _CTX_CBF_LUMA + (0 if tu_depth == 0 else 1)
        else:
            ctx_idx = _CTX_CBF_CHROMA + min(tu_depth, 3)
        return self._ctx[ctx_idx].bit_cost(int(flag))

    def estimate_coeff_bits(self, levels: np.ndarray) -> float:
        """Fast approximate bit cost for a TU's coefficient block."""
        n2 = levels.size
        nnz = int(np.count_nonzero(levels))
        if nnz == 0:
            return 0.0
        log2n = max(1, int(math.log2(math.sqrt(n2))))
        # sig_map bits: 1 bit each
        sig_bits = n2 * 0.5   # average ~0.5 bits per position
        # level bits: greater1/greater2 flags + EG0 for large levels
        abs_lvls = np.abs(levels[levels != 0]).ravel()
        lvl_bits = sum(
            1.0 + (1.0 if l > 1 else 0.0) + (math.log2(l - 1) * 2 if l > 2 else 0.0)
            for l in abs_lvls
        )
        # last position: log2(n) * 2 bits approximately
        last_bits = float(log2n * 2)
        return sig_bits + lvl_bits + last_bits

    # ── Internal: arithmetic coding engine ───────────────────────────────

    def _encode_bin(self, bin_val: int, ctx_idx: int, stat_key: str = "") -> None:
        """Encode one context-coded bin."""
        ctx   = self._ctx[ctx_idx]
        range_ = self._range
        q_idx  = (range_ >> 6) & 3
        lps_r  = _LPS_RANGE_TABLE[ctx.p_state][q_idx]

        # Track fractional bit cost for statistics
        if stat_key:
            bits = ctx.bit_cost(bin_val)
            self.stats[stat_key] = self.stats.get(stat_key, 0.0) + bits

        if bin_val != ctx.mps:
            # LPS branch
            self._low   += range_ - lps_r
            self._range  = lps_r
        else:
            # MPS branch
            self._range  = range_ - lps_r

        ctx.update(bin_val)
        self._renorm()

    def _encode_bypass(self, bin_val: int, stat_key: str = "") -> None:
        """Encode one bypass bin (probability = 0.5)."""
        if stat_key:
            self.stats[stat_key] = self.stats.get(stat_key, 0.0) + 1.0
        self._low  = (self._low << 1)
        if bin_val:
            self._low += self._range
        if self._low >= (1 << 17):
            self._writer.write_bit(1)
            self._low -= (1 << 17)
            for _ in range(self._bits_outstanding):
                self._writer.write_bit(0)
            self._bits_outstanding = 0
        elif self._low < (1 << 16):
            self._writer.write_bit(0)
            self._low = self._low
            for _ in range(self._bits_outstanding):
                self._writer.write_bit(1)
            self._bits_outstanding = 0
        else:
            self._bits_outstanding += 1
            self._low -= (1 << 16)
    def _renorm(self) -> None:
        """Renormalise the coding interval."""
        while self._range < 256:
            if self._low >= (1 << 16):
                self._writer.write_bit(1)
                self._low -= (1 << 16)
                for _ in range(self._bits_outstanding):
                    self._writer.write_bit(0)
                self._bits_outstanding = 0
            elif self._low + self._range <= (1 << 16):
                self._writer.write_bit(0)
                for _ in range(self._bits_outstanding):
                    self._writer.write_bit(1)
                self._bits_outstanding = 0
            else:
                self._bits_outstanding += 1
                self._low -= (1 << 15)
            self._range <<= 1
            self._low   <<= 1

    def _write_bits_from_low(self) -> None:
        """Output remaining bits from low register at flush."""
        low = self._low + self._range
        for bit in [int((low >> 16) & 1), int((low >> 15) & 1)]:
            if bit:
                self._writer.write_bit(1)
                for _ in range(self._bits_outstanding):
                    self._writer.write_bit(0)
                self._bits_outstanding = 0
            else:
                self._writer.write_bit(0)
                for _ in range(self._bits_outstanding):
                    self._writer.write_bit(1)
                self._bits_outstanding = 0

    @staticmethod
    def _init_contexts(qp: int) -> list[ContextModel]:
        """
        Initialise all context models for a given QP.

        HEVC uses init_value tables per slice type (§9.3.2.2).
        For the golden model we use simplified initial states derived
        from QP: state ≈ clip((qp - 26), 0, 63).
        In a full implementation this would use the initValue tables
        from the spec for each context.
        """
        base = max(0, min(63, (qp - 12) // 2))
        ctxs = [ContextModel(base, 1) for _ in range(_TOTAL_CTX)]
        # Override some contexts with better initial values
        for i in range(_NUM_CTX_SPLIT_FLAG):
            ctxs[_CTX_SPLIT_FLAG + i] = ContextModel(20, 1)   # split likely early
        for i in range(_NUM_CTX_CBF_LUMA):
            ctxs[_CTX_CBF_LUMA + i] = ContextModel(33, 1)     # CBF often 1
        return ctxs


# ---------------------------------------------------------------------------
# Standalone bit-cost estimator (for RDO without full encoder state)
# ---------------------------------------------------------------------------

def estimate_bits(
    levels:       np.ndarray,
    log2_tu_size: int,
    qp:           int = 28,
    is_luma:      bool = True,
) -> float:
    """
    Estimate the CABAC bit cost for one TU's coefficient block.

    Returns a fractional bit count suitable for Lagrangian RDO.
    This is a fast approximation — it does not require a full encoder
    instance.

    Parameters
    ----------
    levels       : (N, N) int32 — quantised coefficient levels
    log2_tu_size : int          — log2 of TU size (2=4×4, 3=8×8, 4=16×16, 5=32×32)
    qp           : int          — quantisation parameter (affects context init)
    is_luma      : bool

    Returns
    -------
    float — estimated bit cost (may be fractional)
    """
    enc = CABACEncoder(qp=qp)
    return enc.estimate_coeff_bits(levels)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _diagonal_scan(n: int) -> list[tuple[int, int]]:
    """
    Return (row, col) positions in HEVC diagonal scan order.
    Scans diagonals from top-right to bottom-left, within each 4×4 subblock.
    For simplicity (golden model): returns raster order for 4×4,
    and diagonal scan for larger sizes.
    """
    positions = []
    if n <= 4:
        # Simple diagonal scan for 4×4
        for d in range(2 * n - 1):
            for r in range(max(0, d - n + 1), min(d + 1, n)):
                c = d - r
                if 0 <= c < n:
                    positions.append((r, c))
    else:
        # Scan 4×4 subblocks in diagonal order, each subblock scanned diagonally
        num_sub = n // 4
        sub_scan = _diagonal_scan(num_sub)
        coeff_scan = _diagonal_scan(4)
        for (sr, sc) in sub_scan:
            for (cr, cc) in coeff_scan:
                positions.append((sr * 4 + cr, sc * 4 + cc))
    return positions


def _sig_coeff_ctx(scan_idx: int, log2_size: int) -> int:
    """Context index for sig_coeff_flag at a given scan position."""
    # Simplified: use scan position group (every 16 coefficients)
    group = scan_idx >> 4
    return min(group * 3, _NUM_CTX_SIG_COEFF - 1)


def _last_coord_prefix(coord: int) -> int:
    """Compute prefix for last_sig_coeff coordinate binarisation."""
    if coord <= 3:
        return coord
    length = coord.bit_length() - 1  # Tương đương floor(log2(coord))
    return (length << 1) + ((coord >> (length - 1)) & 1)


def _prefix_to_min_coord(prefix: int) -> int:
    """Minimum coordinate value for a given prefix."""
    if prefix <= 3:
        return prefix
    return (1 << ((prefix >> 1) - 1)) * (2 + (prefix & 1))