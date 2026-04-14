"""
deblocking.py — HEVC Deblocking Filter
Removes blocking artefacts at CU and TU grid boundaries.

Pipeline position
-----------------
    tu_split.py  →  decoded_picture_buffer.py  →  [deblocking.py]  →  sao.py
    (recon frame       (store frame)                (smooth edges)      (sample
     assembled)                                                          offset)

What the deblocking filter does
---------------------------------
After encoding, reconstructed frames contain visible block boundaries because:
    - Each CU/TU is predicted and transformed independently
    - Quantisation introduces discontinuities at block edges

The deblocking filter smooths these edges in-place on the reconstructed luma
and chroma planes. It is applied AFTER reconstruction but BEFORE SAO and
before the frame is stored in the DPB as a reference.

HEVC deblocking filter (ISO/IEC 23008-2 §8.7)
-----------------------------------------------
Two-pass: horizontal edges first, then vertical edges.

For each 8×8 grid boundary:

    Step 1 — Boundary Strength (BS) decision (§8.7.2):
        BS=0: no filtering
        BS=1: weak filter (coded block flag difference, MV difference)
        BS=2: strong filter (intra coding on either side)

    Step 2 — Filter decision (§8.7.3):
        Compute β (beta) threshold from average QP of the two blocks.
        Compute tC threshold from QP and BS.
        If edge samples differ by less than β/2: skip (edge is already smooth)

    Step 3 — Apply filter:
        BS=2 and long edge: strong 7-tap / 3-tap luma filter
        BS=1:               weak 2-tap luma filter + chroma 2-tap filter

β and tC tables (HEVC spec Table 8-10)
----------------------------------------
    beta_table[qp]:  indexed by QP 0..51, values 0..88
    tc_table[qp]:    indexed by (QP + BS_offset) 0..53, values 0..56

Coordinate convention
-----------------------
    All operations are in luma pixel coordinates.
    Chroma is filtered at half the luma grid spacing (4:2:0).

Public API
----------
    deblock_frame(luma, chroma_cb, chroma_cr,
                  cu_grid, qp_map, slice_type)  -> None  (in-place)
    deblock_luma_edge(luma, x, y, is_vertical,
                      bs, qp_p, qp_q)           -> None  (in-place)
    deblock_chroma_edge(chroma, x, y, is_vertical,
                        bs, qp_p, qp_q)          -> None  (in-place)
    compute_boundary_strength(cu_grid, x, y,
                              is_vertical)        -> int   (0, 1, or 2)
    compute_beta(qp)                              -> int
    compute_tc(qp, bs)                            -> int
"""

from __future__ import annotations

import numpy as np
from typing import Literal

# ---------------------------------------------------------------------------
# HEVC β (beta) table — indexed by QP [0..51]
# ISO/IEC 23008-2 Table 8-10
# ---------------------------------------------------------------------------
_BETA_TABLE: list[int] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   # QP  0– 9
     0,  0,  0,  0,  0,  0,  6,  7,  8,  9,   # QP 10–19
    10, 11, 12, 13, 14, 15, 16, 17, 18, 20,   # QP 20–29
    22, 24, 26, 28, 30, 32, 34, 36, 38, 40,   # QP 30–39
    42, 44, 46, 48, 50, 52, 54, 56, 58, 60,   # QP 40–49
    62, 64,                                     # QP 50–51
]

# ---------------------------------------------------------------------------
# HEVC tC table — indexed by clipped_qp [0..53]
# ISO/IEC 23008-2 Table 8-10
# ---------------------------------------------------------------------------
_TC_TABLE: list[int] = [
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   # idx  0– 9
     0,  0,  0,  0,  0,  0,  0,  0,  1,  1,   # idx 10–19
     1,  1,  1,  1,  1,  1,  1,  2,  2,  2,   # idx 20–29
     2,  3,  3,  3,  3,  4,  4,  4,  6,  6,   # idx 30–39
     7,  8,  9, 10, 13, 14, 16, 18, 22, 24,   # idx 40–49
    27, 30, 34, 56,                             # idx 50–53
]

# Default β offset and tC offset (slice-level parameters, typically 0)
DEFAULT_BETA_OFFSET: int = 0
DEFAULT_TC_OFFSET:   int = 0

# Deblocking grid step: filter every 8 luma pixels
FILTER_STEP: int = 8

# Number of samples on each side of the edge used for filtering
LUMA_TAPS:   int = 4   # p0..p3 and q0..q3
CHROMA_TAPS: int = 2   # p0..p1 and q0..q1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deblock_frame(
    luma:      np.ndarray,
    chroma_cb: np.ndarray | None,
    chroma_cr: np.ndarray | None,
    qp:        int = 28,
    qp_map:    np.ndarray | None = None,
    beta_offset: int = DEFAULT_BETA_OFFSET,
    tc_offset:   int = DEFAULT_TC_OFFSET,
) -> None:
    """
    Apply the HEVC deblocking filter to a reconstructed frame (in-place).

    Processes all 8-pixel-aligned luma edges and 4-pixel-aligned chroma
    edges in two passes: horizontal edges first, then vertical.

    Parameters
    ----------
    luma      : np.ndarray — luma plane, shape (H, W), dtype uint8.
                Modified in-place.
    chroma_cb : np.ndarray or None — Cb plane, shape (H/2, W/2), uint8.
    chroma_cr : np.ndarray or None — Cr plane, shape (H/2, W/2), uint8.
    qp        : int — frame-level QP used when qp_map is None.
    qp_map    : np.ndarray or None — per-CU QP map, shape (H//8, W//8),
                int. Each entry is the QP for the corresponding 8×8 block.
                If None, `qp` is used uniformly.
    beta_offset : int — slice-level β offset (default 0).
    tc_offset   : int — slice-level tC offset (default 0).

    Notes
    -----
    The filter is applied in-place. The order is:
        1. Filter all horizontal edges (row boundaries)
        2. Filter all vertical edges   (column boundaries)
    """
    H, W = luma.shape

    def _qp_at(by: int, bx: int) -> int:
        """QP for the 8×8 block at grid position (bx, by)."""
        if qp_map is not None:
            by_c = min(by, qp_map.shape[0] - 1)
            bx_c = min(bx, qp_map.shape[1] - 1)
            return int(qp_map[by_c, bx_c])
        return qp

    # ── Pass 1: Horizontal edges (filter samples along y-boundaries) ────
    for row in range(FILTER_STEP, H, FILTER_STEP):
        for col in range(0, W, FILTER_STEP):
            bx_q = col // FILTER_STEP
            by_q = row // FILTER_STEP
            by_p = by_q - 1
            qp_p = _qp_at(by_p, bx_q)
            qp_q = _qp_at(by_q, bx_q)
            bs   = _default_bs(qp_p, qp_q)
            if bs > 0:
                deblock_luma_edge(
                    luma, col, row, is_vertical=False,
                    bs=bs, qp_p=qp_p, qp_q=qp_q,
                    beta_offset=beta_offset, tc_offset=tc_offset,
                )
                if chroma_cb is not None:
                    deblock_chroma_edge(
                        chroma_cb, col // 2, row // 2, is_vertical=False,
                        bs=bs, qp_p=qp_p, qp_q=qp_q,
                        tc_offset=tc_offset,
                    )
                if chroma_cr is not None:
                    deblock_chroma_edge(
                        chroma_cr, col // 2, row // 2, is_vertical=False,
                        bs=bs, qp_p=qp_p, qp_q=qp_q,
                        tc_offset=tc_offset,
                    )

    # ── Pass 2: Vertical edges (filter samples along x-boundaries) ──────
    for col in range(FILTER_STEP, W, FILTER_STEP):
        for row in range(0, H, FILTER_STEP):
            bx_q = col // FILTER_STEP
            bx_p = bx_q - 1
            by   = row // FILTER_STEP
            qp_p = _qp_at(by, bx_p)
            qp_q = _qp_at(by, bx_q)
            bs   = _default_bs(qp_p, qp_q)
            if bs > 0:
                deblock_luma_edge(
                    luma, col, row, is_vertical=True,
                    bs=bs, qp_p=qp_p, qp_q=qp_q,
                    beta_offset=beta_offset, tc_offset=tc_offset,
                )
                if chroma_cb is not None:
                    deblock_chroma_edge(
                        chroma_cb, col // 2, row // 2, is_vertical=True,
                        bs=bs, qp_p=qp_p, qp_q=qp_q,
                        tc_offset=tc_offset,
                    )
                if chroma_cr is not None:
                    deblock_chroma_edge(
                        chroma_cr, col // 2, row // 2, is_vertical=True,
                        bs=bs, qp_p=qp_p, qp_q=qp_q,
                        tc_offset=tc_offset,
                    )


def deblock_luma_edge(
    luma:        np.ndarray,
    x:           int,
    y:           int,
    is_vertical: bool,
    bs:          int,
    qp_p:        int,
    qp_q:        int,
    beta_offset: int = DEFAULT_BETA_OFFSET,
    tc_offset:   int = DEFAULT_TC_OFFSET,
) -> None:
    """
    Apply the HEVC luma deblocking filter to one edge (in-place).

    The edge is a horizontal or vertical line of 8 pixels.
    For each pair of adjacent sample columns (vertical edge) or rows
    (horizontal edge), the filter is applied if the boundary strength
    and threshold conditions are met.

    Parameters
    ----------
    luma        : np.ndarray — full luma frame, modified in-place
    x, y        : int        — top-left of the 8-pixel edge segment
    is_vertical : bool       — True = filter column boundary at x
                               False = filter row boundary at y
    bs          : int        — boundary strength (1 or 2)
    qp_p, qp_q  : int        — QP of the block on each side of the edge
    beta_offset, tc_offset : int — slice-level offsets
    """
    if bs == 0:
        return

    qp_avg = (qp_p + qp_q + 1) >> 1
    beta   = compute_beta(qp_avg, beta_offset)
    tc     = compute_tc(qp_avg, bs, tc_offset)

    H, W = luma.shape

    for i in range(FILTER_STEP):
        if is_vertical:
            # Filtering column boundary at x: samples are along the column
            r = y + i
            if r < 0 or r >= H:
                continue
            # Extract p3..p0 | q0..q3
            p = [int(luma[r, x - k - 1]) for k in range(4) if x - k - 1 >= 0]
            q = [int(luma[r, x + k    ]) for k in range(4) if x + k     < W]
        else:
            # Filtering row boundary at y: samples are along the row
            c = x + i
            if c < 0 or c >= W:
                continue
            p = [int(luma[y - k - 1, c]) for k in range(4) if y - k - 1 >= 0]
            q = [int(luma[y + k,     c]) for k in range(4) if y + k     < H]

        if len(p) < 2 or len(q) < 2:
            continue

        # ── Threshold check: |p0 - q0| < β/2 to decide filter strength ──
        dp = abs(p[0] - p[min(2, len(p)-1)])   # |p0 - p2|
        dq = abs(q[0] - q[min(2, len(q)-1)])   # |q0 - q2|

        # Strong filter condition (BS=2, long-edge): additional checks
        use_strong = False
        if bs == 2 and len(p) >= 4 and len(q) >= 4:
            d  = dp + dq
            # Strong filter: samples close to boundary
            strong_p = abs(p[0] - p[3]) + abs(p[0] - q[0]) < (beta >> 3)
            strong_q = abs(q[0] - q[3]) + abs(p[0] - q[0]) < (beta >> 3)
            edge_ok  = abs(p[0] - q[0]) < (5 * tc + 1) >> 1
            use_strong = (d < (beta >> 2)) and strong_p and strong_q and edge_ok

        if use_strong:
            # Strong 7-tap filter (HEVC §8.7.2.4)
            p0_f = (_clip3(0, 255, (p[2] + 2*p[1] + 2*p[0] + 2*q[0] + q[1] + 4) >> 3))
            p1_f = (_clip3(0, 255, (p[2] + p[1] + p[0] + q[0] + 2) >> 2))
            p2_f = (_clip3(0, 255, (2*p[3] + 3*p[2] + p[1] + p[0] + q[0] + 4) >> 3))
            q0_f = (_clip3(0, 255, (p[1] + 2*p[0] + 2*q[0] + 2*q[1] + q[2] + 4) >> 3))
            q1_f = (_clip3(0, 255, (p[0] + q[0] + q[1] + q[2] + 2) >> 2))
            q2_f = (_clip3(0, 255, (p[0] + q[0] + 2*q[1] + 3*q[2] + 2*q[3] + 4) >> 3))

            if is_vertical:
                luma[r, x-1], luma[r, x-2], luma[r, x-3] = p0_f, p1_f, p2_f
                luma[r, x  ], luma[r, x+1], luma[r, x+2] = q0_f, q1_f, q2_f
            else:
                luma[y-1, c], luma[y-2, c], luma[y-3, c] = p0_f, p1_f, p2_f
                luma[y,   c], luma[y+1, c], luma[y+2, c] = q0_f, q1_f, q2_f

        else:
            # Weak filter (BS=1 or BS=2 without strong conditions)
            delta = _clip3(-tc, tc, ((q[0] - p[0]) * 4 + (p[1] - q[1]) + 4) >> 3)
            p0_f  = _clip3(0, 255, p[0] + delta)
            q0_f  = _clip3(0, 255, q[0] - delta)

            # Optional p1/q1 update (luma only when dp/dq < β/4)
            tc_div2 = (tc + 1) >> 1
            if dp < (beta >> 4) and len(p) >= 2:
                dp1 = _clip3(-tc_div2, tc_div2, (p[2] + p[0] - 2*p[1]) >> 1)
                p1_f = p[1] + dp1
            else:
                p1_f = p[1]
            if dq < (beta >> 4) and len(q) >= 2:
                dq1 = _clip3(-tc_div2, tc_div2, (q[2] + q[0] - 2*q[1]) >> 1)
                q1_f = q[1] + dq1
            else:
                q1_f = q[1]

            if is_vertical:
                luma[r, x-1] = p0_f; luma[r, x-2] = p1_f
                luma[r, x  ] = q0_f; luma[r, x+1] = q1_f
            else:
                luma[y-1, c] = p0_f; luma[y-2, c] = p1_f
                luma[y,   c] = q0_f; luma[y+1, c] = q1_f


def deblock_chroma_edge(
    chroma:      np.ndarray,
    x:           int,
    y:           int,
    is_vertical: bool,
    bs:          int,
    qp_p:        int,
    qp_q:        int,
    tc_offset:   int = DEFAULT_TC_OFFSET,
) -> None:
    """
    Apply the HEVC chroma deblocking filter to one edge (in-place).

    Chroma uses a simpler 2-tap filter than luma. No strong filter
    for chroma. Applied at every 4-chroma-sample boundary (= every
    8-luma-sample boundary in 4:2:0).

    Parameters
    ----------
    chroma      : np.ndarray — Cb or Cr plane, modified in-place
    x, y        : int        — chroma-coordinate top-left of edge segment
    is_vertical : bool
    bs          : int        — boundary strength (1 or 2)
    qp_p, qp_q  : int        — luma QP of blocks on each side
    tc_offset   : int
    """
    if bs == 0:
        return

    # Chroma QP = luma QP (simplified — proper mapping uses QP table §8.7.1)
    qp_avg  = (qp_p + qp_q + 1) >> 1
    tc      = compute_tc(qp_avg, bs, tc_offset)
    H, W    = chroma.shape

    # Chroma: 4 samples per edge (half of luma 8)
    for i in range(FILTER_STEP // 2):
        if is_vertical:
            r = y + i
            if r < 0 or r >= H or x < 1 or x >= W:
                continue
            p0 = int(chroma[r, x-1])
            q0 = int(chroma[r, x  ])
            p1 = int(chroma[r, max(0, x-2)])
            q1 = int(chroma[r, min(W-1, x+1)])
        else:
            c = x + i
            if c < 0 or c >= W or y < 1 or y >= H:
                continue
            p0 = int(chroma[y-1, c])
            q0 = int(chroma[y,   c])
            p1 = int(chroma[max(0, y-2), c])
            q1 = int(chroma[min(H-1, y+1), c])

        delta = _clip3(-tc, tc, ((q0 - p0) * 4 + (p1 - q1) + 4) >> 3)
        p0_f  = _clip3(0, 255, p0 + delta)
        q0_f  = _clip3(0, 255, q0 - delta)

        if is_vertical:
            chroma[r, x-1] = p0_f
            chroma[r, x  ] = q0_f
        else:
            chroma[y-1, c] = p0_f
            chroma[y,   c] = q0_f


def compute_boundary_strength(
    qp_p:    int,
    qp_q:    int,
    intra_p: bool = False,
    intra_q: bool = False,
    cbf_p:   bool = True,
    cbf_q:   bool = True,
) -> int:
    """
    Compute the Boundary Strength (BS) for one edge (HEVC §8.7.2).

    BS=2: intra prediction on either side
    BS=1: inter-coded with residual, or MV difference > 1 pel (approximated)
    BS=0: inter-coded, no residual, similar MVs — no filtering needed

    Parameters
    ----------
    qp_p, qp_q   : int  — QP of the two blocks (unused in BS logic, kept for future)
    intra_p      : bool — True if the P-side block is intra coded
    intra_q      : bool — True if the Q-side block is intra coded
    cbf_p        : bool — Coded Block Flag: True if P-side has non-zero residual
    cbf_q        : bool — Coded Block Flag: True if Q-side has non-zero residual

    Returns
    -------
    int — 0, 1, or 2
    """
    if intra_p or intra_q:
        return 2
    if cbf_p or cbf_q:
        return 1
    return 0


def compute_beta(qp: int, beta_offset: int = 0) -> int:
    """
    Compute the β threshold for deblocking (HEVC §8.7.2.3).

    β is looked up from the spec table and shifted by the slice-level offset.
    Controls the maximum edge discontinuity that gets filtered.

    Parameters
    ----------
    qp          : int — average QP across the edge
    beta_offset : int — slice-level β offset (default 0)

    Returns
    -------
    int — β threshold ≥ 0
    """
    qp_c = max(0, min(51, qp + beta_offset))
    return _BETA_TABLE[qp_c]


def compute_tc(qp: int, bs: int, tc_offset: int = 0) -> int:
    """
    Compute the tC threshold for deblocking (HEVC §8.7.2.3).

    tC limits the magnitude of filter adjustments (clipping parameter).

    Parameters
    ----------
    qp        : int — average QP across the edge
    bs        : int — boundary strength (1 or 2)
    tc_offset : int — slice-level tC offset (default 0)

    Returns
    -------
    int — tC threshold ≥ 0
    """
    # For BS=2, use QP+2 in the table lookup (spec §8.7.2.3)
    bs_offset = 2 if bs == 2 else 0
    idx = max(0, min(53, qp + bs_offset + tc_offset))
    return _TC_TABLE[idx]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clip3(lo: int, hi: int, val: int) -> int:
    """Clip val to [lo, hi]."""
    return max(lo, min(hi, val))


def _default_bs(qp_p: int, qp_q: int) -> int:
    """
    Default BS for a golden model (no per-CU metadata available).

    Without a proper CU info map, we use a simplified policy:
        - If QP difference between blocks is large: BS=2
        - Otherwise: BS=1 (filter all edges conservatively)

    A production encoder would use actual intra flags and CBF values.
    """
    if abs(qp_p - qp_q) >= 5:
        return 2
    return 1