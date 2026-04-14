"""
sao.py — HEVC Sample Adaptive Offset (SAO) Filter
Reduces quantisation distortion using per-CTU statistics-driven offsets.

Pipeline position
-----------------
    deblocking.py  →  [sao.py]  →  decoded_picture_buffer.py
    (smooth edges)    (offset       (store as reference)
                       samples)

What SAO does
--------------
After deblocking, residual distortion remains due to quantisation rounding.
SAO reduces this distortion by adding a small signed offset to pixels that
fall into specific categories. It is the last in-loop filter before a frame
is stored as a reference.

Unlike deblocking (which targets edge artefacts), SAO targets the ENTIRE
reconstructed signal: both smooth regions and edge regions.

HEVC SAO types (ISO/IEC 23008-2 §8.7.3)
------------------------------------------
Two mutually exclusive SAO types per CTU per colour component:

    Type 0 — BO (Band Offset):
        Divides the 8-bit sample value range [0..255] into 32 bands of 8
        values each. The 4 most-active consecutive bands get an additive
        offset. Good for smooth regions where pixel values cluster.

        Example: band_start=12 (values 96..127 active), offsets=[-2,+1,+3,-1]

    Type 1 — EO (Edge Offset):
        Classifies each sample by comparing it to 2 neighbours in one of
        four directions (0°, 90°, 135°, 45°). The sample gets an offset
        based on whether it is a local minimum, maximum, concave, or convex.

        4 EO direction modes:
            EO_0:   horizontal  → compare with left and right neighbours
            EO_1:   vertical    → compare with above and below
            EO_2:   135°        → compare with top-left and bottom-right
            EO_3:   45°         → compare with top-right and bottom-left

        EO categories per sample (c = current, n1 = neighbour 1, n2 = n2):
            Category 0: c < n1 and c < n2   (local minimum)
            Category 1: c < n1 or  c < n2   (concave ramp)
            Category 3: c > n1 or  c > n2   (convex ramp)
            Category 4: c > n1 and c > n2   (local maximum)
            Category 2: else                 (flat, no offset)

        offsets[0..4]: offset for each category (0 for category 2 by spec)

SAO parameter estimation (encoder side)
-----------------------------------------
The encoder gathers statistics from the residual (original − reconstructed)
to determine which offsets minimise distortion. For each category:

    offset[cat] ≈ average(residual[pixel] for pixel in category[cat])

Then the offset is clipped to a reasonable range (±SAO_MAX_OFFSET).

Public API
----------
    sao_filter_frame(luma, chroma_cb, chroma_cr,
                     luma_params, cb_params, cr_params) -> None  (in-place)
    estimate_sao_params(original, reconstructed, ctu_size, sao_type)
                                                         -> SAOParams
    apply_sao_bo(plane, params, x, y, ctu_size)         -> None  (in-place)
    apply_sao_eo(plane, params, x, y, ctu_size)         -> None  (in-place)
    SAOParams                                            — dataclass
    SAOType                                              — enum
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAO_MAX_OFFSET:   int = 7     # maximum absolute offset value (4-bit signed)
SAO_NUM_BANDS:    int = 32    # number of BO bands in [0, 255]
SAO_BAND_SIZE:    int = 8     # pixel values per band (256 / 32)
SAO_NUM_BO_OFFSETS: int = 4   # number of active consecutive BO bands
SAO_EO_CATEGORIES: int = 5   # EO categories 0..4 (category 2 = no offset)

CTU_SIZE: int = 64   # default CTU size

# EO category labels (for debugging)
_EO_CAT_NAMES = {0: "local_min", 1: "concave", 2: "flat", 3: "convex", 4: "local_max"}


# ---------------------------------------------------------------------------
# Enumerations and data structures
# ---------------------------------------------------------------------------

class SAOType(Enum):
    OFF = auto()   # no SAO for this component
    BO  = auto()   # Band Offset
    EO  = auto()   # Edge Offset


class EODirection(Enum):
    EO_0 = 0   # horizontal (compare left, right)
    EO_1 = 1   # vertical   (compare above, below)
    EO_2 = 2   # 135°       (compare top-left, bottom-right)
    EO_3 = 3   # 45°        (compare top-right, bottom-left)


@dataclass
class SAOParams:
    """
    SAO parameters for one CTU and one colour component.

    Attributes
    ----------
    sao_type   : SAOType — OFF, BO, or EO
    band_start : int     — first active band index [0..28] (BO only)
    offsets    : list[int] — additive offsets:
                   BO: 4 values for bands [band_start .. band_start+3]
                   EO: 5 values for categories 0..4 (category 2 always 0)
    eo_dir     : EODirection — EO direction (EO only, ignored for BO)
    """
    sao_type:   SAOType         = SAOType.OFF
    band_start: int             = 0
    offsets:    list[int]       = field(default_factory=lambda: [0]*5)
    eo_dir:     EODirection     = EODirection.EO_0

    @property
    def is_off(self) -> bool:
        return self.sao_type == SAOType.OFF

    @property
    def is_bo(self) -> bool:
        return self.sao_type == SAOType.BO

    @property
    def is_eo(self) -> bool:
        return self.sao_type == SAOType.EO

    def __repr__(self) -> str:
        if self.is_off:
            return "SAOParams(OFF)"
        if self.is_bo:
            return f"SAOParams(BO, band={self.band_start}, off={self.offsets[:4]})"
        return f"SAOParams(EO, dir={self.eo_dir.name}, off={self.offsets})"


# ---------------------------------------------------------------------------
# Public API — frame-level entry point
# ---------------------------------------------------------------------------

def sao_filter_frame(
    luma:         np.ndarray,
    chroma_cb:    np.ndarray | None,
    chroma_cr:    np.ndarray | None,
    luma_params:  list[list[SAOParams]],
    cb_params:    list[list[SAOParams]] | None = None,
    cr_params:    list[list[SAOParams]] | None = None,
    ctu_size:     int = CTU_SIZE,
) -> None:
    """
    Apply SAO to all CTUs in a frame (in-place).

    SAO is applied independently per CTU, per colour component, based on
    pre-estimated SAOParams. This function iterates over the CTU grid and
    dispatches to apply_sao_bo or apply_sao_eo.

    Parameters
    ----------
    luma         : (H, W) uint8      — luma plane, modified in-place
    chroma_cb    : (H/2, W/2) uint8 or None
    chroma_cr    : (H/2, W/2) uint8 or None
    luma_params  : list[list[SAOParams]] — [row][col] grid of luma SAO params
    cb_params    : list[list[SAOParams]] or None — Cb params (same grid)
    cr_params    : list[list[SAOParams]] or None — Cr params (same grid)
    ctu_size     : int — CTU width/height in luma pixels (default 64)
    """
    H, W = luma.shape
    ctu_rows = (H + ctu_size - 1) // ctu_size
    ctu_cols = (W + ctu_size - 1) // ctu_size

    # Snapshot BEFORE any writes so EO in later CTUs reads original values,
    # not values already modified by earlier CTUs (cascading artefact).
    luma_snap = luma.copy()
    cb_snap   = chroma_cb.copy() if chroma_cb is not None else None
    cr_snap   = chroma_cr.copy() if chroma_cr is not None else None

    def _get(grid, r, c):
        """
        Fetch params from a 2-D grid with BROADCAST semantics.
        If the grid has fewer rows or columns than the actual CTU grid,
        the last available row/column is repeated. This allows callers to
        pass a 1-row grid to apply the same params to ALL CTU rows without
        duplicating entries.
        """
        if not grid:
            return SAOParams()
        r_clamped = min(r, len(grid) - 1)
        row = grid[r_clamped]
        if not row:
            return SAOParams()
        c_clamped = min(c, len(row) - 1)
        return row[c_clamped]

    for r in range(ctu_rows):
        for c in range(ctu_cols):
            lx = c * ctu_size
            ly = r * ctu_size

            # Luma
            params = _get(luma_params, r, c)
            if params.is_bo:
                apply_sao_bo(luma, params, lx, ly, ctu_size)
            elif params.is_eo:
                apply_sao_eo(luma, params, lx, ly, ctu_size, ref_plane=luma_snap)

            # Chroma
            cx, cy, cs = lx // 2, ly // 2, ctu_size // 2
            if chroma_cb is not None and cb_params is not None:
                p = _get(cb_params, r, c)
                if p.is_bo:   apply_sao_bo(chroma_cb, p, cx, cy, cs)
                elif p.is_eo: apply_sao_eo(chroma_cb, p, cx, cy, cs, ref_plane=cb_snap)

            if chroma_cr is not None and cr_params is not None:
                p = _get(cr_params, r, c)
                if p.is_bo:   apply_sao_bo(chroma_cr, p, cx, cy, cs)
                elif p.is_eo: apply_sao_eo(chroma_cr, p, cx, cy, cs, ref_plane=cr_snap)


# ---------------------------------------------------------------------------
# Public API — per-CTU application
# ---------------------------------------------------------------------------

def apply_sao_bo(
    plane:    np.ndarray,
    params:   SAOParams,
    x:        int,
    y:        int,
    ctu_size: int = CTU_SIZE,
) -> None:
    """
    Apply Band Offset SAO to one CTU (in-place).

    For each pixel in the CTU patch, determines its band index
    (value // SAO_BAND_SIZE), checks if it falls in the 4 active bands,
    and adds the corresponding offset.

    Parameters
    ----------
    plane    : np.ndarray — full frame plane (luma or chroma), uint8, in-place
    params   : SAOParams  — must have sao_type == BO
    x, y     : int        — top-left of CTU in plane pixel coordinates
    ctu_size : int        — CTU height/width in this plane's coordinates
    """
    if params.sao_type != SAOType.BO:
        return

    H, W   = plane.shape
    y_end  = min(y + ctu_size, H)
    x_end  = min(x + ctu_size, W)
    patch  = plane[y:y_end, x:x_end].astype(np.int32)

    band_idx  = patch >> 3   # equivalent to patch // SAO_BAND_SIZE
    bs        = params.band_start

    # Build a lookup: band_index → offset (0 for inactive bands)
    # HEVC uses CYCLIC bands: (band_start + k) % 32
    offset_lut = np.zeros(SAO_NUM_BANDS, dtype=np.int32)
    for k in range(SAO_NUM_BO_OFFSETS):
        b = (bs + k) % SAO_NUM_BANDS
        offset_lut[b] = params.offsets[k]

    offsets  = offset_lut[band_idx]
    result   = np.clip(patch + offsets, 0, 255).astype(np.uint8)
    plane[y:y_end, x:x_end] = result


def apply_sao_eo(
    plane:     np.ndarray,
    params:    SAOParams,
    x:         int,
    y:         int,
    ctu_size:  int = CTU_SIZE,
    ref_plane: np.ndarray | None = None,
) -> None:
    """
    Apply Edge Offset SAO to one CTU (in-place).

    For each pixel in the CTU, classifies it into one of 5 EO categories
    by comparing it to two neighbours in the specified direction, then adds
    the offset for that category.

    Parameters
    ----------
    plane     : np.ndarray         — full frame plane, modified in-place
    params    : SAOParams           — must have sao_type == EO
    x, y      : int                 — top-left of CTU in plane coordinates
    ctu_size  : int
    ref_plane : np.ndarray or None  — snapshot of the plane BEFORE any CTU
                                      has been filtered, used to prevent
                                      cascading between neighbouring CTUs.
                                      If None, a local copy is made.

    Edge classification (per pixel value c, neighbours n1, n2):
        Category 0: c < n1 and c < n2   (local minimum — offset should be +)
        Category 1: (c < n1) xor (c < n2) and c < max(n1,n2) (concave)
        Category 2: flat / mixed sign   (no offset, spec mandatory)
        Category 3: (c > n1) xor (c > n2) and c > min(n1,n2) (convex)
        Category 4: c > n1 and c > n2   (local maximum — offset should be -)
    """
    if params.sao_type != SAOType.EO:
        return

    H, W  = plane.shape
    y_end = min(y + ctu_size, H)
    x_end = min(x + ctu_size, W)

    # Neighbour offsets for each EO direction
    _EO_OFFSETS = {
        EODirection.EO_0: ((0, -1), (0,  1)),   # left, right
        EODirection.EO_1: ((-1, 0), (1,  0)),   # above, below
        EODirection.EO_2: ((-1,-1), (1,  1)),   # top-left, bottom-right
        EODirection.EO_3: ((-1, 1), (1, -1)),   # top-right, bottom-left
    }

    (dr1, dc1), (dr2, dc2) = _EO_OFFSETS[params.eo_dir]
    offsets = params.offsets   # [cat0, cat1, cat2, cat3, cat4]

    ph, pw = y_end - y, x_end - x

    # IMPORTANT: read all neighbour values from the ORIGINAL plane snapshot
    # before writing any results. Without this, early writes corrupt the
    # neighbour samples used for later pixel classifications.
    snap = ref_plane if ref_plane is not None else plane.copy()

    result = plane[y:y_end, x:x_end].astype(np.int32)

    for r in range(ph):
        for c in range(pw):
            cur  = int(snap[y + r, x + c])
            r1, c1 = y + r + dr1, x + c + dc1
            r2, c2 = y + r + dr2, x + c + dc2
            n1 = int(snap[max(0, min(H-1, r1)), max(0, min(W-1, c1))])
            n2 = int(snap[max(0, min(H-1, r2)), max(0, min(W-1, c2))])

            cat = _classify_eo(cur, n1, n2)
            result[r, c] = cur + offsets[cat]

    plane[y:y_end, x:x_end] = np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Public API — SAO parameter estimation
# ---------------------------------------------------------------------------

def estimate_sao_params(
    original:      np.ndarray,
    reconstructed: np.ndarray,
    sao_type:      SAOType,
    eo_dir:        EODirection = EODirection.EO_0,
    x:             int = 0,
    y:             int = 0,
    ctu_size:      int = CTU_SIZE,
    max_offset:    int = SAO_MAX_OFFSET,
) -> SAOParams:
    """
    Estimate SAO parameters for one CTU from original and reconstructed pixels.

    The encoder computes statistics (average residual per category) and
    clips the result to the signalled offset range [−max_offset, max_offset].

    For BO:
        For each band [0..31], compute mean(original − reconstructed) for
        pixels in that band. Find 4 CYCLICALLY consecutive bands (wrap-around
        at band 31 → 0) with maximum total |mean|.

    For EO:
        For each EO category [0..4], compute mean(original − reconstructed).
        Category 2 offset is forced to 0 (spec requirement).
        Sign constraints (HEVC CABAC only signals absolute value):
            Categories 0, 1 (local min / concave): offset ≥ 0
            Category  2     (flat):                 offset = 0
            Categories 3, 4 (convex / local max):  offset ≤ 0

    Parameters
    ----------
    original      : (H, W) uint8 — original (uncompressed) frame plane
    reconstructed : (H, W) uint8 — reconstructed (after deblocking) plane
    sao_type      : SAOType      — BO or EO (OFF returns SAOParams(OFF))
    eo_dir        : EODirection  — direction for EO (ignored for BO)
    x, y          : int          — top-left of CTU
    ctu_size      : int          — CTU height/width in this plane
    max_offset    : int          — clip range for offsets

    Returns
    -------
    SAOParams — estimated parameters ready to pass to apply_sao_bo/eo
    """
    if sao_type == SAOType.OFF:
        return SAOParams(SAOType.OFF)

    H, W  = original.shape
    y_end = min(y + ctu_size, H)
    x_end = min(x + ctu_size, W)

    orig = original[y:y_end, x:x_end].astype(np.int32)
    recon= reconstructed[y:y_end, x:x_end].astype(np.int32)
    diff = orig - recon   # positive = recon too dark, negative = too bright

    if sao_type == SAOType.BO:
        return _estimate_bo(diff, recon, max_offset)
    else:
        return _estimate_eo(diff, recon, eo_dir,
                            original, reconstructed, x, y, ctu_size,
                            H, W, max_offset)


def estimate_sao_frame(
    original:      np.ndarray,
    reconstructed: np.ndarray,
    sao_type:      SAOType = SAOType.BO,
    eo_dir:        EODirection = EODirection.EO_0,
    ctu_size:      int = CTU_SIZE,
    max_offset:    int = SAO_MAX_OFFSET,
) -> list[list[SAOParams]]:
    """
    Estimate SAO parameters for all CTUs in a full frame.

    Returns a 2-D grid [row][col] of SAOParams ready to pass to
    sao_filter_frame.

    Parameters
    ----------
    original, reconstructed : (H, W) uint8 — full frame planes
    sao_type  : SAOType — applied uniformly to all CTUs
    eo_dir    : EODirection — used when sao_type == EO
    ctu_size  : int
    max_offset: int

    Returns
    -------
    list[list[SAOParams]] — 2-D grid indexed [ctu_row][ctu_col]
    """
    H, W = original.shape
    ctu_rows = (H + ctu_size - 1) // ctu_size
    ctu_cols = (W + ctu_size - 1) // ctu_size

    grid: list[list[SAOParams]] = []
    for r in range(ctu_rows):
        row_params: list[SAOParams] = []
        for c in range(ctu_cols):
            p = estimate_sao_params(
                original, reconstructed,
                sao_type=sao_type, eo_dir=eo_dir,
                x=c * ctu_size, y=r * ctu_size,
                ctu_size=ctu_size, max_offset=max_offset,
            )
            row_params.append(p)
        grid.append(row_params)
    return grid


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _classify_eo(c: int, n1: int, n2: int) -> int:
    """
    Classify one sample into an EO category (0..4).

    HEVC §8.7.3.1 sign-based category:
        sign(c - n1) + sign(c - n2):
            -2 → category 0  (local minimum)
            -1 → category 1  (concave)
             0 → category 2  (flat)
            +1 → category 3  (convex)
            +2 → category 4  (local maximum)
    """
    s1 = _sign(c - n1)
    s2 = _sign(c - n2)
    edge_idx = s1 + s2    # range: -2..+2
    # Map -2,-1,0,+1,+2 → category 0,1,2,3,4
    return edge_idx + 2


def _sign(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def _estimate_bo(
    diff:       np.ndarray,
    recon:      np.ndarray,
    max_offset: int,
) -> SAOParams:
    """Estimate Band Offset parameters from per-pixel residual statistics."""
    # Accumulate sum and count per band
    bands = recon >> 3   # band index for each pixel
    band_sum   = np.zeros(SAO_NUM_BANDS, dtype=np.int64)
    band_count = np.zeros(SAO_NUM_BANDS, dtype=np.int64)

    flat_bands = bands.ravel()
    flat_diff  = diff.ravel()
    for b, d in zip(flat_bands, flat_diff):
        band_sum[b]   += d
        band_count[b] += 1

    # Compute mean offset per band (0 if no pixels)
    band_mean = np.zeros(SAO_NUM_BANDS, dtype=np.float64)
    for b in range(SAO_NUM_BANDS):
        if band_count[b] > 0:
            band_mean[b] = band_sum[b] / band_count[b]

    # Find the 4 consecutive bands with maximum total absolute mean
    # HEVC uses CYCLIC band selection: band (start+k) % 32
    # e.g. start=30 → active bands 30, 31, 0, 1
    best_start = 0
    best_total = -1.0
    for start in range(SAO_NUM_BANDS):
        total = sum(abs(band_mean[(start + k) % SAO_NUM_BANDS])
                    for k in range(SAO_NUM_BO_OFFSETS))
        if total > best_total:
            best_total = total
            best_start = start

    # Compute clipped offsets for the 4 best bands (cyclic indexing)
    offsets = [0] * 5
    for k in range(SAO_NUM_BO_OFFSETS):
        b   = (best_start + k) % SAO_NUM_BANDS
        raw = band_mean[b]
        offsets[k] = int(np.clip(round(raw), -max_offset, max_offset))

    return SAOParams(
        sao_type=SAOType.BO,
        band_start=best_start,
        offsets=offsets,
    )


def _estimate_eo(
    diff:          np.ndarray,
    recon:         np.ndarray,
    eo_dir:        EODirection,
    original:      np.ndarray,
    reconstructed: np.ndarray,
    x:             int,
    y:             int,
    ctu_size:      int,
    H:             int,
    W:             int,
    max_offset:    int,
) -> SAOParams:
    """Estimate Edge Offset parameters from per-pixel edge statistics."""
    _EO_OFFSETS = {
        EODirection.EO_0: ((0, -1), (0,  1)),
        EODirection.EO_1: ((-1, 0), (1,  0)),
        EODirection.EO_2: ((-1,-1), (1,  1)),
        EODirection.EO_3: ((-1, 1), (1, -1)),
    }
    (dr1, dc1), (dr2, dc2) = _EO_OFFSETS[eo_dir]

    cat_sum   = np.zeros(SAO_EO_CATEGORIES, dtype=np.int64)
    cat_count = np.zeros(SAO_EO_CATEGORIES, dtype=np.int64)

    y_end = min(y + ctu_size, H)
    x_end = min(x + ctu_size, W)

    for r in range(y, y_end):
        for c in range(x, x_end):
            cur   = int(reconstructed[r, c])
            d     = int(original[r, c]) - cur
            r1c1  = (max(0, min(H-1, r+dr1)), max(0, min(W-1, c+dc1)))
            r2c2  = (max(0, min(H-1, r+dr2)), max(0, min(W-1, c+dc2)))
            n1    = int(reconstructed[r1c1[0], r1c1[1]])
            n2    = int(reconstructed[r2c2[0], r2c2[1]])
            cat   = _classify_eo(cur, n1, n2)
            cat_sum[cat]   += d
            cat_count[cat] += 1

    # EO sign constraint (HEVC spec — CABAC only signals absolute values):
    #   cat 0 (local min) and cat 1 (concave): offset MUST be ≥ 0
    #   cat 2 (flat):                          offset MUST be = 0
    #   cat 3 (convex) and cat 4 (local max):  offset MUST be ≤ 0
    offsets = [0] * SAO_EO_CATEGORIES
    for cat in range(SAO_EO_CATEGORIES):
        if cat == 2:
            offsets[cat] = 0   # mandatory zero (spec)
        elif cat_count[cat] > 0:
            raw = cat_sum[cat] / cat_count[cat]
            clipped = int(np.clip(round(raw), -max_offset, max_offset))
            # Enforce sign constraint
            if cat in (0, 1):
                offsets[cat] = max(0, clipped)   # must be ≥ 0
            else:  # cat in (3, 4)
                offsets[cat] = min(0, clipped)   # must be ≤ 0

    return SAOParams(
        sao_type=SAOType.EO,
        eo_dir=eo_dir,
        offsets=offsets,
    )