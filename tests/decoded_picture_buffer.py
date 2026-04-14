"""
decoded_picture_buffer.py — HEVC Decoded Picture Buffer (DPB)
Manages the lifecycle of decoded/reconstructed frames in encoder RAM.

Pipeline position
-----------------
    tu_split.py  →  [decoded_picture_buffer.py]  →  motion_estimation.py
    (recon frame       (store + manage refs)          (fetch ref frame
     assembled)                                        for inter ME)

                    [decoded_picture_buffer.py]  →  reference_manager.py
                    (holds all decoded frames)        (builds RPL lists)

What the DPB does
------------------
After a frame is fully encoded and reconstructed (prediction + residual),
its reconstructed luma/chroma planes are stored in the DPB so that future
frames can use them as inter prediction references.

The DPB also:
    1. Tracks picture order count (POC) and decode order
    2. Marks frames as "short-term" or "long-term" references
    3. Evicts frames that are no longer needed (buffer management)
    4. Exposes frames to motion_estimation.py and reference_manager.py

HEVC DPB rules (ISO/IEC 23008-2 §C.5)
----------------------------------------
    max_dec_pic_buffering: maximum frames in the DPB at any time
        Default for Main profile: NumRefFrames + 1 (typically 5–8)

    Reference picture marking:
        - Used for reference (short-term or long-term): kept in DPB
        - Unused for reference: evicted (marked for removal)

    Short-term reference (STR): referenced by recent P/B frames
    Long-term reference (LTR):  explicitly marked, survives across IDRs

    POC (Picture Order Count): display order, used to sort reference lists

Frame types:
    I  (Intra):        no inter references, refreshes the DPB
    P  (Predictive):   references L0 list only
    B  (Bidirectional): references L0 and L1 lists

Public API
----------
    DecodedPictureBuffer     — the DPB container class
    PictureBuffer            — one decoded frame (luma + chroma planes)
    dpb.store(pic)           — add a newly decoded frame
    dpb.get_ref(poc)         — fetch a reference frame by POC
    dpb.get_short_term_refs()→ list[PictureBuffer] sorted by POC
    dpb.evict_unused()       — remove frames no longer needed
    dpb.mark_used(poc, flag) — mark a frame as used/unused for reference
    dpb.flush()              — clear all frames (IDR reset)
    dpb.fullness             — current number of frames in buffer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_BUFFERING: int = 8    # max frames in DPB (Main profile default)
DEFAULT_CHROMA_FORMAT: str = "420"   # 4:2:0


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class FrameType(Enum):
    I = auto()   # Intra — all CUs are intra predicted
    P = auto()   # Predictive — L0 inter references only
    B = auto()   # Bidirectional — L0 and L1 inter references


class RefStatus(Enum):
    UNUSED        = auto()   # can be evicted
    SHORT_TERM    = auto()   # short-term reference (recent)
    LONG_TERM     = auto()   # long-term reference (explicit marking)


# ---------------------------------------------------------------------------
# PictureBuffer — one decoded frame
# ---------------------------------------------------------------------------

@dataclass
class PictureBuffer:
    """
    One decoded/reconstructed frame stored in the DPB.

    Holds the reconstructed luma and chroma pixel planes after the full
    encode + decode loop (prediction + residual reconstruction).

    Attributes
    ----------
    poc          : int         — Picture Order Count (display order, 0-based)
    decode_order : int         — decode/bitstream order
    frame_type   : FrameType   — I, P, or B
    luma         : np.ndarray  — reconstructed Y plane, (H, W), uint8
    chroma_cb    : np.ndarray  — reconstructed Cb plane, (H/2, W/2), uint8
    chroma_cr    : np.ndarray  — reconstructed Cr plane, (H/2, W/2), uint8
    ref_status   : RefStatus   — UNUSED / SHORT_TERM / LONG_TERM
    qp           : int         — frame-level QP used for encoding
    """
    poc:          int
    decode_order: int
    frame_type:   FrameType
    luma:         np.ndarray              # (H, W) uint8
    chroma_cb:    np.ndarray | None = None  # (H/2, W/2) uint8 — None for 4:0:0
    chroma_cr:    np.ndarray | None = None
    ref_status:   RefStatus = RefStatus.SHORT_TERM
    qp:           int = 28

    # ── Computed properties ──────────────────────────────────────────────

    @property
    def height(self) -> int:
        return self.luma.shape[0]

    @property
    def width(self) -> int:
        return self.luma.shape[1]

    @property
    def resolution(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return (self.width, self.height)

    @property
    def is_reference(self) -> bool:
        return self.ref_status != RefStatus.UNUSED

    @property
    def is_idr(self) -> bool:
        """True if this is an IDR frame (I-frame that resets the DPB)."""
        return self.frame_type == FrameType.I and self.poc == 0

    @property
    def memory_bytes(self) -> int:
        """Approximate RAM footprint of this frame's pixel data."""
        luma_bytes = self.luma.nbytes
        cb_bytes   = self.chroma_cb.nbytes if self.chroma_cb is not None else 0
        cr_bytes   = self.chroma_cr.nbytes if self.chroma_cr is not None else 0
        return luma_bytes + cb_bytes + cr_bytes

    def mark_unused(self) -> None:
        """Mark this frame as unused for reference (candidate for eviction)."""
        self.ref_status = RefStatus.UNUSED

    def mark_short_term(self) -> None:
        self.ref_status = RefStatus.SHORT_TERM

    def mark_long_term(self) -> None:
        self.ref_status = RefStatus.LONG_TERM

    def __repr__(self) -> str:
        return (f"PictureBuffer(poc={self.poc}, type={self.frame_type.name}, "
                f"status={self.ref_status.name}, "
                f"res={self.width}×{self.height})")


# ---------------------------------------------------------------------------
# DecodedPictureBuffer — the DPB container
# ---------------------------------------------------------------------------

class DecodedPictureBuffer:
    """
    HEVC Decoded Picture Buffer — manages the pool of reference frames.

    The DPB stores all reconstructed frames that are currently needed as
    inter prediction references. It enforces the HEVC buffer capacity limit
    and provides ordered access to reference lists.

    Usage
    -----
        dpb = DecodedPictureBuffer(max_buffering=5)

        # After encoding frame 0 (I-frame):
        dpb.store(PictureBuffer(poc=0, decode_order=0,
                                frame_type=FrameType.I, luma=recon_y))

        # After encoding frame 1 (P-frame referencing poc=0):
        ref = dpb.get_ref(poc=0)   # fetch reference for ME
        dpb.store(PictureBuffer(poc=1, decode_order=1,
                                frame_type=FrameType.P, luma=recon_y))

        # When frame 0 is no longer needed:
        dpb.mark_unused(poc=0)
        dpb.evict_unused()
    """

    def __init__(
        self,
        max_buffering: int = DEFAULT_MAX_BUFFERING,
    ) -> None:
        """
        Parameters
        ----------
        max_buffering : int
            Maximum number of frames the DPB can hold simultaneously.
            Matches max_dec_pic_buffering in HEVC VPS/SPS.
        """
        self.max_buffering: int = max_buffering
        self._buffer: list[PictureBuffer] = []
        self._decode_counter: int = 0   # monotonically increasing decode order

    # ── Core operations ──────────────────────────────────────────────────

    def store(self, pic: PictureBuffer) -> None:
        """
        Add a newly reconstructed frame to the DPB.

        If the DPB is full, this raises BufferError — the caller should
        call evict_unused() first, or the encoder should not produce more
        reference frames than max_buffering allows.

        Parameters
        ----------
        pic : PictureBuffer — the reconstructed frame to store

        Raises
        ------
        BufferError  — DPB is at capacity and cannot accept new frames.
        ValueError   — A frame with the same POC already exists in the DPB.
        """
        if self.is_full:
            raise BufferError(
                f"DPB is full ({self.max_buffering} frames). "
                f"Call evict_unused() before storing new frames."
            )
        if self._find(pic.poc) is not None:
            raise ValueError(
                f"A frame with POC={pic.poc} already exists in the DPB."
            )
        self._buffer.append(pic)

    def get_ref(self, poc: int) -> PictureBuffer:
        """
        Fetch a reference frame by POC.

        Parameters
        ----------
        poc : int — picture order count of the desired reference

        Returns
        -------
        PictureBuffer

        Raises
        ------
        KeyError — no frame with this POC in the DPB.
        """
        pic = self._find(poc)
        if pic is None:
            raise KeyError(
                f"No frame with POC={poc} in DPB. "
                f"Available POCs: {self.available_pocs}"
            )
        return pic

    def mark_unused(self, poc: int) -> None:
        """
        Mark a frame as unused for reference (candidate for eviction).

        Parameters
        ----------
        poc : int — picture order count of the frame to mark

        Raises
        ------
        KeyError — frame not found in DPB.
        """
        pic = self._find(poc)
        if pic is None:
            raise KeyError(f"No frame with POC={poc} in DPB.")
        pic.mark_unused()

    def mark_long_term(self, poc: int) -> None:
        """
        Mark a frame as a long-term reference.
        Long-term references survive RASL (Random Access Skipped Leading)
        picture filtering and explicit removal only.
        """
        pic = self._find(poc)
        if pic is None:
            raise KeyError(f"No frame with POC={poc} in DPB.")
        pic.mark_long_term()

    def evict_unused(self) -> list[int]:
        """
        Remove all frames marked as UNUSED for reference.

        Returns
        -------
        list[int] — POCs of the evicted frames (for logging/testing).
        """
        evicted = [p.poc for p in self._buffer if not p.is_reference]
        self._buffer = [p for p in self._buffer if p.is_reference]
        return sorted(evicted)

    def evict_oldest_short_term(self) -> int | None:
        """
        Evict the short-term reference frame with the smallest POC.

        Called automatically by store_and_manage() when the buffer is full.
        Long-term references are never evicted by this method.

        Returns
        -------
        int | None — POC of the evicted frame, or None if no STR found.
        """
        short_term = [p for p in self._buffer
                      if p.ref_status == RefStatus.SHORT_TERM]
        if not short_term:
            return None
        oldest = min(short_term, key=lambda p: p.decode_order)
        oldest.mark_unused()
        self.evict_unused()
        return oldest.poc

    def store_and_manage(self, pic: PictureBuffer) -> list[int]:
        """
        Store a frame, automatically evicting oldest short-term refs if full.

        This is the "managed" entry point that mirrors the HEVC sliding
        window reference picture management (HEVC §C.5.2.2):
            1. If buffer is not full, just store.
            2. If buffer is full, evict oldest short-term reference first.
            3. If still full (all are long-term), raise BufferError.

        Parameters
        ----------
        pic : PictureBuffer — the reconstructed frame to store

        Returns
        -------
        list[int] — POCs of any frames that were evicted (may be empty).

        Raises
        ------
        BufferError — DPB is full and no short-term reference can be evicted.
        ValueError  — Duplicate POC.
        """
        evicted: list[int] = []
        while self.is_full:
            poc_evicted = self.evict_oldest_short_term()
            if poc_evicted is None:
                raise BufferError(
                    f"DPB is full with {self.fullness} long-term references. "
                    f"Cannot evict to make room for POC={pic.poc}."
                )
            evicted.append(poc_evicted)
        self.store(pic)
        return evicted

    def flush(self) -> None:
        """
        Clear all frames from the DPB (IDR reset).

        After an IDR frame, no previously decoded frame can be used as
        a reference. This method mirrors the HEVC decoder's mandatory
        "flush DPB" behaviour at IDR.
        """
        self._buffer.clear()

    # ── Reference list access ────────────────────────────────────────────

    def get_short_term_refs(self) -> list[PictureBuffer]:
        """
        Return all short-term reference frames, sorted by POC ascending.
        Used by reference_manager.py to build RPL lists.
        """
        return sorted(
            (p for p in self._buffer if p.ref_status == RefStatus.SHORT_TERM),
            key=lambda p: p.poc,
        )

    def get_long_term_refs(self) -> list[PictureBuffer]:
        """Return all long-term references, sorted by POC."""
        return sorted(
            (p for p in self._buffer if p.ref_status == RefStatus.LONG_TERM),
            key=lambda p: p.poc,
        )

    def get_all_refs(self) -> list[PictureBuffer]:
        """All reference frames (short + long term), sorted by POC."""
        return sorted(
            (p for p in self._buffer if p.is_reference),
            key=lambda p: p.poc,
        )

    def get_preceding_refs(self, current_poc: int) -> list[PictureBuffer]:
        """
        Return reference frames with POC < current_poc (L0 candidates).
        Sorted by POC descending (closest in time first — HEVC RPL order).
        """
        return sorted(
            (p for p in self._buffer
             if p.is_reference and p.poc < current_poc),
            key=lambda p: p.poc,
            reverse=True,
        )

    def get_following_refs(self, current_poc: int) -> list[PictureBuffer]:
        """
        Return reference frames with POC > current_poc (L1 candidates).
        Sorted by POC ascending (closest in time first).
        """
        return sorted(
            (p for p in self._buffer
             if p.is_reference and p.poc > current_poc),
            key=lambda p: p.poc,
        )

    # ── Iteration ────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[PictureBuffer]:
        """Iterate over all frames in the DPB (in storage order)."""
        return iter(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __contains__(self, poc: int) -> bool:
        return self._find(poc) is not None

    # ── Status / properties ──────────────────────────────────────────────

    @property
    def fullness(self) -> int:
        """Current number of frames in the DPB."""
        return len(self._buffer)

    @property
    def is_full(self) -> bool:
        return len(self._buffer) >= self.max_buffering

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0

    @property
    def available_pocs(self) -> list[int]:
        """Sorted list of POCs currently in the DPB."""
        return sorted(p.poc for p in self._buffer)

    @property
    def total_memory_bytes(self) -> int:
        """Total RAM used by all frames in the DPB."""
        return sum(p.memory_bytes for p in self._buffer)

    def summary(self) -> str:
        """Human-readable DPB status for logging/debugging."""
        lines = [f"DPB ({self.fullness}/{self.max_buffering} frames, "
                 f"{self.total_memory_bytes/1024/1024:.1f} MB):"]
        for p in sorted(self._buffer, key=lambda x: x.poc):
            lines.append(
                f"  POC={p.poc:3d} [{p.frame_type.name}] "
                f"{p.ref_status.name:11s}  {p.width}×{p.height}"
            )
        return "\n".join(lines)

    # ── Private helpers ──────────────────────────────────────────────────

    def _find(self, poc: int) -> PictureBuffer | None:
        """Return the PictureBuffer with this POC, or None."""
        for p in self._buffer:
            if p.poc == poc:
                return p
        return None


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_picture_buffer(
    poc:          int,
    decode_order: int,
    frame_type:   FrameType,
    luma:         np.ndarray,
    chroma_cb:    np.ndarray | None = None,
    chroma_cr:    np.ndarray | None = None,
    qp:           int = 28,
    ref_status:   RefStatus = RefStatus.SHORT_TERM,
) -> PictureBuffer:
    """
    Construct a PictureBuffer with validated inputs.

    Parameters
    ----------
    poc, decode_order : int — picture order and decode order counters
    frame_type        : FrameType
    luma              : (H, W) uint8 — reconstructed luma plane
    chroma_cb         : (H/2, W/2) uint8 or None
    chroma_cr         : (H/2, W/2) uint8 or None
    qp                : int — frame-level QP
    ref_status        : RefStatus — initial reference status

    Returns
    -------
    PictureBuffer

    Raises
    ------
    ValueError — if luma is not 2-D uint8, or chroma shape mismatches.
    """
    if luma.ndim != 2 or luma.dtype != np.uint8:
        raise ValueError(
            f"luma must be a 2-D uint8 array, got shape={luma.shape} "
            f"dtype={luma.dtype}."
        )
    H, W = luma.shape

    if chroma_cb is not None:
        exp_h, exp_w = H // 2, W // 2
        if chroma_cb.shape != (exp_h, exp_w):
            raise ValueError(
                f"chroma_cb shape {chroma_cb.shape} ≠ expected "
                f"({exp_h}, {exp_w}) for 4:2:0."
            )
    if chroma_cr is not None:
        exp_h, exp_w = H // 2, W // 2
        if chroma_cr.shape != (exp_h, exp_w):
            raise ValueError(
                f"chroma_cr shape {chroma_cr.shape} ≠ expected "
                f"({exp_h}, {exp_w}) for 4:2:0."
            )

    return PictureBuffer(
        poc=poc,
        decode_order=decode_order,
        frame_type=frame_type,
        luma=luma,
        chroma_cb=chroma_cb,
        chroma_cr=chroma_cr,
        qp=qp,
        ref_status=ref_status,
    )


def make_grey_frame(
    poc:          int,
    decode_order: int,
    frame_type:   FrameType,
    width:        int,
    height:       int,
    luma_value:   int = 128,
    qp:           int = 28,
) -> PictureBuffer:
    """
    Create a synthetic grey-fill PictureBuffer for testing.

    Parameters
    ----------
    width, height : int — luma frame dimensions
    luma_value    : int — fill value for the luma plane (default 128 = mid-grey)
    """
    luma = np.full((height, width), luma_value, dtype=np.uint8)
    cb   = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    cr   = np.full((height // 2, width // 2), 128, dtype=np.uint8)
    return PictureBuffer(
        poc=poc, decode_order=decode_order,
        frame_type=frame_type,
        luma=luma, chroma_cb=cb, chroma_cr=cr,
        qp=qp,
    )