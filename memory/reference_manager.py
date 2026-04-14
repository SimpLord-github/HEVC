"""
reference_manager.py — HEVC Reference Picture List (RPL) Manager
Builds Reference Picture List 0 (L0) and List 1 (L1) from the DPB.

Pipeline position
-----------------
    decoded_picture_buffer.py  →  [reference_manager.py]  →  motion_estimation.py
    (all decoded frames)           (ordered L0 / L1 lists)    (picks one ref per CU)

                                [reference_manager.py]  →  slice_header.py
                                (ref_pic_list_modification)   (signals RPL in bitstream)

What Reference Picture Lists are
----------------------------------
HEVC inter prediction requires the encoder to tell the decoder exactly which
frames can be used as references. This is done through two ordered lists:

    L0 (List 0) — used by P-slices and B-slices
        Typically sorted with the most-recent frame first (closest in time).
        For P-slices, only L0 exists.

    L1 (List 1) — used by B-slices only
        Typically sorted with the nearest future frame first.

Both lists are built from the frames currently in the DPB that are marked
as SHORT_TERM or LONG_TERM references.

HEVC default list construction (ISO/IEC 23008-2 §8.3.3)
---------------------------------------------------------
For P-slices:
    L0 = [all STR with poc < current, descending POC]
         + [all LTR sorted by long_term_pic_num]

For B-slices:
    L0 = [STR poc < current, descending] + [STR poc > current, ascending] + LTR
    L1 = [STR poc > current, ascending]  + [STR poc < current, descending] + LTR

Both lists are truncated to num_ref_idx_l0/l1_active_minus1 + 1 entries.

Reference Picture Set (RPS) concept
--------------------------------------
HEVC uses a Reference Picture Set to signal which frames must be kept in
the DPB and which can be evicted. The RPS contains:

    StRefPicSet (short-term):
        num_negative_pics — STR frames with poc < current
        num_positive_pics — STR frames with poc > current
        delta_poc values  — POC differences from current

    LtRefPicSet (long-term):
        poc_lsb values    — least-significant bits of LTR POCs

This file implements the construction and RPS signalling logic used by
slice_header.py to write the RPL into the bitstream.

Public API
----------
    ReferenceList       — ordered list of reference frames (L0 or L1)
    ReferencePictureSet — complete RPS for one slice
    ReferenceManager    — stateful manager that wraps a DPB
    rm.build_l0(current_poc, slice_type, max_refs) -> ReferenceList
    rm.build_l1(current_poc, max_refs)             -> ReferenceList
    rm.build_rps(current_poc, slice_type)          -> ReferencePictureSet
    rm.mark_unused_after_encoding(current_poc,
                                  num_active_l0, num_active_l1)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import numpy as np

from decoded_picture_buffer import (
    DecodedPictureBuffer,
    PictureBuffer,
    FrameType,
    RefStatus,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HEVC Main profile maximum reference indices
MAX_REF_IDX_L0: int = 15   # maximum entries in L0 (spec allows up to 15)
MAX_REF_IDX_L1: int = 15   # maximum entries in L1

# Default active reference counts (typical encoder settings)
DEFAULT_NUM_REF_L0: int = 1   # P-slice default: 1 reference
DEFAULT_NUM_REF_L1: int = 1   # B-slice default: 1 reference per list


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReferenceEntry:
    """
    One entry in a reference picture list.

    Attributes
    ----------
    pic        : PictureBuffer — the reference frame
    ref_idx    : int           — 0-based index in the list (0 = first/best ref)
    delta_poc  : int           — signed POC difference from current picture
    is_long_term : bool        — True for long-term references
    """
    pic:          PictureBuffer
    ref_idx:      int
    delta_poc:    int
    is_long_term: bool = False

    @property
    def poc(self) -> int:
        return self.pic.poc

    @property
    def luma(self) -> np.ndarray:
        return self.pic.luma

    def __repr__(self) -> str:
        kind = "LTR" if self.is_long_term else "STR"
        return (f"ReferenceEntry(idx={self.ref_idx}, poc={self.poc}, "
                f"Δpoc={self.delta_poc:+d}, {kind})")


@dataclass
class ReferenceList:
    """
    An ordered reference picture list (L0 or L1).

    Entries are in encoding priority order: index 0 is the preferred reference
    for single-reference inter prediction.

    Attributes
    ----------
    entries     : list[ReferenceEntry] — ordered reference entries
    list_id     : int                  — 0 = L0, 1 = L1
    current_poc : int                  — POC of the frame being encoded
    """
    entries:     list[ReferenceEntry]
    list_id:     int   # 0 = L0, 1 = L1
    current_poc: int

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> ReferenceEntry:
        return self.entries[idx]

    def __iter__(self):
        return iter(self.entries)

    @property
    def is_empty(self) -> bool:
        return len(self.entries) == 0

    @property
    def best_ref(self) -> ReferenceEntry | None:
        """Entry at index 0 — the primary reference for this list."""
        return self.entries[0] if self.entries else None

    @property
    def pocs(self) -> list[int]:
        return [e.poc for e in self.entries]

    @property
    def luma_frames(self) -> list[np.ndarray]:
        """Luma planes in list order — convenience for motion_estimation.py."""
        return [e.luma for e in self.entries]

    def get_by_poc(self, poc: int) -> ReferenceEntry | None:
        for e in self.entries:
            if e.poc == poc:
                return e
        return None

    def __repr__(self) -> str:
        name = f"L{self.list_id}"
        entries = ", ".join(f"poc={e.poc}(Δ{e.delta_poc:+d})" for e in self.entries)
        return f"ReferenceList({name}, current_poc={self.current_poc}, [{entries}])"


@dataclass
class ReferencePictureSet:
    """
    Complete Reference Picture Set for one slice (HEVC §7.3.6.1).

    The RPS is what gets written into the slice header by slice_header.py.
    It tells the decoder which frames to keep in its DPB and which to discard.

    Attributes
    ----------
    current_poc     : int
    l0              : ReferenceList — L0 list (P and B slices)
    l1              : ReferenceList — L1 list (B slices only; empty for P)
    num_negative    : int — number of STR with poc < current (delta_poc < 0)
    num_positive    : int — number of STR with poc > current (delta_poc > 0)
    num_long_term   : int — number of long-term references
    delta_poc_neg   : list[int] — sorted delta_poc values for negative STR
    delta_poc_pos   : list[int] — sorted delta_poc values for positive STR
    ltr_poc_lsb     : list[int] — POC LSBs for long-term references
    """
    current_poc:   int
    l0:            ReferenceList
    l1:            ReferenceList
    num_negative:  int = 0
    num_positive:  int = 0
    num_long_term: int = 0
    delta_poc_neg: list[int] = field(default_factory=list)
    delta_poc_pos: list[int] = field(default_factory=list)
    ltr_poc_lsb:   list[int] = field(default_factory=list)

    @property
    def all_refs(self) -> list[ReferenceEntry]:
        """All unique reference entries across L0 and L1."""
        seen_pocs: set[int] = set()
        result: list[ReferenceEntry] = []
        for lst in (self.l0, self.l1):
            for e in lst:
                if e.poc not in seen_pocs:
                    seen_pocs.add(e.poc)
                    result.append(e)
        return result

    def __repr__(self) -> str:
        return (f"RPS(poc={self.current_poc}, "
                f"L0={self.l0.pocs}, L1={self.l1.pocs}, "
                f"neg={self.num_negative}, pos={self.num_positive}, "
                f"ltr={self.num_long_term})")


# ---------------------------------------------------------------------------
# ReferenceManager — stateful RPL builder
# ---------------------------------------------------------------------------

class ReferenceManager:
    """
    Builds and manages Reference Picture Lists from the DPB.

    One ReferenceManager is created per encoder session and holds a
    reference to the shared DPB. It is called once per slice to produce
    the RPS that drives both inter prediction and slice header writing.

    Usage
    -----
        rm = ReferenceManager(dpb, max_refs_l0=2, max_refs_l1=1)

        # Before encoding a P-slice at POC=5:
        rps = rm.build_rps(current_poc=5, slice_type="P")
        ref_frame = rps.l0.best_ref.luma    # use for ME

        # After encoding:
        rm.mark_unused_after_encoding(current_poc=5,
                                      num_active_l0=1, num_active_l1=0)
    """

    def __init__(
        self,
        dpb:          DecodedPictureBuffer,
        max_refs_l0:  int = DEFAULT_NUM_REF_L0,
        max_refs_l1:  int = DEFAULT_NUM_REF_L1,
        max_poc_lsb:  int = 256,    # 2^log2_max_pic_order_cnt_lsb
    ) -> None:
        """
        Parameters
        ----------
        dpb         : DecodedPictureBuffer — shared DPB instance
        max_refs_l0 : int — maximum active L0 references per slice
        max_refs_l1 : int — maximum active L1 references per slice
        max_poc_lsb : int — modulus for POC LSB computation (2^N from SPS)
        """
        self.dpb         = dpb
        self.max_refs_l0 = min(max_refs_l0, MAX_REF_IDX_L0)
        self.max_refs_l1 = min(max_refs_l1, MAX_REF_IDX_L1)
        self.max_poc_lsb = max_poc_lsb

    # ── Public: list construction ────────────────────────────────────────

    def build_l0(
        self,
        current_poc: int,
        slice_type:  Literal["P", "B"] = "P",
        max_refs:    int | None = None,
    ) -> ReferenceList:
        """
        Build Reference Picture List 0 (L0).

        Default construction order (HEVC §8.3.3):
            1. Short-term refs with poc < current,  sorted by POC descending
               (closest-in-time first — smallest |delta_poc| first for L0)
            2. Short-term refs with poc > current,  sorted by POC ascending
               (used only in B-slice L0 for forward references)
            3. Long-term references, sorted by POC

        Parameters
        ----------
        current_poc : int          — POC of the slice being encoded
        slice_type  : {"P", "B"}   — affects whether forward refs appear in L0
        max_refs    : int or None  — cap on list length (default: self.max_refs_l0)

        Returns
        -------
        ReferenceList
        """
        cap = max_refs if max_refs is not None else self.max_refs_l0

        # Short-term preceding (poc < current): closest first = descending POC
        neg_strs = self.dpb.get_preceding_refs(current_poc)  # already desc

        # Short-term following (poc > current): used in B L0 = ascending
        pos_strs = self.dpb.get_following_refs(current_poc)  # already asc

        # Long-term references sorted by POC
        ltrs = self.dpb.get_long_term_refs()

        ordered: list[PictureBuffer]
        if slice_type == "B":
            ordered = neg_strs + pos_strs + ltrs
        else:
            ordered = neg_strs + ltrs  # P-slice: no forward refs in L0

        entries = _build_entries(ordered, current_poc, cap)
        return ReferenceList(entries=entries, list_id=0, current_poc=current_poc)

    def build_l1(
        self,
        current_poc: int,
        max_refs:    int | None = None,
    ) -> ReferenceList:
        """
        Build Reference Picture List 1 (L1) — B-slices only.

        Default construction order (HEVC §8.3.3):
            1. Short-term refs with poc > current, sorted by POC ascending
               (closest future frame first)
            2. Short-term refs with poc < current, sorted by POC descending
            3. Long-term references

        Parameters
        ----------
        current_poc : int
        max_refs    : int or None

        Returns
        -------
        ReferenceList (empty if no forward references exist)
        """
        cap = max_refs if max_refs is not None else self.max_refs_l1

        pos_strs = self.dpb.get_following_refs(current_poc)   # asc
        neg_strs = self.dpb.get_preceding_refs(current_poc)   # desc
        ltrs     = self.dpb.get_long_term_refs()

        ordered = pos_strs + neg_strs + ltrs
        entries = _build_entries(ordered, current_poc, cap)
        return ReferenceList(entries=entries, list_id=1, current_poc=current_poc)

    def build_rps(
        self,
        current_poc: int,
        slice_type:  Literal["I", "P", "B"] = "P",
        max_refs_l0: int | None = None,
        max_refs_l1: int | None = None,
    ) -> ReferencePictureSet:
        """
        Build the complete Reference Picture Set for one slice.

        Constructs L0 and L1 (if B-slice), then computes the RPS delta_poc
        arrays used by slice_header.py for bitstream signalling.

        Parameters
        ----------
        current_poc  : int
        slice_type   : {"I", "P", "B"}
        max_refs_l0  : int or None — override self.max_refs_l0
        max_refs_l1  : int or None — override self.max_refs_l1

        Returns
        -------
        ReferencePictureSet

        Notes
        -----
        For I-slices, both L0 and L1 are empty lists.
        """
        empty_l0 = ReferenceList([], 0, current_poc)
        empty_l1 = ReferenceList([], 1, current_poc)

        if slice_type == "I":
            return ReferencePictureSet(
                current_poc=current_poc, l0=empty_l0, l1=empty_l1
            )

        l0 = self.build_l0(current_poc, slice_type, max_refs_l0)
        l1 = self.build_l1(current_poc, max_refs_l1) \
             if slice_type == "B" else empty_l1

        # Compute RPS delta_poc arrays for bitstream signalling
        all_strs = self.dpb.get_short_term_refs()
        ltrs     = self.dpb.get_long_term_refs()

        neg_dpocs = sorted(
            [p.poc - current_poc for p in all_strs if p.poc < current_poc],
            reverse=True  # Quan trọng: Gần nhất (ví dụ -1, -2) phải đứng trước!
        )

        pos_dpocs = sorted(
            [p.poc - current_poc for p in all_strs if p.poc > current_poc]
        )  # Positive thì vẫn ascending (tăng dần) đúng chuẩn

        ltr_lsbs = [p.poc % self.max_poc_lsb for p in ltrs]

        return ReferencePictureSet(
            current_poc   = current_poc,
            l0            = l0,
            l1            = l1,
            num_negative  = len(neg_dpocs),
            num_positive  = len(pos_dpocs),
            num_long_term = len(ltrs),
            delta_poc_neg = neg_dpocs,
            delta_poc_pos = pos_dpocs,
            ltr_poc_lsb   = ltr_lsbs,
        )

    # ── Public: post-encoding lifecycle ─────────────────────────────────

    def mark_unused_after_encoding(
        self,
        current_poc:  int,
        num_active_l0: int = 1,
        num_active_l1: int = 0,
    ) -> None:
        """
        Mark frames that are no longer needed as UNUSED for eviction.

        After encoding a frame at `current_poc`, the encoder knows which
        reference frames were actually used (the active entries in L0/L1).
        Frames not used by any future slice can be evicted from the DPB.

        Simple sliding-window policy (HEVC Main profile):
            Keep the `num_active_l0` most-recent preceding frames.
            Mark all older preceding frames as UNUSED.
            (Long-term references are never marked unused here.)

        Parameters
        ----------
        current_poc   : int — POC just encoded
        num_active_l0 : int — how many L0 refs to keep
        num_active_l1 : int — how many L1 refs to keep
        """
        strs = self.dpb.get_short_term_refs()
        neg_strs = [p for p in strs if p.poc < current_poc]  # sorted asc
        pos_strs = [p for p in strs if p.poc > current_poc]  # sorted asc

        # Keep the most recent `num_active_l0` preceding frames
        # (those are the last `num_active_l0` entries in neg_strs since asc)
        keep_neg = set(p.poc for p in neg_strs[-num_active_l0:]) \
                   if num_active_l0 > 0 else set()
        for p in neg_strs:
            if p.poc not in keep_neg:
                try:
                    self.dpb.mark_unused(p.poc)
                except KeyError:
                    pass  # already evicted

        # Keep most recent `num_active_l1` following frames
        keep_pos = set(p.poc for p in pos_strs[:num_active_l1]) \
                   if num_active_l1 > 0 else set()
        for p in pos_strs:
            if p.poc not in keep_pos:
                try:
                    self.dpb.mark_unused(p.poc)
                except KeyError:
                    pass

    def get_best_l0_ref(self, current_poc: int) -> PictureBuffer | None:
        """
        Convenience: return the single best L0 reference (idx=0) for ME.

        This is the frame that motion_estimation.py should search against
        for a standard P-slice CU when only one reference is needed.
        """
        l0 = self.build_l0(current_poc, slice_type="P", max_refs=1)
        return l0.best_ref.pic if l0.best_ref else None

    # ── Statistics / debug ───────────────────────────────────────────────

    def summary(self, current_poc: int, slice_type: str = "P") -> str:
        """Human-readable RPL summary for a given current POC."""
        rps = self.build_rps(current_poc, slice_type)
        lines = [
            f"RPL @ poc={current_poc} [{slice_type}-slice]",
            f"  L0 ({len(rps.l0)} entries): {rps.l0.pocs}",
        ]
        if slice_type == "B":
            lines.append(f"  L1 ({len(rps.l1)} entries): {rps.l1.pocs}")
        lines += [
            f"  delta_poc_neg={rps.delta_poc_neg}",
            f"  delta_poc_pos={rps.delta_poc_pos}",
            f"  LTR POC lsbs={rps.ltr_poc_lsb}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_entries(
    ordered:     list[PictureBuffer],
    current_poc: int,
    cap:         int,
) -> list[ReferenceEntry]:
    """
    Build a list of ReferenceEntry from an ordered list of PictureBuffers.

    Removes duplicates (same POC) and truncates to `cap` entries.
    """
    seen: set[int] = set()
    entries: list[ReferenceEntry] = []

    for pic in ordered:
        if len(entries) >= cap:
            break
        if pic.poc in seen:
            continue
        seen.add(pic.poc)
        entries.append(ReferenceEntry(
            pic=pic,
            ref_idx=len(entries),
            delta_poc=pic.poc - current_poc,
            is_long_term=(pic.ref_status == RefStatus.LONG_TERM),
        ))

    return entries