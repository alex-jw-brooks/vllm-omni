# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Maps each KV slot to the saved CPU row holding its stage outputs."""

import logging
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias

import torch

from vllm_omni.core.prefix_cache.interface import OmniPrefixCacheUnmatchError, TensorName

logger = logging.getLogger(__name__)

# The ratio of referenced vs unreferenced blocks within a save before we collect
# the outputs into a new tensor so that we can free the original save.
SAVE_COMPACTION_RATIO = 0.1

# The outputs one save copied to the CPU, by tensor name.
SavedOutputs: TypeAlias = dict[TensorName, torch.Tensor]


class SavedSlice(NamedTuple):
    """A cache hit subspan (first row idx & num rows in the subspan) with the corresponding
    SavedOutputs. Usually the tokens one request computed in one engine step.
    """

    outputs: SavedOutputs | None  # None for slots no save has written
    first_row: int
    num_rows: int


class PrefixHit:
    """Wraps a request's prefix cache hit stored as its subspans (SavedSlices) in slot order.

    NOTE: This holds the SavedOutputs its subspans read from, so later saves that overwrite the
    hit's slots don't change it. This generally shouldn't happen, but could occur in some cases
    with async outputs under cache pressure.
    """

    def __init__(self, slices: list[SavedSlice]) -> None:
        self._slices = slices

    def all_slots_saved(self, key: TensorName) -> bool:
        """Whether every slot of the hit has a row saved for this key."""
        return all(s.outputs is not None and key in s.outputs for s in self._slices)

    def cat(self, key: TensorName, new_rows: torch.Tensor) -> torch.Tensor:
        """The hit's rows for this key in slot order, followed by new_rows.

        Every slot needs a row saved for the key; check all_slots_saved first.
        """
        rows = [s.outputs[key][s.first_row : s.first_row + s.num_rows] for s in self._slices]
        return torch.cat([*rows, new_rows])


@dataclass
class _SlotsSave:
    """One save's outputs and the index's bookkeeping for them."""

    outputs: SavedOutputs
    # The slots the save was stored with; slots[i] holds row i while it still points at the save.
    slots: torch.Tensor
    # How many slots hold the save. One no slot holds is dropped; hits keep their own reference.
    num_live_slots: int
    # True while its tensors may not hold their final values yet; compaction skips it until mark_final.
    pending: bool


class SlotIndex:
    """Used only on the engine thread, though the hits it returns can be read from any thread.

    A save stays alive while any slot or hit still refers to it. Note that a save's ID is just
    a monotonically increasing ID that we can delete to drop tensors.

    Recorded saves start pending, because their tensors may not hold their final values yet (e.g.,
    if still filling from an async copy), so nothing here reads them until they're passed to
    mark_final.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self._block_size = block_size
        # Each slot holds the id of the save that last wrote it (-1 for none) and its row in that save.
        self._save_of_slot = torch.full((num_blocks * block_size,), -1, dtype=torch.long)
        self._row_of_slot = torch.zeros(num_blocks * block_size, dtype=torch.long)
        self._saves: dict[int, _SlotsSave] = {}
        self._next_save_id = 0

    def _release_stale_saves(self, slots: torch.Tensor) -> list[int]:
        """Unmap reused slots from their saves and decrement those saves' reference counters.
        Returns the saves whose reference counts dropped, to be freed or compacted.
        """
        old_ids, counts = self._save_of_slot[slots].unique(return_counts=True)
        self._save_of_slot[slots] = -1
        stale_saves = []
        for old_id, count in zip(old_ids.tolist(), counts.tolist()):
            if old_id < 0:  # -1 indicates no save holds this slot
                continue
            self._saves[old_id].num_live_slots -= count
            stale_saves.append(old_id)
        return stale_saves

    def _should_compact_save(self, save: _SlotsSave) -> bool:
        """Indicates whether or not the current save should be compacted."""
        num_rows = len(save.slots)  # includes rows whose slots were reused since
        return (save.num_live_slots / num_rows) < SAVE_COMPACTION_RATIO

    def _free_or_compact_saves(self, save_ids: list[int]) -> None:
        """Free saves that no slot references, and compact saves that are mostly stale.
        For tensors to be compacted, we copy any referenced rows out to a new save
        and delete the original save, which frees the mostly unused tensors.
        This prevents us from holding stale outputs in memory for shared prompts, e.g.,
        system prompts, especially since the outputs refs used as is are typically pinned tensors.

        NOTE: we use a new save ID rather than modifying the old save in place because existing
        hits and unmerged batches still hold the old outputs and read them by their old rows.
        """
        for save_id in save_ids:
            # Compaction example: Let's say save_id is 17, and has the following slots:
            # [8, 9, 10, 4, 5] (remember a save pertains to an engine step, so if we have
            # multiple requests, slots won't necessarily be sequential).
            # Then, let's say saves 21 & 22 reuse slots 8, 9, 10, and 4 & compaction ratio is 0.3,
            save = self._saves[save_id]
            if save.num_live_slots > 0:
                # A pending save's tensors may not hold their final values yet, so don't compact it
                if save.pending or not self._should_compact_save(save):
                    continue
                logger.debug("Compacting prefix cache save %d", save_id)
                # Get the rows still pointing at the save. Continuing above, while recording 22
                # [21, 21, 21, -1, 17] == 17 -> [F, F, F, F, T]
                rows = (self._save_of_slot[save.slots] == save_id).nonzero().flatten()
                # For each key, copy the live rows out to a new tensor and link it to a new save
                outputs = {key: tensor.index_select(0, rows) for key, tensor in save.outputs.items()}
                self._add_save(outputs, save.slots[rows], pending=False)
            del self._saves[save_id]

    def _add_save(self, outputs: SavedOutputs, slots: torch.Tensor, *, pending: bool) -> int:
        """Store outputs as a new save, pointing slots[i] at row i of every tensor in outputs."""
        # Create a new save with the new outputs and slots
        save_id = self._next_save_id
        self._next_save_id += 1  # update the next save ID
        self._saves[save_id] = _SlotsSave(outputs, slots, num_live_slots=len(slots), pending=pending)
        self._save_of_slot[slots] = save_id
        self._row_of_slot[slots] = torch.arange(len(slots))
        return save_id

    def mark_final(self, final_ids: list[int]) -> None:
        """Mark recorded saves whose tensors now hold their final values, so compaction may read
        them, and compact the ones that went mostly stale while they were pending.
        """
        # A save whose slots were all overwritten while it was pending is already freed
        unfreed_ids = [save_id for save_id in final_ids if save_id in self._saves]
        for save_id in unfreed_ids:
            self._saves[save_id].pending = False
        self._free_or_compact_saves(unfreed_ids)

    def record(self, outputs: SavedOutputs, slots: torch.Tensor) -> int | None:
        """Record that slots[i]'s outputs are row i of every tensor in outputs. Returns the save's
        ID if anything new must be saved, which stays pending until passed to mark_final.

        Raises if slots repeat or are negative, or if a tensor doesn't have one row per slot.
        """
        # Example: Say we have block_size 2 and this step computed:
        #   - request A's tokens into slots 8-10
        #   - request B's into slots 4-5
        # Then, we would have slots = [8, 9, 10, 4, 5] and every tensor has 5 rows (since we
        # are mapping each token -> tensors, i.e., mapping 2D -> 3D).
        if not len(slots):
            return None
        if slots.min() < 0 or len(slots.unique()) != len(slots):
            raise OmniPrefixCacheUnmatchError("record SlotIndex received repeated or negative KV slots")
        # tensor to be fetched & slots need to be the same length for all tensors in prefix cache
        mismatched = [key for key, tensor in outputs.items() if tensor.shape[0] != len(slots)]
        if mismatched:
            raise OmniPrefixCacheUnmatchError(f"save has {len(slots)} slots but another row count for {mismatched}")

        # Update saves for blocks in the batch that are being reused.
        # e.g., if slots 8-10 were never written and slots 4-5 previously were referencing 12:
        #   old_ids = [-1, 12], counts = [3, 2]
        # This means that 4,5 will now refer to something else and as a result,
        # save 12 needs to have its reference count reduced by 2.
        stale_saves = self._release_stale_saves(slots)
        self._free_or_compact_saves(stale_saves)

        # Point the i-th slot at row i of this save, e.g. if the current save ID is 17:
        #   slot   8   9  10   4   5
        #   save  17  17  17  17  17
        #   row    0   1   2   3   4
        if not outputs:
            return None
        return self._add_save(outputs, slots, pending=True)

    def lookup(self, block_ids: list[int]) -> PrefixHit:
        """Given a set of block IDs, retrieve the corresponding PrefixHit."""
        # Example used below, continuing record's: a later step, save 20, wrote only slots 6-7,
        # with the rows 0-1, and now a request hits blocks [2, 3].
        if not len(block_ids):
            return PrefixHit([])

        # Given the block indices, lookup the KV slots that are owned by our block IDs.
        # e.g., blocks [2, 3] -> slots [4, 5, 6, 7],
        # since a given block b owns slots  [b * block_size, b * block_size + block_size - 1]
        blocks = torch.tensor(block_ids, dtype=torch.long)
        slots = (blocks[:, None] * self._block_size + torch.arange(self._block_size)).flatten()

        # Retrieve the save index where each slot maps to, as well as the row per slot
        # e.g. save_ids = [17, 17, 20, 20], rows = [3, 4, 0, 1]
        save_ids = self._save_of_slot[slots]
        rows = self._row_of_slot[slots]

        # Slot i starts a new slice unless it is the next row of the same save as slot i - 1.
        # e.g., new_slice = [True, False, True, False], so starts = [0, 2] and ends = [2, 4]
        new_slice = torch.ones(len(slots), dtype=torch.bool)
        # NOTE: We need to look for jumps in saves and rows, because we can have row jumps within
        # a save if we have successive requests, where the longer request is adding onto an already
        # cached prefix.
        new_slice[1:] = (save_ids[1:] != save_ids[:-1]) | (rows[1:] != rows[:-1] + 1)
        first_slots = new_slice.nonzero().flatten()
        starts = first_slots.tolist()
        ends = [*starts[1:], len(slots)]

        # Each slice reads from its first slot's save and mapped rows, e.g. save 17 rows 3-4 and save 20 rows 0-1.
        slices = zip(save_ids[first_slots].tolist(), rows[first_slots].tolist(), starts, ends)
        return PrefixHit(
            [
                SavedSlice(self._saves[save_id].outputs if save_id >= 0 else None, row, end - start)
                for save_id, row, start, end in slices
            ]
        )
