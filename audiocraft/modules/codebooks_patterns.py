# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from dataclasses import dataclass
from functools import lru_cache
import logging
import typing as tp

from abc import ABC, abstractmethod
import torch

LayoutCoord = namedtuple('LayoutCoord', ['t', 'q'])  # (timestep, codebook index)
PatternLayout = tp.List[tp.List[LayoutCoord]]  # Sequence of coordinates
logger = logging.getLogger(__name__)


@dataclass
class Pattern:
    """Base implementation of a pattern over a sequence with multiple codebooks.

    The codebook pattern consists in a layout, defining for each sequence step
    the list of coordinates of each codebook timestep in the resulting interleaved sequence.
    The first item of the pattern is always an empty list in order to properly insert a special token
    to start with. For convenience, we also keep track of ``n_q`` the number of codebooks used for the pattern
    and ``timesteps`` the number of timesteps corresponding to the original sequence.

    The pattern provides convenient methods to build and revert interleaved sequences from it:
    ``build_pattern_sequence`` maps a given a dense input tensor of multi-codebook sequence from [B, K, T]
        to the interleaved sequence of shape [B, K, S] applying the pattern, with S being the batch size,
        K being the number of codebooks, T the number of original timesteps and S the number of sequence steps
        for the output sequence. The unfilled positions are replaced with a special token and the built sequence
        is returned along with a mask indicating valid tokens.
    ``revert_pattern_sequence`` maps back an interleaved sequence of shape [B, K, S] to the original alignment
        of codebooks across timesteps to an output tensor of shape [B, K, T], using again a special token and a mask
        to fill and specify invalid positions if needed.
    See the dedicated methods for more details.
    """
    # Pattern layout, for each sequence step, we have a list of coordinates
    # corresponding to the original codebook timestep and position.
    # The first list is always an empty list in order to properly insert
    # a special token to start with.
    layout: PatternLayout
    timesteps: int
    n_q: int

    def __post_init__(self):
        assert len(self.layout) > 0
        assert self.layout[0] == []
        self._validate_layout()
        self._build_reverted_sequence_scatter_indexes = lru_cache(100)(self._build_reverted_sequence_scatter_indexes)
        self._build_pattern_sequence_scatter_indexes = lru_cache(100)(self._build_pattern_sequence_scatter_indexes)
        logger.info("New pattern, time steps: %d, sequence steps: %d", self.timesteps, len(self.layout))

    def _validate_layout(self):
        """Runs checks on the layout to ensure a valid pattern is defined.
        A pattern is considered invalid if:
            - Multiple timesteps for a same codebook are defined in the same sequence step
            - The timesteps for a given codebook are not in ascending order as we advance in the sequence
              (this would mean that we have future timesteps before past timesteps).
        """
        q_timesteps = {q: 0 for q in range(self.n_q)}
        for s, seq_coords in enumerate(self.layout):
            if len(seq_coords) > 0:
                qs = set()
                for coord in seq_coords:
                    qs.add(coord.q)
                    last_q_timestep = q_timesteps[coord.q]
                    assert coord.t >= last_q_timestep, \
                        f"Past timesteps are found in the sequence for codebook = {coord.q} at step {s}"
                    q_timesteps[coord.q] = coord.t
                # each sequence step contains at max 1 coordinate per codebook
                assert len(qs) == len(seq_coords), \
                    f"Multiple entries for a same codebook are found at step {s}"

    @property
    def num_sequence_steps(self):
        return len(self.layout) - 1

    @property
    def max_delay(self):
        max_t_in_seq_coords = 0
        for seq_coords in self.layout[1:]:
            for coords in seq_coords:
                max_t_in_seq_coords = max(max_t_in_seq_coords, coords.t + 1)
        return max_t_in_seq_coords - self.timesteps

    @property
    def valid_layout(self):
        valid_step = len(self.layout) - self.max_delay
        return self.layout[:valid_step]

    def get_sequence_coords_with_timestep(self, t: int, q: tp.Optional[int] = None):
        """Get codebook coordinates in the layout that corresponds to the specified timestep t
        and optionally to the codebook q. Coordinates are returned as a tuple with the sequence step
        and the actual codebook coordinates.
        """
        assert t <= self.timesteps, "provided timesteps is greater than the pattern's number of timesteps"
        if q is not None:
            assert q <= self.n_q, "provided number of codebooks is greater than the pattern's number of codebooks"
        coords = []
        for s, seq_codes in enumerate(self.layout):
            for code in seq_codes:
                if code.t == t and (q is None or code.q == q):
                    coords.append((s, code))
        return coords

    def get_steps_with_timestep(self, t: int, q: tp.Optional[int] = None) -> tp.List[int]:
        return [step for step, coords in self.get_sequence_coords_with_timestep(t, q)]

    def get_first_step_with_timesteps(self, t: int, q: tp.Optional[int] = None) -> tp.Optional[int]:
        steps_with_timesteps = self.get_steps_with_timestep(t, q)
        return steps_with_timesteps[0] if len(steps_with_timesteps) > 0 else None

    def _build_pattern_sequence_scatter_indexes(self, timesteps: int, n_q: int, keep_only_valid_steps: bool,
                                                device: tp.Union[torch.device, str] = 'cpu'):
        """Build scatter indexes corresponding to the pattern, up to the provided sequence_steps.

        Args:
            timesteps (int): Maximum number of timesteps steps to consider.
            keep_only_valid_steps (bool): Restrict the pattern layout to match only valid steps.
            device (torch.device or str): Device for created tensors.
        Returns:
            indexes (torch.Tensor): Indexes corresponding to the sequence, of shape [K, S].
            mask (torch.Tensor): Mask corresponding to indexes that matches valid indexes, of shape [K, S].
        """
        assert n_q == self.n_q, f"invalid number of codebooks for the sequence and the pattern: {n_q} != {self.n_q}"
        assert timesteps <= self.timesteps, "invalid number of timesteps used to build the sequence from the pattern"
        # use the proper layout based on whether we limit ourselves to valid steps only or not,
        # note that using the valid_layout will result in a truncated sequence up to the valid steps
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        # single item indexing being super slow with pytorch vs. numpy, so we use numpy here
        indexes = torch.zeros(n_q, len(ref_layout), dtype=torch.long).numpy()
        mask = torch.zeros(n_q, len(ref_layout), dtype=torch.bool).numpy()
        # fill indexes with last sequence step value that will correspond to our special token
        # the last value is n_q * timesteps as we have flattened z and append special token as the last token
        # which will correspond to the index: n_q * timesteps
        indexes[:] = n_q * timesteps
        # iterate over the pattern and fill scattered indexes and mask
        for s, sequence_coords in enumerate(ref_layout):
            for coords in sequence_coords:
                if coords.t < timesteps:
                    indexes[coords.q, s] = coords.t + coords.q * timesteps
                    mask[coords.q, s] = 1
        indexes = torch.from_numpy(indexes).to(device)
        mask = torch.from_numpy(mask).to(device)
        return indexes, mask

    def build_pattern_sequence(self, z: torch.Tensor, special_token: int, keep_only_valid_steps: bool = False):
        """Build sequence corresponding to the pattern from the input tensor z.
        The sequence is built using up to sequence_steps if specified, and non-pattern
        coordinates are filled with the special token.

        Args:
            z (torch.Tensor): Input tensor of multi-codebooks sequence, of shape [B, K, T].
            special_token (int): Special token used to fill non-pattern coordinates in the new sequence.
            keep_only_valid_steps (bool): Build a sequence from the pattern up to valid (= fully defined) steps.
                Steps that are beyond valid steps will be replaced by the special_token in that case.
        Returns:
            values (torch.Tensor): Interleaved sequence matching the pattern, of shape [B, K, S] with S
                corresponding either to the sequence_steps if provided, otherwise to the length of the pattern.
            indexes (torch.Tensor): Indexes corresponding to the interleaved sequence, of shape [K, S].
            mask (torch.Tensor): Mask corresponding to indexes that matches valid indexes of shape [K, S].
        """
        B, K, T = z.shape
        indexes, mask = self._build_pattern_sequence_scatter_indexes(
            T, K, keep_only_valid_steps=keep_only_valid_steps, device=str(z.device)
        )
        z = z.view(B, -1)
        # we append the special token as the last index of our flattened z tensor
        z = torch.cat([z, torch.zeros_like(z[:, :1]) + special_token], dim=1)
        values = z[:, indexes.view(-1)]
        values = values.view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def _build_reverted_sequence_scatter_indexes(self, sequence_steps: int, n_q: int,
                                                 keep_only_valid_steps: bool = False,
                                                 is_model_output: bool = False,
                                                 device: tp.Union[torch.device, str] = 'cpu'):
        """Builds scatter indexes required to retrieve the original multi-codebook sequence
        from interleaving pattern.

        Args:
            sequence_steps (int): Sequence steps.
            n_q (int): Number of codebooks.
            keep_only_valid_steps (bool): Build a sequence from the pattern up to valid (= fully defined) steps.
                Steps that are beyond valid steps will be replaced by the special_token in that case.
            is_model_output (bool): Whether to keep the sequence item corresponding to initial special token or not.
            device (torch.device or str): Device for created tensors.
        Returns:
            indexes (torch.Tensor): Indexes for reconstructing the output, of shape [K, T].
            mask (torch.Tensor): Mask corresponding to indexes that matches valid indexes of shape [K, T].
        """
        ref_layout = self.valid_layout if keep_only_valid_steps else self.layout
        # TODO(jade): Do we want to further truncate to only valid timesteps here as well?
        timesteps = self.timesteps
        assert n_q == self.n_q, f"invalid number of codebooks for the sequence and the pattern: {n_q} != {self.n_q}"
        assert sequence_steps <= len(ref_layout), \
            f"sequence to revert is longer than the defined pattern: {sequence_steps} > {len(ref_layout)}"

        # ensure we take the appropriate indexes to keep the model output from the first special token as well
        if is_model_output:
            ref_layout = ref_layout[1:]

        # single item indexing being super slow with pytorch vs. numpy, so we use numpy here
        indexes = torch.zeros(n_q, timesteps, dtype=torch.long).numpy()
        mask = torch.zeros(n_q, timesteps, dtype=torch.bool).numpy()
        # fill indexes with last sequence step value that will correspond to our special token
        indexes[:] = n_q * sequence_steps
        for s, sequence_codes in enumerate(ref_layout):
            if s < sequence_steps:
                for code in sequence_codes:
                    if code.t < timesteps:
                        indexes[code.q, code.t] = s + code.q * sequence_steps
                        mask[code.q, code.t] = 1
        indexes = torch.from_numpy(indexes).to(device)
        mask = torch.from_numpy(mask).to(device)
        return indexes, mask

    def revert_pattern_sequence(self, s: torch.Tensor, special_token: int, keep_only_valid_steps: bool = False):
        """Revert a sequence built from the pattern back to the original multi-codebook sequence without interleaving.
        The sequence is reverted using up to timesteps if specified, and non-pattern coordinates
        are filled with the special token.

        Args:
            s (torch.Tensor): Interleaved sequence tensor obtained from the pattern, of shape [B, K, S].
            special_token (int or float): Special token used to fill non-pattern coordinates in the new sequence.
        Returns:
            values (torch.Tensor): Interleaved sequence matching the pattern, of shape [B, K, T] with T
                corresponding either to the timesteps if provided, or the total timesteps in pattern otherwise.
            indexes (torch.Tensor): Indexes corresponding to the interleaved sequence, of shape [K, T].
            mask (torch.Tensor): Mask corresponding to indexes that matches valid indexes of shape [K, T].
        """
        B, K, S = s.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=False, device=str(s.device)
        )
        s = s.view(B, -1)
        # we append the special token as the last index of our flattened z tensor
        s = torch.cat([s, torch.zeros_like(s[:, :1]) + special_token], dim=1)
        values = s[:, indexes.view(-1)]
        values = values.view(B, K, indexes.shape[-1])
        return values, indexes, mask

    def revert_pattern_logits(self, logits: torch.Tensor, special_token: float, keep_only_valid_steps: bool = False):
        """Revert model logits obtained on a sequence built from the pattern
        back to a tensor matching the original sequence.

        This method is similar to ``revert_pattern_sequence`` with the following specificities:
        1. It is designed to work with the extra cardinality dimension
        2. We return the logits for the first sequence item that matches the special_token and
        which matching target in the original sequence is the first item of the sequence,
        while we skip the last logits as there is no matching target
        """
        B, card, K, S = logits.shape
        indexes, mask = self._build_reverted_sequence_scatter_indexes(
            S, K, keep_only_valid_steps, is_model_output=True, device=logits.device
        )
        logits = logits.reshape(B, card, -1)
        # we append the special token as the last index of our flattened z tensor
        logits = torch.cat([logits, torch.zeros_like(logits[:, :, :1]) + special_token], dim=-1)  # [B, card, K x S]
        values = logits[:, :, indexes.view(-1)]
        values = values.view(B, card, K, indexes.shape[-1])
        return values, indexes, mask


class CodebooksPatternProvider(ABC):
    """Abstraction around providing pattern for interleaving codebooks.

    The CodebooksPatternProvider abstraction allows to implement various strategies to
    define interleaving pattern of sequences composed of multiple codebooks. For a given
    number of codebooks `n_q`, the pattern provider can generate a specified pattern
    corresponding to a sequence of `T` timesteps with `n_q` parallel codebooks. This pattern
    can be used to construct a new sequence from the original codes respecting the specified
    pattern. The pattern is defined as a list of list of code coordinates, code coordinate
    being a tuple with the original timestep and codebook to build the new sequence.
    Note that all patterns must start with an empty list that is then used to insert a first
    sequence step of special tokens in the newly generated sequence.

    Args:
        n_q (int): number of codebooks.
        cached (bool): if True, patterns for a given length are cached. In general
            that should be true for efficiency reason to avoid synchronization points.
    """
    def __init__(self, n_q: int, cached: bool = True):
        assert n_q > 0
        self.n_q = n_q
        self.get_pattern = lru_cache(100)(self.get_pattern)  # type: ignore

    @abstractmethod
    def get_pattern(self, timesteps: int) -> Pattern:
        """Builds pattern with specific interleaving between codebooks.

        Args:
            timesteps (int): Total number of timesteps.
        """
        raise NotImplementedError()


class DelayedPatternProvider(CodebooksPatternProvider):
    """Provider for delayed pattern across delayed codebooks.
    Codebooks are delayed in the sequence and sequence steps will contain codebooks
    from different timesteps.

    Example:
        Taking timesteps=4 and n_q=3, delays=None, the multi-codebook sequence:
        [[1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4]]
        The resulting sequence obtained from the returned pattern is:
        [[S, 1, 2, 3, 4],
        [S, S, 1, 2, 3],
        [S, S, S, 1, 2]]
        (with S being a special token)

    Args:
        n_q (int): Number of codebooks.
        delays (list of int, optional): Delay for each of the codebooks.
            If delays not defined, each codebook is delayed by 1 compared to the previous one.
        flatten_first (int): Flatten the first N timesteps.
        empty_initial (int): Prepend with N empty list of coordinates.
    """
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None,
                 flatten_first: int = 0, empty_initial: int = 0):
        super().__init__(n_q)
        if delays is None:
            delays = list(range(n_q))
        self.delays = delays
        self.flatten_first = flatten_first
        self.empty_initial = empty_initial
        assert len(self.delays) == self.n_q
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        max_delay = max(self.delays)
        if self.empty_initial:
            out += [[] for _ in range(self.empty_initial)]
        if self.flatten_first:
            for t in range(min(timesteps, self.flatten_first)):
                for q in range(self.n_q):
                    out.append([LayoutCoord(t, q)])
        for t in range(self.flatten_first, timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= self.flatten_first:
                    v.append(LayoutCoord(t_for_q, q))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class ParallelPatternProvider(DelayedPatternProvider):
    """Provider for parallel pattern across codebooks.
    This pattern provider is a special case of the delayed pattern with actually no delay,
    hence delays=repeat(0, n_q).

    Args:
        n_q (int): Number of codebooks.
    """
    def __init__(self, n_q: int):
        super().__init__(n_q, [0] * n_q)


class UnrolledPatternProvider(CodebooksPatternProvider):
    """Provider for unrolling codebooks pattern.
    This pattern provider enables to represent the codebook flattened completely or only to some extend
    while also specifying a given delay between the flattened codebooks representation, allowing to
    unroll the codebooks in the sequence.

    Example:
        1. Flattening of the codebooks.
        By default, the pattern provider will fully flatten the codebooks such as flattening=range(n_q),
        taking n_q = 3 and timesteps = 4:
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]
        will result into:
        [[S, S, 1, S, S, 2, S, S, 3, S, S, 4],
         [S, 1, S, S, 2, S, S, 3, S, S, 4, S],
         [1, S, S, 2, S, S, 3, S, S, 4, S, S]]
        2. Partial flattening of the codebooks. The ``flattening`` parameter allows to specify the inner step
        for each of the codebook, allowing to define which codebook to flatten (or keep in parallel), for example
        taking n_q = 3, timesteps = 4 and flattening = [0, 1, 1]:
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]
        will result into:
        [[S, 1, S, S, 2, S, S, 3, S, S, 4, S],
         [S, 1, S, S, 2, S, S, 3, S, S, 4, S],
         [1, S, S, 2, S, S, 3, S, S, 4, S, S]]
        3. Flattening with delay. The ``delay`` parameter allows to further unroll the sequence of codebooks
        allowing to specify the delay per codebook. Note that the delay between codebooks flattened to the
        same inner timestep should be coherent. For example, taking n_q = 3, timesteps = 4, flattening = [0, 1, 1]
        and delays = [0, 3, 3]:
        [[1, 2, 3, 4],
         [1, 2, 3, 4],
         [1, 2, 3, 4]]
        will result into:
        [[S, S, S, 1, S, 2, S, 3, S, 4],
         [S, S, S, 1, S, 2, S, 3, S, 4],
         [1, 2, 3, S, 4, S, 5, S, 6, S]]

    Args:
        n_q (int): Number of codebooks.
        flattening (list of int, optional): Flattening schema over the codebooks. If not defined,
            the codebooks will be flattened to 1 codebook per step, meaning that the sequence will
            have n_q extra steps for each timestep.
        delays (list of int, optional): Delay for each of the codebooks. If not defined,
            no delay is added and therefore will default to [0] * ``n_q``.
            Note that two codebooks that will be flattened to the same inner step
            should have the same delay, otherwise the pattern is considered as invalid.
    """
    FlattenedCodebook = namedtuple('FlattenedCodebook', ['codebooks', 'delay'])

    def __init__(self, n_q: int, flattening: tp.Optional[tp.List[int]] = None,
                 delays: tp.Optional[tp.List[int]] = None):
        super().__init__(n_q)
        if flattening is None:
            flattening = list(range(n_q))
        if delays is None:
            delays = [0] * n_q
        assert len(flattening) == n_q
        assert len(delays) == n_q
        assert sorted(flattening) == flattening
        assert sorted(delays) == delays
        self._flattened_codebooks = self._build_flattened_codebooks(delays, flattening)
        self.max_delay = max(delays)

    def _build_flattened_codebooks(self, delays: tp.List[int], flattening: tp.List[int]):
        """Build a flattened codebooks representation as a dictionary of inner step
        and the actual codebook indices corresponding to the flattened codebook. For convenience, we
        also store the delay associated to the flattened codebook to avoid maintaining an extra mapping.
        """
        flattened_codebooks: dict = {}
        for q, (inner_step, delay) in enumerate(zip(flattening, delays)):
            if inner_step not in flattened_codebooks:
                flat_codebook = UnrolledPatternProvider.FlattenedCodebook(codebooks=[q], delay=delay)
            else:
                flat_codebook = flattened_codebooks[inner_step]
                assert flat_codebook.delay == delay, (
                    "Delay and flattening between codebooks is inconsistent: ",
                    "two codebooks flattened to the same position should have the same delay."
                )
                flat_codebook.codebooks.append(q)
            flattened_codebooks[inner_step] = flat_codebook
        return flattened_codebooks

    @property
    def _num_inner_steps(self):
        """Number of inner steps to unroll between timesteps in order to flatten the codebooks.
        """
        return max([inner_step for inner_step in self._flattened_codebooks.keys()]) + 1

    def num_virtual_steps(self, timesteps: int) -> int:
        return timesteps * self._num_inner_steps + 1

    def get_pattern(self, timesteps: int) -> Pattern:
        """Builds pattern for delay across codebooks.

        Args:
            timesteps (int): Total number of timesteps.
        """
        # the PatternLayout is built as a tuple of sequence position and list of coordinates
        # so that it can be reordered properly given the required delay between codebooks of given timesteps
        indexed_out: list = [(-1, [])]
        max_timesteps = timesteps + self.max_delay
        for t in range(max_timesteps):
            # for each timestep, we unroll the flattened codebooks,
            # emitting the sequence step with the corresponding delay
            for step in range(self._num_inner_steps):
                if step in self._flattened_codebooks:
                    # we have codebooks at this virtual step to emit
                    step_codebooks = self._flattened_codebooks[step]
                    t_for_q = t + step_codebooks.delay
                    coords = [LayoutCoord(t, q) for q in step_codebooks.codebooks]
                    if t_for_q < max_timesteps and t < max_timesteps:
                        indexed_out.append((t_for_q, coords))
                else:
                    # there is no codebook in this virtual step so we emit an empty list
                    indexed_out.append((t, []))
        out = [coords for _, coords in sorted(indexed_out)]
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class CoarseFirstPattern(CodebooksPatternProvider):
    """First generates all the codebooks #1 (e.g. coarser), then the remaining ones,
    potentially with delays.

    ..Warning:: You must always generate the full training duration at test time, for instance,
        30 seconds, as otherwise, the fine codebooks will start being generated in an unexpected
        location. This is due to the non causality of the remaining codebooks with respect to
        the first ones.

    Args:
        n_q (int): Number of codebooks.
        delays (list of int, optional): Delay for each of the codebooks.
            If delays not defined, each codebook is delayed by 1 compared to the previous one.
    """
    def __init__(self, n_q: int, delays: tp.Optional[tp.List[int]] = None):
        super().__init__(n_q)
        if delays is None:
            delays = [0] * (n_q - 1)
        self.delays = delays
        assert len(self.delays) == self.n_q - 1
        assert sorted(self.delays) == self.delays

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        for t in range(timesteps):
            out.append([LayoutCoord(t, 0)])
        max_delay = max(self.delays)
        for t in range(timesteps + max_delay):
            v = []
            for q, delay in enumerate(self.delays):
                t_for_q = t - delay
                if t_for_q >= 0:
                    v.append(LayoutCoord(t_for_q, q + 1))
            out.append(v)
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)


class MusicLMPattern(CodebooksPatternProvider):
    """Almost MusicLM style pattern. This is equivalent to full flattening
    but in a different order.

    Args:
        n_q (int): Number of codebooks.
        group_by (int): Number of codebooks to group together.
    """
    def __init__(self, n_q: int, group_by: int = 2):
        super().__init__(n_q)
        self.group_by = group_by

    def get_pattern(self, timesteps: int) -> Pattern:
        out: PatternLayout = [[]]
        for offset in range(0, self.n_q, self.group_by):
            for t in range(timesteps):
                for q in range(offset, offset + self.group_by):
                    out.append([LayoutCoord(t, q)])
        return Pattern(out, n_q=self.n_q, timesteps=timesteps)
