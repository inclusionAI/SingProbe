"""
Guardrail Dataset for multi-task learning

Supports two dataset types:
1. Safety: Query risk classification (8 categories, multi-label) + Response
   safety (binary), both broadcast onto Response tokens
2. Response_Hallu: Token-level hallucination detection
"""

import os
from typing import Dict, List, Optional
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import split_data_paths
from .utils import calculate_response_boundaries_with_padding, map_char_to_token_labels, map_spans_to_token_labels
from .converters import SafetyConverter, ResponseHalluConverter


class GuardrailDataset(Dataset):
    """
    Unified Guardrail Dataset

    Handles two types of datasets with a unified interface:
    - safety: Query risk classification (8 categories) + Response safety (binary),
              both broadcast onto Response tokens
    - response_hallu: Token-level hallucination detection

    Output format:
        - input_ids: [max_length]
        - attention_mask: [max_length]
        - labels: [max_length, 10]  # 8 Query labels + 2 Response labels
        - response_mask: [max_length]  # 1 for Response tokens, 0 otherwise
        - dataset_type: str
        - sample_id: str
    """

    # Dataset type to converter mapping
    CONVERTERS = {
        'safety': SafetyConverter,
        'response_hallu': ResponseHalluConverter
    }

    def __init__(
        self,
        data_path: str,
        dataset_type: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        ignore_index: int = -100,
        split_filter: Optional[str] = None,
        use_single_token_as_query: bool = False,
        use_hallu_assertion_mask: bool = False
    ):
        """
        Args:
            data_path: Path to the JSONL data file
            dataset_type: One of 'safety', 'response_hallu'
            tokenizer: Tokenizer with padding_side='left'
            max_length: Maximum sequence length
            ignore_index: Label value for ignored positions (default: -100)
            split_filter: For 'response_hallu' only -- route rows by their `split`
                field: 'train' keeps split in {None,'train','test'}, 'validation'
                keeps only 'validation', None keeps all (caller partitions). Passed
                through to ResponseHalluConverter; ignored by other converters.
            use_single_token_as_query: When True (safety dataset only) place the
                Query labels (dims 0-7) on a SINGLE token -- the Query's last
                token (query_end_pos) -- instead of broadcasting them across all
                Response tokens. Response tokens and Query prefix tokens are then
                -100 for dims 0-7. Response safety (dim 8) / hallucination (dim 9)
                are unaffected. When False, legacy broadcast behavior is kept.
            use_hallu_assertion_mask: When True (response_hallu only) compute a
                per-response-token POS "assertion mask" (True only on content
                words -- spaCy en_core_web_sm POS not in {PUNCT,ADP,DET,AUX,
                CCONJ,SCONJ,PRON,SPACE,PART}) and emit it as a `hallu_assertion_mask`
                [seq] bool tensor; the loss ANDs it into `hallu_mask` so function
                words become unsupervised for the Hallu task (dim 9 only; Safety /
                Query are unaffected). Off by default -- when False the mask is
                all-False and the loss ignores it, so runs are byte-identical to
                pre-feature behavior and require no spaCy dependency. Requires
                spacy + en_core_web_sm when True.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ignore_index = ignore_index
        self.dataset_type = dataset_type
        self.use_single_token_as_query = use_single_token_as_query
        self.use_hallu_assertion_mask = use_hallu_assertion_mask
        # Per-sample in-memory cache of the POS assertion mask (np.ndarray[bool]),
        # keyed by sample_id. Lives in each DataLoader worker process; skips
        # spaCy recompute across epochs. Only populated when the flag is on.
        self._assertion_mask_cache: dict = {}

        # Ensure tokenizer uses left padding
        if tokenizer.padding_side != 'left':
            print(f"Warning: tokenizer.padding_side is '{tokenizer.padding_side}', setting to 'left'")
            tokenizer.padding_side = 'left'

        # Load and convert data
        raw_data = self._load_data(data_path)
        converter_class = self.CONVERTERS.get(dataset_type)
        if converter_class is None:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        # response_hallu accepts a split_filter to self-route by the `split` field;
        # other converters ignore it (BaseConverter.__init__ takes no args).
        try:
            self.converter = converter_class(split_filter=split_filter)
        except TypeError:
            self.converter = converter_class()
        self.data = self.converter.convert_batch(raw_data)

        # Note: Logging of dataset size is handled by train.py for better clarity

    def _load_data(self, path: str) -> list:
        """Load data from one or more JSONL / JSON files and concatenate.

        ``path`` may be a single file path or a ';'-separated list of paths
        (``"a.jsonl;b.jsonl"``); each file is loaded in order and its rows
        appended, so multiple datasets of the same type are merged into one
        in-memory list. Files may freely mix JSONL (.jsonl) and JSON-array
        (.json) formats. A path that does not exist on disk is skipped with a
        warning, so one missing file in a multi-path list does not abort the
        load -- callers still get the rows from the files that do exist.

        Args:
            path: A single path or a ';'-separated multi-path string.

        Returns:
            Concatenated list of raw rows from every readable file.
        """
        data = []
        for sub_path in split_data_paths(path):
            if not os.path.exists(sub_path):
                print(f"Warning: data file not found, skipping: {sub_path}")
                continue
            data.extend(self._load_one_file(sub_path))
        return data

    def _load_one_file(self, path: str) -> list:
        """Load a single JSONL ('.jsonl') or JSON-array ('.json') file."""
        rows = []

        if path.endswith('.jsonl'):
            # JSONL format -- one JSON object per line
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        elif path.endswith('.json'):
            # JSON array format -- one file, one top-level list
            with open(path, 'r', encoding='utf-8') as f:
                rows = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        return rows

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample

        Returns:
            Dict with keys:
                - input_ids: [max_length]
                - attention_mask: [max_length]
                - labels: [max_length, 10]
                - response_mask: [max_length]
                - dataset_type: str
                - sample_id: str
        """
        item = self.data[idx]

        # Calculate boundaries (considering left padding and chat template)
        boundaries = calculate_response_boundaries_with_padding(
            self.tokenizer,
            item['messages'],
            self.max_length
        )

        # Generate labels [seq_len, 10]
        labels = self._generate_labels(
            item,
            boundaries['input_ids'],
            boundaries['query_end_pos'],
            boundaries['response_start_pos'],
            boundaries['attention_mask'],
            boundaries['has_response'],
            boundaries['padding_end'],
        )

        # Generate response_mask [seq_len]
        response_mask = self._generate_response_mask(
            boundaries['input_ids'],
            boundaries['attention_mask'],
            boundaries['response_start_pos'],
            boundaries['has_response']
        )

        # Generate hallu_assertion_mask [seq_len] (Hallu-only POS filter; all-False
        # unless use_hallu_assertion_mask is on AND this is a response_hallu sample).
        hallu_assertion_mask = self._generate_hallu_assertion_mask(
            item,
            boundaries['response_start_pos'],
            boundaries['attention_mask'],
            boundaries['has_response'],
        )

        return {
            'input_ids': boundaries['input_ids'],
            'attention_mask': boundaries['attention_mask'],
            'labels': labels,
            'response_mask': response_mask,
            'hallu_assertion_mask': hallu_assertion_mask,
            'dataset_type': item['dataset_type'],
            'sample_id': item['sample_id'],
            # Carry the tokenizer's pad id so the collator can left-pad
            # variable-length samples (max_length<=0 sentinel) correctly.
            'pad_token_id': self.tokenizer.pad_token_id,
        }

    def _generate_labels(
        self,
        item: Dict,
        input_ids: torch.Tensor,
        query_end_pos: int,
        response_start_pos: int,
        attention_mask: torch.Tensor,
        has_response: bool,
        padding_end: int = 0
    ) -> torch.Tensor:
        """
        Generate token-level labels [seq_len, 10]

        Label structure:
            - First 8 dimensions: Query risk categories (multi-label)
            - Dimension 8: Response safety (binary)
            - Dimension 9: Response hallucination (binary)

        Label assignment strategy:
            - Query labels (dims 0-7), 'safety' dataset:
                * use_single_token_as_query=False (legacy): broadcast the same
                  multi-hot on EVERY response token (Query tokens stay -100).
                * use_single_token_as_query=True: place the multi-hot on a SINGLE
                  token, the Query's last token (query_end_pos, i.e. the token
                  immediately before the Response). Every other position --
                  Response tokens and Query prefix tokens -- is -100 for dims 0-7.
            - Response tokens, 'safety' dataset:
                * dim 8 = response_safety label (always on Response tokens,
                  regardless of use_single_token_as_query)
                * dim 9 = -100 (no hallucination labels)
            - Response tokens, 'response_hallu' dataset:
                * dims 0-7 = -100 ; dim 8 = -100 ; dim 9 = token-level hallu labels
            - Padding tokens: -100

        IMPORTANT: use_single_token_as_query only relocates the QUERY labels
        (dims 0-7). Response safety (dim 8) and hallucination (dim 9) are
        unaffected and stay on the Response tokens.
        """
        seq_len = len(input_ids)
        labels = torch.full((seq_len, 10), self.ignore_index, dtype=torch.long)

        # ---- Query labels (dims 0-7): broadcast vs single-token placement ----
        if (
            item['dataset_type'] == 'safety'
            and self.use_single_token_as_query
            and 0 <= query_end_pos < seq_len
            # query_end_pos must be a real (non-padding) content token
            and attention_mask[query_end_pos] == 1
        ):
            # Single-token: put the Query multi-hot ONLY on the Query's last
            # token (query_end_pos). Response tokens and Query prefix tokens are
            # left -100 for dims 0-7. query_end_pos is the token right before the
            # Response starts (or the last content token when there IS no Response).
            query_labels = torch.tensor(item['query_labels'], dtype=torch.long)
            labels[query_end_pos, :8] = query_labels

        # Assign Response labels (vectorized over the response region)
        if has_response and response_start_pos < seq_len:
            # Determine the contiguous (non-padding) response region [start, end)
            attn_tail = attention_mask[response_start_pos:]
            # Number of leading non-padding tokens in the tail = response length
            resp_len = int(attn_tail.sum().item()) if attn_tail.numel() > 0 else 0
            # Guard against response_start_pos landing in padding (right-truncation edge case)
            if resp_len == 0 and response_start_pos < seq_len and attention_mask[response_start_pos] == 1:
                resp_len = 1
            resp_end = min(response_start_pos + resp_len, seq_len)

            if resp_end > response_start_pos:
                if item['dataset_type'] == 'safety':
                    # Query labels (dims 0-7): when use_single_token_as_query is
                    # False (legacy) broadcast the multi-hot on every Response
                    # token. When True, dims 0-7 were already placed on the Query
                    # last token above and the Response region stays -100.
                    if not self.use_single_token_as_query:
                        query_labels = torch.tensor(item['query_labels'], dtype=torch.long)
                        labels[response_start_pos:resp_end, :8] = query_labels
                    # Sample-level safety label broadcast across all response tokens
                    labels[response_start_pos:resp_end, 8] = item['response_safety']
                    # dim 9 remains -100 (no hallucination labels for safety dataset)

                elif item['dataset_type'] == 'response_hallu':
                    # dims 0-7 stay -100; dim 8 remains -100; dim 9 gets token-level
                    # hallucination labels
                    token_labels = None
                    if item.get('response_hallucination_tokens') is not None:
                        hallu_tokens = item['response_hallucination_tokens']
                        if isinstance(hallu_tokens, list) and len(hallu_tokens) > 0:
                            if isinstance(hallu_tokens[0], dict):
                                token_labels = [t.get('label', 0) for t in hallu_tokens]
                            else:
                                token_labels = list(hallu_tokens)

                    if token_labels is None:
                        # Compute on-the-fly from character-level annotations
                        token_labels = self._compute_hallucination_labels(
                            item, input_ids, response_start_pos
                        )

                    n = resp_end - response_start_pos
                    # Truncate or pad token_labels to the actual response length
                    if len(token_labels) >= n:
                        seg = token_labels[:n]
                    else:
                        seg = token_labels + [0] * (n - len(token_labels))
                    labels[response_start_pos:resp_end, 9] = torch.tensor(seg, dtype=torch.long)

        return labels

    def _compute_hallucination_labels(
        self,
        item: Dict,
        input_ids: torch.Tensor,
        response_start_pos: int
    ) -> List[int]:
        """
        Compute token-level hallucination labels from character-level annotations

        Args:
            item: Data item with metadata['annotations'] and metadata['response_text']
            input_ids: Tokenized sequence
            response_start_pos: Response start position

        Returns:
            List of labels for each token in the Response
        """
        annotations = item.get('metadata', {}).get('annotations', [])
        response_text = item.get('metadata', {}).get('response_text', '')

        # Get actual response length (excluding padding)
        response_mask = input_ids[response_start_pos:] != self.tokenizer.pad_token_id
        response_length = response_mask.sum().item()

        if not response_text:
            # No response text to map onto
            return [0] * int(response_length)

        # Detect annotation shape: the token-level format stores spans as
        # {"start", "end", ...} dicts (char offsets relative to response); the
        # legacy format stores {"index", "label", "span"} dicts. Route to the
        # matching char->token mapper. When NO span annotations are present
        # (sentence/response-level Hallucination dataset, which only carries a
        # whole-response "hallucinated"/"clean" label), fall through to the
        # response-level branch below instead of silently treating it as clean.
        spans_shaped = (
            isinstance(annotations, list)
            and len(annotations) > 0
            and isinstance(annotations[0], dict)
            and ('start' in annotations[0] or 'end' in annotations[0])
        )
        legacy_shaped = (
            isinstance(annotations, list)
            and len(annotations) > 0
            and isinstance(annotations[0], dict)
            and ({'index', 'label', 'span'} & set(annotations[0].keys()))
        )

        try:
            if spans_shaped:
                # New token-level format: map response-relative spans via the
                # tokenizer's offset_mapping (half-open [start,end) intersection).
                # Empty spans -> all zeros (clean row).
                token_labels, total_tokens = map_spans_to_token_labels(
                    self.tokenizer,
                    response_text,
                    annotations,
                    max_tokens=int(response_length)
                )
            elif legacy_shaped:
                # Legacy format: {"index": char_offset, "label", "span"}.
                token_labels, total_tokens = map_char_to_token_labels(
                    self.tokenizer,
                    response_text,
                    annotations,
                    max_tokens=int(response_length)
                )
            else:
                # No span annotations -> response-level labeling. The
                # sentence/response-level Hallucination dataset only has a whole-
                # response "hallucinated" | "clean" label (stored by the converter
                # as metadata['original_label']); mark the ENTIRE response 1 when
                # hallucinated, else all 0. A missing/unknown label defaults to 0
                # (clean) so the row still trains safely.
                original_label = item.get('metadata', {}).get('original_label')
                is_hallucinated = str(original_label).lower() == 'hallucinated'
                fill = 1 if is_hallucinated else 0
                token_labels, total_tokens = (
                    [fill] * int(response_length),
                    int(response_length),
                )

            # Log truncation info for debugging (only once per unique sample to avoid spam)
            if total_tokens > response_length:
                # Truncation occurred - some annotations at the end are lost
                # This is expected behavior when response exceeds max_length
                import hashlib
                sample_id = item.get('sample_id', 'unknown')
                # Use a simple flag to avoid repeated warnings for same sample
                cache_key = hashlib.md5(f"{sample_id}_{total_tokens}_{response_length}".encode()).hexdigest()[:8]
                if not hasattr(self, '_truncation_warned'):
                    self._truncation_warned = set()
                if cache_key not in self._truncation_warned:
                    self._truncation_warned.add(cache_key)
                    print(f"INFO: Response truncated for hallucination labels: {total_tokens} -> {response_length} tokens")
                    print(f"  Sample ID: {sample_id}")
                    print(f"  Annotations beyond truncation point are dropped")

            # Pad if needed (shouldn't happen with max_tokens, but handle edge cases)
            if len(token_labels) < int(response_length):
                token_labels.extend([0] * (int(response_length) - len(token_labels)))

            return token_labels

        except Exception as e:
            print(f"ERROR computing hallucination labels: {e}")
            print(f"  Sample ID: {item.get('sample_id', 'unknown')}")
            print(f"  Response text: {response_text[:100]}...")
            print(f"  Annotations: {annotations[:3]}...")
            # Return default labels on error
            return [0] * int(response_length)

    def _generate_hallu_assertion_mask(
        self,
        item: Dict,
        response_start_pos: int,
        attention_mask: torch.Tensor,
        has_response: bool,
    ) -> torch.Tensor:
        """Per-response-token content-word mask for the Hallu task, [seq_len] bool.

        True only at response tokens that are content words (POS not in
        POS_BLACKLIST, via spaCy en_core_web_sm). The loss ANDs this into
        `hallu_mask`, so function-word Response tokens become unsupervised for
        dim 9 only. Safety/Query are unaffected (the loss never reads this mask
        for dims 0-8).

        Returns an all-False [seq_len] tensor (no spaCy import, no work) when the
        flag is off, the sample is not response_hallu, or there is no response --
        so default-off runs are dependency-free and byte-identical.

        Args:
            item: Converted data item (must carry metadata['response_text'] for
                response_hallu samples).
            response_start_pos: First Response token index (on the left-padded grid).
            attention_mask: Attention mask [seq_len].
            has_response: Whether the sample has a Response region.

        Returns:
            torch.Tensor[bool, seq_len]: 1 where a Response token is a content word.
        """
        seq_len = int(attention_mask.shape[0])
        empty = torch.zeros(seq_len, dtype=torch.bool)

        # Default-off / non-hallu / no-response: no work, no spaCy import.
        if (
            not self.use_hallu_assertion_mask
            or item.get('dataset_type') != 'response_hallu'
            or not has_response
            or response_start_pos >= seq_len
        ):
            return empty

        response_text = item.get('metadata', {}).get('response_text', '')
        if not response_text:
            return empty

        sample_id = item.get('sample_id', '')
        # Per-sample cache: skip spaCy recompute across epochs within this worker.
        cached = self._assertion_mask_cache.get(sample_id)
        if cached is None:
            # Lazy import so default-off runs never import spaCy / the module.
            from .assertion_mask import build_assertion_mask_from_offsets
            try:
                enc = self.tokenizer(
                    response_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                offsets = enc.get('offset_mapping') or []
                cached = build_assertion_mask_from_offsets(response_text, offsets)
            except Exception as e:
                print(
                    f"ERROR computing hallu assertion mask: {e}\n"
                    f"  Sample ID: {sample_id}\n"
                    f"  Response text: {response_text[:100]}... -> using all-supervised"
                )
                # Fallback: supervise all response tokens (no POS filter).
                try:
                    cached = np.ones(len(offsets), dtype=bool)
                except Exception:
                    cached = np.ones(len(response_text.split()), dtype=bool)
            self._assertion_mask_cache[sample_id] = cached

        content_mask = np.asarray(cached, dtype=bool)

        # Surviving response token count after max_length truncation (mirror the
        # resp_len computation in _generate_labels so the mask stays aligned with
        # the dim-9 label window [response_start_pos, response_start_pos+resp_len)).
        attn_tail = attention_mask[response_start_pos:]
        resp_len = int(attn_tail.sum().item()) if attn_tail.numel() > 0 else 0
        # Edge-case guard mirroring _generate_labels (response_start_pos in padding).
        if resp_len == 0 and response_start_pos < seq_len and attention_mask[response_start_pos] == 1:
            resp_len = 1

        n = min(resp_len, content_mask.shape[0])
        if n <= 0:
            return empty

        out = torch.zeros(seq_len, dtype=torch.bool)
        out[response_start_pos:response_start_pos + n] = torch.from_numpy(
            content_mask[:n].astype(bool)
        )
        return out

    def _generate_response_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_start_pos: int,
        has_response: bool
    ) -> torch.Tensor:
        """
        Generate response mask indicating which tokens belong to Response

        Vectorized: a position is a Response token iff it is a non-padding token
        (attention_mask==1) AND its index is >= response_start_pos.

        Args:
            input_ids: Tokenized sequence [seq_len]
            attention_mask: Attention mask [seq_len]
            response_start_pos: Response start position
            has_response: Whether there's a Response

        Returns:
            response_mask: [seq_len], 1 for Response tokens, 0 otherwise
        """
        # Use the actual tokenized length (== len(input_ids)). This stays correct both
        # with a fixed max_length grid and with the max_length<=0 "no truncation"
        # sentinel, where input_ids is variable-length per sample.
        seq_len = int(attention_mask.shape[0])
        response_mask = torch.zeros(seq_len, dtype=torch.long)

        if not has_response or response_start_pos >= seq_len:
            return response_mask

        positions = torch.arange(seq_len)
        response_mask = (
            (attention_mask == 1) & (positions >= response_start_pos)
        ).long()

        return response_mask