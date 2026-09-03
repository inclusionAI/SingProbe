"""
Converter for Response_Hallu dataset (token-level hallucination format)

Dataset: path/to/token_level_hallucination_dataset.jsonl
Format: JSON Lines, each row:
    {
        "id": str,
        "dataset": str,                 # source dataset
        "split": None | "train" | "validation" | "test",
        "prompt": str,
        "context": str | null,
        "question": str | null,
        "response": str,               # span offsets are relative to this string
        "label": "hallucinated" | "clean",
        "hallu_class": "faithfulness" | "factuality" | "mixed",
        "spans": [{"start": int, "end": int, "text": str, "type"?: str}, ...],
        "confidence": "high" | "adjudicated" | "default",
        ...
    }

Task: Token-level hallucination detection
Labels: a response token is hallucination (1) iff its char range intersects any
span's [start, end); clean rows have spans=[] -> all-zero labels. The actual
char->token mapping happens on-the-fly in GuardrailDataset (it needs the
sequence's response boundary), so the converter only normalizes into the
standard internal format and preserves the spans + response_text.

A single file may serve both train and val: pass split_filter to route rows by
their `split` field without re-reading the file.
"""

from typing import List, Optional
from .base import BaseConverter


class ResponseHalluConverter(BaseConverter):
    """
    Convert token-level hallucination data to the standard internal format.

    Key features:
    - Query + Response messages (chat format) -- boundary detection unchanged.
    - Token-level hallucination labels span-preserved for char->token mapping.
    - No Query risk categories (all zeros).
    - No Response safety labels (ignored).
    - Optional split_filter routes rows by their `split` field so one file can
      build a train view or a validation view.

    split_filter:
        None           -> keep all rows (caller partitions by item['metadata']['split'])
        'train'        -> keep split in {None, 'train'} ('validation' and 'test' held out)
        'validation'   -> keep only split == 'validation'
        'test'         -> keep only split == 'test'   (reserved for offline eval)

    Note: 'test' is NEVER used by train or val -- it is held out for offline
    evaluation only. Train and val are guaranteed disjoint from test.
    """

    def __init__(self, split_filter: Optional[str] = None):
        self.split_filter = split_filter

    def _keep(self, split):
        if self.split_filter is None:
            return True
        if self.split_filter == 'validation':
            return split == 'validation'
        if self.split_filter == 'train':
            return split in (None, 'train')
        if self.split_filter == 'test':
            return split == 'test'
        return True

    def convert(self, raw_data: dict) -> dict:
        """
        Convert a single raw token-level hallucination row to standard format.

        Args:
            raw_data: row from token_level_hallucination_dataset.jsonl

        Returns:
            Standard format data with spans preserved for later char->token
            mapping in GuardrailDataset.
        """
        prompt = raw_data.get('prompt', '')
        response = raw_data.get('response', '')

        # Generate unique sample ID
        sample_id = raw_data.get('id', f"response_hallu_{hash(prompt)}")

        # Query labels: no risk categories for the hallucination dataset
        query_labels = [0] * 8

        # spans are response-relative char offsets; preserve verbatim for the
        # Dataset's on-the-fly char->token mapping.
        spans = raw_data.get('spans', []) or []

        return {
            'sample_id': sample_id,
            'dataset_type': 'response_hallu',
            'messages': [
                {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': response}
            ],
            'query_labels': query_labels,
            'response_safety': None,  # No safety labels
            'response_hallucination_tokens': None,  # Computed on-the-fly in Dataset
            'metadata': {
                'source_file': 'token_level_hallucination_dataset.jsonl',
                'source_dataset': raw_data.get('dataset'),
                'original_label': raw_data.get('label'),  # 'hallucinated' | 'clean'
                'hallu_class': raw_data.get('hallu_class'),
                'confidence': raw_data.get('confidence'),
                'has_response': True,
                # Span-shaped annotations (new format). Kept under the existing
                # 'annotations' key so GuardrailDataset._compute_hallucination_labels
                # can detect the shape and route to the span mapper.
                'annotations': spans,
                'response_text': response,  # spans are relative to this
                'split': raw_data.get('split'),  # train.py carves train/val by this
            }
        }

    def convert_batch(self, raw_data_list: List[dict]) -> List[dict]:
        """
        Convert a batch of raw rows, applying split_filter if set.

        Args:
            raw_data_list: List of raw token-level rows

        Returns:
            List of standard-format items whose `split` passes split_filter.
        """
        if self.split_filter is None:
            return [self.convert(item) for item in raw_data_list]
        return [
            self.convert(item) for item in raw_data_list
            if self._keep(item.get('split'))
        ]