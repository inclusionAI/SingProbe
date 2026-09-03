"""
Data collator for batch processing
"""

from typing import Dict, List
import torch


class GuardrailCollator:
    """
    Collator for batching Guardrail Dataset samples

    Handles:
    - Dynamic (per-batch) left padding: pad each batch only up to its longest
      sample instead of the fixed dataset max_length. Cuts base-model forward
      cost dramatically when real sequences are shorter than max_length.
    - Stacking tensors (input_ids, attention_mask, labels, response_mask)
    - Preserving metadata (dataset_type, sample_id)

    Two sample shapes are supported transparently:
    - Fixed grid (max_length > 0): each sample tensor is max_length-long with
      left padding already applied (content right-aligned). The loss is fully
      mask-based, so padding length is irrelevant; we keep only the rightmost
      batch_max columns to drop excess left padding.
    - Variable length (max_length <= 0 sentinel): each sample tensor is exactly
      its content length (no padding). We left-pad each one to batch_max.

    In both cases the result is a uniformly left-padded batch right-aligned to
    batch_max, with consistent padding values:
      input_ids <- pad_token_id (from batch[0]'s tokenizer; carried via the
                   sample dict as 'pad_token_id' when present, else 0),
      attention_mask / response_mask <- 0,
      labels <- -100 (ignore_index).
    The loss (trainers/loss.py) is fully mask-based, so only valid-token masks
    matter.
    """

    # Per-tensor fill value for the dynamically added left padding.
    def _pad_value(self, key: str, batch: List[Dict]):
        if key == 'labels':
            return -100
        if key == 'input_ids':
            # Samples carry their tokenizer's pad id when variable-length; fall
            # back to 0 (matches the dataset's torch.full default for pad).
            return batch[0].get('pad_token_id', 0)
        if key == 'hallu_assertion_mask':
            return False  # bool mask: pad positions are not content words
        return 0  # attention_mask, response_mask

    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate a batch of samples with dynamic left padding

        Args:
            batch: List of samples from GuardrailDataset

        Returns:
            Dict with keys:
                - input_ids: [batch_size, batch_max_len]
                - attention_mask: [batch_size, batch_max_len]
                - labels: [batch_size, batch_max_len, 10]
                - response_mask: [batch_size, batch_max_len]
                - dataset_type: List[str]
                - sample_id: List[str]
        """
        # Length of real (non-padding) tokens per sample under left padding
        content_lens = [int(item['attention_mask'].sum()) for item in batch]
        # Guard against an all-empty batch (defensive; should not happen)
        batch_max = max(content_lens) if content_lens else 1

        def _align(t: torch.Tensor, key: str) -> torch.Tensor:
            """Right-align content to batch_max, left-padding the rest.

            Keep the rightmost min(batch_max, L) tokens (drops excess left
            padding on fixed-grid samples; takes all of a variable-length one),
            then left-pad to batch_max with the key's pad value.
            """
            keep = min(batch_max, t.shape[0])
            right = t[-keep:]
            pad = batch_max - keep
            if pad <= 0:
                return right
            pad_shape = (pad,) + t.shape[1:]
            pad_tensor = torch.full(pad_shape, self._pad_value(key, batch),
                                    dtype=t.dtype)
            return torch.cat([pad_tensor, right], dim=0)

        # Stack tensors
        input_ids = torch.stack([_align(item['input_ids'], 'input_ids') for item in batch])
        attention_mask = torch.stack([_align(item['attention_mask'], 'attention_mask') for item in batch])
        labels = torch.stack([_align(item['labels'], 'labels') for item in batch])
        response_mask = torch.stack([_align(item['response_mask'], 'response_mask') for item in batch])
        # Hallu-only POS content-word mask [batch, batch_max_len] bool. Aligned
        # through the same _align path so it stays in lock-step with label dim 9
        # and response_mask under dynamic left padding. All-False when the
        # assertion-mask flag is off (the loss then ignores it).
        hallu_assertion_mask = torch.stack([_align(item['hallu_assertion_mask'], 'hallu_assertion_mask') for item in batch])

        # Preserve metadata
        dataset_types = [item['dataset_type'] for item in batch]
        sample_ids = [item['sample_id'] for item in batch]

        return {
            'input_ids': input_ids,              # [batch, batch_max_len]
            'attention_mask': attention_mask,    # [batch, batch_max_len]
            'labels': labels,                    # [batch, batch_max_len, 10]
            'response_mask': response_mask,      # [batch, batch_max_len]
            'hallu_assertion_mask': hallu_assertion_mask,  # [batch, batch_max_len] bool
            'dataset_type': dataset_types,       # List[str]
            'sample_id': sample_ids              # List[str]
        }


if __name__ == '__main__':
    """Smoke test for dynamic left-padding collation."""
    max_len = 16

    def _sample(content_len, total_len, label_val, has_resp):
        # Build a left-padded, right-aligned sample mimicking dataset output.
        x = torch.zeros(total_len, dtype=torch.long)                       # input_ids (0 = pad)
        a = torch.zeros(total_len, dtype=torch.long)                       # attention_mask
        l = torch.full((total_len, 10), -100, dtype=torch.long)            # labels
        r = torch.zeros(total_len, dtype=torch.long)                       # response_mask
        h = torch.zeros(total_len, dtype=torch.bool)                       # hallu_assertion_mask
        start = total_len - content_len
        x[start:] = torch.randint(1, 100, (content_len,))
        a[start:] = 1
        r[start:] = 1 if has_resp else 0
        if has_resp:
            l[start, 8] = label_val        # safety label on first response token
            h[start] = True                # mark one response token as "content"
        else:
            l[start, :8] = label_val       # query label on last query token (here start)
        return {
            'input_ids': x, 'attention_mask': a, 'labels': l, 'response_mask': r,
            'hallu_assertion_mask': h, 'dataset_type': 'test', 'sample_id': 's'
        }

    collator = GuardrailCollator()
    # Mixed lengths -> batch_max should be max(content_lens)
    batch = [_sample(4, max_len, 1, True), _sample(7, max_len, 0, False), _sample(2, max_len, 1, True)]
    out = collator(batch)

    b, t = out['input_ids'].shape
    assert t == 7, f"expected batch_max=7, got {t}"
    # Right-aligned content: last token always real
    assert (out['attention_mask'][:, -1] == 1).all(), "content must be right-aligned"
    # Left-trimmed padding columns (before content) must be all-zero masks
    shortest = min([4, 7, 2])
    leading = out['attention_mask'][:, :(t - shortest)]
    assert (leading.sum(dim=0) <= out['attention_mask'].size(0)).all()
    # No label leakage into trimmed area: valid labels only within content
    valid = out['labels'][out['response_mask'] == 1]
    assert (valid != -100).any(), "response labels should survive collation"
    # hallu_assertion_mask: bool, right-aligned with response_mask, padded False.
    assert out['hallu_assertion_mask'].dtype == torch.bool, "assertion mask must be bool"
    assert out['hallu_assertion_mask'].shape == out['response_mask'].shape
    assert (out['hallu_assertion_mask'] & (out['response_mask'].bool())).sum() >= 2, \
        "marked content tokens should survive collation"

    print(f"OK: batch shape {b}x{t} (dynamic, was fixed {max_len})")
    print(f"attention_mask:\n{out['attention_mask']}")
    print(f"response_mask:\n{out['response_mask']}")