"""
Converter for the unified Safety dataset

Dataset: path/to/safety_both.jsonl
Format: {
    "Query": str,
    "Response": str,
    "Query_Label": str,     # A-H characters concatenated w/o separator, e.g. "B", "BF", "ABC"
    "Response_Label": str   # "Safe" | "Unsafe"
}

This dataset unifies the two former datasets (query_safety + response_safety).
Both labels are produced from the same sample:
- Query_Label: 8-category risk classification (multi-label), broadcast onto all
  Response tokens in _generate_labels (NOT on the query last token anymore).
- Response_Label: binary safety classification (safe=0 / unsafe=1), also on all
  Response tokens.

Note: Query_Label is a concatenation of characters (e.g. "BF"), NOT comma
separated like the former risk_category. Parse it character-by-character.
"""

from typing import List
from .base import BaseConverter


class SafetyConverter(BaseConverter):
    """
    Convert the unified Safety dataset to standard format

    Key features:
    - Query + Response (both present)
    - Query_Label: 8-category multi-hot (parsed char-by-char)
    - Response_Label: binary safety (Safe=0, Unsafe=1)
    - Labels are sample-level and broadcast across Response tokens in the Dataset
    """

    # 7 risk categories + 1 Safe (mutually exclusive with the risk classes):
    #   risk classes A-G map to dims 0-6 (multi-label, may co-occur)
    #   H maps to dim 7 = Safe; by label construction H=1 iff A-G are all 0.
    CATEGORY_TO_IDX = {
        'A': 0,  # Sexual Content Risk
        'B': 1,  # Real-World Crimes & Public Safety
        'C': 2,  # Unethical Behavior
        'D': 3,  # Cybersecurity & Information Manipulation
        'E': 4,  # Agent Safety
        'F': 5,  # Politically Sensitive Content
        'G': 6,  # Animal Abuse
        'H': 7,  # Safe (mutually exclusive with A-G)
    }

    def convert(self, raw_data: dict) -> dict:
        """
        Convert raw Safety data to standard format

        Args:
            raw_data: {
                "Query": str,
                "Response": str,
                "Query_Label": str (e.g. "A", "B", "BF", "ABC"),
                "Response_Label": "Safe" | "Unsafe"
            }

        Returns:
            Standard format data
        """
        query_label_str = raw_data.get('Query_Label', '')
        query_labels = self._parse_query_label(query_label_str)

        response_safety = 1 if raw_data.get('Response_Label', '') == 'Unsafe' else 0

        # Generate unique sample ID
        sample_id = raw_data.get('id', f"safety_{hash(raw_data['Query'] + raw_data['Response'])}")

        return {
            'sample_id': sample_id,
            'dataset_type': 'safety',
            'messages': [
                {'role': 'user', 'content': raw_data['Query']},
                {'role': 'assistant', 'content': raw_data['Response']}
            ],
            'query_labels': query_labels,
            'response_safety': response_safety,
            'response_hallucination_tokens': None,  # No token-level labels
            'metadata': {
                'source_file': 'safety_both.jsonl',
                'original_response_label': raw_data.get('Response_Label'),
                'has_response': True,
                'query_label': query_label_str,
            }
        }

    def _parse_query_label(self, query_label: str) -> List[int]:
        """
        Parse Query_Label string to 8-dim multi-hot vector.

        Query_Label is a concatenation of single-character category codes
        (e.g. "BF", "ABC"), so parse it character-by-character. Unknown
        characters are silently skipped (defensive against dirty data).

        Category H (dim 7) means "Safe" and is mutually exclusive with the
        seven risk classes A-G (dims 0-6): a clean query is labeled "H", a
        risky one is labeled with one or more of A-G. If a row erroneously
        mixes H with any risk letter (e.g. "BH"), we treat H as authoritative
        (mutual-exclusion highest priority): drop the risk letters and keep
        only Safe, warning once per process about the dirty data. This keeps
        the label distribution self-consistent with the Safe-vs-risk training
        objective in BalancedMultiTaskLoss.

        Args:
            query_label: Category string (e.g. "A", "BF", "ABC", "H", or "")

        Returns:
            List[int]: 8-dim multi-hot vector (e.g. [0, 1, 0, 0, 0, 1, 0, 0])
        """
        query_labels = [0] * 8

        if not query_label:
            return query_labels

        # Defensive cleanup: H (Safe) may not co-occur with any risk class.
        has_safe = 'H' in query_label
        has_risk = any(ch in self.CATEGORY_TO_IDX and ch != 'H' for ch in query_label)
        if has_safe and has_risk:
            risk_chars = [ch for ch in query_label
                          if ch in self.CATEGORY_TO_IDX and ch != 'H']
            if not getattr(self, '_warned_safe_risk_mix', False):
                import warnings
                warnings.warn(
                    f"SafetyConverter: Query_Label '{query_label}' mixes H (Safe) "
                    f"with risk class(es) {''.join(risk_chars)}; Safe is mutually "
                    f"exclusive with risk classes -- dropping the risk letters and "
                    f"keeping only Safe. (warned once)",
                    stacklevel=2,
                )
                self._warned_safe_risk_mix = True
            # Keep only H (Safe), drop the risk letters for this row.
            query_labels[self.CATEGORY_TO_IDX['H']] = 1
            return query_labels

        for ch in query_label:
            if ch in self.CATEGORY_TO_IDX:
                query_labels[self.CATEGORY_TO_IDX[ch]] = 1

        return query_labels