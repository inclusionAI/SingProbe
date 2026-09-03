"""
Base converter for dataset format transformation
"""

from abc import ABC, abstractmethod
from typing import Dict, List


class BaseConverter(ABC):
    """
    Abstract base class for dataset converters

    All dataset converters must implement the convert() and convert_batch() methods
    to transform raw data into the standard format.
    """

    @abstractmethod
    def convert(self, raw_data: Dict) -> Dict:
        """
        Convert a single raw data item to standard format

        Args:
            raw_data: Raw data from the source dataset

        Returns:
            Standard format data with fields:
                - sample_id: str
                - dataset_type: str
                - messages: List[Dict] (chat format)
                - query_labels: List[int] (8-dim multi-hot)
                - response_safety: Optional[int]
                - response_hallucination_tokens: Optional[List[Dict]]
                - metadata: Dict
        """
        pass

    def convert_batch(self, raw_data_list: List[Dict]) -> List[Dict]:
        """
        Convert a batch of raw data items to standard format

        Args:
            raw_data_list: List of raw data items

        Returns:
            List of standard format data items
        """
        return [self.convert(item) for item in raw_data_list]