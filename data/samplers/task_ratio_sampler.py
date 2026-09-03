"""
Task Ratio Sampler for multi-dataset balanced sampling

Implements periodic complete traversal:
- All samples from all datasets are guaranteed to be visited
- Smaller datasets are repeated to match the desired ratio
- Each epoch maintains the specified task ratio
- Samples are shuffled within each epoch
"""

import math
import random
from typing import Iterator, List
import torch
from torch.utils.data import Sampler


class TaskRatioSampler(Sampler):
    """
    Sampler that maintains task ratios while ensuring all samples are visited.

    Strategy:
    1. Each dataset is assigned a ratio (e.g., query:0.4, safety:0.35, hallu:0.25)
    2. The largest dataset determines the epoch size
    3. Smaller datasets are repeated (cycled) to match their ratio
    4. All samples from all datasets are guaranteed to be visited at least once per epoch

    Example:
        Dataset sizes: query=1000, safety=500, hallu=200
        Ratios: query:0.4, safety:0.35, hallu:0.25

        Epoch calculation:
        - Base samples = max(1000, 500, 200) = 1000
        - Total epoch size = base_samples / max_ratio = 1000 / 0.4 = 2500
        - Query samples = 2500 * 0.4 = 1000 (1 full pass)
        - Safety samples = 2500 * 0.35 = 875 (1.75 passes)
        - Hallu samples = 2500 * 0.25 = 625 (3.125 passes)

    Args:
        dataset_sizes: List of dataset sizes [size1, size2, size3, ...]
        task_ratios: List of task ratios [ratio1, ratio2, ratio3, ...] (must sum to 1.0)
        shuffle: Whether to shuffle indices within each epoch (default: True)
        seed: Random seed for reproducibility (default: 42)
        drop_last: Whether to drop the last incomplete batch (default: False)
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        task_ratios: List[float],
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False
    ):
        # Don't call super().__init__() for custom Sampler
        # PyTorch's Sampler expects a data_source, but we manage indices ourselves

        assert len(dataset_sizes) == len(task_ratios), \
            f"Number of datasets ({len(dataset_sizes)}) must match number of ratios ({len(task_ratios)})"

        # Normalize task ratios to sum to 1.0
        ratio_sum = sum(task_ratios)
        task_ratios = [r / ratio_sum for r in task_ratios]

        self.dataset_sizes = dataset_sizes
        self.task_ratios = task_ratios
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Calculate epoch size and samples per task
        self._calculate_epoch_config()

    def _calculate_epoch_config(self):
        """Calculate the number of samples per task and total epoch size"""
        max_size = max(self.dataset_sizes)
        max_ratio = max(self.task_ratios)

        # Base epoch size: ensure the largest dataset at max ratio is fully traversed
        # This guarantees ALL samples are visited at least once
        self.base_epoch_size = math.ceil(max_size / max_ratio)

        # Calculate samples per task
        self.samples_per_task = [
            math.ceil(self.base_epoch_size * ratio)
            for ratio in self.task_ratios
        ]

        # Total epoch size
        self.total_samples = sum(self.samples_per_task)

        # Calculate dataset offsets for global indexing
        # When used with ConcatDataset, indices are: [0..size1-1, size1..size1+size2-1, ...]
        self.dataset_offsets = [0]
        for size in self.dataset_sizes[:-1]:
            self.dataset_offsets.append(self.dataset_offsets[-1] + size)

    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch"""
        # Set random seed for this epoch
        rng = random.Random(self.seed + self.epoch)

        # Generate indices for each task
        all_indices = []

        for task_idx, (size, num_samples, offset) in enumerate(
            zip(self.dataset_sizes, self.samples_per_task, self.dataset_offsets)
        ):
            # Generate base indices for this dataset [0, 1, 2, ..., size-1]
            base_indices = list(range(size))

            # Shuffle if requested
            if self.shuffle:
                rng.shuffle(base_indices)

            # Repeat indices to reach num_samples (cycle through dataset)
            # This ensures all samples are visited, smaller datasets are repeated
            repeated_indices = []
            for i in range(num_samples):
                # Use modulo to cycle through the dataset
                idx = base_indices[i % size]
                # Add offset to convert to global index (for ConcatDataset)
                repeated_indices.append(offset + idx)

            all_indices.extend(repeated_indices)

        # Shuffle all indices together to mix tasks
        if self.shuffle:
            rng.shuffle(all_indices)

        return iter(all_indices)

    def __len__(self) -> int:
        """Return the total number of samples in one epoch"""
        return self.total_samples

    def set_epoch(self, epoch: int):
        """Set the current epoch (for deterministic shuffling across epochs)"""
        self.epoch = epoch

    def get_stats(self) -> dict:
        """Return statistics about the sampling configuration"""
        return {
            'dataset_sizes': self.dataset_sizes,
            'task_ratios': self.task_ratios,
            'samples_per_task': self.samples_per_task,
            'total_samples': self.total_samples,
            'dataset_cycles': [
                round(samples / size, 2) if size > 0 else 0
                for samples, size in zip(self.samples_per_task, self.dataset_sizes)
            ]
        }


class DistributedTaskRatioSampler(Sampler):
    """TaskRatioSampler + per-rank sharding of its index stream.

    Wraps a ``TaskRatioSampler`` and exposes, on each rank, the disjoint
    round-robin slice ``all_indices[rank::num_replicas]`` of the base sampler's
    full epoch order. Because the base re-seeds its RNG with ``seed + epoch``
    (identical across ranks), the *pre-shard* order is the same on every rank;
    the ``rank::num_replicas`` stride then guarantees disjoint coverage across
    all ranks.

    The base index stream is PADDED (with wrapped copies of its leading
    indices) up to an even multiple of ``num_replicas`` before the stride, so
    every rank yields the SAME number of samples. This is required for DDP:
    ``train.py`` synchronizes gradients per micro-batch, so a per-rank
    micro-batch-count mismatch deadlocks the all-reduce (ranks that finish
    early exit the training loop while others still call ``backward()``).
    Mirrors ``torch.utils.data.distributed.DistributedSampler``, which pads
    rather than truncating the tail rank.

    This is to DDP what ``torch.utils.data.distributed.DistributedSampler`` is
    to a plain shuffle: it preserves the task-ratio-balanced ordering while
    partitioning it across replicas. ``num_replicas == 1`` (single-process /
    world_size==1) yields the base order unchanged.

    Args:
        base_sampler: the configured ``TaskRatioSampler`` (train side).
        num_replicas: world size (number of DDP ranks).
        rank: this rank's id in ``[0, num_replicas)``.
    """

    def __init__(
        self,
        base_sampler: TaskRatioSampler,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if not (0 <= rank < num_replicas):
            raise ValueError(
                f"rank {rank} out of range [0, {num_replicas})"
            )
        self.base = base_sampler
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        # Full deterministic epoch order (same on every rank), then stride.
        all_indices = list(self.base)
        # Pad to an even multiple of num_replicas so the round-robin stride
        # (rank::num_replicas) yields the SAME count on every rank. Without
        # this, a non-divisible total_samples leaves the tail rank(s) one
        # sample short, making per-rank micro-batch counts diverge under DDP
        # and deadlocking the gradient all-reduce. Padding is with wrapped
        # copies of the leading indices (DistributedSampler's approach); since
        # the base order is already shuffled, the repeat distribution is no
        # more biased than a regular cyclic dataset.
        num_pad = (-len(all_indices)) % self.num_replicas
        if num_pad:
            all_indices = all_indices + all_indices[:num_pad]
        yield from iter(all_indices[self.rank::self.num_replicas])

    def __len__(self) -> int:
        # After padding, every rank holds exactly total_samples // num_replicas
        # samples (the ceil before padding, now identical across ranks).
        n = self.base.total_samples
        total = n + (-n) % self.num_replicas
        return total // self.num_replicas

    def set_epoch(self, epoch: int) -> None:
        """Forwarded to the base so each epoch reshuffles identically per rank."""
        self.base.set_epoch(epoch)

    def get_stats(self) -> dict:
        stats = self.base.get_stats()
        stats["num_replicas"] = self.num_replicas
        stats["rank"] = self.rank
        stats["rank_total_samples"] = len(self)
        return stats


class BalancedBatchSampler(Sampler):
    """
    Batch sampler that ensures each batch contains samples from all tasks.

    Alternative to TaskRatioSampler that guarantees balanced batches.

    Args:
        dataset_sizes: List of dataset sizes
        task_ratios: List of task ratios (must sum to 1.0)
        batch_size: Total batch size
        drop_last: Whether to drop incomplete batches
        shuffle: Whether to shuffle within each task
        seed: Random seed
    """

    def __init__(
        self,
        dataset_sizes: List[int],
        task_ratios: List[float],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42
    ):
        # Don't call super().__init__() for custom Sampler

        assert len(dataset_sizes) == len(task_ratios)

        # Normalize task ratios to sum to 1.0
        ratio_sum = sum(task_ratios)
        task_ratios = [r / ratio_sum for r in task_ratios]

        self.dataset_sizes = dataset_sizes
        self.task_ratios = task_ratios
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Calculate samples per task in each batch
        self.samples_per_task_per_batch = [
            max(1, round(batch_size * ratio))
            for ratio in task_ratios
        ]

        # Adjust to match batch_size exactly
        total = sum(self.samples_per_task_per_batch)
        if total != batch_size:
            # Adjust the largest task
            max_idx = self.samples_per_task_per_batch.index(max(self.samples_per_task_per_batch))
            self.samples_per_task_per_batch[max_idx] += batch_size - total

        # Calculate dataset offsets
        self.dataset_offsets = [0]
        for size in self.dataset_sizes[:-1]:
            self.dataset_offsets.append(self.dataset_offsets[-1] + size)

        # Calculate number of batches (based on smallest dataset)
        min_cycles = min(
            size / samples_per_batch
            for size, samples_per_batch in zip(self.dataset_sizes, self.samples_per_task_per_batch)
        )
        self.num_batches = math.floor(min_cycles) if drop_last else math.ceil(min_cycles)

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches for one epoch"""
        rng = random.Random(self.seed + self.epoch)

        # Generate and shuffle indices for each task
        task_indices = []
        for size in self.dataset_sizes:
            indices = list(range(size))
            if self.shuffle:
                rng.shuffle(indices)
            task_indices.append(indices)

        # Generate batches
        task_positions = [0] * len(self.dataset_sizes)

        for _ in range(self.num_batches):
            batch = []

            for task_idx, (indices, offset, num_samples) in enumerate(
                zip(task_indices, self.dataset_offsets, self.samples_per_task_per_batch)
            ):
                for _ in range(num_samples):
                    # Get next index for this task (cycle if needed)
                    pos = task_positions[task_idx] % len(indices)
                    batch.append(offset + indices[pos])
                    task_positions[task_idx] += 1

            # Shuffle within batch
            if self.shuffle:
                rng.shuffle(batch)

            yield batch

    def __len__(self) -> int:
        """Return the number of batches"""
        return self.num_batches

    def set_epoch(self, epoch: int):
        """Set the current epoch"""
        self.epoch = epoch


def test_sampler():
    """Test TaskRatioSampler"""
    print("\n" + "="*80)
    print("Testing TaskRatioSampler")
    print("="*80)

    # Test case: imbalanced datasets
    dataset_sizes = [1000, 200]  # safety, hallucination
    task_ratios = [0.5, 0.5]

    sampler = TaskRatioSampler(
        dataset_sizes=dataset_sizes,
        task_ratios=task_ratios,
        shuffle=True,
        seed=42
    )

    stats = sampler.get_stats()
    print(f"\nDataset sizes: {stats['dataset_sizes']}")
    print(f"Task ratios: {stats['task_ratios']}")
    print(f"Samples per task: {stats['samples_per_task']}")
    print(f"Total samples per epoch: {stats['total_samples']}")
    print(f"Dataset cycles (passes per epoch): {stats['dataset_cycles']}")

    # Test iteration
    indices = list(sampler)
    print(f"\nActual indices generated: {len(indices)}")

    # Verify coverage
    for i, (size, offset) in enumerate(zip(dataset_sizes, [0, 1000])):
        task_indices = [idx for idx in indices if offset <= idx < offset + size]
        unique_indices = set(idx - offset for idx in task_indices)
        print(f"Task {i}: {len(task_indices)} samples, {len(unique_indices)} unique (coverage: {len(unique_indices)/size*100:.1f}%)")

    # Test multiple epochs
    print("\nTesting multiple epochs (should have different shuffle orders):")
    epoch_indices = []
    for epoch in range(3):
        sampler.set_epoch(epoch)
        indices = list(sampler)
        epoch_indices.append(indices[:10])  # First 10 indices
        print(f"  Epoch {epoch}: {indices[:5]}...")

    print("\n✅ TaskRatioSampler test passed!")
    return True


if __name__ == '__main__':
    test_sampler()