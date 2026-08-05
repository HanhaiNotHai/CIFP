from __future__ import annotations

from cifp.data.sampler import EnvironmentBalancedBatchSampler


def test_environment_balanced_sampler_is_deterministic_and_rank_partitioned() -> None:
    labels = [0, 1] * 12
    environments = [index % 4 for index in range(24)]
    rank0 = EnvironmentBalancedBatchSampler(
        labels, environments, batch_size=4, num_replicas=2, rank=0, seed=42
    )
    rank1 = EnvironmentBalancedBatchSampler(
        labels, environments, batch_size=4, num_replicas=2, rank=1, seed=42
    )

    first_rank0 = list(rank0)
    first_rank1 = list(rank1)
    assert first_rank0 == list(rank0)
    assert set(index for batch in first_rank0 for index in batch).isdisjoint(
        index for batch in first_rank1 for index in batch
    )
    assert all({labels[index] for index in batch} == {0, 1} for batch in first_rank0)

    rank0.set_epoch(1)
    assert list(rank0) != first_rank0


def test_sampler_records_balance_fallbacks() -> None:
    sampler = EnvironmentBalancedBatchSampler(
        labels=[0, 0, 0, 1],
        environments=[0, 0, 0, 0],
        batch_size=4,
        seed=1,
    )
    list(sampler)
    assert sampler.fallback_count > 0
