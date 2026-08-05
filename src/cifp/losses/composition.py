from __future__ import annotations

import torch


def composition_regularizers(
    assignments: torch.Tensor, normalized_dictionary: torch.Tensor, *, eps: float = 1e-8
) -> dict[str, torch.Tensor]:
    """Return activation entropy, usage balance, and dictionary diversity losses."""
    if assignments.ndim != 3:
        raise ValueError(f"assignments must have shape [B, N, K], got {assignments.shape}")
    primitive_count = assignments.shape[-1]
    if normalized_dictionary.shape[0] != primitive_count:
        raise ValueError("dictionary and assignment primitive counts differ")
    sparse = -(assignments * torch.log(assignments + eps)).sum(dim=-1).mean()
    mean_usage = assignments.mean(dim=(0, 1))
    balance = (mean_usage * torch.log(mean_usage * primitive_count + eps)).sum()
    gram = normalized_dictionary @ normalized_dictionary.transpose(0, 1)
    if primitive_count == 1:
        diversity = gram.new_zeros(())
    else:
        off_diagonal = gram[~torch.eye(primitive_count, dtype=torch.bool, device=gram.device)]
        diversity = off_diagonal.square().mean()
    return {"sparse": sparse, "balance": balance, "diversity": diversity}
