from __future__ import annotations

import torch

from cifp.models.adversarial import gradient_reverse, grl_coefficient


def test_gradient_reversal_identity_and_direction() -> None:
    value = torch.tensor([1.0, -2.0], requires_grad=True)
    output = gradient_reverse(value, coefficient=0.25)
    assert torch.equal(output, value)
    output.sum().backward()
    assert torch.allclose(value.grad, torch.full_like(value, -0.25))


def test_grl_schedule() -> None:
    assert grl_coefficient(epoch=0, warmup_epochs=5, ramp_end_epoch=20, maximum=1.0) == 0.0
    assert grl_coefficient(epoch=5, warmup_epochs=5, ramp_end_epoch=20, maximum=1.0) == 0.0
    assert grl_coefficient(epoch=12.5, warmup_epochs=5, ramp_end_epoch=20, maximum=1.0) == 0.5
    assert grl_coefficient(epoch=20, warmup_epochs=5, ramp_end_epoch=20, maximum=1.0) == 1.0
    assert grl_coefficient(epoch=50, warmup_epochs=5, ramp_end_epoch=20, maximum=1.0) == 1.0
