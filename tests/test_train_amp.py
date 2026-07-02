import torch
import torch.nn as nn

from src.training_utils import _cast_trainable_float_params_to_fp32


def test_cast_trainable_float_params_to_fp32_keeps_frozen_fp16():
    model = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2)).half()
    for param in model[0].parameters():
        param.requires_grad = False

    changed = _cast_trainable_float_params_to_fp32(model)

    assert changed == 2
    assert {param.dtype for param in model[0].parameters()} == {torch.float16}
    assert {param.dtype for param in model[1].parameters()} == {torch.float32}
