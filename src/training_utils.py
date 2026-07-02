import torch


@torch.no_grad()
def _cast_trainable_float_params_to_fp32(model: torch.nn.Module) -> int:
    """Keep AMP trainable weights in FP32 so GradScaler can unscale grads.

    Local pretrained checkpoints may store frozen backbones in FP16. If later
    config unfreezes a tail layer, that layer remains FP16 unless we promote it
    before building the optimizer, causing GradScaler to reject FP16 grads.
    """
    changed = 0
    for param in model.parameters():
        if param.requires_grad and param.is_floating_point() and param.dtype != torch.float32:
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()
            changed += 1
    return changed
