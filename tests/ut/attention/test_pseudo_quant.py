import torch

from vllm_ascend.attention.pseudo_quant import pseudo_quantize_fp4_per_block


def test_pseudo_quantize_fp4_per_block_preserves_shape_and_dtype():
    x = torch.tensor(
        [[0.0, 1.0, -2.0, 3.0, 8.0, -4.0]],
        dtype=torch.bfloat16,
    )

    output = pseudo_quantize_fp4_per_block(x, block_size=4)

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert torch.isfinite(output.float()).all()


def test_pseudo_quantize_fp4_per_block_uses_independent_block_scales():
    x = torch.tensor([[1.0, 7.0, 10.0, 70.0]], dtype=torch.float32)

    output = pseudo_quantize_fp4_per_block(
        x,
        block_size=2,
        scale_dtype=torch.float32,
    )

    torch.testing.assert_close(output, x)
