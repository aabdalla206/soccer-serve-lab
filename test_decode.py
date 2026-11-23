import pytest
import torch

import backends
from backends import get_backend, make_inputs

SEQ_CONFIGS = [[1], [17], [128, 129], [1000, 37, 512, 2048]]


def _devices():
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("seq_lens", SEQ_CONFIGS)
def test_sdpa_matches_naive(seq_lens):
    args = make_inputs(seq_lens, H=4, D=64, block_size=16, dtype=torch.float32, device="cpu")
    ref = get_backend("naive")(*args)
    out = get_backend("sdpa")(*args)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("seq_lens", SEQ_CONFIGS)
@pytest.mark.parametrize("head_dim,block_size", [(64, 16), (64, 32), (128, 16)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_triton_matches_naive(seq_lens, head_dim, block_size, dtype):
    import triton_decode  # noqa: F401  (registers the backend)

    args = make_inputs(seq_lens, H=8, D=head_dim, block_size=block_size, dtype=dtype, device="cuda")
    ref = get_backend("naive")(*args)
    out = get_backend("triton")(*args)
    tol = 2e-2 if dtype == torch.float16 else 1e-4
    torch.testing.assert_close(out, ref, rtol=tol, atol=tol)
