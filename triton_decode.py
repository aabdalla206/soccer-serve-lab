"""Triton decode attention over a paged KV cache (one program per sequence and head).

Online softmax over the sequence's physical blocks, fp32 accumulation.
Constraints: HEAD_DIM and BLOCK_SIZE must be powers of two (e.g. 64 or 128, and 16 or 32).
Triton compiles and caches one variant per (HEAD_DIM, BLOCK_SIZE, dtype), and autotune picks
num_warps per config, which is the small specialization layer described in the README.
"""
import torch
import triton
import triton.language as tl

from backends import AttentionBackend, register


@triton.autotune(
    configs=[triton.Config({}, num_warps=w) for w in (1, 2, 4, 8)],
    key=["BLOCK_SIZE", "HEAD_DIM"],
)
@triton.jit
def _decode_kernel(
    Q, K, V, BT, SL, O,
    stride_qb, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_btb,
    stride_ob, stride_oh,
    scale,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(SL + b)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_s = tl.arange(0, BLOCK_SIZE)

    q = tl.load(Q + b * stride_qb + h * stride_qh + offs_d).to(tl.float32)

    # Scalar running max, running sum, and output accumulator (online softmax state).
    m_i = tl.max(tl.full([BLOCK_SIZE], float("-inf"), tl.float32), axis=0)
    l_i = tl.sum(tl.zeros([BLOCK_SIZE], tl.float32), axis=0)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    n_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(0, n_blocks):
        blk = tl.load(BT + b * stride_btb + i).to(tl.int64)
        tok = i * BLOCK_SIZE + offs_s
        mask = tok < seq_len

        k_ptrs = K + blk * stride_kb + offs_s[:, None] * stride_ks + h * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        s = tl.sum(k * q[None, :], axis=1) * scale
        s = tl.where(mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        p = tl.exp(s - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)

        v_ptrs = V + blk * stride_vb + offs_s[:, None] * stride_vs + h * stride_vh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(O + b * stride_ob + h * stride_oh + offs_d, out.to(O.dtype.element_ty))


@register
class TritonBackend(AttentionBackend):
    name = "triton"

    def __call__(self, q, k_cache, v_cache, block_tables, seq_lens):
        B, H, D = q.shape
        block_size = k_cache.shape[1]
        assert D & (D - 1) == 0 and block_size & (block_size - 1) == 0, "D and block_size must be powers of two"
        assert k_cache.stride(3) == 1 and v_cache.stride(3) == 1 and q.stride(2) == 1
        out = torch.empty_like(q)
        _decode_kernel[(B, H)](
            q, k_cache, v_cache, block_tables, seq_lens, out,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            block_tables.stride(0),
            out.stride(0), out.stride(1),
            D ** -0.5,
            BLOCK_SIZE=block_size,
            HEAD_DIM=D,
        )
        return out
