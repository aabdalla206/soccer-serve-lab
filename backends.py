"""Attention backend interface, registry, and baseline decode-attention implementations.

Decode attention = one new query token per sequence attending over a paged KV cache.
Plain multi-head attention (no GQA) to keep the first version simple.

Tensor layout
  q             [B, H, D]
  k_cache       [num_blocks, block_size, H, D]
  v_cache       [num_blocks, block_size, H, D]
  block_tables  [B, max_blocks] int32   (logical block i of sequence b -> physical block id)
  seq_lens      [B] int32
  returns       [B, H, D]
"""
import math

import torch
import torch.nn.functional as F

REGISTRY = {}


class AttentionBackend:
    """Every implementation (naive, SDPA, Triton, later FlashInfer) plugs in here."""

    name = "base"

    def __call__(self, q, k_cache, v_cache, block_tables, seq_lens):
        raise NotImplementedError


def register(cls):
    REGISTRY[cls.name] = cls()
    return cls


def get_backend(name):
    return REGISTRY[name]


def gather_kv(k_cache, v_cache, block_table, seq_len):
    """Collect one sequence's K and V from its (non-contiguous) physical blocks: [L, H, D]."""
    block_size = k_cache.shape[1]
    n_blocks = (seq_len + block_size - 1) // block_size
    ids = block_table[:n_blocks].long()
    k = k_cache[ids].reshape(-1, *k_cache.shape[2:])[:seq_len]
    v = v_cache[ids].reshape(-1, *v_cache.shape[2:])[:seq_len]
    return k, v


@register
class NaiveBackend(AttentionBackend):
    """Reference implementation in fp32. Slow, but the ground truth for tests."""

    name = "naive"

    def __call__(self, q, k_cache, v_cache, block_tables, seq_lens):
        B, H, D = q.shape
        scale = D ** -0.5
        out = torch.empty_like(q)
        for b in range(B):
            k, v = gather_kv(k_cache, v_cache, block_tables[b], int(seq_lens[b]))
            scores = torch.einsum("hd,lhd->hl", q[b].float(), k.float()) * scale
            probs = torch.softmax(scores, dim=-1)
            out[b] = torch.einsum("hl,lhd->hd", probs, v.float()).to(q.dtype)
        return out


@register
class SDPABackend(AttentionBackend):
    """PyTorch scaled_dot_product_attention on gathered K/V (includes the gather cost)."""

    name = "sdpa"

    def __call__(self, q, k_cache, v_cache, block_tables, seq_lens):
        out = torch.empty_like(q)
        for b in range(q.shape[0]):
            k, v = gather_kv(k_cache, v_cache, block_tables[b], int(seq_lens[b]))
            o = F.scaled_dot_product_attention(
                q[b].unsqueeze(1),      # [H, 1, D]
                k.permute(1, 0, 2),     # [H, L, D]
                v.permute(1, 0, 2),
            )
            out[b] = o.squeeze(1)
        return out


def make_inputs(seq_lens, H=16, D=64, block_size=16, dtype=torch.float16, device="cuda", seed=0):
    """Random paged-KV inputs with shuffled physical blocks, so paging is actually exercised."""
    torch.manual_seed(seed)
    B = len(seq_lens)
    max_blocks = math.ceil(max(seq_lens) / block_size)
    total = B * max_blocks
    q = torch.randn(B, H, D, device=device, dtype=dtype)
    k_cache = torch.randn(total, block_size, H, D, device=device, dtype=dtype)
    v_cache = torch.randn(total, block_size, H, D, device=device, dtype=dtype)
    perm = torch.randperm(total).to(device=device, dtype=torch.int32)
    block_tables = perm.view(B, max_blocks).contiguous()
    sl = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    return q, k_cache, v_cache, block_tables, sl
