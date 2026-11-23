"""Kernel microbenchmark: decode attention latency and achieved memory bandwidth.

Usage: python bench.py
Writes results.csv and decode_bench.png. Bandwidth counts K and V bytes read once.
FlashInfer comparison is not wired in yet (see README, Next steps).
"""
import csv
import itertools

import torch
import triton

import triton_decode  # noqa: F401  (registers the triton backend)
from backends import get_backend, make_inputs

H, D, BLOCK = 16, 64, 16
BATCHES = [1, 8, 32]
SEQ_LENS = [1024, 4096, 16384]
BACKENDS = ["sdpa", "triton"]
DTYPE = torch.float16


def kv_bytes(seq_lens):
    return 2 * sum(seq_lens) * H * D * 2  # K and V, fp16


def main():
    dev = torch.cuda.get_device_name(0)
    rows = []
    for B, L in itertools.product(BATCHES, SEQ_LENS):
        seq_lens = [L] * B
        args = make_inputs(seq_lens, H=H, D=D, block_size=BLOCK, dtype=DTYPE, device="cuda")
        ref = get_backend("sdpa")(*args)
        for name in BACKENDS:
            fn = get_backend(name)
            out = fn(*args)
            torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
            ms = triton.testing.do_bench(lambda: fn(*args), warmup=25, rep=100)
            gbs = kv_bytes(seq_lens) / (ms * 1e-3) / 1e9
            rows.append(dict(device=dev, backend=name, batch=B, seq_len=L, ms=round(ms, 4), gb_per_s=round(gbs, 1)))
            print(rows[-1])
        del args

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(BATCHES), figsize=(5 * len(BATCHES), 4), sharey=False)
    for ax, B in zip(axes, BATCHES):
        for name in BACKENDS:
            pts = [(r["seq_len"], r["gb_per_s"]) for r in rows if r["backend"] == name and r["batch"] == B]
            ax.plot(*zip(*pts), marker="o", label=name)
        ax.set_xscale("log", base=2)
        ax.set_title(f"batch={B}")
        ax.set_xlabel("sequence length")
        ax.set_ylabel("achieved GB/s")
        ax.legend()
    fig.suptitle(f"Decode attention on {dev} (H={H}, D={D}, block={BLOCK}, fp16)")
    fig.tight_layout()
    fig.savefig("decode_bench.png", dpi=150)
    print("wrote results.csv and decode_bench.png")


if __name__ == "__main__":
    main()
