# soccer-serve-lab

A small LLM serving and GPU kernel lab, driven by a soccer workload.

**Workload.** One long match event log (StatsBomb open data) shared by many analyst questions.
Every request has the same long prefix, which is the case prefix caching is built for.

**What is here**
- `data.py`: loads a match with `statsbombpy`, builds the shared-prefix prompt and a 20-question set.
- `backends.py`: `AttentionBackend` interface, a registry, and two baselines (naive fp32 reference, PyTorch SDPA) over a paged KV cache.
- `triton_decode.py`: Triton paged-KV decode attention (online softmax, fp32 accumulation). Triton caches one compiled variant per (head_dim, block_size, dtype), and autotune picks `num_warps` per config. This is a small kernel specialization and caching layer, not a compiler.
- `test_decode.py`: correctness tests against the naive reference. Triton tests need a GPU.
- `bench.py`: latency and achieved memory bandwidth across batch size and sequence length.
- `bench_prefix_cache.py`: vLLM time-to-first-token and throughput with prefix caching off and on.

**Run it (Colab T4 or any CUDA GPU)**
```
pip install torch triton statsbombpy matplotlib pytest vllm
pytest -q test_decode.py
python bench.py
python bench_prefix_cache.py --prefix-caching off
python bench_prefix_cache.py --prefix-caching on
```

## Results

**Decode attention kernel**, Tesla T4 (Colab), fp16, 16 heads, head_dim 64, KV block size 16.
All 28 correctness tests pass against the fp32 reference. Bandwidth counts K and V bytes read once.
The T4's peak memory bandwidth is about 320 GB/s.

| Batch | Seq len | Gather + SDPA (ms) | Triton (ms) | Triton GB/s | Speedup |
|---|---|---|---|---|---|
| 1 | 1,024 | 0.36 | 0.10 | 42 | 3.6x |
| 1 | 16,384 | 1.78 | 1.44 | 46 | 1.2x |
| 8 | 4,096 | 3.90 | 0.68 | 198 | 5.7x |
| 8 | 16,384 | 14.03 | 2.72 | 198 | 5.2x |
| 32 | 1,024 | 7.33 | 0.62 | 215 | 11.7x |
| 32 | 16,384 | 56.14 | 12.19 | 176 | 4.6x |

Full sweep: `results.csv`.

**Read the speedup carefully.** The baseline gathers each sequence's paged K/V into contiguous memory with a Python loop and then calls SDPA. Much of the gap comes from that extra copy and per-sequence launches, not from SDPA's attention math. The more honest measure is achieved bandwidth: the kernel reaches roughly two thirds of the T4's peak at batch 8 and above.

**vLLM prefix caching:** not run yet. See `bench_prefix_cache.py`.

## What I learned

- Decode attention is memory bound, so achieved GB/s against peak bandwidth is the right scorecard, not FLOPs.
- Reading K/V straight through the block table avoids materializing a contiguous copy, which is the whole point of a paged-KV kernel.
- At batch 1 the kernel launches only 16 programs (one per head), which cannot fill the T4's 40 SMs, so it stalls near 45 GB/s however long the sequence. Splitting each sequence's KV across several programs and reducing the partial softmax results (flash-decoding) is the fix.
- Bandwidth dips at batch 32 and 16K tokens (176 vs ~210 GB/s), worth profiling with Nsight Compute.

## Limitations

- Plain multi-head attention only (no grouped-query attention), fp16/fp32, powers-of-two head dim and block size.
- The kernel parallelizes over (batch, head) only. It does not split long sequences across programs, so it underuses the GPU at small batch sizes.
- The vLLM run uses vLLM's own kernels. My Triton kernel is benchmarked separately and is not integrated into vLLM.
- Match text is a filtered, truncated event log, not the full match.

## Next steps

- FlashInfer decode comparison, and a split-KV (flash-decoding style) Triton kernel.
- Grouped-query attention support.
- A CUDA C++ version of one kernel, and a JAX reimplementation with an XLA HLO dump.
- Compare `torch.compile` output and Triton IR to see what fusion each does.

## Attribution

Event data: StatsBomb Open Data (https://github.com/statsbomb/open-data). Check their license and terms before publishing anything built on it.
