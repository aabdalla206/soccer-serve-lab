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

**vLLM prefix caching**, Tesla T4, Qwen2.5-0.5B-Instruct (fp16), 2018 World Cup final event log as a 5,303-token shared prefix, 20 analyst questions.

| | Prefix caching off | Prefix caching on | Change |
|---|---|---|---|
| Mean time to first token (one request at a time) | 2,148 ms | 74.5 ms | 29x faster |
| Batch throughput (20 questions, 64 output tokens each) | 27 tok/s | 378 tok/s | 14x |

With caching on, the 5.3K-token match log is prefilled once and every later question reuses its KV cache, so each request only prefills its own question. Without it, every request recomputes the whole log, and that prefill dominates both latency and batch time. Raw data: `prefix_cache_results.jsonl`. These are vLLM's own kernels; my Triton kernel is benchmarked separately above.

## What I learned

- Decode attention is memory bound, so achieved GB/s against peak bandwidth is the right scorecard, not FLOPs.
- Reading K/V straight through the block table avoids materializing a contiguous copy, which is the whole point of a paged-KV kernel.
- At batch 1 the kernel launches only 16 programs (one per head), which cannot fill the T4's 40 SMs, so it stalls near 45 GB/s however long the sequence. Splitting each sequence's KV across several programs and reducing the partial softmax results (flash-decoding) is the fix.
- Bandwidth dips at batch 32 and 16K tokens (176 vs ~210 GB/s), worth profiling with Nsight Compute.
- For a shared-context workload, the biggest win is not a faster kernel but not doing the work at all: prefix caching cut time to first token 29x by reusing the match log's KV cache.

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
