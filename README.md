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

Fill these in from your own runs. Do not publish numbers you did not measure.

| Experiment | Result |
|---|---|
| GPU | |
| Triton vs SDPA, batch=32, seq=16384 (GB/s) | |
| vLLM mean TTFT, prefix caching off (ms) | |
| vLLM mean TTFT, prefix caching on (ms) | |
| Prompt length (tokens) | |

Plot: `decode_bench.png`. Raw data: `results.csv`, `prefix_cache_results.jsonl`.

## What I learned

(Write this after benchmarking. Where does the kernel lose to SDPA, and why? Decode attention is memory bound, so compare achieved GB/s to the GPU's peak bandwidth.)

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
