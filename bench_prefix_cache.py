"""vLLM prefix-caching benchmark on the soccer match prompts.

Run once per mode, in separate processes (vLLM holds the GPU):
    python bench_prefix_cache.py --prefix-caching off
    python bench_prefix_cache.py --prefix-caching on

Each run appends a JSON line to prefix_cache_results.jsonl.
TTFT proxy = latency of a max_tokens=1 request, issued one at a time, since prefill dominates.
Throughput = tokens/sec generating 64 tokens for all questions in one batch.
"""
import argparse
import json
import time

from data import QUESTIONS, build_prompt, build_timeline, load_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-caching", choices=["on", "off"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--max-chars", type=int, default=20000)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    timeline = build_timeline(load_events(), max_chars=args.max_chars)
    prompts = [build_prompt(timeline, q) for q in QUESTIONS]

    llm = LLM(
        model=args.model,
        dtype="half",
        max_model_len=8192,
        enable_prefix_caching=(args.prefix_caching == "on"),
        gpu_memory_utilization=0.8,
    )
    n_prompt_tokens = len(llm.get_tokenizer().encode(prompts[0]))

    llm.generate(prompts[:1], SamplingParams(max_tokens=1))  # warmup (also warms the cache when on)

    one = SamplingParams(max_tokens=1, temperature=0)
    ttfts = []
    for p in prompts:
        t0 = time.perf_counter()
        llm.generate([p], one)
        ttfts.append(time.perf_counter() - t0)

    gen = SamplingParams(max_tokens=64, temperature=0)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, gen)
    dt = time.perf_counter() - t0
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

    result = dict(
        prefix_caching=args.prefix_caching,
        model=args.model,
        prompt_tokens=n_prompt_tokens,
        n_requests=len(prompts),
        mean_ttft_ms=round(1000 * sum(ttfts) / len(ttfts), 2),
        batch_tokens_per_s=round(gen_tokens / dt, 1),
    )
    print(result)
    with open("prefix_cache_results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
