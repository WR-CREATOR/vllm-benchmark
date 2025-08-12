import asyncio
import json
import time
import argparse
from vllm_benchmark import run_benchmark

async def run_all_benchmarks(vllm_url, api_key, use_long_context,model_name):
    configurations = [
        {"num_requests": 10, "concurrency": 1, "output_tokens": 100},
        {"num_requests": 100, "concurrency": 50, "output_tokens": 100},
        {"num_requests": 200, "concurrency": 100, "output_tokens": 100},
        {"num_requests": 1000, "concurrency": 500, "output_tokens": 100},
        {"num_requests": 2000, "concurrency": 1000, "output_tokens": 100},
        # {"num_requests": 3000, "concurrency": 1000, "output_tokens": 200},
    ]

    all_results = []

    for config in configurations:
        print(f"Running benchmark with concurrency {config['concurrency']}...")
        results = await run_benchmark(config['num_requests'], config['concurrency'], 30, config['output_tokens'], vllm_url, api_key, use_long_context,model_name)
        all_results.append(results)
        time.sleep(5)  # Wait a bit between runs to let the system cool down

    return all_results

def main():
    parser = argparse.ArgumentParser(description="Run vLLM benchmarks with various configurations")
    parser.add_argument("--vllm_url", type=str, required=True, help="URL of the vLLM server")
    parser.add_argument("--api_key", type=str, required=True, help="API key for vLLM server")
    parser.add_argument("--use_long_context", action="store_true", help="Use long context prompt pairs instead of short prompts")
    parser.add_argument("--model_name", type=str, default="Qwen3-32B", help="Model Name")
    args = parser.parse_args()

    all_results = asyncio.run(run_all_benchmarks(args.vllm_url, args.api_key, args.use_long_context,args.model_name))

    with open('benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("Benchmark results saved to benchmark_results.json")

if __name__ == "__main__":
    main()

