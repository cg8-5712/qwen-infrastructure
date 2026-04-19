"""
Benchmark for Qwen Inference Engine

Tests performance characteristics:
- Memory usage at different sequence lengths
- Throughput for various batch sizes
- Latency for incremental token generation
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

import torch
import time
import json
from typing import List, Dict
import logging

from qwen_infer import InferenceEngine, Config
from qwen_infer.utils.memory_utils import get_gpu_memory_info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Benchmark:
    """Benchmark suite for inference engine"""

    def __init__(self, config: Config):
        self.config = config
        self.results = []

    def benchmark_memory_scalability(self) -> Dict:
        """Test memory usage at different sequence lengths"""
        logger.info("=" * 60)
        logger.info("Benchmark: Memory Scalability")
        logger.info("=" * 60)

        sequence_lengths = [1000, 10000, 50000, 100000, 200000]
        results = []

        engine = InferenceEngine(self.config)

        try:
            engine.initialize()

            for length in sequence_lengths:
                if length > self.config.max_sequence_length:
                    continue

                # Record memory before
                mem_before = get_gpu_memory_info(0)['allocated']

                # Create sequence
                tokens = [i % 1000 for i in range(length)]
                start = time.time()
                seq_id = engine.create_sequence(tokens)
                create_time = time.time() - start

                # Record memory after
                mem_after = get_gpu_memory_info(0)['allocated']
                mem_used = mem_after - mem_before

                result = {
                    'sequence_length': length,
                    'create_time_ms': create_time * 1000,
                    'memory_used_gb': mem_used,
                    'memory_per_token_mb': (mem_used * 1024) / length if length > 0 else 0
                }
                results.append(result)

                logger.info(
                    f"Length: {length:>8} | "
                    f"Time: {create_time*1000:>8.2f}ms | "
                    f"Memory: {mem_used:>6.2f}GB | "
                    f"Per token: {result['memory_per_token_mb']:.4f}MB"
                )

                engine.free_sequence(seq_id)

        finally:
            engine.__exit__(None, None, None)

        return {
            'test': 'memory_scalability',
            'results': results
        }

    def benchmark_throughput(self) -> Dict:
        """Test generation throughput"""
        logger.info("=" * 60)
        logger.info("Benchmark: Generation Throughput")
        logger.info("=" * 60)

        batch_sizes = [1, 2, 4]
        prompt_length = 1000
        generate_tokens = 100

        results = []
        engine = InferenceEngine(self.config)

        try:
            engine.initialize()

            for batch_size in batch_sizes:
                if batch_size > self.config.max_batch_size:
                    continue

                # Create sequences
                seq_ids = []
                for i in range(batch_size):
                    tokens = [j % 1000 for j in range(prompt_length)]
                    seq_id = engine.create_sequence(tokens)
                    seq_ids.append(seq_id)

                # Generate tokens
                start = time.time()
                tokens_generated = 0

                for _ in range(generate_tokens):
                    for seq_id in seq_ids:
                        try:
                            next(engine.generate(seq_id, max_new_tokens=1))
                            tokens_generated += 1
                        except StopIteration:
                            pass

                duration = time.time() - start
                throughput = tokens_generated / duration if duration > 0 else 0

                result = {
                    'batch_size': batch_size,
                    'prompt_length': prompt_length,
                    'tokens_generated': tokens_generated,
                    'duration_s': duration,
                    'throughput_tok_s': throughput,
                    'latency_ms_per_tok': (duration / tokens_generated) * 1000 if tokens_generated > 0 else 0
                }
                results.append(result)

                logger.info(
                    f"Batch: {batch_size} | "
                    f"Tokens: {tokens_generated} | "
                    f"Time: {duration:.2f}s | "
                    f"Throughput: {throughput:.2f} tok/s | "
                    f"Latency: {result['latency_ms_per_tok']:.2f}ms/tok"
                )

                for seq_id in seq_ids:
                    engine.free_sequence(seq_id)

        finally:
            engine.__exit__(None, None, None)

        return {
            'test': 'throughput',
            'results': results
        }

    def benchmark_long_sequence(self) -> Dict:
        """Test long sequence handling"""
        logger.info("=" * 60)
        logger.info("Benchmark: Long Sequence (>200k)")
        logger.info("=" * 60)

        target_length = 200000
        engine = InferenceEngine(self.config)

        try:
            engine.initialize()

            # Create long sequence
            tokens = [i % 1000 for i in range(target_length)]

            start = time.time()
            seq_id = engine.create_sequence(tokens)
            create_time = time.time() - start

            # Verify allocation
            actual_length = engine.get_sequence_length(seq_id)

            # Test incremental generation
            extend_times = []
            for i in range(10):
                start = time.time()
                success = engine.paged_attention.append_tokens(seq_id, 1)
                extend_times.append(time.time() - start)
                if not success:
                    logger.error(f"Failed to extend at step {i}")
                    break

            avg_extend_time = sum(extend_times) / len(extend_times) * 1000  # ms

            # Memory stats
            status = engine.memory_manager.get_balanced_memory_status()

            result = {
                'target_length': target_length,
                'actual_length': actual_length,
                'create_time_s': create_time,
                'avg_extend_time_ms': avg_extend_time,
                'memory_allocated_gb': status.allocated_gb,
                'memory_free_gb': status.free_gb,
                'memory_pressure': status.pressure_level.value
            }

            logger.info(f"Long sequence test:")
            logger.info(f"  Created: {actual_length} tokens in {create_time:.2f}s")
            logger.info(f"  Avg extend: {avg_extend_time:.2f}ms")
            logger.info(f"  Memory: {status.allocated_gb:.2f}GB allocated, {status.free_gb:.2f}GB free")

            engine.free_sequence(seq_id)

            return {
                'test': 'long_sequence',
                'result': result
            }

        finally:
            engine.__exit__(None, None, None)

    def run_all(self) -> List[Dict]:
        """Run all benchmarks"""
        logger.info("\n" + "=" * 60)
        logger.info("Qwen Inference Engine Benchmark Suite")
        logger.info("Model: Qwen3.5-35B-A3B-GPTQ-Int4")
        logger.info("GPUs: 2,3")
        logger.info("=" * 60 + "\n")

        benchmarks = [
            self.benchmark_memory_scalability,
            self.benchmark_throughput,
            self.benchmark_long_sequence,
        ]

        results = []
        for benchmark_fn in benchmarks:
            try:
                result = benchmark_fn()
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark {benchmark_fn.__name__} failed: {e}", exc_info=True)

        return results

    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filename}")

if __name__ == "__main__":
    config = Config()
    benchmark = Benchmark(config)

    try:
        results = benchmark.run_all()
        benchmark.results = results
        benchmark.save_results()
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)