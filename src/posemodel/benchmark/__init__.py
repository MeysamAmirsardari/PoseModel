"""Leakage-safe representation benchmark framework."""

from posemodel.benchmark.manifest import BenchmarkConfig
from posemodel.benchmark.runner import run_benchmark, write_benchmark_index

__all__ = ["BenchmarkConfig", "run_benchmark", "write_benchmark_index"]
