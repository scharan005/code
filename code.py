import gc
import importlib
import logging
import os
import re
import sys
import warnings
from collections import namedtuple
from os.path import abspath, exists

import torch

try:
    from .common import BenchmarkRunner, load_yaml_file, main
except ImportError:
    from common import BenchmarkRunner, load_yaml_file, main

from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs

# We are primarily interested in tf32 datatype
torch.backends.cuda.matmul.allow_tf32 = True

# Enable FX graph caching
if "TORCHINDUCTOR_FX_GRAPH_CACHE" not in os.environ:
    torch._inductor.config.fx_graph_cache = True


class TorchBenchmarkRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "torchbench"
        self.optimizer = None

    def load_model(
        self,
        device,
        model_name,
        batch_size=None,
        part=None,
        extra_args=None,
    ):
        if self.args.enable_activation_checkpointing:
            raise NotImplementedError(
                "Activation checkpointing not implemented for Torchbench models"
            )
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode

        # Import the model from available paths
        candidates = [
            f"torchbenchmark.models.{model_name}",
            f"torchbenchmark.canary_models.{model_name}",
            f"torchbenchmark.models.fb.{model_name}",
        ]
        for c in candidates:
            try:
                module = importlib.import_module(c)
                break
            except ModuleNotFoundError as e:
                if e.name != c:
                    raise
        else:
            raise ImportError(f"could not import any of {candidates}")

        benchmark_cls = getattr(module, "Model", None)
        if benchmark_cls is None:
            raise NotImplementedError(f"{model_name}.Model is None")

        if not hasattr(benchmark_cls, "name"):
            benchmark_cls.name = model_name

        benchmark = benchmark_cls(
            test="eval",  # Set to inference only
            device=device,
            batch_size=batch_size,
            extra_args=extra_args or [],
        )

        model, example_inputs = benchmark.get_module()

        # [MODIFICATION FOR ZENTORCH/IPEX]
        if self.args.backend == "zentorch":
            import zentorch
            model = zentorch.optimize(model)

        elif self.args.backend == "ipex":
            import intel_extension_for_pytorch as ipex
            model, example_inputs = ipex.optimize(model, dtype=torch.float32)

        # Models that must be in eval mode for inference
        model.eval()

        gc.collect()
        return device, benchmark.name, model, example_inputs, batch_size

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast(**self.autocast_arg):
            # [MODIFICATION FOR ZENTORCH/IPEX]
            if self.args.backend == "zentorch":
                import zentorch
                return zentorch.forward(mod, *inputs)

            elif self.args.backend == "ipex":
                import intel_extension_for_pytorch as ipex
                with ipex.amp.autocast():
                    if isinstance(inputs, dict):
                        return mod(**inputs)
                    else:
                        return mod(*inputs)

            # Default case
            if isinstance(inputs, dict):
                return mod(**inputs)
            else:
                return mod(*inputs)


def torchbench_main():
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")
    main(TorchBenchmarkRunner(), os.getcwd())


if __name__ == "__main__":
    torchbench_main()


#!/bin/bash

set -ex

# Variables
TEST_REPORTS_DIR=$(pwd)/test/test-reports
BACKENDS=("eager" "aot_eager" "inductor" "zentorch" "ipex")  # Include new backends
DEVICE="cpu"

# Ensure the test reports directory exists
mkdir -p "$TEST_REPORTS_DIR"

run_torchbench() {
  local backend=$1
  local output_file="$TEST_REPORTS_DIR/${backend}_torchbench.csv"

  echo "Running TorchBench with backend: $backend"

  python torchbench.py \
    --device "$DEVICE" \
    --backend "$backend" \
    --inference \
    --performance \
    --output "$output_file"
}

# Run TorchBench for all backends
for backend in "${BACKENDS[@]}"; do
  run_torchbench "$backend"
done

# Summary of Results
echo "TorchBench test results available in: $TEST_REPORTS_DIR"


Abstract
This paper presents an optimized approach for deploying the Deep Learning Recommendation Model (DLRM) in both Python and C++ environments. Leveraging the oneDNN PyTorch Plugin and Ahead-Of-Time (AOT) compilation through the Inductor path, we provide a seamless workflow to enhance model performance. Our solution focuses on generating efficient .so files for C++ while retaining compatibility with Python exports, ensuring interoperability and scalability. Benchmark results demonstrate that the C++ environment outperforms the PyTorch environment, showcasing the effectiveness of the proposed solution.

Introduction
Recommendation systems like the Deep Learning Recommendation Model (DLRM) are pivotal in applications that require personalized experiences, such as e-commerce and social media platforms. While PyTorch provides an excellent framework for model development, deploying models in high-performance C++ environments remains challenging. Transitioning from Python to C++ often involves trade-offs in compatibility and performance.

This paper addresses these challenges by introducing a pipeline that integrates the oneDNN PyTorch Plugin with AOT Inductor to optimize DLRM for deployment in C++ environments. By aligning the deployment pipeline with hardware optimizations from oneDNN, this solution ensures reduced latency and higher throughput, meeting the performance demands of modern recommendation systems.

Problem Statement
Deploying machine learning models like DLRM in Python and C++ environments poses the following challenges:

Performance Disparity: Models optimized for Python often underperform when transitioned to C++ without additional tuning.
Complex Workflows: Transitioning models between Python development environments and C++ production requires extensive engineering, leading to increased development time.
Hardware Underutilization: Existing workflows fail to fully leverage hardware-specific optimizations such as those provided by oneDNN, resulting in inefficient inference performance.
Interoperability Gaps: Maintaining compatibility between Python and C++ exports for iterative development and deployment remains a bottleneck.
Relevance to Intel
Intel, as a leader in advancing high-performance computing and AI solutions, seeks to demonstrate the capabilities of oneDNN in optimizing machine learning workloads. By focusing on integrating the oneDNN PyTorch Plugin and leveraging Intel hardware capabilities, this work addresses Intel’s goals of enabling scalable, efficient model deployments. The proposed solution highlights the tangible benefits of Intel's oneDNN library, showcasing its role in enhancing the performance of recommendation systems like DLRM.

Proposed Solution
The solution builds a robust pipeline to optimize DLRM deployment across Python and C++ environments. The workflow includes the following steps:

Workflow Explanation:
Model Preparation:

The DLRM model is developed and trained in PyTorch, leveraging its user-friendly APIs for iterative experimentation.
Export:

The trained model is exported into an intermediate representation, enabling compatibility with further optimization processes.
Integration of oneDNN PyTorch Plugin:

The exported model passes through the oneDNN PyTorch Plugin, which:
Applies hardware-specific optimizations to maximize performance.
Prepares the model for subsequent compilation with AOT Inductor.
AOT Inductor Compilation:

The AOT Inductor path compiles the optimized model into:
.so files for high-performance C++ deployment.
Python-compatible artifacts for iterative research and debugging.
Deployment:

The generated .so files are deployed in a C++ environment, leveraging oneDNN optimizations to ensure superior runtime performance.
Workflow Diagram:
rust
Copy code
Model --> Export --> oneDNN PyTorch Plugin --> AOT Inductor --> Python Export
                                                    |
                                                    --> .so files for C++ Export
Results
The benchmark results demonstrate that the C++ environment, enhanced by oneDNN optimizations and AOT Inductor compilation, outperforms the PyTorch environment in terms of runtime efficiency and throughput. Key metrics include:

Reduced Latency: The C++ deployment achieves lower inference latency compared to the Python environment.
Higher Throughput: The optimized C++ pipeline processes a greater volume of requests per second, proving its scalability for large-scale recommendation systems.
