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
