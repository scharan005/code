#!/usr/bin/env python3

import importlib
import logging
import os
import re
import subprocess
import sys
import warnings

try:
    from .common import (
        BenchmarkRunner,
        download_retry_decorator,
        load_yaml_file,
        main,
        reset_rng_state,
    )
except ImportError:
    from common import (
        BenchmarkRunner,
        download_retry_decorator,
        load_yaml_file,
        main,
        reset_rng_state,
    )

import torch
from torch._dynamo.testing import collect_results
from torch._dynamo.utils import clone_inputs


log = logging.getLogger(__name__)

# ✅ **DISABLE INDUCTOR COMPLETELY** (Ensures we don't default to Inductor)
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

# ✅ **Force the user-defined backend** instead of letting it default to Inductor.
if "TORCH_COMPILE_BACKEND" in os.environ:
    chosen_backend = os.environ["TORCH_COMPILE_BACKEND"]
    log.info(f"Using explicitly set backend: {chosen_backend}")
else:
    chosen_backend = "eager"  # Default to eager mode if no backend is set.
    log.info("No backend specified. Using eager mode.")

# ✅ **Check available backends to verify user-defined backend exists**
available_backends = torch._dynamo.list_backends()
if chosen_backend not in available_backends:
    raise ValueError(
        f"Backend '{chosen_backend}' is not available. Available backends: {available_backends}"
    )

# ✅ **Manually override the backend setting**
torch._dynamo.reset()  # Reset any previously set backend
os.environ["TORCH_COMPILE_BACKEND"] = chosen_backend

# ✅ **Disable FX Graph Caching for testing (Remove Inductor dependence)**
if "TORCHINDUCTOR_FX_GRAPH_CACHE" in os.environ:
    del os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"]

# ✅ **Ensure Autograd cache is disabled for our custom backend**
torch._functorch.config.enable_autograd_cache = False


def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


# ✅ **Force Import Required Libraries**
try:
    import transformers
except ImportError:
    print("Installing HuggingFace Transformers...")
    pip_install("git+https://github.com/huggingface/transformers.git#egg=transformers")


class HuggingfaceRunner(BenchmarkRunner):
    def __init__(self):
        super().__init__()
        self.suite_name = "huggingface"

    @property
    def _config(self):
        return load_yaml_file("huggingface.yaml")

    @property
    def _skip(self):
        return self._config["skip"]

    @property
    def _accuracy(self):
        return self._config["accuracy"]

    @property
    def skip_models(self):
        return self._skip["all"]

    @property
    def skip_models_for_cpu(self):
        return self._skip["device"]["cpu"]

    @property
    def fp32_only_models(self):
        return self._config["only_fp32"]

    @property
    def skip_models_due_to_control_flow(self):
        return self._skip["control_flow"]

    def _get_model_cls_and_config(self, model_name):
        model_cls = getattr(transformers, model_name)
        config_cls = model_cls.config_class
        config = config_cls()
        return model_cls, config

    @download_retry_decorator
    def _download_model(self, model_name):
        model_cls, config = self._get_model_cls_and_config(model_name)
        model = model_cls.from_config(config)
        return model

    def load_model(self, device, model_name, batch_size=None, extra_args=None):
        """
        Load the model with the user-specified backend (ipex, zentorch, etc.).
        """
        is_training = self.args.training
        use_eval_mode = self.args.use_eval_mode
        dtype = torch.float32
        reset_rng_state()

        model_cls, config = self._get_model_cls_and_config(model_name)
        model = self._download_model(model_name)
        model = model.to(device, dtype=dtype)

        if self.args.enable_activation_checkpointing:
            model.gradient_checkpointing_enable()

        batch_size = batch_size or 16  # Default batch size if not specified

        example_inputs = {
            "input_ids": torch.randint(
                0, config.vocab_size, (batch_size, 128), device=device
            )
        }

        if (
            is_training
            and not use_eval_mode
            and not (self.args.accuracy and model_name in self._config["only_inference"])
        ):
            model.train()
        else:
            model.eval()

        # ✅ **Force Custom Backend Instead of Inductor**
        if chosen_backend:
            torch._dynamo.reset()
            os.environ["TORCH_COMPILE_BACKEND"] = chosen_backend
            model = torch.compile(model, backend=chosen_backend)
            log.info(f"Using backend: {chosen_backend} for model: {model_name}")

        self.validate_model(model, example_inputs)
        return device, model_name, model, example_inputs, batch_size

    def compute_loss(self, pred):
        return pred[0]

    def forward_pass(self, mod, inputs, collect_outputs=True):
        with self.autocast(**self.autocast_arg):
            return mod(**inputs)

    def forward_and_backward_pass(self, mod, inputs, collect_outputs=True):
        cloned_inputs = clone_inputs(inputs)
        self.optimizer_zero_grad(mod)
        with self.autocast(**self.autocast_arg):
            pred = mod(**cloned_inputs)
            loss = self.compute_loss(pred)
        self.grad_scaler.scale(loss).backward()
        self.optimizer_step()
        if collect_outputs:
            return collect_results(mod, pred, loss, cloned_inputs)
        return None


def huggingface_main():
    """
    Main function to run HuggingFace benchmarks with a custom backend (ipex, zentorch).
    """
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore")

    # ✅ **Check if user-specified backend is set**
    if "TORCH_COMPILE_BACKEND" in os.environ:
        log.info(f"Running with backend: {os.environ['TORCH_COMPILE_BACKEND']}")
    else:
        log.warning("Backend not set. Running in eager mode.")

    main(HuggingfaceRunner())


if __name__ == "__main__":
    huggingface_main()


