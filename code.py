python -c "
import json
old = json.load(open('$combined_json'))
for line in '''$result'''.split('\n'):
    line = line.strip()
    if line.startswith('{') and line.endswith('}'):
        old.append(json.loads(line))
json.dump(old, open('$combined_json','w'), indent=2)
"

"""
test_bench.py
Runs hub models in benchmark mode using pytest-benchmark on CPU.

Usage:
  python install.py
  pytest test_bench.py --backend=zentorch

See pytest-benchmark help (pytest test_bench.py -h) for additional options
e.g. --benchmark-autosave
     --benchmark-compare
     -k <filter expression>
     --ignore_machine_config
"""

import os
import time
import pytest
import torch
from torchbenchmark import _list_model_paths, get_metadata_from_yaml, ModelTask
from torchbenchmark._components._impl.workers import subprocess_worker
from torchbenchmark.util.machine_config import get_machine_state
from torchbenchmark.util.metadata_utils import skip_by_metadata


# 🛠️ Register Custom Zentorch Backend
def custom_zentorch_backend(model, inputs):
    print("Using custom Zentorch backend for compilation.")
    return model  # Placeholder for actual Zentorch integration

torch._dynamo.register_backend("zentorch", custom_zentorch_backend)


# 🛠️ Add `--backend` Flag for Custom Backend Support
def pytest_addoption(parser):
    parser.addoption(
        "--backend", action="store", default=None, help="Specify custom backend for torch.compile"
    )


# 🛠️ Generate Model Tests (ONLY CPU MODE)
def pytest_generate_tests(metafunc):
    devices = ["cpu"]  # Run only on CPU

    if metafunc.cls and metafunc.cls.__name__ == "TestBenchNetwork":
        paths = _list_model_paths()
        metafunc.parametrize(
            "model_path",
            paths,
            ids=[os.path.basename(path) for path in paths],
            scope="class",
        )

        metafunc.parametrize("device", devices, scope="class")


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
    group="hub",
)
class TestBenchNetwork:
    """Run Benchmarks for Each Model on CPU"""

    def test_train(self, model_path, device, benchmark, pytestconfig):
        """Runs training benchmark on CPU"""
        try:
            model_name = os.path.basename(model_path)
            if skip_by_metadata(test="train", device=device, extra_args=[], metadata=get_metadata_from_yaml(model_path)):
                raise NotImplementedError("Test skipped by its metadata.")

            if "quantized" in model_name:
                return

            task = ModelTask(model_name)
            if not task.model_details.exists:
                return  # Model is not supported.

            task.make_model_instance(test="train", device=device)

            # Apply custom backend if specified
            backend = pytestconfig.getoption("backend")
            if backend:
                print(f"⚙️ Using {backend} backend for training {model_name} on CPU")
                task.model = torch.compile(task.model, backend=backend)

            benchmark(task.invoke)
            benchmark.extra_info["machine_state"] = get_machine_state()
            benchmark.extra_info["batch_size"] = task.get_model_attribute("batch_size")
            benchmark.extra_info["precision"] = task.get_model_attribute("dargs", "precision")
            benchmark.extra_info["test"] = "train"

        except NotImplementedError:
            print(f"⚠️ Test train on {device} is not implemented, skipping...")

    def test_eval(self, model_path, device, benchmark, pytestconfig):
        """Runs evaluation benchmark on CPU"""
        try:
            model_name = os.path.basename(model_path)
            if skip_by_metadata(test="eval", device=device, extra_args=[], metadata=get_metadata_from_yaml(model_path)):
                raise NotImplementedError("Test skipped by its metadata.")

            if "quantized" in model_name:
                return

            task = ModelTask(model_name)
            if not task.model_details.exists:
                return  # Model is not supported.

            task.make_model_instance(test="eval", device=device)

            # Apply custom backend if specified
            backend = pytestconfig.getoption("backend")
            if backend:
                print(f"⚙️ Using {backend} backend for evaluation {model_name} on CPU")
                task.model = torch.compile(task.model, backend=backend)

            benchmark(task.invoke)
            benchmark.extra_info["machine_state"] = get_machine_state()
            benchmark.extra_info["batch_size"] = task.get_model_attribute("batch_size")
            benchmark.extra_info["precision"] = task.get_model_attribute("dargs", "precision")
            benchmark.extra_info["test"] = "eval"

        except NotImplementedError:
            print(f"⚠️ Test eval on {device} is not implemented, skipping...")


@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    disable_gc=False,
    timer=time.perf_counter,
    group="hub",
)
class TestWorker:
    """Benchmark SubprocessWorker to make sure we aren't skewing results."""

    def test_worker_noop(self, benchmark):
        worker = subprocess_worker.SubprocessWorker()
        benchmark(lambda: worker.run("pass"))

    def test_worker_store(self, benchmark):
        worker = subprocess_worker.SubprocessWorker()
        benchmark(lambda: worker.store("x", 1))

    def test_worker_load(self, benchmark):
        worker = subprocess_worker.SubprocessWorker()
        worker.store("x", 1)
        benchmark(lambda: worker.load("x"))

