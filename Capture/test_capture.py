"""Dependency-free exporter/contract tests; real PyTorch integration is not run."""
import copy
import math
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch
from capture_activations import ActivationCapture, validate_trace, write_trace
from make_synthetic_trace import synthetic_trace
from host_example import record_probe


class Tensor:
    calls = []
    def __init__(self, data, detached=False, ndim=3):
        self.data, self.detached, self.ndim = data, detached, ndim
        self.shape = (1, len(data), len(data[0])) if ndim == 3 else (len(data), len(data[0]))
        self.device, self.dtype = "fake-gpu", "torch.float32"
    def detach(self):
        Tensor.calls.append("detach")
        return Tensor(self.data, True)
    def __getitem__(self, batch):
        if not self.detached or batch != 0:
            raise AssertionError("detach must precede batch slicing")
        return Tensor(self.data, True, ndim=2)
    def index_select(self, dim, indices):
        if not self.detached or dim != 0:
            raise AssertionError("only detached selected token rows may be copied")
        Tensor.calls.append("sample")
        return Tensor([self.data[i] for i in indices], True, ndim=2)
    def to(self, device, dtype):
        if not self.detached or device != "cpu":
            raise AssertionError("only detached samples go to CPU")
        Tensor.calls.append("cpu")
        return self
    def clone(self):
        return self
    def tolist(self):
        return copy.deepcopy(self.data)


class Finite:
    def __init__(self, value): self.value = value
    def all(self): return self.value


class Handle:
    def __init__(self, module, hook): self.module, self.hook = module, hook
    def remove(self): self.module.hooks.remove(self.hook)


class Module:
    def __init__(self): self.hooks = []; self.training = True
    def train(self, mode): self.training = mode; return self
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return Handle(self, hook)
    def forward(self, output):
        for hook in list(self.hooks): hook(self, (), output)


class Model:
    def __init__(self): self.module = Module(); self.training = True
    def modules(self): return [self, self.module]
    def eval(self): return self.train(False)
    def train(self, mode):
        self.training = mode; self.module.train(mode); return self
    def __call__(self, output):
        if self.training or self.module.training:
            raise AssertionError("probe must execute in eval mode")
        self.module.forward(output)
    def named_modules(self): return [("block", self.module)]


def recorder():
    return ActivationCapture(model_name="existing-local-model", tokenizer_name="local-tokenizer",
                             prompt="fixed probe", token_ids=[10, 11, 12], token_texts=["a", "b", "c"],
                             positions=[0, 2], layers=["block"], mode="training")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        Tensor.calls = []
        fake = types.SimpleNamespace(Tensor=Tensor, long="long", float32="float32",
                                     tensor=lambda positions, **kw: positions,
                                     no_grad=nullcontext,
                                     isfinite=lambda sample: Finite(all(math.isfinite(x) for row in sample.data for x in row)))
        self.torch_patch = patch.dict(sys.modules, {"torch": fake})
        self.torch_patch.start()
        self.addCleanup(self.torch_patch.stop)

    def test_detached_sampling_and_hook_cleanup(self):
        capture, model = recorder(), Model()
        output = Tensor([[1, 2, 3], [100, 200, 300], [4, 5, 6]])
        with capture.installed(model):
            model.module.forward(output)  # inactive training calls are ignored
            self.assertEqual(capture.trace["snapshots"], [])
            with capture.snapshot(checkpoint="step-10", training_step=10):
                model.module.forward((output, "unused output"))
        self.assertEqual(model.module.hooks, [])
        self.assertEqual(Tensor.calls, ["detach", "sample", "cpu"])
        frame = capture.trace["snapshots"][0]
        self.assertEqual(frame["activations"], [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(frame["trainingStep"], 10)
        self.assertEqual([t["position"] for t in capture.trace["tokens"]], [0, 2])
        validate_trace(capture.trace)
        # Final record consists of serializable metadata/numbers, not tensor objects.
        with tempfile.TemporaryDirectory() as tmp:
            capture.write(Path(tmp) / "trace.json")
            self.assertTrue((Path(tmp) / "trace.json").is_file())

    def test_failed_capture_rolls_back_and_removes_hooks(self):
        capture, model = recorder(), Model()
        with self.assertRaises(ValueError), capture.installed(model):
            with capture.snapshot(checkpoint="step-1", training_step=1):
                model.module.forward(Tensor([[1, 2, 3]] * 3))
                model.module.forward(Tensor([[1, 2, 3]] * 3))
        self.assertEqual(capture.trace["snapshots"], [])
        self.assertEqual(capture._values, 0)
        self.assertEqual(model.module.hooks, [])
        self.assertIsNone(capture._scope)

    def test_resource_bound_preserves_prior_snapshot(self):
        capture, model = recorder(), Model()
        with patch("capture_activations.MAX_VALUES", 10), capture.installed(model):
            with capture.snapshot(checkpoint="first", training_step=1):
                model.module.forward(Tensor([[1, 2, 3]] * 3))
            with self.assertRaises(ValueError), capture.snapshot(checkpoint="second", training_step=2):
                model.module.forward(Tensor([[1, 2, 3]] * 3))
        self.assertEqual(len(capture.trace["snapshots"]), 1)
        self.assertEqual(capture._values, 6)

    def test_missing_selected_module_and_nonfinite_fail(self):
        capture, model = recorder(), Model()
        with capture.installed(model):
            with self.assertRaises(ValueError), capture.snapshot(checkpoint="missing"):
                pass
            with self.assertRaises(ValueError), capture.snapshot(checkpoint="nan"):
                model.module.forward(Tensor([[float("nan"), 2, 3]] * 3))
        self.assertEqual(capture.trace["snapshots"], [])

    def test_training_probe_restores_modes_even_on_failure(self):
        capture, model = recorder(), Model()
        model.module.training = False  # preserve a custom child mode
        with capture.installed(model):
            record_probe(model, capture, {"output": Tensor([[1, 2, 3]] * 3)}, checkpoint="step-1", training_step=1)
            self.assertTrue(model.training)
            self.assertFalse(model.module.training)
            with self.assertRaises(ValueError):
                record_probe(model, capture, {"output": Tensor([[float("nan"), 2, 3]] * 3)}, checkpoint="step-2", training_step=2)
            self.assertTrue(model.training)
            self.assertFalse(model.module.training)
        self.assertEqual(len(capture.trace["snapshots"]), 1)

    def test_synthetic_portable_contract(self):
        trace = synthetic_trace()
        validate_trace(trace)
        trace["tokens"][1]["id"] = trace["tokens"][0]["id"]
        with self.assertRaises(ValueError): validate_trace(trace)
        trace = synthetic_trace()
        trace["snapshots"][0]["activations"][0].append(1)
        with self.assertRaises(ValueError): validate_trace(trace)


if __name__ == "__main__":
    unittest.main(verbosity=2)
