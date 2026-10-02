"""Local-only activation recorder for an already-loaded PyTorch model.

No model loading, downloads, installs, network clients, or training loop here.
Only explicitly selected named modules and fixed probe token positions are sampled.
PyTorch is required in the caller's existing environment, not installed by this file.
"""
from contextlib import contextmanager
import json
from pathlib import Path

MAX_TOKENS = 128
MAX_DIMENSIONS = 4096
MAX_SNAPSHOTS = 16
MAX_VALUES = 1_048_576
MAX_BYTES = 32_000_000


def validate_trace(trace):
    """Dependency-free validation for the portable v1 viewer envelope."""
    if trace.get("schemaVersion") != 1:
        raise ValueError("schemaVersion must be 1")
    if trace.get("mode") not in ("inference", "training", "synthetic"):
        raise ValueError("invalid mode")
    if not trace.get("model") or not trace.get("tokenizer"):
        raise ValueError("model/tokenizer metadata required")
    tokens, snapshots = trace["tokens"], trace["snapshots"]
    if not 2 <= len(tokens) <= MAX_TOKENS or not 1 <= len(snapshots) <= MAX_SNAPSHOTS:
        raise ValueError("token/snapshot count bound exceeded")
    if len({t["id"] for t in tokens}) != len(tokens):
        raise ValueError("duplicate token identity")
    if len({(t["sequenceID"], t["position"]) for t in tokens}) != len(tokens):
        raise ValueError("duplicate sequence/position")
    if any(not t["id"] or not t["sequenceID"] or t["position"] < 0 for t in tokens):
        raise ValueError("invalid token identity")
    if len({s["id"] for s in snapshots}) != len(snapshots):
        raise ValueError("duplicate snapshot identity")
    import math
    dimensions = len(snapshots[0]["activations"][0])
    if not 3 <= dimensions <= MAX_DIMENSIONS:
        raise ValueError("feature dimension bound exceeded")
    total = 0
    for snapshot in snapshots:
        if any(not snapshot[key] for key in ("id", "layer", "checkpoint", "dtype")):
            raise ValueError("snapshot metadata required")
        if snapshot.get("trainingStep") is not None and snapshot["trainingStep"] < 0:
            raise ValueError("negative training step")
        if len(snapshot["activations"]) != len(tokens):
            raise ValueError("ordered token rows must match")
        for row in snapshot["activations"]:
            if len(row) != dimensions or not all(math.isfinite(x) and abs(x) <= 1e100 for x in row):
                raise ValueError("ragged/nonfinite activation row")
            total += len(row)
            if total > MAX_VALUES:
                raise ValueError("total activation bound exceeded")


def write_trace(trace, path):
    validate_trace(trace)
    encoded = json.dumps(trace, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > MAX_BYTES:
        raise ValueError("serialized trace exceeds 32 MB")
    # Caller chooses the local path. No upload or remote storage.
    Path(path).write_bytes(encoded)


class ActivationCapture:
    """Capture a fixed probe, batch index zero, selected [batch, token, feature] outputs.

    Construct metadata from the exact fixed probe inputs. Use named-module paths
    from model.named_modules(); output must be a Tensor or a tuple whose first item
    is a Tensor. Model-specific adapters are needed for other output structures.
    Snapshot scope labels a probe forward, not the ordinary training minibatches.
    """
    def __init__(self, *, model_name, tokenizer_name, prompt, token_ids,
                 token_texts, positions, layers, mode="inference", sequence_id="probe0"):
        if not 2 <= len(positions) <= MAX_TOKENS or len(set(positions)) != len(positions):
            raise ValueError("select 2–128 distinct token positions")
        if not positions or min(positions) < 0 or max(positions) >= len(token_ids):
            raise ValueError("positions outside fixed probe")
        if len(token_ids) != len(token_texts):
            raise ValueError("token text metadata length mismatch")
        if not layers or len(set(layers)) != len(layers) or len(layers) > MAX_SNAPSHOTS:
            raise ValueError("select 1–16 distinct named modules")
        if mode not in ("inference", "training"):
            raise ValueError("capture mode must be inference or training")
        self.positions = list(positions)
        self.layers = list(layers)
        self.trace = {
            "schemaVersion": 1, "model": model_name, "tokenizer": tokenizer_name,
            "prompt": prompt, "mode": mode,
            "tokens": [{"id": f"{sequence_id}:{i}", "sequenceID": sequence_id,
                        "tokenID": int(token_ids[i]), "text": str(token_texts[i]),
                        "position": i} for i in positions], "snapshots": []}
        self._scope = None
        self._seen = set()
        self._values = 0
        self._dimensions = None

    @contextmanager
    def installed(self, model):
        """All hooks are removed even when capture or the caller forward fails."""
        modules = dict(model.named_modules())
        if any(name not in modules for name in self.layers):
            raise ValueError("unknown selected module path")
        handles = []
        try:
            for layer in self.layers:
                def hook(_module, _args, output, layer=layer):
                    if self._scope is None:
                        return
                    self._record(layer, output)
                handles.append(modules[layer].register_forward_hook(hook))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def snapshot(self, *, checkpoint, training_step=None):
        if self._scope is not None:
            raise ValueError("nested snapshot scopes are unsupported")
        if not checkpoint or (training_step is not None and training_step < 0):
            raise ValueError("checkpoint and nonnegative step required")
        if len(self.trace["snapshots"]) + len(self.layers) > MAX_SNAPSHOTS:
            raise ValueError("snapshot bound reached; write a separate trace")
        old_count, old_values, old_dimensions = len(self.trace["snapshots"]), self._values, self._dimensions
        self._scope = (str(checkpoint), training_step)
        self._seen = set()
        try:
            yield
            if self._seen != set(self.layers):
                raise ValueError("not all selected modules ran in the probe forward")
        except BaseException:
            del self.trace["snapshots"][old_count:]
            self._values, self._dimensions = old_values, old_dimensions
            raise
        finally:
            self._scope = None
            self._seen = set()

    def _record(self, layer, output):
        import torch  # existing caller environment only
        if layer in self._seen:
            raise ValueError("selected module ran more than once; use a model-specific adapter")
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError("expected selected module output [batch, token, feature]")
        if tensor.shape[0] < 1 or max(self.positions) >= tensor.shape[1]:
            raise ValueError("probe token positions outside module output")
        dimensions = int(tensor.shape[2])
        count = len(self.positions) * dimensions
        if not 3 <= dimensions <= MAX_DIMENSIONS or self._values + count > MAX_VALUES:
            raise ValueError("activation resource bound reached")
        if self._dimensions is not None and self._dimensions != dimensions:
            raise ValueError("all selected layers must share feature coordinates/dimension")
        checkpoint, step = self._scope
        identity = f"{layer}@{checkpoint}:step={step}"
        if any(s["id"] == identity for s in self.trace["snapshots"]):
            raise ValueError("duplicate layer/checkpoint/step snapshot")
        # Detach BEFORE slicing/indexing, then copy ONLY the bounded sample to CPU.
        # No Tensor or autograd graph is stored in trace; tolist materializes numbers.
        indices = torch.tensor(self.positions, device=tensor.device, dtype=torch.long)
        with torch.no_grad():
            sampled = tensor.detach()[0].index_select(0, indices).to(device="cpu", dtype=torch.float32).clone()
            if not bool(torch.isfinite(sampled).all()):
                raise ValueError("nonfinite sampled activations")
            values = sampled.tolist()
        self.trace["snapshots"].append({"id": identity, "layer": layer,
            "checkpoint": checkpoint, "trainingStep": step, "dtype": str(tensor.dtype),
            "activations": values})
        self._seen.add(layer); self._values += count; self._dimensions = dimensions

    def write(self, path):
        write_trace(self.trace, path)
