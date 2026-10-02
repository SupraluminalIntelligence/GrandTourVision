# Portable activation trace v1

The host runs an existing model and exports a local UTF-8 JSON file. The Vision Pro app only views numbers and metadata. It does not load a model, execute inference, train, or call an API.

Use **Import CSV / trace** and select the JSON file, or choose **Synthetic trace**. The packaged `Samples/synthetic-trace.json` has 128 matching token points, 32 features, two layers and two synthetic training steps. It is not a model capture. Regenerate it with `python3 Capture/make_synthetic_trace.py`, using only the Python standard library.

## JSON contract

```json
{
  "schemaVersion": 1,
  "model": "local-model-name-or-fingerprint",
  "tokenizer": "tokenizer-name-and-version",
  "prompt": "the fixed probe prompt, or a redacted label",
  "mode": "training",
  "tokens": [
    {"id":"probe0:0","sequenceID":"probe0","tokenID":10,"text":"Hello","position":0},
    {"id":"probe0:1","sequenceID":"probe0","tokenID":11,"text":" world","position":1}
  ],
  "snapshots": [
    {
      "id":"block.0@step-100",
      "layer":"block.0",
      "checkpoint":"step-100",
      "trainingStep":100,
      "dtype":"torch.float32",
      "activations":[[0.1,0.2,0.3],[0.2,0.4,0.6]]
    }
  ]
}
```

`mode` is inference, training or synthetic. Inference `trainingStep` is null or omitted. Activations have shape [sampled-token, feature]. Row order must exactly match the shared `tokens` array in every snapshot. IDs and (sequenceID, position) pairs must be unique; token metadata is shared, so different prompts/tokenizations should be separate traces. Each snapshot has a unique ID and explicit layer/checkpoint/source-dtype metadata. Values are serialized numbers; the capture example converts sampled values to float32, recording the source dtype. Use layers with compatible feature coordinates and the same dimension; unequal dimensions are rejected rather than padded, truncated or independently projected.

Current envelope: 2–128 token points, 3–4,096 features, 1–16 snapshots, at most 1,048,576 scalar values **across all snapshots**, and 32 MB UTF-8 JSON. The total-value cap also applies when individual limits are satisfied. For example, 128 tokens × 4,096 features × 2 snapshots uses the full value budget; choose fewer tokens for more layers/checkpoints. Imports reject invalid/ragged/nonfinite values and unsupported versions. These are bounded implementation limits, not measured headset throughput promises.

## Capturing an existing host model

`capture_activations.py` selects explicitly named PyTorch modules whose outputs are [batch, token, feature] tensors (or tuples with such a tensor first). The sample uses batch index zero and fixed token positions. It is an adapter for common transformer blocks, not every architecture. Choose paths from `model.named_modules()`. Other output layouts, repeated module calls, generation streams/KV caches, variable token sequences and different layer dimensions need adapters and are not accepted silently.

`host_example.py` accepts an already-loaded model/tokenizer; it performs no downloads or package installation. PyTorch/tokenizer dependencies must already exist in the user's own host environment. Do not invoke it until model execution is authorized and resource requirements are understood.

Inference example (caller supplies all named objects):

```python
from host_example import record_inference
record_inference(
    model, tokenizer, prompt="your fixed probe prompt",
    positions=[0, 1, 2, 3], layers=["model.layers.0", "model.layers.5"],
    model_name="local-model/fingerprint", tokenizer_name="local-tokenizer/version",
    device="your-existing-device", output_path="inference-trace.json",
    checkpoint="checkpoint-name", forward_kwargs={"use_cache": False},
)
```

Paths and optional kwargs are model-specific. This code has not been run against a real model here.

Training snapshot integration (illustrative; no training was run):

```python
from host_example import make_probe_capture, record_probe
capture, probe = make_probe_capture(
    model, tokenizer, prompt="same probe at every step", positions=[0, 1, 2, 3],
    layers=["model.layers.0", "model.layers.5"], model_name="same-training-run",
    tokenizer_name="same-tokenizer/version", mode="training", device=device,
)
with capture.installed(model):
    for step, batch in enumerate(existing_training_loader):
        # Existing caller-owned training code. Hooks are inactive here.
        optimizer.zero_grad()
        loss = existing_loss_function(model, batch)
        loss.backward()
        optimizer.step()
        if step in {100, 200}:
            # Fixed probe, recorded after this optimizer update. It uses eval/no_grad,
            # restores previous module training modes, and doesn't retain the output.
            record_probe(model, capture, probe, checkpoint=f"step-{step}",
                         training_step=step, forward_kwargs={"use_cache": False})
capture.write("training-trace.json")
```

The selected capture hook first detaches the layer output, samples only bounded token rows, copies those rows to CPU and materializes plain numbers. It does not keep tensors, full layer outputs or autograd graphs in the trace. Ordinary training forward/backward is not wrapped in no_grad. Snapshot scopes enforce value/snapshot bounds, rollback incomplete captures and remove hooks on exit. Capturing incurs extra probe forward/CPU synchronization cost; frequency is chosen by the caller. Activation checkpointing, compiled/fused models, parallel/sharded models and arbitrary custom mode behavior may need additional adapters and validation.

## Comparison semantics

Normalization statistics are fitted once across ALL snapshots in an imported trace; choosing a frame never refits them. The same dense orthonormal projection and fitted display scale are shared across layers/checkpoints. Selecting a new frame pauses playback and preserves selected token identity and physical plot pose. Lock projection to pause and guard tour/manual changes while comparing; explicitly starting animation unlocks it. Normalization controls are disabled while locked. There is no per-frame PCA or coordinate rotation. The initial dense basis is deterministic; it is a generic projection, not a learned interpretability direction.

The trace fit uses the maximum projected radius across all snapshots in the current shared basis. A tour/manual projection can leave that fixed display reference; in the mixed immersive room all projected points remain visible, without clipping or clamping. **Fit this projection across all snapshots** creates a new shared display scale explicitly. Re-lock before comparing. CSV/demo mode retains its full-dimensional norm bound.

The layer selector retains the selected checkpoint when available and vice versa. If the requested combination is absent, it selects the first available matching layer/checkpoint and shows the actual snapshot ID/step; check those fields. Selecting a token in the controls or tapping its point shows ID, sequence, position, tokenizer ID and token text. Selected points are enlarged; colors are stable by sampled point index, not inferred semantic clusters.

This is a local exploratory viewer, not proof of causal interpretability. Matching dimensions alone does not guarantee aligned feature meaning. Even layers within one model can use changing representations; separately trained models/checkpoints may require explicit alignment. Trace files can contain sensitive prompts/token text and model activations. The capture/viewer code uses no network; keep actual traces local and transfer them deliberately. Synthetic data, the attributed public teaching-model fixture, and source code were saved to private Library for delivery.

API references: [PyTorch Module hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html), [detach](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.detach.html), [no_grad](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html).

Optional token `category` is a literal categorical label. If every token has one and there are at most four distinct labels, the viewer colors points by those labels; otherwise it uses one neutral group. Optional top-level `source` has `title`, `url`, and `license` strings and is shown in the viewer. Category does not enter projection or normalization.

The optional `Samples/room-demo-trace.json` is a seeded full-dimensional fabricated 128-token × 64-feature fixture with four designed snapshots. It is explicitly synthetic, not a residual/model capture or unit-RMS measurement. It supports diffuse room-scale tour previews and defaults to token-position coloring. Regenerate with `python3 Capture/make_synthetic_trace.py`.
