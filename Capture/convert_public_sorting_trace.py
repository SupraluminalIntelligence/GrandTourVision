"""Convert a pinned MIT public activation fixture; no model execution/dependencies."""
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
from capture_activations import write_trace

COMMIT = "9da93742382f1bf36c020c38a1ace454e82c4490"
SHA256 = "642b29b966fc4d90e4672a54c7cfda72b5249427b2b1561c83cd536b857195e9"
SOURCE_URL = f"https://github.com/bbycroft/llm-viz/blob/{COMMIT}/public/gpt-nano-sort-t0-partials.json"


def convert(path):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError("pinned source checksum mismatch")
    source = json.loads(data)
    assert source["config"]["n_layer"] == 3 and source["config"]["n_embd"] == 48
    def tensor(name, shape):
        item = source[name]
        if item["shape"] != shape or item["dtype"] != "torch.float32":
            raise ValueError("unexpected tensor shape/dtype")
        binary = base64.b64decode(item["data"], validate=True)
        count = math.prod(shape)
        if len(binary) != count * 4:
            raise ValueError("unexpected tensor byte count")
        values = struct.unpack("<" + "f" * count, binary)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite public data")
        return values
    idx = tensor("idx", [3, 11])
    tokens, prompts = [], []
    for batch in range(3):
        prompt = ""
        for position in range(6):  # only actual input; exclude diagnostic/padding positions 6–10
            token_id = int(idx[batch * 11 + position])
            if token_id not in [0, 1, 2] or token_id != idx[batch * 11 + position]:
                raise ValueError("invalid sorting token ID")
            text = "ABC"[token_id]; prompt += text
            tokens.append({"id": f"batch{batch}:{position}", "sequenceID": f"batch{batch}",
                           "tokenID": token_id, "text": text, "position": position,
                           "category": f"Input {text}"})
        prompts.append(f"batch{batch}: {prompt}")
    frames = []
    for layer in ["x", "block0", "block1", "block2"]:
        values = tensor(layer, [3, 11, 48])
        rows = []
        for batch in range(3):
            for position in range(6):
                offset = (batch * 11 + position) * 48
                rows.append(list(values[offset:offset + 48]))
        frames.append({"id": f"{layer}@published-t0", "layer": layer, "checkpoint": "published t0",
                       "trainingStep": None, "dtype": "torch.float32", "activations": rows})
    return {"schemaVersion": 1, "model": "Tiny GPT sorting — real recorded activations",
            "tokenizer": "ABC sort vocabulary: A=0, B=1, C=2", "prompt": "; ".join(prompts),
            "mode": "inference", "tokens": tokens, "snapshots": frames,
            "source": {"title": "bbycroft / llm-viz", "url": SOURCE_URL, "license": "MIT"}}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    trace = convert(root / "Samples/llm-viz/gpt-nano-sort-t0-partials.json")
    write_trace(trace, root / "Samples/public-sorting-trace.json")
    print(trace["prompt"])
    print("18 actual input-token points × 48 features × 4 recorded snapshots; no synthetic augmentation")
