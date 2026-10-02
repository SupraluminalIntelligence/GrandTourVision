"""Dependency-free engineered fixture. No model run; clusters are designed."""
import math
from pathlib import Path
from capture_activations import write_trace


def dense_basis(dimensions=32):
    seed = 2026
    def uniform():
        nonlocal seed
        seed = (seed * 6364136223846793005 + 1) & ((1 << 64) - 1)
        return max(1e-12, (seed >> 11) / (((1 << 64) - 1) >> 11))
    basis = []
    for _ in range(3):
        v = [math.sqrt(-2*math.log(uniform())) * math.cos(2*math.pi*uniform()) for _ in range(dimensions)]
        for previous in basis:
            dot = sum(x*y for x, y in zip(v, previous))
            v = [x-dot*y for x, y in zip(v, previous)]
        norm = math.sqrt(sum(x*x for x in v))
        basis.append([x/norm for x in v])
    return basis


def synthetic_trace():
    tokens = [{"id": f"prompt0:{i}", "sequenceID": "prompt0", "tokenID": 100+i,
               "text": ["The", "model", "learns", "patterns"][i % 4], "position": i,
               "category": ["Designed A", "Designed B", "Designed C", "Designed D"][i // 32]} for i in range(128)]
    basis = dense_basis()
    centers = [[-2.5, -1.6, -0.7], [2.4, -1.5, 0.6], [-2.1, 1.7, 0.9], [2.0, 1.7, -0.7]]
    frames = []
    for step in [0, 100]:
        for layer in ["block.0", "block.1"]:
            vectors = []
            for i in range(128):
                group = i // 32
                phase = (i % 32) * 2.3999632297
                spread = 0.18 + ((i % 7) + 1) * 0.028
                center = list(centers[group])
                if layer == "block.1":
                    center[0] *= 0.82; center[1] *= 1.1
                    center[2] += (1 if group % 2 == 0 else -1) * 0.6
                if step == 100:
                    center[0] += (-1 if group < 2 else 1) * 0.4
                    center[1] += (1 if group % 2 == 0 else -1) * 0.25
                local = [math.cos(phase)*spread, math.sin(phase)*spread*0.72, math.sin(phase*1.37)*spread*0.8]
                vectors.append([sum((center[a]+local[a])*basis[a][d] for a in range(3))
                                + math.sin((i+1)*(d+1)*0.17)*0.035 for d in range(32)])
            frames.append({"id": f"{layer}@{step}", "layer": layer, "checkpoint": f"step-{step}",
                           "trainingStep": step, "dtype": "float32", "activations": vectors})
    return {"schemaVersion": 1, "model": "Synthetic activations (no LLM run)",
            "tokenizer": "synthetic-v1", "prompt": "Engineered synthetic clusters; repeated placeholder tokens, no language model capture",
            "mode": "synthetic", "tokens": tokens, "snapshots": frames}


if __name__ == "__main__":
    write_trace(synthetic_trace(), Path(__file__).resolve().parents[1] / "Samples/synthetic-trace.json")


def room_demo_trace():
    """Diffuse, full-dimensional fabricated vectors for room-scale tour demos."""
    import random
    rng = random.Random(20261001)
    base = [[rng.gauss(0, 1) for _ in range(64)] for _ in range(128)]
    tokens = [{"id": f"demo:{i}", "position": i, "tokenID": i,
               "text": f"synthetic {i}", "sequenceID": "fabricated-probe"} for i in range(128)]
    frames = []
    for step in [0, 100]:
        for layer in ["block.0", "block.1"]:
            rows = []
            for i, row in enumerate(base):
                rows.append([(v * (0.85 + 0.18 * math.sin(d * 0.3)) if layer == "block.1" else v)
                             + (0.18 * math.sin(i * 0.15 + d * 0.4) if step == 100 else 0)
                             for d, v in enumerate(row)])
            frames.append({"id": f"{layer}@{step}", "layer": layer, "checkpoint": f"step-{step}",
                           "trainingStep": step, "dtype": "float32", "activations": rows})
    return {"schemaVersion": 1, "model": "Room demo — synthetic 64D vectors (NO MODEL RUN)",
            "tokenizer": "fabricated identities", "prompt": "Seeded full-dimensional Gaussian vectors; designed layer/step changes, no language model capture",
            "mode": "synthetic", "tokens": tokens, "snapshots": frames}


if __name__ == "__main__":
    write_trace(room_demo_trace(), Path(__file__).resolve().parents[1] / "Samples/room-demo-trace.json")
