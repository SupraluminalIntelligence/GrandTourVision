"""Record the bundled FlowScope samples: real training runs of Karpathy's Zero-to-Hero GPT (12 blocks).

    uv run --project ../flowscope python Capture/record_flowscope_samples.py

Writes Samples/flowscope/{healthy,no-residual}-12-layers.jsonl: one tour frame every 200 steps, from step 0
(rank collapse is visible at initialization) to step 1000. Needs ../flowscope (and its data/input.txt).
"""
import importlib.util, os, pathlib

import torch
from flowscope import FlowScope

ROOT = pathlib.Path(__file__).resolve().parents[1]
FLOWSCOPE = ROOT.parent / "flowscope"
spec = importlib.util.spec_from_file_location("gpt", FLOWSCOPE / "examples" / "gpt.py")
gpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpt)

text = (FLOWSCOPE / "data" / "input.txt").read_text()
chars = sorted(set(text))
data = torch.tensor([{c: i for i, c in enumerate(chars)}[c] for c in text])

for name, flags in (("healthy", []), ("no-residual", ["--no-residual"])):
    cfg = gpt.parse_args(["--layers", "12", *flags])
    torch.manual_seed(1337)
    model = gpt.GPT(cfg, len(chars))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    out = ROOT / "Samples" / "flowscope" / f"{name}-12-layers.jsonl"
    if out.exists():
        out.unlink()
    FlowScope(model, opt, every=200, label=f"12 blocks, {name.replace('-', ' ')}", sink=None, record=str(out))
    for step in range(1001):
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        x = torch.stack([data[i:i + cfg.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + cfg.block_size + 1] for i in ix])
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"{out.relative_to(ROOT)}: loss {loss.item():.3f}, {os.path.getsize(out) // 1024} KB")
