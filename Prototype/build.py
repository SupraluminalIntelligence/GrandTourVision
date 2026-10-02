"""Build inside-a-block.html: the prototype page with real FlowScope recordings embedded.

    python3 Prototype/build.py   (needs ../flowscope/data/input.txt for the character vocabulary)
"""
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
text = (ROOT.parent / "flowscope" / "data" / "input.txt").read_text()
runs = {}
for name, title in (("no-residual", "No residual"), ("healthy", "Healthy")):
    frames = [json.loads(line) for line in (ROOT / "Samples" / "flowscope" / f"{name}-12-layers.jsonl").read_text().splitlines()]
    keep = {str(f["step"]): {k: f[k] for k in ("step", "C", "N", "T", "k", "layers", "tokens")} for f in frames if f["step"] in (0, 1000)}
    runs[name] = {"title": title, "frames": keep}
data = "window.FLOW_DATA = " + json.dumps({"vocab": "".join(sorted(set(text))), "runs": runs}, separators=(",", ":")) + ";"
src = (ROOT / "Prototype" / "inside-a-block.src.html").read_text()
out = ROOT / "Prototype" / "inside-a-block.html"
out.write_text(src.replace("/*FLOW_DATA*/", data))
print(f"{out.relative_to(ROOT)}: {out.stat().st_size // 1024} KB")
