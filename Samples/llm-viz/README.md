# Public recorded tiny-GPT activations

Source: Brendan Bycroft, llm-viz, MIT license (see LICENSE).
Pinned commit: `9da93742382f1bf36c020c38a1ace454e82c4490`.
[Original fixture](https://github.com/bbycroft/llm-viz/blob/9da93742382f1bf36c020c38a1ace454e82c4490/public/gpt-nano-sort-t0-partials.json).
SHA-256: `642b29b966fc4d90e4672a54c7cfda72b5249427b2b1561c83cd536b857195e9`.

The source records a tiny trained sorting GPT: three transformer layers, three heads, 48 embedding features, vocabulary A/B/C. We decode existing float32 tensors, without downloading weights or running the model. Tensors x, block0, block1, block2 each have shape [3,11,48]. We retain only the first six input tokens in each batch (AACBAB, ABCCCA, BACBCC), giving 18 points, 48 features and four actual snapshots. Positions 6–10 are diagnostic padding and excluded. Training step is unknown. Colors are literal input labels; they do not claim semantic clustering.

Run `python3 Capture/convert_public_sorting_trace.py` from the project root to verify the pinned checksum and regenerate `Samples/public-sorting-trace.json`. The converter preserves signed activation values and stable batch/position IDs. Normalization uses one reference across all four snapshots. No independently fitted per-layer scale or synthetic expansion is applied. This is a tiny public teaching-model example, not a production-LLM performance benchmark.
