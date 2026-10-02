# Grand Tour Vision — one room-scale representation

Native visionOS prototype with one selected point cloud in a **mixed ImmersiveSpace**, blended with passthrough. It is independent of window/volume bounds: you can walk around and into the representation, animate its high-dimensional projection, and move/rotate/resize the physical plot at the same time.

## FlowScope layer gallery (start here)

The app opens on the layer gallery: real training runs, every block of a transformer around you. Each block is a
cloud of 256 token vectors from the 64-dimensional residual stream, placed on an arc at arm's length (embeddings on
the left, last block on the right), all seen through the same still 3D window. Each block shows `dim` (effective
dimensionality) and `% seen` (share of its variance inside the view's principal directions).

1. Pick a bundled run under **Run**: **No residual · 12 blocks** (loaded by default) or **Healthy · 12 blocks**.
   Both are real recordings of Karpathy's Zero-to-Hero GPT, steps 0-1000. Regenerate with
   `uv run --project ../flowscope python Capture/record_flowscope_samples.py`.
2. **Enter gallery.** Nothing moves: it's a still view with X/Y/Z = PC1/PC2/PC3.
3. **Step inside a block**: tap it (or pick it under Space). It becomes a fixed window in front of you, a 2.4 m cube
   (or sphere) you can walk to and into. The window never moves; you move the data through it:
   - **Pinch the data with both hands** to zoom: zoom in and tokens spread apart and leave through the boundary;
     zoom out and data from outside comes in. Faint dots on the boundary hint at data just outside.
   - **Pinch and drag the data** to slide it through the window.
   - **Pull a direction knob** toward an axis to **tilt the window** inside the high-dimensional space. Each knob is
     one principal direction, labelled with how much of it you see: it sits on its spoke tip inside the window when
     the direction is in view, and waits in a column along the window's front-left edge when it's hidden.
     Tilting is the only motion that changes the shape: with 8 directions toured there are 15 ways to tilt
     (3 axes x 5 hidden directions); page the handles and raise "directions toured" to reach all 64.
   - **Tap a token** to see its character in the sentence around it; a line traces that sentence through the window.
   - **Slice** mode shows only tokens near this 3D slice of the space; **Depth** slides the slice along a hidden
     direction, so tokens enter and leave without anything moving.
   The other blocks wait on a rail to your left; tap one to switch. Every gesture has a button in the panel too.
4. **Tour (optional)**: Start tour moves the window on its own (the grand tour from [FlowScope](../flowscope), same
   math: the Swift port matches `tour.js` to 1e-9), starting from wherever you left it; stopping keeps that view.
5. **Training step** compares the same view across training. Step 0 of the no-residual run already shows rank
   collapse: from block 3 on, the tokens lie on thin strands (~2 effective dims).

The interaction was designed in `Prototype/` first: a walkable browser version on the same data
(`python3 Prototype/build.py` builds `Prototype/inside-a-block.html`).

**Live:** on the Mac run `flowscope run train.py --host 0.0.0.0` (it prints the address, e.g. `192.168.1.20:8765`),
enter it under **Run**, and **Connect live**: the gallery updates while you train. The first time, the app asks for
local-network permission (plain HTTP to your Mac only). Record a run for later with `--record run.jsonl`.

The classic single-cloud plot (CSV import, toy 4-D clusters, synthetic traces) opens from **Run**.

## Try it (classic single-cloud room plot)

1. Open `GrandTourVision.xcodeproj` in Xcode and run on a Vision Pro simulator.
2. Choose **Room demo (synthetic)** for 128 diffuse fabricated token vectors × 64 dimensions × two layers × two designed checkpoints. The demo is seeded Gaussian data, explicitly labeled **NO MODEL RUN**; it is not a model result.
3. Choose **Enter room-scale plot**. This starts the tour and unlocks comparison. A single cloud appears 3.5 metres ahead at 1.5 m height, with a 4 m coordinate scale.
4. **Pause animation / Animate projection** and the speed slider control the tour. Grab a point and drag, rotate with two hands, or pinch to resize while animation continues. Physical placement never changes projection weights. **Smaller / Larger / Move left / Move right / Rotate scene** provide accessible alternatives.
5. **Hide controls** reduces the console to a small bar with show/play/pause/exit actions. **Recenter scene** restores the initial location, orientation and 4 m scale without pausing animation. Scale can range from 1.5–6 m; actual point-cloud extent changes with the projection.
6. For comparison, pause and enable **Lock projection for layer comparison**, then select a layer and training-step/checkpoint. Selection pauses the tour and preserves projection, shared normalization/display reference, token identities and physical pose. One representation is shown at a time.
7. Expand **Manual high-dimensional projection** to choose an input-dimension plane and move its angle, or use **Nudge projection**. This pauses animation and changes the orthonormal basis; it does not move the scene or create a data slice. **Reset projection** pauses and restores the initial basis/display reference while keeping physical placement.
8. **Data & axis contributions** opens detailed controls for import, normalization, signed X/Y/Z weights and session projection bookmarks. **Return to room controls** closes that window. **Exit room** returns to data selection; reentry resets scene placement and resumes the tour.

The console uses system windows so its controls remain interactive; the in-scene attachment is an informational legend. The immersive scene has one RealityKit root/point set. No grid, duplicated layer panels or oversized bounded volume is used.

## Data and mathematical contract

- **Room demo (synthetic)**: diffuse 128 × 64 vectors, four fabricated snapshots; color by token position by default. Regenerate with `python3 Capture/make_synthetic_trace.py`.
- **Synthetic trace**: the earlier 128 × 32 designed four-group fixture, two layers/two synthetic checkpoints.
- **Synthetic 4D**: 300 points in four separated clusters.
- **Public tiny GPT**: 18 real recorded input-token activations × 48 features × four snapshots from Brendan Bycroft's MIT-licensed [llm-viz fixture](https://github.com/bbycroft/llm-viz/blob/9da93742382f1bf36c020c38a1ace454e82c4490/public/gpt-nano-sort-t0-partials.json). Three six-token inputs are retained; diagnostic padding is excluded. Training step is unknown. No model was run here. See [provenance](Samples/llm-viz/README.md).
- **CSV / portable JSON import**: see [trace schema and capture examples](Capture/TRACE_SCHEMA.md). CSV supports 3–12 complete numeric columns, 2–1,000 rows, 2 MB and 64 source columns. JSON supports 2–128 sampled tokens, 3–4,096 features, 1–16 snapshots, at most 1,048,576 total scalars and 32 MB. These are implementation bounds, not measured headset throughput promises.

Three orthonormal basis vectors project the input coordinates to 3D. Smooth connected ambient rotations advance the tour in O(D); no finite run is claimed to cover every projection. Physical placement is a separate metre-based `ScenePlacement` with a unit quaternion. Animation updates point coordinates; gestures update placement. Neither overwrites the other.

Normalization is explicit: column centering or z-score, fitted once across every trace snapshot. The initial display reference fits the current shared projection across all snapshots and stays fixed during animation and layer selection. In the unbounded immersive room, projected excursions remain visible; there is no clipping, per-frame fitting or independent per-layer scaling. **Fit this projection across all snapshots** changes the reference explicitly. Colors denote supplied categories or token position, not inferred semantics. The reference image's unit-RMS, participation-ratio and variance-percentage metrics are not fabricated or claimed by this MVP.

Saved projection viewpoints are limited to eight per session/dataset; imports or relaunch clear them. Matching feature dimensions do not establish aligned feature meaning across independently trained models.

## Validation and recording

Xcode 27.0 (27A266a), visionOS / Simulator SDK 27.0. Official Apple visionOS 27.0 arm64 simulator runtime (24M362) installed earlier with user approval. Final unsigned arm64 simulator and physical-device SDK builds pass. `./check-core.sh` passes 119 Swift checks. `python3 Capture/test_capture.py` passes six dependency-free tests using a mocked tensor protocol; actual PyTorch integration was not run.

Native `ImmersivePlotTests` pass: immersive entry, 128 × 64 demo, animated tour time, placement/scale changes while animation remains on, recenter, hide/show, pause/lock, layer/checkpoint selection, exit/reentry, manual projection and resume. Simulator screenshots and a 10-second video are actual framebuffers, visually inspected. They do not establish physical headset gesture tracking or device performance. See [validation](VALIDATION.md).

To reproduce captures, run the native UI tests with log `ui-immersive-final.log`, and alongside them run:

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer python3 Scripts/capture_simulator_frames.py --immersive --video --udid <simulator-UDID> --log ui-immersive-final.log
```

The app uses system-targeted gestures; it does not request ARKit hand-tracking or room-mesh permissions. No model downloads/runs, package installs, paid services, Apple credentials, signing/provisioning changes, device pairing, remote pushes or publication were performed.

## Run on your Vision Pro

Open the project in Xcode, pair/select your Vision Pro through Devices and Simulators, complete the headset's trust/Developer Mode steps, and use your own valid signing/provisioning configuration when you choose to authorize device deployment. Then Run, select Room demo, and enter the room-scale plot. SDK compilation is verified; those device steps and actual headset operation are not.
