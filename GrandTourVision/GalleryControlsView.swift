import SwiftUI

// Controls for the FlowScope layer gallery: pick a run (bundled recording or a live FlowScope server),
// scrub training steps, and drive the shared tour.
struct GalleryControlsView: View {
    @Bindable var model: GalleryModel
    var roomPlotOpen: Bool
    var compact = false  // the in-space utility panel: same controls, less prose
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openWindow) private var openWindow
    @State private var busy = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                Text("Layer gallery").font(compact ? .title.bold() : .largeTitle.bold())
                if !compact { Text("Every block of a transformer as a cloud of token vectors around you, all seen through the same 3D window into the 64-dimensional residual stream. Step inside any block to explore it at room scale.")
                    .foregroundStyle(.secondary) }

                GroupBox("Space") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            if model.galleryOpen {
                                Button("Exit gallery") { Task { await dismissImmersiveSpace() } }
                            } else {
                                Button("Enter gallery") { enter() }
                                    .buttonStyle(.borderedProminent)
                                    .disabled(busy || model.clouds.isEmpty || roomPlotOpen)
                                    .accessibilityIdentifier("enter-gallery")
                            }
                            Picker("Step inside", selection: $model.focused) {
                                Text("Gallery (no block)").tag(Int?.none)
                                ForEach(model.clouds.indices, id: \.self) { Text(model.clouds[$0].name).tag(Int?.some($0)) }
                            }.accessibilityIdentifier("focus-block")
                        }
                        if model.focused != nil {
                            HStack {
                                Button("Smaller") { model.focusScale = max(0.15, model.focusScale / 1.4) }
                                Button("Larger") { model.focusScale = min(5, model.focusScale * 1.4) }
                                Button("Recenter") { model.resetPlacement() }
                                Button("Back to gallery") { model.focused = nil }
                            }
                        }
                        if roomPlotOpen { Text("Exit the classic room plot first.").font(.caption) }
                        Text("Tap a block in space (or pick it here) to step inside: it becomes a ~2.4 m window in front of you. Pinch to zoom, drag to move, rotate with two hands, walk into it. The other blocks wait on your left; tap one to switch.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }

                GroupBox("View") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack(spacing: 18) {
                            ForEach(0..<3, id: \.self) { axis in
                                HStack(spacing: 6) {
                                    Button { model.stepAxis(axis, by: -1) } label: { Image(systemName: "chevron.left") }
                                        .accessibilityLabel("Previous direction for \(["X", "Y", "Z"][axis])")
                                    Text(verbatim: "\(["X", "Y", "Z"][axis]) = PC\(model.axes[axis] + 1)").monospacedDigit().fixedSize()
                                        .accessibilityIdentifier("axis-\(axis)-label")
                                    Button { model.stepAxis(axis, by: 1) } label: { Image(systemName: "chevron.right") }
                                        .accessibilityIdentifier("axis-\(axis)-next")
                                }
                            }
                        }
                        Toggle("Color by token (instead of position in sequence)", isOn: $model.colorByToken)
                            .disabled(model.frame?.tokens == nil)
                        Text("A still 3D window: each axis shows one principal direction of the 64-dim residual stream (PC1 = the direction with the most spread). Pick any three to look at the data from there.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }

                GroupBox("Tour (optional motion)") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Button(model.playing ? "Stop tour" : "Start tour") { model.playing.toggle() }
                                .accessibilityIdentifier("tour-toggle")
                            Button("Back to chosen axes") { model.resetView() }
                        }
                        HStack { Text("Speed \(model.speed, specifier: "%.1f")×"); Slider(value: $model.speed, in: 0.1...3) }
                        HStack {
                            Text("Dims toured \(model.toured)")
                            Slider(value: Binding(get: { Double(model.toured) }, set: { model.setToured(Int($0.rounded())) }),
                                   in: 3...Double(max(4, model.frame?.C ?? 64)), step: 1)
                        }
                        Text("The tour rotates the window smoothly through the toured principal directions (a grand tour), starting from your chosen axes. dim = how many directions a block's cloud really uses; % seen = share of its variance inside the toured dims.")
                            .font(.caption).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }

                if model.frames.count > 1 {
                    GroupBox("Training step") {
                        Picker("Training step", selection: Binding(get: { model.frameIndex }, set: { try? model.select(frame: $0) })) {
                            ForEach(model.frames.indices, id: \.self) { Text(verbatim: String(model.frames[$0].step)).tag($0) }
                        }.pickerStyle(.segmented)
                        Text("Changing step keeps the same window, so you compare the same view across training. Step 0 shows what the architecture does before learning anything.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }

                GroupBox("Run") {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            ForEach(GalleryModel.samples, id: \.file) { sample in
                                Button(sample.title) { model.loadSample(sample.file) }
                            }
                        }
                        HStack {
                            TextField("Mac address, e.g. 192.168.1.20:8765", text: $model.liveAddress)
                                .textFieldStyle(.roundedBorder)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled()
                            if model.isLive {
                                Button("Disconnect") { model.disconnect() }
                            } else {
                                Button("Connect live") { model.connect() }
                            }
                        }
                        Text("Live: on the Mac run `flowscope run train.py --host 0.0.0.0`; it prints the address to enter here.")
                            .font(.caption).foregroundStyle(.secondary)
                        Text(model.status).font(.callout).accessibilityIdentifier("gallery-status")
                        Button("Classic single-cloud plot (CSV, toy datasets)") { openWindow(id: "controls", value: "controls") }
                            .accessibilityIdentifier("open-classic")
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
            }.padding(28)
        }
        .alert("Gallery", isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })) {
            Button("OK") { model.error = nil }
        } message: { Text(model.error ?? "") }
        .onAppear { if model.frames.isEmpty && !model.isLive { model.loadSample("no-residual-12-layers") } }
    }

    private func enter() {
        busy = true
        Task { @MainActor in
            defer { busy = false }
            switch await openImmersiveSpace(id: "layer-gallery") {
            case .opened:
                // Swap this big window for a small utility panel low in front of you, so the arc stays visible.
                openWindow(id: "gallery-panel", value: "gallery-panel")
                dismissWindow(id: "gallery")
                dismissWindow(id: "controls")
            case .error: model.error = "The gallery space could not open."
            default: break
            }
        }
    }
}
