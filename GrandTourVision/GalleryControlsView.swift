import SwiftUI

// Controls for the FlowScope layer gallery, in two sizes:
// - the home window (before you enter): pick a run and a training step, then enter;
// - the in-space panel (compact): only what you need while exploring, the rest folded under "More".
// Both use grouped forms so spacing and insets follow the system.
struct GalleryControlsView: View {
    @Bindable var model: GalleryModel
    var roomPlotOpen: Bool
    var compact = false
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openWindow) private var openWindow
    @State private var busy = false
    @State private var tiltSlot = 3  // PC4: the first direction the default view doesn't show
    @State private var showMore = false

    var body: some View {
        Form {
            if compact { panel } else { home }
        }
        .formStyle(.grouped)
        .alert("Gallery", isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })) {
            Button("OK") { model.error = nil }
        } message: { Text(model.error ?? "") }
        .onAppear { if model.frames.isEmpty && !model.isLive { model.loadSample("no-residual-12-layers") } }
        .task { if DemoScript.enabled && !compact && !model.galleryOpen { enter(demo: true) } }
    }

    // MARK: home window

    @ViewBuilder private var home: some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                Text("Layer gallery").font(.largeTitle.bold())
                Text("Every block of a transformer as a cloud of token vectors around you. Step inside one to explore its high-dimensional space at room scale.")
                    .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
        }
        Section("Run") {
            runRows
            Text(model.status).font(.callout).foregroundStyle(.secondary).accessibilityIdentifier("gallery-status")
        }
        Section {
            HStack {
                TextField("Mac address, e.g. 192.168.1.20:8765", text: $model.liveAddress)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                if model.isLive { Button("Disconnect") { model.disconnect() } } else { Button("Connect live") { model.connect() } }
            }
        } header: { Text("Live from your Mac") } footer: {
            Text("Run `flowscope run train.py --host 0.0.0.0`; it prints the address to enter here.")
        }
        if model.frames.count > 1 { Section(model.framesAreCheckpoints ? "Checkpoint" : "Training step") { stepPicker } }
        Section {
            Button { enter() } label: {
                Text("Enter gallery").font(.title3.bold()).frame(maxWidth: .infinity).padding(.vertical, 6)
            }
            .buttonStyle(.borderedProminent)
            .disabled(busy || model.clouds.isEmpty || roomPlotOpen)
            .accessibilityIdentifier("enter-gallery")
            if roomPlotOpen { Text("Exit the classic room plot first.").font(.caption) }
        } footer: {
            Button("Classic single-cloud plot (CSV, toy datasets)") { openWindow(id: "controls", value: "controls") }
                .font(.caption).accessibilityIdentifier("open-classic")
        }
    }

    // MARK: in-space panel

    @ViewBuilder private var panel: some View {
        Section {
            HStack {
                Picker("Block", selection: $model.focused) {
                    Text("Gallery").tag(Int?.none)
                    ForEach(model.clouds.indices, id: \.self) { Text(model.clouds[$0].name).tag(Int?.some($0)) }
                }
                .accessibilityIdentifier("focus-block")
                Spacer()
                if model.focused != nil { Button("Back to gallery") { model.focused = nil } }
                Button("Exit gallery") { Task { await dismissImmersiveSpace() } }
            }
            .buttonStyle(.bordered)  // several buttons in one row: each must be its own tap target
            if model.frames.count > 1 { stepPicker }
        } footer: {
            Text(model.focused == nil ? "Tap a block to step inside it." : "\(model.insideCount) of \(model.frame?.N ?? 0) tokens in the window")
                .accessibilityIdentifier("inside-count")
        }

        if model.focused != nil {
            Section("Data") {
                Picker("Show", selection: $model.sliceMode) {
                    Text("Projection").tag(false)
                    Text("Slice").tag(true)
                }
                .pickerStyle(.segmented).accessibilityIdentifier("show-mode")
                HStack {
                    Text("Zoom")
                    Spacer()
                    Button { model.view.zoom = max(0.3, model.view.zoom / 1.4) } label: { Label("Zoom out", systemImage: "minus") }
                    Text(verbatim: String(format: "%.1f×", model.view.zoom)).monospacedDigit().frame(minWidth: 48)
                        .accessibilityIdentifier("zoom-label")
                    Button { model.view.zoom = min(20, model.view.zoom * 1.4) } label: { Label("Zoom in", systemImage: "plus") }
                    Button("Re-center data") { model.recenter() }
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.bordered)
                if model.sliceMode {
                    LabeledContent("Thickness") { Slider(value: $model.sliceThickness, in: 0.15...2.5) }
                    LabeledContent(String(format: "Depth %+.1fσ", model.depth)) { Slider(value: $model.depth, in: -3...3) }
                }
            }
        }

        Section {
            Text(model.axisSummary).accessibilityIdentifier("axis-summary")
            let handles = model.handleDirections
            HStack {
                Picker("Tilt toward", selection: $tiltSlot) {
                    ForEach(handles.indices, id: \.self) { Text(verbatim: "PC\(handles[$0].pc + 1)").tag($0) }
                }
                .accessibilityIdentifier("tilt-direction")
                ForEach(0..<3, id: \.self) { a in
                    Button(["X", "Y", "Z"][a]) {
                        guard tiltSlot < handles.count else { return }
                        var angles = SIMD3<Double>.zero; angles[a] = .pi / 12
                        model.tilt(toward: handles[tiltSlot].q, by: angles)
                    }
                    .accessibilityIdentifier("tilt-\(["x", "y", "z"][a])")
                }
            }
            .buttonStyle(.bordered)
            Button(model.playing ? "Stop tour" : "Start tour") { model.setPlaying(!model.playing) }
                .accessibilityIdentifier("tour-toggle")
        } header: { Text("Direction") } footer: {
            Text("In space: pull a direction's knob toward an axis to tilt the window. Each tilt button turns it 15°.")
        }

        Section {
            DisclosureGroup("More", isExpanded: $showMore) {
                Picker("Run", selection: Binding(get: { currentSample }, set: { model.loadSample($0) })) {
                    ForEach(GalleryModel.samples, id: \.file) { Text($0.title).tag($0.file) }
                }
                .accessibilityIdentifier("run-picker")
                Picker("Window", selection: $model.view.shape) {
                    ForEach(WindowShape.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                LabeledContent(String(format: "Window size %.1f m", 2 * GalleryScene.windowRadius * model.windowSize)) {
                    Slider(value: $model.windowSize, in: 0.5...2)
                }
                Toggle("Edge hints for data just outside", isOn: $model.showHints)
                if model.frame?.groups != nil {
                    Toggle("Color by \(model.framesAreCheckpoints ? "harness" : "group")", isOn: $model.colorByGroup)
                }
                Toggle("Color by token", isOn: $model.colorByToken).disabled(model.frame?.tokens == nil || (model.frame?.groups != nil && model.colorByGroup))
                axisSteppers
                Button("Back to PC1–PC3") { model.resetView() }
                HStack {
                    Text(verbatim: "Handles PC\(handles.first.map { $0.pc + 1 } ?? 1)–\(handles.last.map { $0.pc + 1 } ?? 8)")
                    Spacer()
                    Button { model.handlePage = max(0, model.handlePage - 1) } label: { Label("Previous handles", systemImage: "chevron.left") }
                        .disabled(model.handlePage == 0)
                    Button { model.handlePage += 1 } label: { Label("Next handles", systemImage: "chevron.right") }
                        .disabled((model.handlePage + 1) * 8 >= model.dims.count)
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.bordered)
                LabeledContent(String(format: "Tour speed %.1f×", model.speed)) { Slider(value: $model.speed, in: 0.1...3) }
                LabeledContent("Tour through the top \(model.toured)") {
                    Slider(value: Binding(get: { Double(model.toured) }, set: { model.setToured(Int($0.rounded())) }),
                           in: 3...Double(max(4, model.frame?.C ?? 64)), step: 1)
                }
            }
        }
    }

    private var handles: [(q: Int, pc: Int)] { model.handleDirections }

    private var currentSample: String {
        if case .recording(let file) = model.source { return file }
        return GalleryModel.samples[0].file
    }

    private var runRows: some View {
        ForEach(GalleryModel.samples, id: \.file) { sample in
            Button { model.loadSample(sample.file) } label: {
                HStack {
                    Text(sample.title)
                    Spacer()
                    if currentSample == sample.file && !model.isLive { Image(systemName: "checkmark").foregroundStyle(.tint) }
                }
            }
        }
    }

    // checkpoint names are long ("multi-harness SFT"): a menu shows them whole, where segments would truncate
    @ViewBuilder private var stepPicker: some View {
        let picker = Picker(model.framesAreCheckpoints ? "Checkpoint" : "Training step", selection: Binding(get: { model.frameIndex }, set: { try? model.select(frame: $0) })) {
            // iterate the frames themselves: switching to a run with fewer frames must not re-read stale positions
            ForEach(Array(model.frames.enumerated()), id: \.offset) { i, f in Text(verbatim: f.name ?? String(f.step)).tag(i) }
        }
        if model.framesAreCheckpoints {
            picker.pickerStyle(.menu).accessibilityIdentifier("step-picker")
        } else {
            picker.pickerStyle(.segmented).accessibilityIdentifier("step-picker")
        }
    }

    private var axisSteppers: some View {
        HStack(spacing: 16) {
            ForEach(0..<3, id: \.self) { axis in
                HStack(spacing: 4) {
                    Button { model.stepAxis(axis, by: -1) } label: { Label("Previous direction for \(["X", "Y", "Z"][axis])", systemImage: "chevron.left") }
                    Text(verbatim: "\(["X", "Y", "Z"][axis]) = PC\(model.axes[axis] + 1)").monospacedDigit().fixedSize()
                        .accessibilityIdentifier("axis-\(axis)-label")
                    Button { model.stepAxis(axis, by: 1) } label: { Label("Next direction for \(["X", "Y", "Z"][axis])", systemImage: "chevron.right") }
                        .accessibilityIdentifier("axis-\(axis)-next")
                }
            }
        }
        .labelStyle(.iconOnly)
        .buttonStyle(.bordered)
    }

    private func enter(demo: Bool = false) {
        busy = true
        Task { @MainActor in
            defer { busy = false }
            switch await openImmersiveSpace(id: "layer-gallery") {
            case .opened:
                // Swap this big window for the compact panel, so the arc stays visible. The recorded demo shows no panel.
                if !demo { openWindow(id: "gallery-panel", value: "gallery-panel") }
                dismissWindow(id: "gallery")
                dismissWindow(id: "controls")
            case .error: model.error = "The gallery space could not open."
            default: break
            }
        }
    }
}
