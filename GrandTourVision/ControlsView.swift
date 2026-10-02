import SwiftUI
import UniformTypeIdentifiers

struct ControlsView: View {
    @Bindable var model: TourModel
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.openImmersiveSpace) private var openImmersiveSpace
    @State private var entering = false
    @State private var importing = false
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Text("Grand Tour").font(.largeTitle.bold())
                Text("Explore multidimensional data and local activation traces.")
                Button("FlowScope layer gallery: real training runs, every block around you") {
                    openWindow(id: "gallery", value: "gallery")
                }
                .buttonStyle(.borderedProminent)
                .accessibilityIdentifier("open-gallery")
                Text(model.dataset.label).font(.headline)
                HStack {
                    Button("Enter room-scale plot") { enterRoom() }
                        .disabled(entering || model.immersiveActive)
                    Button("Import CSV / trace") { importing = true }
                    Button("Synthetic 4D") { model.load(.synthetic()) }
                    Button("Synthetic trace") {
                        do { try model.loadTrace(.synthetic()) } catch { model.error = error.localizedDescription }
                    }
                }
                HStack {
                    Button("Room demo (synthetic)") {
                        do {
                            guard let url = Bundle.main.url(forResource: "room-demo-trace", withExtension: "json") else { throw CSVError.invalid("Bundled room demo is missing.") }
                            try model.loadTrace(ActivationTrace.parse(Data(contentsOf: url)))
                            model.colorByPosition = true
                        } catch { model.error = error.localizedDescription }
                    }
                    Button("Public tiny GPT") {
                        do {
                            guard let url = Bundle.main.url(forResource: "public-sorting-trace", withExtension: "json") else {
                                throw CSVError.invalid("Bundled public trace is missing.")
                            }
                            try model.loadTrace(ActivationTrace.parse(Data(contentsOf: url)))
                        } catch { model.error = error.localizedDescription }
                    }
                    if model.immersiveActive {
                        Button("Return to room controls") {
                            openWindow(id: "room-controls", value: "room-controls")
                            dismissWindow(id: "controls")
                        }
                    }
                }
                if model.trace != nil { TraceControlsView(model: model) }
                Picker("Normalization", selection: $model.normalization) {
                    ForEach(Normalization.allCases, id: \.self) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented)
                    .disabled(model.trace != nil && model.comparisonLocked)
                Text("Both modes center each column. Z-score divides by its population standard deviation; constant columns become zero. CSV uses a fixed data norm bound. Traces use one reference fitted across all snapshots; switching layers/checkpoints never refits normalization or display scale.").font(.caption)
                Divider()
                Text("High-dimensional projection").font(.title2)
                HStack {
                    Button(model.playing ? "Pause tour" : "Play tour") { if model.playing { model.playing = false } else { model.startAnimation() } }
                    Button("Reset projection & plot") { model.playing = false; model.reset() }
                }
                HStack { Text("Speed \(model.speed, specifier: "%.1f")×"); Slider(value: $model.speed, in: 0.1...3) }
                Text("Manual ambient plane").font(.headline)
                HStack {
                    DimensionControl(title: "Dimension 1", names: model.dataset.names, index: Binding(get: { model.planeI }, set: { model.planeI = $0; model.manualAngle = 0 }))
                    DimensionControl(title: "Dimension 2", names: model.dataset.names, index: Binding(get: { model.planeJ }, set: { model.planeJ = $0; model.manualAngle = 0 }))
                }
                Slider(value: Binding(get: { model.manualAngle }, set: { model.turnManual(to: $0) }), in: -.pi...Double.pi)
                    .disabled(model.planeI == model.planeJ || (model.trace != nil && model.comparisonLocked))
                Text("The slider pauses the tour and rotates the projection basis in the selected input-dimension plane. Selecting equal dimensions disables it. This changes the projection, not a data slice.").font(.caption)
                Text("Saved projection viewpoints").font(.headline)
                HStack {
                    Button("Save current projection") { model.saveViewpoint() }
                        .disabled(model.savedViewpoints.count >= 8)
                    Button("Clear saved") { model.clearViewpoints() }
                        .disabled(model.savedViewpoints.isEmpty)
                }
                ForEach(model.savedViewpoints) { view in
                    Button("Restore \(view.name)") { model.restoreViewpoint(view.id) }
                }
                Text("Up to eight for this session and dataset. Restore pauses the tour, restores projection and normalization, and resets the physical plot pose. Importing data or relaunching clears saved viewpoints.").font(.caption)
                Text("Axis contributions").font(.headline)
                Text("Signed weights for X / Y / Z (red / green / blue); at most 24 dimensions with the largest combined absolute weights.").font(.caption)
                ForEach(model.contributionDimensions, id: \.self) { d in
                    HStack {
                        Text(model.dataset.names[d]).frame(width: 120, alignment: .leading)
                        ForEach(0..<3) { axis in
                            Text(String(format: "%+.3f", model.projection.basis[axis][d]))
                                .monospacedDigit().frame(maxWidth: .infinity)
                        }
                    }
                }
                Divider()
                Text("Manipulating the plot").font(.headline)
                Text("Pinch and drag the cloud to move it; use two hands to rotate or scale it. These gestures only change the physical 3D plot while animation continues. They never change projection weights. The mixed immersive plot can span 1.5–6 metres; reset places it three metres ahead at a comfortable height.")
                Text("The automatic tour uses smooth ambient rotations with an orthonormal basis. It offers exploration; no finite run covers all possible projections.").font(.caption)
                Text("CSV: UTF-8, header row, 3–12 complete numeric columns, at least 2 rows; maximum 1,000 rows / 2 MB / 64 source columns. Text or incomplete columns are excluded. Imported points share one color.").font(.caption)
            }.padding(28)
        }
        .fileImporter(isPresented: $importing, allowedContentTypes: [.commaSeparatedText, .json], allowsMultipleSelection: false) { result in
            do {
                guard let url = try result.get().first else { return }
                let access = url.startAccessingSecurityScopedResource()
                defer { if access { url.stopAccessingSecurityScopedResource() } }
                let isCSV = url.pathExtension.lowercased() == "csv"
                let limit = isCSV ? CSVImport.maxBytes : ActivationTrace.maxBytes
                let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
                guard size <= limit else { throw CSVError.invalid("File exceeds the import size limit.") }
                let data = try Data(contentsOf: url)
                guard data.count <= limit else { throw CSVError.invalid("File exceeds the import size limit.") }
                if isCSV {
                    guard let text = String(data: data, encoding: .utf8) else { throw CSVError.invalid("Use UTF-8 CSV.") }
                    model.load(try CSVImport.parse(text))
                } else { try model.loadTrace(ActivationTrace.parse(data)) }
            } catch { model.error = error.localizedDescription }
        }
        .alert("Import failed", isPresented: Binding(get: { model.error != nil }, set: { if !$0 { model.error = nil } })) {
            Button("OK") { model.error = nil }
        } message: { Text(model.error ?? "") }
    }
    private func enterRoom() {
        guard !entering, !model.immersiveActive else { return }
        entering = true
        Task { @MainActor in
            defer { entering = false }
            switch await openImmersiveSpace(id: "room-plot") {
            case .opened:
                model.immersiveActive = true
                model.roomControlsHidden = false
                model.scenePlacement.reset()
                model.startAnimation()
                openWindow(id: "room-controls", value: "room-controls")
                dismissWindow(id: "controls")
            case .userCancelled: break
            case .error: model.error = "The room-scale space could not open. Try again from the data controls."
            @unknown default: model.error = "The immersive-space request did not complete."
            }
        }
    }

}


struct TraceControlsView: View {
    @Bindable var model: TourModel
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Activation comparison").font(.title2)
            if let trace = model.trace {
                if let source = trace.source {
                    Text("Source: \(source.title) · \(source.license) · public recorded fixture")
                        .font(.caption).accessibilityIdentifier("trace-source")
                }
                Text("\(trace.mode) • tokenizer: \(trace.tokenizer)").font(.caption)
                Text("Prompt: \(trace.prompt)").font(.caption).textSelection(.enabled)
                HStack {
                    Picker("Layer", selection: Binding(get: { model.currentFrame?.layer ?? "" }, set: { model.selectLayer($0) })) {
                        ForEach(model.layers, id: \.self) { Text($0).tag($0) }
                    }.accessibilityIdentifier("layer-picker")
                    Picker("Checkpoint", selection: Binding(get: { model.currentFrame?.checkpoint ?? "" }, set: { model.selectCheckpoint($0) })) {
                        ForEach(model.checkpoints, id: \.self) { Text($0).tag($0) }
                    }.accessibilityIdentifier("checkpoint-picker")
                }
                if let frame = model.currentFrame {
                    Text("Snapshot \(frame.id) • step: \(frame.trainingStep.map(String.init) ?? "inference") • dtype: \(frame.dtype)").font(.caption).accessibilityIdentifier("snapshot-metadata")
                }
                Toggle("Lock projection for comparison", isOn: $model.comparisonLocked).accessibilityIdentifier("comparison-lock")
                Text("One shared basis, normalization reference and display scale across all snapshots. Frame selection pauses the tour. Unlock for exploration; re-lock before comparing. No per-frame PCA.").font(.caption)
                Button("Fit this projection across all snapshots") { model.fitProjection() }
                Text("\(model.clippedCount) points outside the fixed display reference remain visible in the immersive room. Fit after changing the projection if needed.").font(.caption)
                Picker("Selected token (also tap a point)", selection: $model.selectedTokenIndex) {
                    ForEach(trace.tokens.indices, id: \.self) { i in
                        Text("\(trace.tokens[i].position): \(trace.tokens[i].text)").tag(i)
                    }
                }.accessibilityIdentifier("token-picker")
                if let token = model.selectedToken {
                    Text("ID: \(token.id) • sequence: \(token.sequenceID) • position: \(token.position) • token ID: \(token.tokenID) • text: \(token.text)").font(.caption).textSelection(.enabled).accessibilityIdentifier("token-metadata")
                }
                Text("Patterns are exploratory, not evidence of causal interpretability. Separately trained models may need coordinate alignment before comparing activations.").font(.caption)
            }
        }.padding().background(.thinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}


struct DimensionControl: View {
    let title: String
    let names: [String]
    @Binding var index: Int
    @State private var entered = "0"
    var body: some View {
        VStack(alignment: .leading) {
            Text("\(title): \(names[index])").font(.caption)
            TextField("Zero-based index", text: $entered)
                .textFieldStyle(.roundedBorder)
                .onSubmit {
                    if let value = Int(entered), names.indices.contains(value) { index = value }
                    entered = String(index)
                }
            Stepper("0–\(names.count - 1)", value: $index, in: 0...(names.count - 1))
                .font(.caption)
        }
        .onAppear { entered = String(index) }
        .onChange(of: index) { entered = String(index) }
    }
}
