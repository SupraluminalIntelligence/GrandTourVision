import SwiftUI
import simd

struct RoomControlsView: View {
    @Bindable var model: TourModel
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    @Environment(\.dismissImmersiveSpace) private var dismissImmersiveSpace
    @State private var exiting = false
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Room-scale tour").font(.headline)
                Spacer()
                Button(model.playing ? "Pause animation" : "Animate projection") {
                    if model.playing { model.playing = false } else { model.startAnimation() }
                }.accessibilityIdentifier("room-play")
                Button(model.roomControlsHidden ? "Show room controls" : "Hide controls") {
                    model.roomControlsHidden.toggle()
                }
                Button("Exit room") { exitRoom() }.disabled(exiting)
            }
            if !model.roomControlsHidden {
                Text("\(model.dataset.rows.count) points · \(model.dataset.names.count) dimensions · ONE selected representation")
                    .font(.caption).accessibilityIdentifier("room-summary")
                if model.trace != nil {
                    HStack {
                        Picker("Layer", selection: Binding(get: { model.currentFrame?.layer ?? "" }, set: { model.selectLayer($0) })) {
                            ForEach(model.layers, id: \.self) { Text($0).tag($0) }
                        }.accessibilityIdentifier("room-layer")
                        Picker("Training step / checkpoint", selection: Binding(get: { model.currentFrame?.checkpoint ?? "" }, set: { model.selectCheckpoint($0) })) {
                            ForEach(model.checkpoints, id: \.self) { Text($0).tag($0) }
                        }.accessibilityIdentifier("room-checkpoint")
                    }
                    Text("\(model.currentFrame?.id ?? "") · \(model.comparisonLocked ? "locked comparison" : "shared projection")")
                        .font(.caption).accessibilityIdentifier("room-snapshot")
                    Toggle("Lock projection for layer comparison", isOn: $model.comparisonLocked)
                        .accessibilityIdentifier("room-lock")
                }
                HStack {
                    Text("Speed \(model.speed, specifier: "%.1f")×")
                    Slider(value: $model.speed, in: 0.1...3).accessibilityIdentifier("room-speed")
                    Text("Tour \(model.time, specifier: "%.1f")s").monospacedDigit()
                        .accessibilityIdentifier("room-time")
                }
                HStack {
                    Text("Plot scale \(model.scenePlacement.diameter, specifier: "%.1f") m")
                        .accessibilityIdentifier("room-span")
                    Button("Smaller") { model.scenePlacement.resize(to: model.scenePlacement.diameter / 1.2) }
                    Button("Larger") { model.scenePlacement.resize(to: model.scenePlacement.diameter * 1.2) }
                    Button("Recenter scene") { model.scenePlacement.reset() }
                }
                Slider(value: Binding(get: { Double(model.scenePlacement.diameter) }, set: { model.scenePlacement.resize(to: Float($0)) }), in: 1.5...6)
                    .accessibilityIdentifier("room-scale")
                HStack {
                    Text("Physical placement")
                    Button("Move left") { model.scenePlacement.move(to: model.scenePlacement.center + [-0.25, 0, 0]) }
                    Button("Move right") { model.scenePlacement.move(to: model.scenePlacement.center + [0.25, 0, 0]) }
                    Button("Rotate scene") { model.scenePlacement.rotate(to: simd_quatf(angle: .pi / 12, axis: [0, 1, 0]) * model.scenePlacement.orientation) }
                }.font(.caption)
                Toggle("Color by token position", isOn: $model.colorByPosition)
                DisclosureGroup("Manual high-dimensional projection") {
                    HStack {
                        DimensionControl(title: "Dimension 1", names: model.dataset.names, index: $model.planeI)
                        DimensionControl(title: "Dimension 2", names: model.dataset.names, index: $model.planeJ)
                    }
                    Slider(value: Binding(get: { model.manualAngle }, set: {
                        model.comparisonLocked = false; model.turnManual(to: $0)
                    }), in: -.pi...Double.pi).disabled(model.planeI == model.planeJ)
                        .accessibilityIdentifier("room-manual")
                    Button("Nudge projection") {
                        model.comparisonLocked = false
                        model.turnManual(to: model.manualAngle + 0.15)
                    }.disabled(model.planeI == model.planeJ)
                    Button("Reset projection") { model.reset() }
                    Text("Changes input-dimension weights; hand gestures change only physical placement.").font(.caption)
                }
                HStack {
                    Button("Data & axis contributions") { openWindow(id: "controls", value: "controls") }
                    Spacer()
                    Text(model.playing ? "Animation keeps one fixed display reference" : "Layer changes pause; projection and scale stay shared")
                        .font(.caption)
                }
                Text("Grab a point and drag, rotate with two hands, or pinch to resize while the tour runs. Walk around or into the single cloud. Recenter restores its initial position and 4 m span.")
                    .font(.caption)
            }
        }
        .padding(18).frame(width: model.roomControlsHidden ? 610 : 620)
        .fixedSize(horizontal: false, vertical: true)
    }
    private func exitRoom() {
        guard !exiting else { return }; exiting = true
        model.playing = false
        openWindow(id: "controls", value: "controls")
        Task { @MainActor in
            await dismissImmersiveSpace()
            model.immersiveActive = false
            dismissWindow(id: "room-controls")
        }
    }
}
