import SwiftUI
import RealityKit
import Spatial
import simd

struct PlotView: View {
    var model: TourModel
    @State private var scene = PlotScene()
    private var root: Entity { scene.root }
    @State private var dragStart: SIMD3<Float>?
    @State private var rotationStart: simd_quatf?
    @State private var scaleStart: Float?
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    private let colors: [UIColor] = [.systemOrange, .systemTeal, .systemPink, .systemYellow]

    var body: some View {
        RealityView { content, attachments in
            root.name = "single-room-cloud"
            root.components.set(InputTargetComponent())
            content.add(root)
            rebuild()
            if let info = attachments.entity(for: "plot-info") {
                // Informational, separated from the cloud and physical gestures.
                info.position = SIMD3(-1.25, 0.65, -2.8)
                info.scale = SIMD3(repeating: 2.2)
                content.add(info)
            }
        } update: { _, _ in
            if scene.points.count != model.dataset.rows.count || scene.lastNames != model.dataset.names { rebuild() }
            applyPlacement()
            updatePositions()
        } attachments: {
            Attachment(id: "plot-info") {
                VStack(alignment: .leading, spacing: 6) {
                    Text(model.trace?.mode == "synthetic" ? "SYNTHETIC — NO MODEL RUN" : model.dataset.label).font(.headline)
                    Text("\(model.dataset.rows.count) points · \(model.dataset.names.count)D · \(model.currentFrame?.layer ?? "3D projection") · \(model.currentFrame?.checkpoint ?? "")")
                        .font(.subheadline)
                    Text(model.playing ? "MOVING PROJECTION" : (model.comparisonLocked && model.trace != nil ? "LOCKED COMPARISON" : "PAUSED PROJECTION")).font(.caption)
                    if model.colorByPosition {
                        Text("Color: token position · blue → teal → yellow").font(.caption)
                    } else {
                        HStack(spacing: 12) {
                            ForEach(model.plotLegend.indices, id: \.self) { i in
                                HStack(spacing: 4) {
                                    Circle().fill(Color(uiColor: colors[i % colors.count])).frame(width: 10, height: 10)
                                    Text(model.plotLegend[i])
                                }
                            }
                        }.font(.caption)
                    }
                    Text("X red · Y green · Z blue · \(model.scenePlacement.diameter, specifier: "%.1f") m span").font(.caption)
                    if let source = model.trace?.source { Text("\(source.title) · \(source.license) · recorded tiny GPT").font(.caption) }
                }.padding(14).frame(width: 550).glassBackgroundEffect()
            }
        }
        .simultaneousGesture(DragGesture().targetedToEntity(root).onChanged { value in
            if dragStart == nil { dragStart = model.scenePlacement.center }
            let movement = value.convert(value.translation3D, from: .local, to: .scene)
            model.scenePlacement.move(to: (dragStart ?? model.scenePlacement.center) + movement)
        }.onEnded { _ in dragStart = nil })
        .simultaneousGesture(RotateGesture3D().targetedToEntity(root).onChanged { value in
            if rotationStart == nil { rotationStart = model.scenePlacement.orientation }
            model.scenePlacement.rotate(to: simd_quatf(value.rotation) * (rotationStart ?? model.scenePlacement.orientation))
        }.onEnded { _ in rotationStart = nil })
        .simultaneousGesture(MagnifyGesture().targetedToEntity(root).onChanged { value in
            if scaleStart == nil { scaleStart = model.scenePlacement.diameter }
            model.scenePlacement.resize(to: (scaleStart ?? model.scenePlacement.diameter) * Float(value.magnification))
        }.onEnded { _ in scaleStart = nil })
        .simultaneousGesture(SpatialTapGesture().targetedToEntity(root).onEnded { value in
            if value.entity.name.hasPrefix("point-"), let index = Int(value.entity.name.dropFirst(6)) { model.selectedTokenIndex = index }
        })
        .onChange(of: model.plotReset) { rebuild() }
        .onAppear { model.immersiveActive = true }
        .onDisappear {
            model.immersiveActive = false; model.playing = false
            openWindow(id: "controls", value: "controls")
            dismissWindow(id: "room-controls")
        }
        .task { await model.runTourClock() }
    }
    private func applyPlacement() {
        // Projection ticks update point coordinates, never this physical transform.
        root.position = model.scenePlacement.center
        root.orientation = model.scenePlacement.orientation
        root.scale = SIMD3(repeating: model.scenePlacement.diameter / 0.72)
    }
    private func rebuild() {
        for child in Array(root.children) { child.removeFromParent() }
        scene.points = []; scene.lastNames = model.dataset.names
        let mesh = MeshResource.generateSphere(radius: 0.0045)
        for index in model.dataset.rows.indices {
            let point = ModelEntity(mesh: mesh, materials: [UnlitMaterial(color: color(index))])
            point.name = "point-\(index)"
            point.components.set(InputTargetComponent())
            point.components.set(CollisionComponent(shapes: [.generateSphere(radius: 0.011)]))
            root.addChild(point); scene.points.append(point)
        }
        for axis in 0..<3 {
            var size = SIMD3<Float>(repeating: 0.001); size[axis] = 0.40
            root.addChild(ModelEntity(mesh: .generateBox(size: size), materials: [UnlitMaterial(color: [UIColor.systemRed, .systemGreen, .systemBlue][axis])]))
            let label = ModelEntity(mesh: .generateText(["X", "Y", "Z"][axis], extrusionDepth: 0.0002, font: .systemFont(ofSize: 0.012)), materials: [UnlitMaterial(color: [UIColor.systemRed, .systemGreen, .systemBlue][axis])])
            var position = SIMD3<Float>(repeating: 0); position[axis] = 0.23
            label.position = position; root.addChild(label)
        }
        applyPlacement(); updatePositions()
    }
    private func color(_ index: Int) -> UIColor {
        guard model.colorByPosition else { return colors[model.dataset.groups[index] % colors.count] }
        let positions = model.trace?.tokens.map(\.position) ?? Array(model.dataset.rows.indices)
        let t = CGFloat(positions[index]) / CGFloat(max(1, positions.max() ?? 1))
        return UIColor(red: 0.15 + 0.8 * t, green: 0.35 + 0.55 * t, blue: 0.9 - 0.75 * t, alpha: 1)
    }
    private func updatePositions() {
        let projected = model.projected
        for i in scene.points.indices {
            guard projected.indices.contains(i) else { continue }
            let p = projected[i]
            scene.points[i].isEnabled = true // No volume boundary or hidden projection excursions.
            scene.points[i].scale = SIMD3(repeating: model.trace != nil && i == model.selectedTokenIndex ? Float(1.5) : 1)
            scene.points[i].position = SIMD3(Float(p[0] / model.radius * 0.36), Float(p[1] / model.radius * 0.36), Float(p[2] / model.radius * 0.36))
            if scene.lastColorByPosition != model.colorByPosition { scene.points[i].model?.materials = [UnlitMaterial(color: color(i))] }
        }
        scene.lastColorByPosition = model.colorByPosition
    }
}

@MainActor private final class PlotScene {
    let root = Entity()
    var points: [ModelEntity] = []
    var lastNames: [String] = []
    var lastColorByPosition: Bool?
}
