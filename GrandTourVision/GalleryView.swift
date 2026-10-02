import SwiftUI
import RealityKit
import simd

// The FlowScope layer gallery: one token cloud per block on an arc around you, all seen through the same
// 3D window of the residual stream. Pick a block (tap it, or the Focus menu) and it becomes a room-scale
// window in the middle that you pinch to zoom, drag to move, rotate with two hands, and walk into; the other
// blocks move to a rail on your left. Nothing moves until you start the tour. Projection runs on the CPU into
// one LowLevelMesh per block (a small octahedron per token), so thousands of points stay ~13 entities.
struct GalleryView: View {
    var model: GalleryModel
    @State private var scene = GalleryScene()
    @State private var dragStart: SIMD3<Float>?
    @State private var scaleStart: Float?
    @State private var rotationStart: simd_quatf?
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        RealityView { content, attachments in
            content.add(scene.root)
            scene.subscription = content.subscribe(to: SceneEvents.Update.self) { event in
                MainActor.assumeIsolated { scene.tick(model: model, dt: event.deltaTime) }
            }
            scene.sync(model: model, attachments: attachments)
        } update: { _, attachments in
            scene.sync(model: model, attachments: attachments)
        } attachments: {
            Attachment(id: "title") { GalleryTitle(model: model) }
            ForEach(model.clouds.indices, id: \.self) { i in
                Attachment(id: "label-\(i)") {
                    CloudLabel(cloud: model.clouds[i], seen: model.percentSeen(i), compact: model.focused != nil)
                }
            }
        }
        .gesture(SpatialTapGesture().targetedToAnyEntity().onEnded { value in
            if let i = cloudIndex(value.entity), i != model.focused { model.focused = i }
        })
        .simultaneousGesture(DragGesture().targetedToAnyEntity().onChanged { value in
            guard isFocused(value.entity) else { return }
            if dragStart == nil { dragStart = model.focusOffset }
            model.focusOffset = (dragStart ?? .zero) + value.convert(value.translation3D, from: .local, to: .scene)
        }.onEnded { _ in dragStart = nil })
        .simultaneousGesture(MagnifyGesture().targetedToAnyEntity().onChanged { value in
            guard isFocused(value.entity) else { return }
            if scaleStart == nil { scaleStart = model.focusScale }
            model.focusScale = min(5, max(0.15, (scaleStart ?? 1) * Float(value.magnification)))
        }.onEnded { _ in scaleStart = nil })
        .simultaneousGesture(RotateGesture3D().targetedToAnyEntity().onChanged { value in
            guard isFocused(value.entity) else { return }
            if rotationStart == nil { rotationStart = model.focusRotation }
            model.focusRotation = simd_normalize(simd_quatf(value.rotation) * (rotationStart ?? model.focusRotation))
        }.onEnded { _ in rotationStart = nil })
        .onAppear { model.galleryOpen = true }
        .onDisappear {
            model.galleryOpen = false
            openWindow(id: "gallery", value: "gallery")
            dismissWindow(id: "gallery-panel")
        }
    }

    private func cloudIndex(_ entity: Entity) -> Int? {
        var e: Entity? = entity
        while let current = e, !current.name.hasPrefix("cloud-") { e = current.parent }
        return e.flatMap { Int($0.name.dropFirst(6)) }
    }

    private func isFocused(_ entity: Entity) -> Bool { model.focused != nil && cloudIndex(entity) == model.focused }
}

private struct GalleryTitle: View {
    var model: GalleryModel
    var body: some View {
        VStack(spacing: 4) {
            if let f = model.focused, model.clouds.indices.contains(f) {
                let cloud = model.clouds[f]
                Text("\(cloud.name) · \(model.title)").font(.title2.bold())
                Text("dim \(cloud.dim, specifier: "%.1f") · \(model.percentSeen(f))% seen · step \(model.frame?.step ?? 0)")
                    .font(.callout).foregroundStyle(.secondary)
                Text("Pinch to zoom · drag to move · two hands to rotate · walk in").font(.caption).foregroundStyle(.tertiary)
            } else {
                Text(model.title).font(.title2.bold())
                Text("step \(model.frame?.step ?? 0) · \(model.frame?.N ?? 0) tokens × \(model.frame?.C ?? 0) dims")
                    .font(.callout).foregroundStyle(.secondary)
                Text("Tap a block to step inside it").font(.caption).foregroundStyle(.tertiary)
            }
            Text(model.playing ? "Touring \(model.dims.count) dims" : "Still view: X = PC\(model.axes[0] + 1) · Y = PC\(model.axes[1] + 1) · Z = PC\(model.axes[2] + 1)")
                .font(.caption).foregroundStyle(.secondary)
        }
        .padding(.horizontal, 24).padding(.vertical, 12)
        .glassBackgroundEffect()
    }
}

private struct CloudLabel: View {
    let cloud: FlowCloud
    let seen: Int
    let compact: Bool
    var body: some View {
        VStack(spacing: 2) {
            Text(cloud.name).font(compact ? .callout.bold() : .title3.bold())
            Text("dim \(cloud.dim, specifier: "%.1f")")
                .font(compact ? .caption2 : .callout).foregroundStyle(cloud.dim < 3 ? .orange : .secondary)
            if !compact {
                Text("\(seen)% seen").font(.caption).foregroundStyle(seen < 50 ? .yellow : .secondary)
            }
        }
        .padding(.horizontal, compact ? 8 : 12).padding(.vertical, compact ? 4 : 6)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 14))
    }
}

// MARK: - scene

@MainActor
final class GalleryScene {
    let root = Entity()
    var subscription: EventSubscription?
    private var clouds: [CloudNode] = []
    private var title: Entity?
    private var shape = (layers: 0, points: 0, colorKey: "")
    private var lastView: (time: Double, dims: [Int], fits: [Double], frame: Int) = (-1, [], [], -1)

    /// Rebuild when the data's shape or coloring changes; otherwise just (re)attach labels.
    func sync(model: GalleryModel, attachments: RealityViewAttachments) {
        let colorKey = "\(model.colorByToken)-\(model.frame?.T ?? 0)-\(model.frame?.tokens?.prefix(8).map(String.init).joined() ?? "")"
        let n = model.frame?.N ?? 0
        if shape.layers != model.clouds.count || shape.points != n || shape.colorKey != colorKey {
            clouds.forEach { $0.entity.removeFromParent() }
            clouds = model.clouds.indices.compactMap { i in
                try? CloudNode(index: i, colors: colors(model), parent: root)
            }
            shape = (model.clouds.count, n, colorKey)
            lastView.frame = -1
        }
        for (i, node) in clouds.enumerated() {
            if let label = attachments.entity(for: "label-\(i)"), label.parent !== node.entity {
                node.entity.addChild(label)
            }
            node.label = attachments.entity(for: "label-\(i)")
        }
        if let t = attachments.entity(for: "title"), t.parent !== root {
            root.addChild(t)
            title = t
        }
    }

    /// Per-frame: advance the tour if it's running, ease layout, and re-project only when the view changed.
    func tick(model: GalleryModel, dt: TimeInterval) {
        model.advance(dt)
        guard !clouds.isEmpty, clouds.count == model.clouds.count, model.fits.count == model.clouds.count else { return }
        let ease = Float(1 - exp(-dt * 6))
        let view = (model.time, model.dims, model.fits, model.frameIndex)
        let reproject = view != lastView
        let basis = reproject ? model.basis() : []
        for (i, node) in clouds.enumerated() {
            let target = layout(i, count: clouds.count, model: model)
            node.ease(to: target, by: model.focused == i ? max(ease, 0.35) : ease)  // gestures feel direct
            if reproject { model.clouds[i].project(basis, dims: view.1, fit: model.fits[i], into: &node.points) }
            node.writeVertices(force: reproject)
        }
        lastView = view
        if let title {
            let target: SIMD3<Float> = model.focused == nil ? [0, 2.25, -2.6] : [0.95, 2.05, -1.7]
            title.position += (target - title.position) * ease
            let yaw = atan2(-title.position.x, -title.position.z)
            title.orientation = simd_quatf(angle: yaw, axis: [0, 1, 0])
            title.scale = SIMD3(repeating: model.focused == nil ? 2.2 : 1.5)
        }
    }

    private func colors(_ model: GalleryModel) -> [SIMD3<Float>] {
        guard let f = model.frame else { return [] }
        return (0..<f.N).map { p in
            if model.colorByToken, let tokens = f.tokens {
                let h = Float((Double(tokens[p]) * 137.508).truncatingRemainder(dividingBy: 360) / 360)
                return hsv(h, 0.65, 0.95)
            }
            return FlowColors.viridis(Float(p % f.T) / Float(max(1, f.T - 1)))
        }
    }

    /// No focus: an arc at arm's length, embeddings on the left, last block on the right.
    /// Focus: the picked block becomes a ~2.4 m window in front of you (plus your pinch/drag/rotate);
    /// the others line up on a rail to your left, still tappable.
    private func layout(_ i: Int, count: Int, model: GalleryModel) -> Placement {
        func facing(_ a: Float) -> simd_quatf { simd_quatf(angle: -a, axis: [0, 1, 0]) }
        guard let f = model.focused else {
            let radius: Float = 1.6
            let span = min(Float.pi * 0.83, Float(max(count - 1, 1)) * 0.25)
            let angle = count == 1 ? 0 : -span / 2 + span * Float(i) / Float(count - 1)
            let size = min(0.17, 0.42 * radius * span / Float(max(count - 1, 1)))
            return Placement(position: [radius * sin(angle), 1.5, -radius * cos(angle)], rotation: facing(angle),
                             size: size, opacity: 1, labelScale: 1.1)
        }
        if f == i {
            return Placement(position: SIMD3<Float>(0, 1.45, -2.0) + model.focusOffset, rotation: model.focusRotation,
                             size: 1.2 * model.focusScale, opacity: 1, labelScale: 0)
        }
        let others = count - 1, k = i < f ? i : i - 1
        let angle = -1.75 + 1.0 * Float(k) / Float(max(others - 1, 1))
        let radius: Float = 1.35
        return Placement(position: [radius * sin(angle), 1.3, -radius * cos(angle)], rotation: facing(angle),
                         size: min(0.06, 0.4 * radius / Float(max(others, 1))), opacity: 0.9, labelScale: 0.55)
    }
}

struct Placement {
    var position: SIMD3<Float>
    var rotation: simd_quatf
    var size: Float
    var opacity: Float
    var labelScale: Float
}

@MainActor
final class CloudNode {
    let entity = Entity()
    var label: Entity? { didSet { label?.position = [0, -1.3, 0] } }
    var points: [SIMD3<Float>] = []
    private let model: ModelEntity
    private let mesh: LowLevelMesh
    private let slot: [Int]  // point index -> vertex block, grouped by color so each color is one mesh part
    private var placement = Placement(position: [0, 1.5, -1.6], rotation: simd_quatf(angle: 0, axis: [0, 1, 0]),
                                      size: 0.001, opacity: 0, labelScale: 1)
    private var writtenRadius: Float = -1
    private let count: Int

    // Octahedron: 6 vertices, 8 triangles per point.
    private static let corners: [SIMD3<Float>] = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    private static let faces: [UInt32] = [0, 2, 4, 2, 1, 4, 1, 3, 4, 3, 0, 4, 2, 0, 5, 1, 2, 5, 3, 1, 5, 0, 3, 5]

    init(index: Int, colors: [SIMD3<Float>], parent: Entity) throws {
        count = colors.count
        // Quantize colors into a few buckets: one unlit material and one mesh part per bucket.
        let bucket = colors.map { c -> Int in
            let key = Int((c.x * 3).rounded()) * 16 + Int((c.y * 3).rounded()) * 4 + Int((c.z * 3).rounded())
            return key
        }
        let keys = Array(Set(bucket)).sorted()
        let order = colors.indices.sorted { (bucket[$0], $0) < (bucket[$1], $1) }
        var slot = [Int](repeating: 0, count: count)
        for (s, p) in order.enumerated() { slot[p] = s }
        self.slot = slot

        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = count * 6
        desc.indexCapacity = count * 24
        desc.indexType = .uint32
        desc.vertexAttributes = [.init(semantic: .position, format: .float3, layoutIndex: 0, offset: 0)]
        desc.vertexLayouts = [.init(bufferIndex: 0, bufferStride: MemoryLayout<SIMD3<Float>>.stride)]  // 16: we write SIMD3<Float>
        let mesh = try LowLevelMesh(descriptor: desc), n = colors.count, faces = Self.faces
        mesh.withUnsafeMutableIndices { raw in
            let idx = raw.bindMemory(to: UInt32.self)
            for s in 0..<n { for (k, f) in faces.enumerated() { idx[s * 24 + k] = UInt32(s * 6) + f } }
        }
        self.mesh = mesh
        let bounds = BoundingBox(min: [-1.8, -1.8, -1.8], max: [1.8, 1.8, 1.8])
        var materials: [any RealityKit.Material] = []
        var parts: [LowLevelMesh.Part] = []
        var start = 0
        for (m, key) in keys.enumerated() {
            let members = order.filter { bucket[$0] == key }
            let mean = members.reduce(SIMD3<Float>.zero) { $0 + colors[$1] } / Float(members.count)
            materials.append(UnlitMaterial(color: UIColor(red: CGFloat(mean.x), green: CGFloat(mean.y), blue: CGFloat(mean.z), alpha: 1)))
            // indexOffset is in bytes (Apple docs: "The offset, in bytes, of the first index").
            parts.append(.init(indexOffset: start * 24 * MemoryLayout<UInt32>.stride, indexCount: members.count * 24, topology: .triangle, materialIndex: m, bounds: bounds))
            start += members.count
        }
        mesh.parts.replaceAll(parts)
        model = ModelEntity(mesh: try MeshResource(from: mesh), materials: materials)
        points = [SIMD3<Float>](repeating: .zero, count: count)

        entity.name = "cloud-\(index)"
        entity.addChild(model)
        entity.components.set(InputTargetComponent())
        entity.components.set(CollisionComponent(shapes: [.generateSphere(radius: 1.05)]))
        entity.components.set(HoverEffectComponent())
        entity.components.set(OpacityComponent(opacity: 0))
        parent.addChild(entity)
        writeVertices(force: true)
    }

    func ease(to target: Placement, by k: Float) {
        placement.position += (target.position - placement.position) * k
        placement.rotation = simd_slerp(placement.rotation, target.rotation, k)
        placement.size += (target.size - placement.size) * k
        placement.opacity += (target.opacity - placement.opacity) * k
        entity.position = placement.position
        entity.orientation = placement.rotation
        entity.scale = SIMD3(repeating: placement.size)
        entity.components.set(OpacityComponent(opacity: placement.opacity))
        placement.labelScale += (target.labelScale - placement.labelScale) * k
        // Labels keep a readable physical size whatever the cloud's scale; the focused block's label is the title.
        label?.isEnabled = placement.labelScale > 0.05
        label?.scale = SIMD3(repeating: placement.labelScale / max(placement.size, 0.01))
    }

    /// Points stay 4-9 mm in the world: zooming in spreads them apart instead of inflating them.
    func writeVertices(force: Bool) {
        let r = min(0.009, max(0.0042, 0.012 * placement.size)) / max(placement.size, 0.01)
        guard force || abs(r - writtenRadius) > 0.003 * writtenRadius else { return }
        writtenRadius = r
        mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
            let v = raw.bindMemory(to: SIMD3<Float>.self)
            for p in 0..<min(count, points.count) {
                let base = slot[p] * 6, c = points[p]
                for k in 0..<6 { v[base + k] = c + Self.corners[k] * r }
            }
        }
    }
}

private func hsv(_ h: Float, _ s: Float, _ v: Float) -> SIMD3<Float> {
    let i = Int(h * 6) % 6, f = h * 6 - Float(Int(h * 6))
    let p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s)
    switch i {
    case 0: return [v, t, p]
    case 1: return [q, v, p]
    case 2: return [p, v, t]
    case 3: return [p, q, v]
    case 4: return [t, p, v]
    default: return [v, p, q]
    }
}
