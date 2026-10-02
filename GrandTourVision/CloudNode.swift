import RealityKit
import UIKit
import simd

/// Where a block sits and how big it is (in metres), eased toward each frame.
struct Placement {
    var position: SIMD3<Float>
    var size: Float
    var opacity: Float
    var labelScale: Float
    var yaw: Float
}

/// How a token is drawn in the window: fully, as a faint hint on the boundary, or not at all.
enum PointState { case shown, hint, hidden }

/// One block's token cloud: a single LowLevelMesh with a small octahedron per token. Tokens are grouped by color
/// bucket so each bucket is one mesh part; in the window each bucket also gets a faint part for edge hints.
@MainActor
final class CloudNode {
    let entity = Entity()
    var label: Entity? { didSet { label?.position = [0, -1.3, 0] } }
    var points: [SIMD3<Float>] = []
    private(set) var placement = Placement(position: [0, 1.5, -1.6], size: 0.001, opacity: 0, labelScale: 1, yaw: 0)
    private let model: ModelEntity
    private let mesh: LowLevelMesh
    private let count: Int
    private let slot: [Int]          // token -> vertex block
    private let bucketOf: [Int]      // token -> color bucket
    private let buckets: Int
    private var staticParts: [LowLevelMesh.Part] = []
    private var writtenRadius: Float = -1
    private var windowed = false
    private var tappable: Bool?
    var wasWindowed: Bool { windowed }  // leaving the window: points must be re-projected for the gallery

    // Each token is a small icosphere (12 vertices, 20 faces) with smooth normals: it reads as a sphere at this size,
    // and lighting gives it a depth cue that flat shapes don't have.
    private static let corners: [SIMD3<Float>] = {
        let t: Float = (1 + Float(5).squareRoot()) / 2
        let raw: [SIMD3<Float>] = [[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                                   [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]]
        return raw.map(simd_normalize)
    }()
    private static let faces: [UInt32] = {
        var f: [UInt32] = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                           3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]
        // make every triangle wind counter-clockwise seen from outside, whatever the source table used
        for i in stride(from: 0, to: f.count, by: 3) {
            let a = corners[Int(f[i])], b = corners[Int(f[i + 1])], c = corners[Int(f[i + 2])]
            if simd_dot(simd_cross(b - a, c - a), a + b + c) < 0 { f.swapAt(i + 1, i + 2) }
        }
        return f
    }()
    private static let vertsPer = 12, indicesPer = 60
    private static let bounds = BoundingBox(min: [-3.2, -3.2, -3.2], max: [3.2, 3.2, 3.2])

    init(index: Int, colors: [SIMD3<Float>], parent: Entity) throws {
        count = colors.count
        let key = colors.map { c in Int((c.x * 3).rounded()) * 16 + Int((c.y * 3).rounded()) * 4 + Int((c.z * 3).rounded()) }
        let keys = Array(Set(key)).sorted()
        let bucketOf = key.map { keys.firstIndex(of: $0)! }
        self.bucketOf = bucketOf
        buckets = keys.count
        let order = colors.indices.sorted { (bucketOf[$0], $0) < (bucketOf[$1], $1) }
        var slot = [Int](repeating: 0, count: count)
        for (s, p) in order.enumerated() { slot[p] = s }
        self.slot = slot

        var desc = LowLevelMesh.Descriptor()
        desc.vertexCapacity = count * Self.vertsPer
        desc.indexCapacity = count * Self.indicesPer
        desc.indexType = .uint32
        // interleaved position + normal, written as two SIMD3<Float> (16 bytes each)
        desc.vertexAttributes = [.init(semantic: .position, format: .float3, layoutIndex: 0, offset: 0),
                                 .init(semantic: .normal, format: .float3, layoutIndex: 0, offset: 16)]
        desc.vertexLayouts = [.init(bufferIndex: 0, bufferStride: 32)]
        let mesh = try LowLevelMesh(descriptor: desc)
        self.mesh = mesh

        // Two materials per bucket: solid tokens, and faint edge hints for data just outside the window.
        var materials: [any RealityKit.Material] = []
        for b in 0..<keys.count {
            let members = order.filter { bucketOf[$0] == b }
            let mean = members.reduce(SIMD3<Float>.zero) { $0 + colors[$1] } / Float(members.count)
            let color = UIColor(red: CGFloat(mean.x), green: CGFloat(mean.y), blue: CGFloat(mean.z), alpha: 1)
            materials.append(Self.material(color, opacity: 1))
            materials.append(Self.material(color, opacity: 0.3))
        }
        model = ModelEntity()
        let n = count
        mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in  // normals once; positions are rewritten every frame
            let v = raw.bindMemory(to: SIMD3<Float>.self)
            for s in 0..<n { for i in 0..<Self.vertsPer { v[2 * (s * Self.vertsPer + i) + 1] = Self.corners[i] } }
        }
        writeStaticIndices()
        model.model = ModelComponent(mesh: try MeshResource(from: mesh), materials: materials)
        points = [SIMD3<Float>](repeating: .zero, count: count)

        entity.name = "cloud-\(index)"
        entity.addChild(model)
        entity.components.set(InputTargetComponent())
        entity.components.set(HoverEffectComponent())
        entity.components.set(OpacityComponent(opacity: 0))
        setTappable(true)
        parent.addChild(entity)
        write(force: true)
    }

    /// Small multiples are tappable as a whole; the focused window is not (its tokens are, individually).
    func setTappable(_ on: Bool) {
        guard tappable != on else { return }
        tappable = on
        if on { entity.components.set(CollisionComponent(shapes: [.generateSphere(radius: 1.05)])) }
        else { entity.components.remove(CollisionComponent.self) }
    }

    func ease(to target: Placement, by k: Float) {
        placement.position += (target.position - placement.position) * k
        placement.yaw += (target.yaw - placement.yaw) * k
        placement.size += (target.size - placement.size) * k
        placement.opacity += (target.opacity - placement.opacity) * k
        placement.labelScale += (target.labelScale - placement.labelScale) * k
        entity.position = placement.position
        entity.orientation = simd_quatf(angle: placement.yaw, axis: [0, 1, 0])
        entity.scale = SIMD3(repeating: placement.size)
        entity.components.set(OpacityComponent(opacity: placement.opacity))
        // Labels keep a readable physical size whatever the cloud's scale; the focused block's label is the title.
        label?.isEnabled = placement.labelScale > 0.05
        label?.scale = SIMD3(repeating: placement.labelScale / max(placement.size, 0.01))
    }

    /// Points stay 4-9 mm in the world in the gallery; zooming the data spreads them apart instead of inflating them.
    private var radius: Float { DemoScript.pointScale * min(0.009, max(0.0042, 0.012 * placement.size)) / max(placement.size, 0.01) }

    /// Gallery mode: every token drawn at its projected position.
    func write(force: Bool) {
        if windowed { writeStaticIndices(); windowed = false }
        let r = radius
        guard force || abs(r - writtenRadius) > 0.003 * writtenRadius else { return }
        writtenRadius = r
        writeVertices(r) { p in (self.points[p], 1) }
    }

    /// Window mode: tokens inside are drawn; hints sit on the boundary, faint and smaller; the rest are skipped.
    func writeWindow(states: [PointState], hintPositions: [SIMD3<Float>], worldRadius: Float) {
        windowed = true
        let r = worldRadius / max(placement.size, 0.01)
        writeVertices(r) { p in states[p] == .hint ? (hintPositions[p], 0.6) : (self.points[p], 1) }
        var parts: [LowLevelMesh.Part] = []
        mesh.withUnsafeMutableIndices { raw in
            let idx = raw.bindMemory(to: UInt32.self)
            var at = 0
            for b in 0..<buckets {
                for (state, material) in [(PointState.shown, 2 * b), (.hint, 2 * b + 1)] {
                    let start = at
                    for p in 0..<count where bucketOf[p] == b && states[p] == state {
                        let base = UInt32(slot[p] * Self.vertsPer)
                        for f in Self.faces { idx[at] = base + f; at += 1 }
                    }
                    if at > start {
                        // indexOffset is in bytes (Apple docs: "The offset, in bytes, of the first index").
                        parts.append(.init(indexOffset: start * MemoryLayout<UInt32>.stride, indexCount: at - start,
                                           topology: .triangle, materialIndex: material, bounds: Self.bounds))
                    }
                }
            }
        }
        mesh.parts.replaceAll(parts)
    }

    private func writeVertices(_ r: Float, _ at: (Int) -> (SIMD3<Float>, Float)) {
        mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
            let v = raw.bindMemory(to: SIMD3<Float>.self)
            for p in 0..<min(count, points.count) {
                let (c, k) = at(p), base = slot[p] * Self.vertsPer
                for i in 0..<Self.vertsPer { v[2 * (base + i)] = c + Self.corners[i] * (r * k) }  // normals stay as written at init
            }
        }
    }

    private func writeStaticIndices() {
        let n = count, faces = Self.faces
        mesh.withUnsafeMutableIndices { raw in
            let idx = raw.bindMemory(to: UInt32.self)
            for s in 0..<n { for (k, f) in faces.enumerated() { idx[s * Self.indicesPer + k] = UInt32(s * Self.vertsPer) + f } }
        }
        if staticParts.isEmpty {
            var start = 0
            for b in 0..<buckets {
                let members = bucketOf.filter { $0 == b }.count
                staticParts.append(.init(indexOffset: start * Self.indicesPer * MemoryLayout<UInt32>.stride, indexCount: members * Self.indicesPer,
                                         topology: .triangle, materialIndex: 2 * b, bounds: Self.bounds))
                start += members
            }
        }
        mesh.parts.replaceAll(staticParts)
        writtenRadius = -1
    }

    private static func material(_ color: UIColor, opacity: Float) -> PhysicallyBasedMaterial {
        var m = PhysicallyBasedMaterial()
        m.baseColor = .init(tint: color)
        m.emissiveColor = .init(color: color)
        m.emissiveIntensity = 0.55
        m.roughness = .init(floatLiteral: 0.35)
        m.metallic = .init(floatLiteral: 0)
        if opacity < 1 { m.blending = .transparent(opacity: .init(floatLiteral: opacity)) }
        return m
    }
}
