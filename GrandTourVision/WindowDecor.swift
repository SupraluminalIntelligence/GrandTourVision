import RealityKit
import UIKit
import simd

/// Everything around the window you step into. The window itself never moves; this draws its boundary and
/// floor footprint, marks the data's center, shows a spoke per handled direction with a grabbable knob on its tip
/// (hidden directions park in a column along the window's front-left edge, clear of the control panel), gives every visible token a grab target, and
/// traces the pinned token's sentence.
@MainActor
final class WindowDecor {
    static let handleCount = 8
    static let knobRadius: Float = 0.012     // metres, whatever the window size

    let frame = Entity()       // follows the window: inside it, the window spans -1...1
    let footprint = Entity()
    var handleLabels: [Entity?] = Array(repeating: nil, count: WindowDecor.handleCount)
    var tokenCard: Entity?
    private let cubeEdges = Entity(), cubeFoot = Entity()
    private let sphere = ModelEntity(mesh: .generateSphere(radius: 1), materials: [WindowDecor.material(.cyan, 0.05)])
    private let sphereFoot = ModelEntity(mesh: .generateCylinder(height: 0.002, radius: 1), materials: [WindowDecor.material(.cyan, 0.12)])
    private let origin = Entity()
    private var spokes: [ModelEntity] = []
    private(set) var knobs: [ModelEntity] = []
    private var segments: [ModelEntity] = []
    private var grabbers: [Entity] = []

    init(parent: Entity) {
        let accent = UIColor(red: 0.13, green: 0.83, blue: 0.93, alpha: 1)
        let corners: [SIMD3<Float>] = [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)] {
            let e = Self.bar(0.004, accent, 0.55); Self.stretch(e, from: corners[a], to: corners[b]); cubeEdges.addChild(e)
        }
        for (a, b) in [(0, 1), (1, 5), (5, 4), (4, 0)] {  // floor square, drawn at y = 0 of the footprint entity
            let e = Self.bar(0.006, accent, 0.45)
            Self.stretch(e, from: [corners[a].x, 0, corners[a].z], to: [corners[b].x, 0, corners[b].z]); cubeFoot.addChild(e)
        }
        for axis in 0..<3 {
            var d = SIMD3<Float>.zero; d[axis] = 0.07
            let e = Self.bar(0.006, .white, 0.9); Self.stretch(e, from: -d, to: d); origin.addChild(e)
        }
        frame.addChild(cubeEdges); frame.addChild(sphere); frame.addChild(origin)
        footprint.addChild(cubeFoot); footprint.addChild(sphereFoot)

        for j in 0..<Self.handleCount {
            let spoke = Self.bar(0.004, .white, 0.5); frame.addChild(spoke); spokes.append(spoke)
            // unit sphere scaled each frame to a fixed size in metres; generous collision so it's easy to pinch
            let knob = ModelEntity(mesh: .generateSphere(radius: 1), materials: [Self.material(accent, 1)])
            knob.name = "handle-\(j)"
            knob.components.set(InputTargetComponent())
            knob.components.set(CollisionComponent(shapes: [.generateSphere(radius: 2.6)]))
            knob.components.set(HoverEffectComponent())
            frame.addChild(knob); knobs.append(knob)
        }
        for _ in 0..<64 { let s = Self.bar(0.006, .white, 0.6); frame.addChild(s); segments.append(s) }
        parent.addChild(frame); parent.addChild(footprint)
        setVisible(false)
    }

    func setVisible(_ on: Bool) {
        frame.isEnabled = on; footprint.isEnabled = on
        grabbers.forEach { $0.isEnabled = on && $0.isEnabled }
        handleLabels.forEach { $0?.isEnabled = on }
        if !on { tokenCard?.isEnabled = false }
    }

    /// Per frame, while a block is open: boundary, spokes and knobs, grab targets, the pinned sentence.
    func update(center: SIMD3<Float>, scale: Float, model: GalleryModel, frame windowFrame: WindowFrame,
                local: [SIMD3<Float>], states: [PointState], parent: Entity) {
        frame.position = center; frame.scale = SIMD3(repeating: scale)
        footprint.position = [center.x, 0.003, center.z]; footprint.scale = SIMD3(repeating: scale)
        let cube = model.view.shape == .cube
        cubeEdges.isEnabled = cube; cubeFoot.isEnabled = cube; sphere.isEnabled = !cube; sphereFoot.isEnabled = !cube

        // the data's center (the mean token) moves with your hand
        let o = model.view.local(.zero)
        origin.isEnabled = model.view.reach(o) <= 1; origin.position = o

        // spokes and their knobs: a visible direction's knob sits on its spoke tip inside the window; a hidden one
        // waits in a column along the front-left edge. Grab a knob and pull: the window tilts toward that direction.
        let handles = model.handleDirections
        for j in 0..<Self.handleCount {
            guard j < handles.count else { spokes[j].isEnabled = false; knobs[j].isEnabled = false; handleLabels[j]?.isEnabled = false; continue }
            let v = SIMD3<Float>(windowFrame.spoke(handles[j].q)), vis = min(1, simd_length(v))
            Self.stretch(spokes[j], from: .zero, to: v * 0.92)
            spokes[j].components.set(OpacityComponent(opacity: 0.08 + 0.3 * vis))
            let tray = SIMD3<Float>(-0.9, 0.3 - 0.75 * Float(j) / Float(Self.handleCount - 1), 0.92)
            let tip = vis > 1e-3 ? simd_normalize(v) * 0.92 : tray
            let at = tip * vis + tray * (1 - vis)
            knobs[j].isEnabled = true
            knobs[j].position = at
            knobs[j].scale = SIMD3(repeating: Self.knobRadius * (0.8 + 0.5 * vis) / scale)
            knobs[j].components.set(OpacityComponent(opacity: 0.5 + 0.5 * vis))
            if let label = handleLabels[j] {
                label.isEnabled = true
                if label.parent !== parent { parent.addChild(label) }
                label.position = center + at * scale + [0.05, 0.03, 0]
            }
        }

        // grab targets: one per visible token, so you can pinch the data itself to move or zoom it
        while grabbers.count < local.count {
            let g = Entity(); g.name = "token-\(grabbers.count)"
            g.components.set(InputTargetComponent())
            g.components.set(CollisionComponent(shapes: [.generateSphere(radius: 0.02)]))
            parent.addChild(g); grabbers.append(g)
        }
        for p in grabbers.indices {
            let on = p < local.count && states[p] == .shown
            grabbers[p].isEnabled = on
            if on { grabbers[p].position = center + local[p] * scale }
        }

        // the pinned token's whole sentence, in order, where it runs inside the window
        var used = 0
        if let pinned = model.pinned, let T = model.frame?.T, pinned < local.count {
            let start = (pinned / T) * T
            for i in start..<min(start + T, local.count) - 1 where states[i] == .shown && states[i + 1] == .shown && used < segments.count {
                Self.stretch(segments[used], from: local[i], to: local[i + 1]); used += 1
            }
            if let card = tokenCard {
                if card.parent !== parent { parent.addChild(card) }
                card.isEnabled = states[pinned] == .shown
                card.position = center + local[pinned] * scale + [0, 0.12, 0.02]
            }
        } else { tokenCard?.isEnabled = false }
        for k in used..<segments.count { segments[k].isEnabled = false }
    }

    static func material(_ color: UIColor, _ opacity: Float) -> UnlitMaterial {
        var m = UnlitMaterial(color: color)
        if opacity < 1 { m.blending = .transparent(opacity: .init(floatLiteral: opacity)) }
        return m
    }

    static func bar(_ thickness: Float, _ color: UIColor, _ opacity: Float) -> ModelEntity {
        ModelEntity(mesh: .generateBox(size: [thickness, thickness, 1]), materials: [material(color, opacity)])
    }

    /// Stretch a unit bar between two points (in its parent's space).
    static func stretch(_ e: Entity, from a: SIMD3<Float>, to b: SIMD3<Float>) {
        let d = b - a, len = simd_length(d)
        e.isEnabled = len > 1e-4
        guard len > 1e-4 else { return }
        e.position = (a + b) / 2
        e.orientation = simd_quatf(from: [0, 0, 1], to: d / len)
        e.scale = [1, 1, len]
    }
}
