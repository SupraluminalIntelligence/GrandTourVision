import SwiftUI
import RealityKit
import simd

// The FlowScope layer gallery: one token cloud per block on an arc around you, all seen through the same 3D window
// of the residual stream. Pick a block and it becomes a fixed room-scale window: you walk to it and into it, pinch
// the data to zoom, drag the data to pull regions in and out, and tilt the window through the high-dimensional
// space by pulling the direction knobs inside it. Nothing moves until you start the tour.
struct GalleryView: View {
    var model: GalleryModel
    @State private var scene = GalleryScene()
    @State private var lastDrag: SIMD3<Float>?
    @State private var zoomStart: Float?
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
                    CloudLabel(cloud: model.clouds[i], seen: model.percentSeen(i), compact: model.focused != nil,
                               dense: model.clouds.count > denseLabelCount,
                               similarity: model.frameIndex > 0 ? model.frame?.similarity?[i] : nil,
                               change: model.frameIndex > 0 ? model.frame?.change?[i] : nil)
                }
            }
            ForEach(0..<WindowDecor.handleCount, id: \.self) { j in
                Attachment(id: "handle-\(j)") { HandleLabel(model: model, slot: j) }
            }
            Attachment(id: "token-card") { TokenCard(model: model) }
            Attachment(id: "caption") {
                if let caption = model.caption {
                    Text(caption).font(.system(size: 34, weight: .semibold)).multilineTextAlignment(.center)
                        .padding(.horizontal, 28).padding(.vertical, 16).glassBackgroundEffect()
                }
            }
        }
        .gesture(SpatialTapGesture().targetedToAnyEntity().onEnded { value in
            switch Target(value.entity) {
            case .cloud(let i): if i != model.focused { model.focused = i }
            case .token(let p): model.pinned = model.pinned == p ? nil : p
            default: break
            }
        })
        .simultaneousGesture(DragGesture().targetedToAnyEntity().onChanged { value in
            let now = value.convert(value.translation3D, from: .local, to: .scene)
            let delta = (now - (lastDrag ?? .zero)) / scene.windowScale  // hand movement in window units
            lastDrag = now
            switch Target(value.entity) {
            case .handle(let j):  // pull a direction toward the axes: the window tilts and the shape changes
                let handles = model.handleDirections
                if j < handles.count { model.tilt(toward: handles[j].q, by: SIMD3<Double>(delta) * 3) }
            case .token: model.view.pan(byWindowUnits: delta)  // the data follows your hand
            default: break
            }
        }.onEnded { _ in lastDrag = nil })
        .simultaneousGesture(MagnifyGesture().targetedToAnyEntity().onChanged { value in
            guard case .token = Target(value.entity) else { return }
            if zoomStart == nil { zoomStart = model.view.zoom }
            model.view.zoom = min(20, max(0.3, (zoomStart ?? 1) * Float(value.magnification)))
        }.onEnded { _ in zoomStart = nil })
        .onAppear { model.galleryOpen = true }
        .task { if DemoScript.enabled { await DemoScript.run(model) } }
        .onDisappear {
            model.galleryOpen = false
            openWindow(id: "gallery", value: "gallery")
            dismissWindow(id: "gallery-panel")
        }
    }
}

/// What a gesture landed on: a block in the gallery, a direction handle, or a token in the window.
private enum Target {
    case cloud(Int), handle(Int), token(Int), none
    init(_ entity: Entity) {
        var e: Entity? = entity
        while let current = e {
            for (prefix, make) in [("cloud-", Target.cloud), ("handle-", Target.handle), ("token-", Target.token)]
            where current.name.hasPrefix(prefix) {
                if let i = Int(current.name.dropFirst(prefix.count)) { self = make(i); return }
            }
            e = current.parent
        }
        self = .none
    }
}

private struct GalleryTitle: View {
    var model: GalleryModel
    var body: some View {
        VStack(spacing: 4) {
            if let f = model.focused, model.clouds.indices.contains(f) {
                let cloud = model.clouds[f]
                Text("\(cloud.name) · \(model.title)").font(.title2.bold())
                Text("dim \(cloud.dim, specifier: "%.1f") · \(model.frameLabel) · \(model.insideCount) of \(model.frame?.N ?? 0) tokens in the window · zoom \(model.view.zoom, specifier: "%.1f")×")
                    .font(.callout).foregroundStyle(.secondary)
                if model.frameIndex > 0, let sim = model.frame?.similarity?[f], let moved = model.frame?.change?[f] {
                    // comparison runs: how this block differs from the first checkpoint
                    Text("vs \(model.frameTitle(0)): similarity \(sim, specifier: "%.3f") (linear CKA) · tokens moved \(moved * 100, specifier: "%.0f")%")
                        .font(.callout).foregroundStyle(sim < 0.95 ? .orange : .secondary)
                }
                Text("Pinch the data to zoom · drag it to move · pull a handle to tilt · walk in")
                    .font(.caption).foregroundStyle(.tertiary)
            } else {
                Text(model.title).font(.title2.bold())
                Text("\(model.frameLabel) · \(model.frame?.N ?? 0) tokens × \(model.frame?.C ?? 0) dims")
                    .font(.callout).foregroundStyle(.secondary)
                Text("Tap a block to step inside it").font(.caption).foregroundStyle(.tertiary)
            }
            Text(model.playing ? "Touring \(model.dims.count) directions" : model.axisSummary)
                .font(.caption).foregroundStyle(.secondary)
            if model.colorByGroup, let names = model.frame?.groupNames, model.frame?.groups != nil {
                HStack(spacing: 14) {
                    ForEach(names.indices, id: \.self) { g in
                        let c = GalleryModel.groupColors[g % GalleryModel.groupColors.count]
                        HStack(spacing: 5) {
                            Circle().fill(Color(red: Double(c.x), green: Double(c.y), blue: Double(c.z))).frame(width: 10, height: 10)
                            Text(names[g]).font(.caption)
                        }
                    }
                }
            }
        }
        .padding(.horizontal, 24).padding(.vertical, 12)
        .glassBackgroundEffect()
    }
}

/// Above this many blocks the arc gets crowded: labels show the name and one number.
private let denseLabelCount = 16

private struct CloudLabel: View {
    let cloud: FlowCloud
    let seen: Int
    let compact: Bool
    var dense = false              // many blocks share the arc: name and one number only
    var similarity: Double? = nil  // linear CKA against the first checkpoint, for comparison runs
    var change: Double? = nil      // how far its token vectors moved from the first checkpoint
    var body: some View {
        VStack(spacing: 2) {
            Text(cloud.name).font(compact || dense ? .callout.bold() : .title3.bold())
            if dense {
                if let change {
                    Text(String(format: "moved %.0f%%", change * 100)).font(.caption)
                        .foregroundStyle((similarity ?? 1) < 0.95 ? .orange : .secondary)
                } else {
                    Text("dim \(cloud.dim, specifier: "%.1f")").font(.caption).foregroundStyle(cloud.dim < 3 ? .orange : .secondary)
                }
            } else if let similarity {
                Text(change.map { String(format: "vs base %.2f · moved %.0f%%", similarity, $0 * 100) } ?? String(format: "vs base %.2f", similarity))
                    .font(compact ? .caption2 : .callout).foregroundStyle(similarity < 0.9 ? .orange : .secondary)
            }
            if !dense {
                Text("dim \(cloud.dim, specifier: "%.1f")")
                    .font(compact ? .caption2 : .callout).foregroundStyle(cloud.dim < 3 ? .orange : .secondary)
            }
            if !compact && !dense {
                Text("\(seen)% seen").font(.caption).foregroundStyle(seen < 50 ? .yellow : .secondary)
            }
        }
        .padding(.horizontal, compact ? 8 : 12).padding(.vertical, compact ? 4 : 6)
        .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 14))
    }
}

/// Beside each direction knob: which direction it is and how much of it the window shows right now.
private struct HandleLabel: View {
    var model: GalleryModel
    let slot: Int
    var body: some View {
        let handles = model.handleDirections
        if slot < handles.count {
            let _ = model.tourTick
            let seen = Int((model.currentFrame().visibility(handles[slot].q) * 100).rounded())
            VStack(spacing: 1) {
                Text(verbatim: "PC\(handles[slot].pc + 1)").font(.caption.bold())
                Text(verbatim: "\(seen)%").font(.caption2).foregroundStyle(seen > 50 ? .primary : .secondary)
            }
            .padding(.horizontal, 6).padding(.vertical, 3)
            .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 8))
        }
    }
}

/// The token you tapped: its character in the sentence around it.
private struct TokenCard: View {
    var model: GalleryModel
    var body: some View {
        if let p = model.pinned, let f = model.frame, let tokens = f.tokens, p < tokens.count {
            let start = (p / f.T) * f.T, i = p - start
            let text: [String] = (start..<min(start + f.T, tokens.count)).map { k in
                if let texts = f.texts { return visible(texts[k]) }  // decoded tokens from a comparison run
                return character(tokens[k], f.vocab)                // character-level runs carry a vocabulary
            }
            let group = f.groups.flatMap { g in f.groupNames.map { $0[g[p]] } }
            VStack(alignment: .leading, spacing: 4) {
                Text("\(group.map { "\($0) · " } ?? "")sequence \(p / f.T + 1), position \(i)").font(.caption).foregroundStyle(.secondary)
                text.indices.reduce(Text("")) { line, k in
                    line + Text(verbatim: text[k]).foregroundColor(k == i ? .cyan : .secondary).bold(k == i)
                }
                .font(.system(.body, design: .monospaced))
                Text("The line traces this sequence through the window").font(.caption2).foregroundStyle(.tertiary)
            }
            .padding(12)
            .glassBackgroundEffect(in: RoundedRectangle(cornerRadius: 14))
        }
    }

    private func visible(_ s: String) -> String {
        s.replacingOccurrences(of: "\n", with: "⏎").replacingOccurrences(of: " ", with: "␣")
    }

    private func character(_ id: Int, _ vocab: String?) -> String {
        guard let vocab, id >= 0, id < vocab.count else { return "·\(id)" }
        let c = vocab[vocab.index(vocab.startIndex, offsetBy: id)]
        return c == "\n" ? "⏎" : c == " " ? "␣" : String(c)
    }
}

// MARK: - scene

@MainActor
final class GalleryScene {
    static let windowRadius: Float = 1.2  // a 2.4 m window at size 1
    let root = Entity()
    var subscription: EventSubscription?
    private(set) var windowScale: Float = windowRadius
    private var clouds: [CloudNode] = []
    private var title: Entity?
    private lazy var decor = WindowDecor(parent: root)
    private var shape = (layers: 0, points: 0, colorKey: "")
    private var lastView: (time: Double, dims: [Int], fits: [Double], frame: Int, window: WindowFrame?) = (-1, [], [], -1, nil)
    private var sample = WindowSample()
    private var states: [PointState] = []
    private var hints: [SIMD3<Float>] = []

    /// Rebuild when the data's shape or coloring changes; otherwise just (re)attach labels.
    func sync(model: GalleryModel, attachments: RealityViewAttachments) {
        let colorKey = "\(model.colorByToken)-\(model.colorByGroup)-\(model.frame?.T ?? 0)-\(model.frame?.tokens?.prefix(8).map(String.init).joined() ?? "")"
        let n = model.frame?.N ?? 0
        if shape.layers != model.clouds.count || shape.points != n || shape.colorKey != colorKey {
            clouds.forEach { $0.entity.removeFromParent() }
            clouds = model.clouds.indices.compactMap { i in try? CloudNode(index: i, colors: colors(model), parent: root) }
            shape = (model.clouds.count, n, colorKey)
            lastView.frame = -1
        }
        for (i, node) in clouds.enumerated() {
            if let label = attachments.entity(for: "label-\(i)"), label.parent !== node.entity { node.entity.addChild(label) }
            node.label = attachments.entity(for: "label-\(i)")
        }
        for j in 0..<WindowDecor.handleCount { decor.handleLabels[j] = attachments.entity(for: "handle-\(j)") }
        decor.tokenCard = attachments.entity(for: "token-card")
        if let t = attachments.entity(for: "title"), t.parent !== root { root.addChild(t); title = t }
        if let c = attachments.entity(for: "caption"), c.parent !== root {
            root.addChild(c); c.position = [0, 0.98, -1.25]
            c.orientation = simd_quatf(angle: -0.35, axis: [1, 0, 0])  // tipped back toward you, like a teleprompter
        }
    }

    /// Per frame: advance the tour if it's running, ease the layout, draw the gallery and the open window.
    func tick(model: GalleryModel, dt: TimeInterval) {
        model.advance(dt)
        guard !clouds.isEmpty, clouds.count == model.clouds.count, model.fits.count == model.clouds.count else {
            decor.setVisible(false); return
        }
        let ease = Float(1 - exp(-dt * 6))
        let frame = model.currentFrame(), dims = model.dims
        let view = (model.time, dims, model.fits, model.frameIndex, model.windowFrame)
        let reproject = view != lastView
        for (i, node) in clouds.enumerated() {
            let open = model.focused == i
            node.ease(to: layout(i, count: clouds.count, model: model), by: open ? max(ease, 0.35) : ease)
            node.setTappable(!open)
            if open {
                drawWindow(node, cloud: model.clouds[i], model: model, frame: frame, dims: dims)
            } else {
                if reproject || node.wasWindowed { model.clouds[i].project(frame.vectors, dims: dims, fit: model.fits[i], into: &node.points) }
                node.write(force: reproject)
            }
        }
        lastView = view
        decor.setVisible(model.focused != nil)
        if let title {
            let target: SIMD3<Float> = model.focused == nil ? [0, 2.25, -2.6] : [-0.55, 2.1, -1.4]
            title.position += (target - title.position) * ease
            title.orientation = simd_quatf(angle: atan2(-title.position.x, -title.position.z), axis: [0, 1, 0])
            title.scale = SIMD3(repeating: model.focused == nil ? 2.2 : 1.5)
        }
    }

    /// The open block: clipped at the window, faint hints just outside, and in slice mode only tokens near the slice.
    private func drawWindow(_ node: CloudNode, cloud: FlowCloud, model: GalleryModel, frame: WindowFrame, dims: [Int]) {
        cloud.project(frame, dims: dims, into: &sample)
        let n = cloud.N
        if states.count != n { states = .init(repeating: .hidden, count: n); hints = .init(repeating: .zero, count: n) }
        // slice thickness scales with how far tokens typically sit from this view; depth is in standard deviations
        let typical = (0..<n).map { sliceDistance(residual: sample.residual[$0], hidden: sample.hidden[$0], depth: 0) }.sorted()[n / 2]
        let thickness = model.sliceThickness * max(typical, 1e-6)
        let mean = sample.hidden.reduce(0, +) / Float(n)
        let spread = sqrt(max(sample.hidden.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(n), 1e-12))
        let depth = mean + model.depth * spread
        var inside = 0
        for p in 0..<n {
            let l = model.view.local(sample.base[p]), r = model.view.reach(l)
            let near = !model.sliceMode || sliceDistance(residual: sample.residual[p], hidden: sample.hidden[p], depth: depth) < thickness
            node.points[p] = l
            hints[p] = l / max(r, 1e-6)
            states[p] = r <= 1 && near ? .shown : (model.showHints && near && r < 3 ? .hint : .hidden)
            if states[p] == .shown { inside += 1 }
        }
        node.writeWindow(states: states, hintPositions: hints, worldRadius: 0.006 * DemoScript.pointScale)
        windowScale = node.placement.size
        decor.update(center: node.entity.position, scale: windowScale, model: model, frame: frame,
                     local: node.points, states: states, parent: root)
        if model.insideCount != inside { model.insideCount = inside }
    }

    private func colors(_ model: GalleryModel) -> [SIMD3<Float>] {
        guard let f = model.frame else { return [] }
        return (0..<f.N).map { p in
            if model.colorByGroup, let groups = f.groups {
                return GalleryModel.groupColors[groups[p] % GalleryModel.groupColors.count]
            }
            if model.colorByToken, let tokens = f.tokens {
                let h = Float((Double(tokens[p]) * 137.508).truncatingRemainder(dividingBy: 360) / 360)
                return hsv(h, 0.65, 0.95)
            }
            return FlowColors.viridis(Float(p % f.T) / Float(max(1, f.T - 1)))
        }
    }

    /// No block open: an arc at arm's length, embeddings on the left, last block on the right.
    /// A block open: a fixed 2.4 m window in front of you; the other blocks wait on a rail to your left.
    private func layout(_ i: Int, count: Int, model: GalleryModel) -> Placement {
        guard let f = model.focused else {
            let radius: Float = 1.6
            let span = min(Float.pi * 0.83, Float(max(count - 1, 1)) * 0.25)
            let angle = count == 1 ? 0 : -span / 2 + span * Float(i) / Float(count - 1)
            let pitch = radius * span / Float(max(count - 1, 1))  // metres between neighbours
            let size = min(0.17, 0.42 * pitch)
            // labels shrink with the spacing so neighbours never overlap; many blocks get the narrow dense label
            let room: Float = count > denseLabelCount ? 0.12 : 0.3  // metres a label needs at full size
            return Placement(position: [radius * sin(angle), 1.5, -radius * cos(angle)], size: size, opacity: 1,
                             labelScale: min(1.1, 1.1 * pitch / room), yaw: -angle)
        }
        if f == i {
            let size = Self.windowRadius * model.windowSize
            // 2.3 m away: the window's front face stays behind the control panel, so the panel is never covered
            return Placement(position: [0, max(1.45, size + 0.05), -2.3], size: size, opacity: 1, labelScale: 0, yaw: 0)
        }
        let others = count - 1, k = i < f ? i : i - 1
        let angle = -1.75 + 1.0 * Float(k) / Float(max(others - 1, 1))
        let radius: Float = 1.35
        return Placement(position: [radius * sin(angle), 1.3, -radius * cos(angle)],
                         size: min(0.06, 0.4 * radius / Float(max(others, 1))), opacity: 0.9, labelScale: 0.55, yaw: -angle)
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
