import Foundation
import Observation
import simd

// State for the FlowScope layer gallery: which run we're looking at (a bundled recording or a live FlowScope
// server), the decoded per-block clouds, and the shared tour. Tour time is read by the render loop every
// frame, so it is deliberately not observed (it would re-run SwiftUI on every frame).
@MainActor @Observable
final class GalleryModel {
    enum Source: Equatable { case none, recording(String), live(String) }

    static let samples: [(file: String, title: String)] = [
        ("healthy-12-layers", "Healthy · 12 blocks (recorded)"),
        ("no-residual-12-layers", "No residual · 12 blocks (recorded)"),
    ]

    private(set) var source: Source = .none
    private(set) var frames: [FlowFrame] = []
    private(set) var frameIndex = 0
    private(set) var clouds: [FlowCloud] = []
    private(set) var fits: [Double] = []
    private(set) var toured = 8
    private(set) var planes = TorusTour.speeds(8)
    private(set) var status = "Choose a recording or connect to FlowScope."
    private(set) var axes = [0, 1, 2]  // principal directions shown on X/Y/Z when the tour isn't running
    var playing = false  // static by default: you explore; the tour starts only when asked
    var speed = 1.0
    var colorByToken = false
    var focused: Int? { didSet { if focused != oldValue { resetPlacement() } } }
    // Your placement of the focused block (pinch, drag, two-hand rotate), on top of its default room-scale pose.
    var focusOffset = SIMD3<Float>.zero
    var focusScale: Float = 1
    var focusRotation = simd_quatf(angle: 0, axis: [0, 1, 0])
    var galleryOpen = false
    var liveAddress = "127.0.0.1:8765"
    var error: String?
    @ObservationIgnored var time = 0.0
    @ObservationIgnored private var liveTask: Task<Void, Never>?

    var frame: FlowFrame? { frames.indices.contains(frameIndex) ? frames[frameIndex] : nil }
    var isLive: Bool { if case .live = source { return true } else { return false } }
    var title: String { frame?.label ?? (isLive ? "Live FlowScope run" : "FlowScope run") }

    func loadSample(_ file: String) {
        do {
            guard let url = Bundle.main.url(forResource: file, withExtension: "jsonl") else {
                throw CSVError.invalid("Bundled recording \(file) is missing.")
            }
            try load(FlowRecording.parse(Data(contentsOf: url)), source: .recording(file))
        } catch { self.error = error.localizedDescription }
    }

    func load(_ frames: [FlowFrame], source: Source) throws {
        disconnect()
        self.frames = frames
        self.source = source
        focused = nil
        try select(frame: 0)
        status = "\(frames.count) snapshots · steps \(frames.first!.step)–\(frames.last!.step)"
    }

    /// Switching training step keeps the tour time and toured dims, so the same view is compared across steps.
    func select(frame index: Int) throws {
        guard frames.indices.contains(index) else { return }
        let f = frames[index]
        clouds = try f.layers.map { try FlowCloud($0, N: f.N, C: f.C) }
        frameIndex = index
        if focused.map({ $0 >= clouds.count }) ?? false { focused = nil }
        if axes.contains(where: { $0 >= f.C }) { axes = [0, 1, 2] }
        setToured(toured > f.C ? min(f.k, f.C) : toured)
    }

    /// The principal directions in play: chosen axes first, then the rest of the toured set.
    var dims: [Int] { viewDims(axes: axes, toured: toured, C: frame?.C ?? 64) }

    func setToured(_ n: Int) {
        toured = max(3, min(n, frame?.C ?? n))
        refit()
    }

    /// Pick which principal direction each display axis shows. Pauses the tour and returns to that exact view.
    func setAxis(_ axis: Int, to direction: Int) {
        guard axes.indices.contains(axis), (0..<(frame?.C ?? 64)).contains(direction) else { return }
        var next = axes
        if let other = next.firstIndex(of: direction) { next.swapAt(axis, other) } else { next[axis] = direction }
        axes = next
        playing = false; time = 0
        refit()
    }

    /// Step an axis to the next principal direction not already shown on another axis.
    func stepAxis(_ axis: Int, by delta: Int) {
        let C = frame?.C ?? 64
        var d = axes[axis]
        repeat { d = (d + delta + C) % C } while axes.enumerated().contains { $0.offset != axis && $0.element == d }
        setAxis(axis, to: d)
    }

    private func refit() {
        let d = dims
        if planes.count != d.count * (d.count - 1) / 2 { planes = TorusTour.speeds(d.count) }
        fits = clouds.map { $0.fit(dims: d) }
    }

    func percentSeen(_ i: Int) -> Int { clouds[i].percentSeen(dims: dims) }

    /// Back to the chosen axes (tour time 0), paused.
    func resetView() { time = 0; playing = false }

    func resetPlacement() { focusOffset = .zero; focusScale = 1; focusRotation = simd_quatf(angle: 0, axis: [0, 1, 0]) }

    func advance(_ dt: Double) { if playing { time += min(max(dt, 0), 0.1) * speed } }

    func basis() -> [[Double]] { TorusTour.basis(dims.count, t: time, planes: planes) }

    // MARK: live FlowScope connection (Server-Sent Events from `flowscope run ... --host 0.0.0.0`)

    func connect() {
        disconnect()
        let address = liveAddress.trimmingCharacters(in: .whitespaces)
        guard let url = URL(string: "http://\(address)/events"), url.host != nil else {
            error = "Use host:port, e.g. 192.168.1.20:8765"; return
        }
        source = .live(address)
        frames = []; clouds = []; fits = []; frameIndex = 0
        status = "Connecting to \(address)…"
        liveTask = Task.detached { [weak self] in
            while !Task.isCancelled {
                do {
                    let (bytes, response) = try await URLSession.shared.bytes(from: url)
                    guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
                    await self?.setStatus("Connected · waiting for the next frame")
                    for try await line in bytes.lines {
                        if let frame = FlowEvents.frame(fromLine: line) { await self?.receive(frame) }
                    }
                    await self?.setStatus("Run ended · keeping the last frame · reconnecting")
                } catch {
                    if Task.isCancelled { return }
                    await self?.setStatus("Can't reach \(address) (\(error.localizedDescription)) · retrying")
                }
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func disconnect() {
        liveTask?.cancel(); liveTask = nil
        if isLive { source = .none; status = "Disconnected." }
    }

    private func setStatus(_ text: String) { status = text }

    private func receive(_ frame: FlowFrame) {
        guard isLive else { return }
        frames = [frame]
        do {
            try select(frame: 0)
            status = "Live · step \(frame.step)"
        } catch { status = "Bad frame: \(error.localizedDescription)" }
    }
}
