import Foundation
import Observation

@MainActor @Observable
final class TourModel {
    var dataset = Dataset.synthetic()
    var normalization = Normalization.standardized {
        didSet { if normalization != oldValue { prepare() } }
    }
    var projection = Projection(dimensions: 4)
    var playing = false
    var speed = 1.0
    var time = 0.0
    var planeI = 0
    var planeJ = 3
    var manualAngle = 0.0
    var plotReset = 0
    // A return or newer focus request invalidates a pending controls dismissal.
    var focusRequest: UUID?
    var scenePlacement = ScenePlacement()
    var immersiveActive = false
    var roomControlsHidden = false
    var colorByPosition = false
    var error: String?
    private(set) var savedViewpoints: [SavedViewpoint] = []
    private(set) var trace: ActivationTrace?
    private(set) var frameIndex = 0
    var selectedTokenIndex = 0
    var comparisonLocked = true {
        didSet { if comparisonLocked { playing = false } }
    }
    private(set) var normalized: [[Double]] = []
    private(set) var normalizedFrames: [[[Double]]] = []
    private(set) var radius = 1.0
    @ObservationIgnored private var clockRunning = false
    var layers: [String] { Array(Set(trace?.snapshots.map(\.layer) ?? [])).sorted() }
    var checkpoints: [String] { Array(Set(trace?.snapshots.map(\.checkpoint) ?? [])).sorted() }
    var currentFrame: TraceSnapshot? { trace?.snapshots[frameIndex] }
    var selectedToken: TraceToken? {
        guard let trace, trace.tokens.indices.contains(selectedTokenIndex) else { return nil }
        return trace.tokens[selectedTokenIndex]
    }
    var plotLegend: [String] {
        if let trace { return trace.colorLabels }
        return Set(dataset.groups).count > 1 ? ["Cluster A", "Cluster B", "Cluster C", "Cluster D"] : ["CSV points"]
    }
    var projected: [[Double]] { normalized.map { projection.project($0) } }
    var clippedCount: Int {
        projected.filter { sqrt($0.reduce(0) { $0 + $1*$1 }) > radius * 1.000001 }.count
    }
    var contributionDimensions: [Int] {
        guard dataset.names.count > 24 else { return Array(dataset.names.indices) }
        return dataset.names.indices.sorted { a, b in
            projection.basis.reduce(0) { $0 + abs($1[a]) } > projection.basis.reduce(0) { $0 + abs($1[b]) }
        }.prefix(24).sorted()
    }
    init() { prepare() }
    func runTourClock() async {
        while clockRunning {
            do { try await Task.sleep(for: .milliseconds(100)) } catch { return }
        }
        guard !Task.isCancelled else { return }
        clockRunning = true
        defer { clockRunning = false }
        var last = Date()
        while !Task.isCancelled {
            do { try await Task.sleep(for: .milliseconds(33)) } catch { break }
            let now = Date(); tick(now.timeIntervalSince(last)); last = now
        }
    }
    func prepare() {
        if let trace {
            let reference = ReferenceNormalization(trace: trace, mode: normalization)
            normalizedFrames = trace.snapshots.map { $0.activations.map { reference.apply($0) } }
            normalized = normalizedFrames[frameIndex]
            fitProjection()
        } else {
            normalizedFrames = []
            normalized = dataset.normalized(normalization)
            // A fixed rotation-invariant bound avoids pulsing during the CSV tour.
            radius = max(1e-12, normalized.map { sqrt($0.reduce(0) { $0 + $1*$1 }) }.max() ?? 1)
        }
    }
    func startAnimation() {
        comparisonLocked = false
        // Keep the existing shared display reference. Immersive space is
        // unbounded: projection excursions remain visible, without per-frame
        // refitting, clipping, or independently scaling different layers.
        playing = true
    }
    func fitProjection() {
        guard trace != nil else { return }
        playing = false
        // One display scale fitted across ALL frames at the current shared basis.
        // Frame selection never changes this scale. Tour excursions outside it are
        // visibly counted and hidden, rather than clamped into misleading geometry.
        var bound = 0.0
        for frame in normalizedFrames { for row in frame {
            let p = projection.project(row)
            bound = max(bound, sqrt(p.reduce(0) { $0 + $1*$1 }))
        } }
        radius = max(1e-12, bound)
    }
    func load(_ value: Dataset) {
        trace = nil; normalizedFrames = []; savedViewpoints = []
        playing = false; dataset = value; projection = Projection(dimensions: value.names.count)
        planeI = 0; planeJ = value.names.count - 1; manualAngle = 0; time = 0
        prepare(); plotReset += 1
    }
    func loadTrace(_ value: ActivationTrace) throws {
        try value.validate()
        playing = false; savedViewpoints = []; trace = value; frameIndex = 0
        selectedTokenIndex = 0; comparisonLocked = true
        dataset = value.dataset(frame: 0)
        projection = Projection(dimensions: value.dimensions, dense: true)
        planeI = 0; planeJ = value.dimensions - 1; manualAngle = 0; time = 0
        if normalization != .centered { normalization = .centered } else { prepare() }
        plotReset += 1
    }
    func selectFrame(_ index: Int) {
        guard let trace, trace.snapshots.indices.contains(index) else { return }
        playing = false
        frameIndex = index; dataset = trace.dataset(frame: index)
        normalized = normalizedFrames[index]
        // Projection, statistics, display radius, token selection and physical pose
        // deliberately remain unchanged for a matched comparison.
    }
    func selectLayer(_ layer: String) {
        guard let trace else { return }
        let checkpoint = currentFrame?.checkpoint
        if let index = trace.snapshots.firstIndex(where: { $0.layer == layer && $0.checkpoint == checkpoint })
            ?? trace.snapshots.firstIndex(where: { $0.layer == layer }) { selectFrame(index) }
    }
    func selectCheckpoint(_ checkpoint: String) {
        guard let trace else { return }
        let layer = currentFrame?.layer
        if let index = trace.snapshots.firstIndex(where: { $0.checkpoint == checkpoint && $0.layer == layer })
            ?? trace.snapshots.firstIndex(where: { $0.checkpoint == checkpoint }) { selectFrame(index) }
    }
    func saveViewpoint() {
        guard savedViewpoints.count < 8 else { return }
        savedViewpoints.append(SavedViewpoint(name: "Viewpoint \(savedViewpoints.count + 1)",
            projection: projection, normalization: normalization, time: time,
            planeI: planeI, planeJ: planeJ, manualAngle: manualAngle, radius: radius))
    }
    func restoreViewpoint(_ id: UUID) {
        guard let view = savedViewpoints.first(where: { $0.id == id }) else { return }
        playing = false; normalization = view.normalization; projection = view.projection
        time = view.time; planeI = view.planeI; planeJ = view.planeJ
        manualAngle = view.manualAngle; radius = view.radius; plotReset += 1
    }
    func clearViewpoints() { savedViewpoints = [] }
    func reset() {
        projection = Projection(dimensions: dataset.names.count, dense: trace != nil)
        manualAngle = 0; time = 0; playing = false; plotReset += 1
        if trace != nil { fitProjection() }
    }
    func turnManual(to value: Double) {
        guard trace == nil || !comparisonLocked else { return }
        playing = false
        projection.rotate(planeI, planeJ, radians: value - manualAngle)
        manualAngle = value
    }
    func tick(_ elapsed: Double) {
        guard playing, trace == nil || !comparisonLocked else { return }
        let dt = min(max(elapsed, 0), 0.1) * speed
        projection.advance(time: time, delta: dt); time += dt
    }
}
