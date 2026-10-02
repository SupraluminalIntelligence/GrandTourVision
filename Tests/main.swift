import simd
import Foundation
func check(_ condition: @autoclosure () -> Bool, _ name: String) {
    guard condition() else { fatalError("FAILED: \(name)") }
    print("PASS: \(name)")
}
func rejects(_ text: String) -> Bool { do { _ = try CSVImport.parse(text); return false } catch { return true } }
for d in [3,4,8,12] {
    var p = Projection(dimensions: d)
    for t in 0..<10000 { p.advance(time: Double(t)/30, delta: 1/30) }
    for a in 0..<3 { for b in 0..<3 {
        let dot = zip(p.basis[a], p.basis[b]).reduce(0) { $0 + $1.0*$1.1 }
        check(abs(dot - (a == b ? 1 : 0)) < 1e-10, "orthonormal \(d)D \(a),\(b)")
    } }
}
var p = Projection(dimensions: 4)
p.rotate(0,3,radians: .pi/2)
check(abs(p.project([0,0,0,1])[0]-1)<1e-12, "fourth dimension becomes visible")
p.rotate(0,3,radians: -.pi/2)
check(abs(p.basis[0][0]-1)<1e-12, "manual inverse rotation")
let csv = "\u{FEFF}a,\"b, label\",c,d,name\r\n1,2,3,4,\"a\"\"b\"\r\n2,3,4,5,\"multi\nline\""
let dataset = try CSVImport.parse(csv)
check(dataset.names == ["a","b, label","c","d"] && dataset.rows.count == 2, "BOM, quoted comma, escaped quote, CRLF, newline, text exclusion")
let norm = dataset.normalized(.standardized)
check(norm[0].allSatisfy { abs($0+1)<1e-12 } && norm[1].allSatisfy { abs($0-1)<1e-12 }, "z-score mean and variance")
let constant = try CSVImport.parse("a,b,c\n1,2,3\n1,2,3")
check(constant.normalized(.standardized).flatMap { $0 }.allSatisfy { $0 == 0 }, "constant columns")
check(rejects("a,b,c\n1,2,3\n4,5"), "ragged rows rejected")
check(rejects("a,b,c\n1,2,nan\n4,5,6"), "nonfinite column excluded, insufficient dimensions rejected")
check(rejects("a,b,c\n1,2,\"3\n4,5,6"), "unclosed quote rejected")
check(rejects("a,b,c\n1,2,\"3\"oops\n4,5,6"), "trailing quoted garbage rejected")
check(rejects("a,b,c\n" + Array(repeating: "1,2,3", count: 1001).joined(separator: "\n")), "row bound")
check(rejects(String(repeating: "x", count: CSVImport.maxBytes+1)), "byte bound")
check(rejects((0..<13).map { "x\($0)" }.joined(separator: ",")+"\n"+Array(repeating: Array(repeating: "1", count: 13).joined(separator: ","), count: 2).joined(separator: "\n")), "dimension bound")
let synth = Dataset.synthetic()
check(synth.rows.count == 300 && Set(synth.groups).count == 4, "synthetic clusters")
let centered = synth.normalized(.centered)
for c in 0..<4 { check(abs(centered.reduce(0) { $0+$1[c] })<1e-10, "centered column \(c)") }
MainActor.assumeIsolated {
    let model = TourModel()
    model.playing = true; model.tick(0.05)
    model.planeI = 1; model.planeJ = 3
    model.turnManual(to: 0.7)
    model.normalization = .centered
    let expected = model.projection
    let savedTime = model.time
    model.saveViewpoint()
    let id = model.savedViewpoints[0].id
    model.reset(); model.normalization = .standardized; model.playing = true
    model.restoreViewpoint(id)
    check(model.projection == expected, "viewpoint restores exact orthonormal basis")
    check(!model.playing && model.normalization == .centered && model.time == savedTime,
          "viewpoint pauses and restores normalization and tour time")
    check(model.planeI == 1 && model.planeJ == 3 && model.manualAngle == 0.7,
          "viewpoint restores manual plane controls")
    model.turnManual(to: 0.9)
    check(model.savedViewpoints[0].projection == expected, "viewpoint is immutable snapshot")
    for _ in 0..<10 { model.saveViewpoint() }
    check(model.savedViewpoints.count == 8, "viewpoint capacity bound")
    model.load(.synthetic())
    check(model.savedViewpoints.isEmpty, "dataset replacement clears viewpoints")
    let afterLoad = model.projection
    model.restoreViewpoint(id)
    check(model.projection == afterLoad, "stale viewpoint restore safely ignored")
    model.saveViewpoint(); model.clearViewpoints()
    check(model.savedViewpoints.isEmpty, "clear saved viewpoints")
}
let traceFixture = try ActivationTrace.parse(Data(contentsOf: URL(fileURLWithPath: "Samples/synthetic-trace.json")))
check(traceFixture.dimensions == 32 && traceFixture.tokens.count == 128 && traceFixture.snapshots.count == 4,
      "portable synthetic trace imports layers and checkpoints")
let reference = ReferenceNormalization(trace: traceFixture, mode: .standardized)
for c in [0, 15, 31] {
    let values = traceFixture.snapshots.flatMap { $0.activations.map { reference.apply($0)[c] } }
    check(abs(values.reduce(0,+)) < 1e-10, "trace shared reference centered across all frames \(c)")
}
var dense = Projection(dimensions: 4096, dense: true)
for tick in 0..<120 { dense.advance(time: Double(tick)/30, delta: 1/30) }
for a in 0..<3 { for b in 0..<3 {
    let dot = zip(dense.basis[a], dense.basis[b]).reduce(0) { $0 + $1.0 * $1.1 }
    check(abs(dot - (a == b ? 1 : 0)) < 1e-10, "4096D dense sparse-tour orthonormal \(a),\(b)")
} }
check(dense.basis.allSatisfy { abs($0[4095]) > 1e-8 }, "dense basis includes last hidden feature")
func traceRejects(_ mutate: (inout [String: Any]) -> Void) -> Bool {
    do {
        var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(traceFixture)) as! [String: Any]
        mutate(&object)
        _ = try ActivationTrace.parse(JSONSerialization.data(withJSONObject: object))
        return false
    } catch { return true }
}
check(traceRejects { $0["schemaVersion"] = 2 }, "trace rejects unknown schema")
check(traceRejects { object in
    var tokens = object["tokens"] as! [[String: Any]]; tokens[1] = tokens[0]; object["tokens"] = tokens
}, "trace rejects duplicate point identities")
check(traceRejects { object in
    var frames = object["snapshots"] as! [[String: Any]]
    var rows = frames[1]["activations"] as! [[Double]]; rows.removeLast()
    frames[1]["activations"] = rows; object["snapshots"] = frames
}, "trace rejects mismatched point rows")
check(traceRejects { object in
    var frames = object["snapshots"] as! [[String: Any]]
    var rows = frames[1]["activations"] as! [[Double]]; rows[0].append(1)
    frames[1]["activations"] = rows; object["snapshots"] = frames
}, "trace rejects unequal feature dimensions")
check(traceRejects { object in
    var frames = object["snapshots"] as! [[String: Any]]; frames[0]["trainingStep"] = -1; object["snapshots"] = frames
}, "trace rejects invalid training step")
check(traceRejects { object in
    var frames = object["snapshots"] as! [[String: Any]]; frames[0]["dtype"] = ""; object["snapshots"] = frames
}, "trace requires dtype metadata")
MainActor.assumeIsolated {
    let model = TourModel()
    try! model.loadTrace(traceFixture)
    let basis = model.projection
    let scale = model.radius
    let tokens = model.trace!.tokens
    model.selectedTokenIndex = 7
    model.playing = true; model.tick(0.1)
    check(model.projection == basis, "locked comparison prevents tour drift")
    model.turnManual(to: 0.9)
    check(model.projection == basis, "locked comparison prevents manual drift")
    model.selectLayer("block.1")
    check(model.currentFrame?.layer == "block.1" && model.currentFrame?.checkpoint == "step-0",
          "layer selection preserves checkpoint when available")
    model.selectCheckpoint("step-100")
    check(model.currentFrame?.layer == "block.1" && model.currentFrame?.trainingStep == 100,
          "checkpoint selection preserves layer")
    check(model.projection == basis && model.radius == scale && !model.playing,
          "frame switch locks shared projection, scale and pauses tour")
    check(model.trace!.tokens == tokens && model.selectedToken?.id == tokens[7].id,
          "point identity and selection preserved across comparisons")
    let expected = ReferenceNormalization(trace: traceFixture, mode: model.normalization).apply(traceFixture.snapshots[3].activations[7])
    check(model.normalized[7] == expected, "frame uses frozen global normalization reference")
    model.saveViewpoint()
    let bookmark = model.savedViewpoints[0].id
    model.comparisonLocked = false; model.turnManual(to: 0.8); model.fitProjection()
    model.restoreViewpoint(bookmark)
    check(model.projection == basis && model.radius == scale, "trace bookmark restores comparison basis and display scale")
    model.selectFrame(0)
    let frame0 = model.projected
    model.selectFrame(3); model.selectFrame(0)
    check(model.projected == frame0, "round-trip frame selection creates no artificial motion")
    model.load(.synthetic())
    check(model.trace == nil && model.normalizedFrames.isEmpty, "CSV/demo replacement removes trace reference")
}
let wideTokens = (0..<128).map { TraceToken(id: "p:\($0)", sequenceID: "p", tokenID: $0, text: "t", position: $0) }
let wideRows = Array(repeating: Array(repeating: 0.0, count: 4096), count: 128)
let oversized = ActivationTrace(schemaVersion: 1, model: "synthetic-bound-test", tokenizer: "synthetic", prompt: "none", mode: "synthetic",
    tokens: wideTokens, snapshots: (0..<3).map { (index: Int) in TraceSnapshot(id: "s\(index)", layer: "l", checkpoint: "c\(index)", trainingStep: index, dtype: "float32", activations: wideRows) })
var rejected = false
 do { try oversized.validate() } catch { rejected = true }
check(rejected, "trace total-scalar bound enforced across frames")
let acceptedWide = ActivationTrace(schemaVersion: 1, model: "synthetic-bound-test", tokenizer: "synthetic", prompt: "none", mode: "synthetic",
    tokens: wideTokens, snapshots: Array(oversized.snapshots.prefix(2)))
try acceptedWide.validate()
check(acceptedWide.dimensions == 4096, "maximum trace dimension and total budget validate")
let equalPointTrace = ActivationTrace(schemaVersion: 1, model: "synthetic-matching-test", tokenizer: "synthetic", prompt: "none", mode: "synthetic",
    tokens: Array(wideTokens.prefix(2)), snapshots: [
        TraceSnapshot(id: "a", layer: "l", checkpoint: "a", trainingStep: 0, dtype: "float32", activations: [[1,2,3],[2,4,6]]),
        TraceSnapshot(id: "b", layer: "l", checkpoint: "b", trainingStep: 1, dtype: "float32", activations: [[1,2,3],[4,8,12]])])
MainActor.assumeIsolated {
    let model = TourModel()
    try! model.loadTrace(equalPointTrace)
    let point = model.projected[0]
    let radius = model.radius
    model.selectFrame(1)
    check(model.projected[0] == point && model.radius == radius,
          "unchanged token activation stays fixed while other token changes")
    model.comparisonLocked = false; model.normalization = .standardized
    model.selectFrame(0); let standardizedPoint = model.projected[0]
    model.selectFrame(1)
    check(model.projected[0] == standardizedPoint, "shared z-score also prevents artificial token motion")
}
MainActor.assumeIsolated {
    let model = TourModel()
    try! model.loadTrace(traceFixture)
    check(model.plotLegend == ["Designed A", "Designed B", "Designed C", "Designed D"], "engineered fixture legend is explicit")
    check((0..<4).allSatisfy { g in model.dataset.groups.filter { $0 == g }.count == 32 }, "128 fixture points form four designed groups")
    let rows = model.projected
    let centers = (0..<4).map { g in (0..<3).map { axis in rows[(g*32)..<(g*32+32)].reduce(0) { $0 + $1[axis] } / 32 } }
    let spread = (0..<128).map { i in sqrt((0..<3).reduce(0.0) { $0 + pow(rows[i][$1] - centers[i/32][$1], 2) }) }.max()!
    var separation = Double.infinity
    for a in 0..<4 { for b in (a+1)..<4 {
        separation = min(separation, sqrt((0..<3).reduce(0.0) { $0 + pow(centers[a][$1] - centers[b][$1], 2) }))
    } }
    check(separation > 5 * spread, "engineered clusters separate clearly in default projection")
    let basis = model.projection
    let initial = model.projected
    model.selectFrame(3)
    check(model.projection == basis && model.projected != initial, "designed layer/checkpoint changes preserve shared projection")
}
let publicTrace = try ActivationTrace.parse(Data(contentsOf: URL(fileURLWithPath: "Samples/public-sorting-trace.json")))
check(publicTrace.tokens.count == 18 && publicTrace.dimensions == 48 && publicTrace.snapshots.count == 4, "public trace is 18 genuine input points by 48 features and four snapshots")
check(publicTrace.tokens.allSatisfy { $0.position < 6 }, "public trace excludes diagnostic/padding positions")
check(publicTrace.tokens.prefix(6).map(\.text).joined() == "AACBAB", "public first batch matches pinned input sequence")
check(Set(publicTrace.tokens.map(\.sequenceID)) == ["batch0", "batch1", "batch2"], "public batch identities preserved")
check(publicTrace.source?.license == "MIT" && publicTrace.source?.title == "bbycroft / llm-viz", "public attribution carried in trace")
check(publicTrace.snapshots[3].activations.flatMap { $0 }.contains(where: { $0 < -1 }) && publicTrace.snapshots[3].activations.flatMap { $0 }.contains(where: { $0 > 1 }), "public signed activation values preserved")
MainActor.assumeIsolated {
    let model = TourModel()
    try! model.loadTrace(publicTrace)
    let basis = model.projection
    let radius = model.radius
    model.selectLayer("block2")
    check(model.projection == basis && model.radius == radius && model.trace!.tokens == publicTrace.tokens, "public layer switch preserves coordinates scale and identities")
    check(model.plotLegend == ["Input A", "Input B", "Input C"], "public colors reflect literal input categories not inferred clusters")
}
MainActor.assumeIsolated {
    let model = TourModel()
    try! model.loadTrace(.synthetic())
    model.startAnimation()
    check(model.playing && !model.comparisonLocked, "room animation explicitly unlocks comparison")
    let initialBasis = model.projection
    model.scenePlacement.resize(to: 100)
    model.scenePlacement.move(to: [100, -10, 100])
    model.scenePlacement.rotate(to: simd_quatf(angle: 0.4, axis: [0,1,0]))
    let physical = model.scenePlacement
    check(physical.diameter == 6 && physical.center == SIMD3<Float>(5,0.5,-0.5), "room physical transforms have metre bounds")
    check(model.projection == initialBasis && model.playing, "physical manipulation preserves projection and running animation")
    model.tick(0.25)
    check(model.projection != initialBasis && model.scenePlacement == physical, "animation changes basis without overwriting physical placement")
    let radius = model.radius
    for _ in 0..<300 { model.tick(0.02) }
    check(model.radius == radius, "animation keeps the fixed shared display reference without pulsing")
    let movingBasis = model.projection
    model.scenePlacement.reset()
    check(model.scenePlacement == ScenePlacement() && model.projection == movingBasis && model.playing, "recenter restores initial pose and scale while animation continues")
    model.turnManual(to: 0.2)
    check(!model.playing && model.scenePlacement == ScenePlacement(), "manual projection pauses animation and preserves physical pose")
    let lockedBasis = model.projection
    model.comparisonLocked = true
    model.selectFrame(3)
    check(model.projection == lockedBasis && model.radius == radius && model.scenePlacement == ScenePlacement(), "layer comparison preserves basis scale and physical placement")
    var placement = model.scenePlacement
    placement.resize(to: .nan); placement.move(to: [Float.infinity, 0, 0])
    check(placement == model.scenePlacement, "nonfinite physical transforms are rejected")
}
let roomTrace = try ActivationTrace.parse(Data(contentsOf: URL(fileURLWithPath: "Samples/room-demo-trace.json")))
check(roomTrace.mode == "synthetic" && roomTrace.tokens.count == 128 && roomTrace.dimensions == 64, "room demo is explicitly synthetic 128 by 64")
MainActor.assumeIsolated {
    let model = TourModel(); try! model.loadTrace(roomTrace)
    let radius = model.radius
    model.startAnimation()
    for _ in 0..<100 { model.tick(0.03) }
    check(model.radius == radius, "diffuse room tour retains shared reference across animation")
    let norms = model.projected.map { sqrt($0.reduce(0) { $0 + $1*$1 }) }
    check(norms.max()! > 2 && Set(norms.map { Int($0 * 10) }).count > 15, "diffuse room demo projects to a spread of distinct positions")
}

// FlowScope gallery: the Swift tour must match flowscope/tour.js exactly. Expected values were produced by
// the real tour.js engine in a browser on the same recorded frame (no-residual run, step 1000, t = 3.7).
func close(_ a: Double, _ b: Double, _ tol: Double) -> Bool { abs(a - b) <= tol * max(1, abs(b)) }
let recording = try FlowRecording.parse(Data(contentsOf: URL(fileURLWithPath: "Samples/flowscope/no-residual-12-layers.jsonl")))
check(recording.map(\.step) == [0, 200, 400, 600, 800, 1000] && recording[0].layers.count == 13, "FlowScope recording parses")
let flowFrame = recording[5]
let clouds = try flowFrame.layers.map { try FlowCloud($0, N: flowFrame.N, C: flowFrame.C) }
let jsExpected: [Int: (sp0: [Double], count: Int, b0: [Double], b2last: Double, fit0: Double, fit12: Double, seen: [Int], p0: [Double])] = [
    8: ([0.3989224925285357, 0.5685338263460434], 28, [-0.00006830300591555804, 0.0017277315602132616, -0.0036933729895401593],
        0.206975353294156, 0.282289873581579, 0.2384603163807605, [13, 25], [0.20428200918290318, -0.0809779795470803, -0.21917591533539446]),
    64: ([0.13129472950698795, 0.1871177894045097], 2016, [0.000022322329506740287, -0.0004369193441452472, 0.0025846960008031805],
         0.09520043059062433, 0.11846226692367648, 0.11182594836113638, [100, 100], [-0.20611744032994203, -0.13769991305118245, -0.10637559298685224]),
]
for (n, js) in jsExpected.sorted(by: { $0.key < $1.key }) {
    let planes = TorusTour.speeds(n)
    check(planes.count == js.count && close(planes[0].speed, js.sp0[0], 1e-12) && close(planes[1].speed, js.sp0[1], 1e-12), "torus speeds match tour.js (n=\(n))")
    let basis = TorusTour.basis(n, t: 3.7, planes: planes)
    check(zip(basis[0].prefix(3), js.b0).allSatisfy { close($0, $1, 1e-9) } && close(basis[2][n - 1], js.b2last, 1e-9), "tour basis matches tour.js (n=\(n))")
    for a in 0..<3 { for b in 0..<3 {
        let dot = zip(basis[a], basis[b]).reduce(0) { $0 + $1.0 * $1.1 }
        check(abs(dot - (a == b ? 1 : 0)) < 1e-9, "tour basis orthonormal (n=\(n)) \(a),\(b)")
    } }
    check(close(clouds[0].fit(n), js.fit0, 1e-9) && close(clouds[12].fit(n), js.fit12, 1e-9), "cloud zoom matches tour.js (n=\(n))")
    check([clouds[0].percentSeen(n), clouds[12].percentSeen(n)] == js.seen, "% seen matches tour.js (n=\(n))")
    var points: [SIMD3<Float>] = []
    clouds[12].project(basis, n: n, fit: clouds[12].fit(n), into: &points)
    check(points.count == 256 && zip([points[0].x, points[0].y, points[0].z].map(Double.init), js.p0).allSatisfy { close($0, $1, 1e-5) }, "projected point matches tour.js (n=\(n))")
}
check(clouds[12].dim < 2 && clouds[0].dim > 20, "no-residual run: last block collapsed, embeddings not")
let liveLine = "data: " + String(decoding: try JSONEncoder().encode(["html": "<svg/>"]), as: UTF8.self).dropLast() + ",\"tour\":" + String(decoding: try JSONEncoder().encode(flowFrame), as: UTF8.self) + "}"
check(FlowEvents.frame(fromLine: liveLine)?.step == 1000, "live SSE line yields its tour frame")
check(FlowEvents.frame(fromLine: ": keepalive") == nil && FlowEvents.frame(fromLine: "data: {\"html\":\"x\",\"tour\":null}") == nil, "keepalives and frames without a tour are ignored")
check(abs(FlowColors.viridis(0).x - 68.0 / 255) < 1e-6 && abs(FlowColors.viridis(1).y - 231.0 / 255) < 1e-6, "viridis endpoints")

check(viewDims(axes: [5, 0, 9], toured: 8, C: 64) == [5, 0, 9, 1, 2, 3, 4, 6, 7], "chosen axes lead the toured set")
do {
    var pts: [SIMD3<Float>] = []
    let dims = viewDims(axes: [5, 0, 9], toured: 8, C: 64)
    let still = TorusTour.basis(dims.count, t: 0, planes: TorusTour.speeds(dims.count))
    clouds[3].project(still, dims: dims, fit: 1, into: &pts)
    let z = clouds[3].z
    check(abs(pts[7].x - z[7 * 64 + 5] * 0.88) < 1e-6 && abs(pts[7].y - z[7 * 64] * 0.88) < 1e-6 && abs(pts[7].z - z[7 * 64 + 9] * 0.88) < 1e-6,
          "a static view shows exactly the chosen principal directions on X/Y/Z")
}
print("All checks passed")
