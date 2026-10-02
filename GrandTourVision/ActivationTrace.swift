import Foundation

struct TraceToken: Codable, Identifiable, Equatable {
    let id: String
    let sequenceID: String
    let tokenID: Int
    let text: String
    let position: Int
    var category: String? = nil
}
struct TraceSnapshot: Codable, Identifiable {
    let id: String
    let layer: String
    let checkpoint: String
    let trainingStep: Int?
    let dtype: String
    let activations: [[Double]]
}
struct TraceSource: Codable {
    let title: String
    let url: String
    let license: String
}
struct ActivationTrace: Codable {
    let schemaVersion: Int
    let model: String
    let tokenizer: String
    let prompt: String
    let mode: String
    let tokens: [TraceToken]
    let snapshots: [TraceSnapshot]
    var source: TraceSource? = nil
    var dimensions: Int { snapshots[0].activations[0].count }
    static let maxBytes = 32_000_000
    static let maxValues = 1_048_576
    static let maxDimensions = 4096
    static let maxTokens = 128
    static let maxSnapshots = 16

    static func parse(_ data: Data) throws -> ActivationTrace {
        guard data.count <= maxBytes else { throw CSVError.invalid("Trace exceeds 32 MB.") }
        let trace = try JSONDecoder().decode(ActivationTrace.self, from: data)
        try trace.validate()
        return trace
    }
    func validate() throws {
        func require(_ value: Bool, _ message: String) throws {
            guard value else { throw CSVError.invalid(message) }
        }
        try require(schemaVersion == 1, "Trace schemaVersion must be 1.")
        try require(["inference", "training", "synthetic"].contains(mode), "Invalid trace mode.")
        try require(!model.isEmpty && !tokenizer.isEmpty, "Trace requires model and tokenizer metadata.")
        try require((2...Self.maxTokens).contains(tokens.count), "Trace needs 2–128 sampled token points.")
        try require((1...Self.maxSnapshots).contains(snapshots.count), "Trace needs 1–16 snapshots.")
        try require(Set(tokens.map(\.id)).count == tokens.count && tokens.allSatisfy { !$0.id.isEmpty && !$0.sequenceID.isEmpty && $0.position >= 0 }, "Token IDs must be unique; sequence IDs and nonnegative positions are required.")
        try require(Set(tokens.map { "\($0.sequenceID):\($0.position)" }).count == tokens.count, "Repeated sequence/position identity.")
        try require(Set(snapshots.map(\.id)).count == snapshots.count, "Snapshot IDs must be unique.")
        try require(snapshots[0].activations.count == tokens.count && snapshots[0].activations.first != nil, "Snapshot row count must match token identities.")
        let d = dimensions
        try require((3...Self.maxDimensions).contains(d), "Trace feature dimension must be 3–4096.")
        var total = 0
        for frame in snapshots {
            try require(!frame.id.isEmpty && !frame.layer.isEmpty && !frame.checkpoint.isEmpty && !frame.dtype.isEmpty && (frame.trainingStep ?? 0) >= 0, "Snapshot metadata is incomplete or invalid.")
            try require(frame.activations.count == tokens.count, "Every snapshot must contain the same ordered token points.")
            for vector in frame.activations {
                try require(vector.count == d, "All layers/checkpoints must share the same feature dimension.")
                total += vector.count
                try require(total <= Self.maxValues, "Trace exceeds 1,048,576 total scalar activations.")
                try require(vector.allSatisfy { $0.isFinite && abs($0) <= 1e100 }, "Activations must be finite and bounded.")
            }
        }
    }
    var colorLabels: [String] {
        let labels = Set(tokens.compactMap(\.category)).sorted()
        return labels.count <= 4 && tokens.allSatisfy({ $0.category != nil }) ? labels : ["Sampled tokens"]
    }
    func dataset(frame: Int) -> Dataset {
        let snapshot = snapshots[frame]
        return Dataset(names: (0..<dimensions).map { "h\($0)" }, rows: snapshot.activations,
                       groups: tokens.map { colorLabels.firstIndex(of: $0.category ?? "") ?? 0 },
                       label: "\(model) • \(snapshot.layer) • \(snapshot.checkpoint) • \(tokens.count) tokens × \(dimensions)D")
    }
    static func synthetic() -> ActivationTrace {
        let tokens = (0..<128).map { i in TraceToken(id: "prompt0:\(i)", sequenceID: "prompt0", tokenID: i + 100,
            text: ["The", "model", "learns", "patterns"][i % 4], position: i,
            category: ["Designed A", "Designed B", "Designed C", "Designed D"][i / 32]) }
        let scaffold = Projection(dimensions: 32, dense: true)
        let centers: [[Double]] = [[-2.5,-1.6,-0.7], [2.4,-1.5,0.6], [-2.1,1.7,0.9], [2.0,1.7,-0.7]]
        var frames: [TraceSnapshot] = []
        for step in [0, 100] {
            for layer in ["block.0", "block.1"] {
                var vectors: [[Double]] = []
                for i in 0..<128 {
                    let group = i / 32
                    let phase = Double(i % 32) * 2.3999632297
                    let spread = 0.18 + Double((i % 7) + 1) * 0.028
                    var center = centers[group]
                    if layer == "block.1" {
                        center[0] *= 0.82; center[1] *= 1.1
                        center[2] += Double(group % 2 == 0 ? 1 : -1) * 0.6
                    }
                    if step == 100 {
                        center[0] += Double(group < 2 ? -1 : 1) * 0.4
                        center[1] += Double(group % 2 == 0 ? 1 : -1) * 0.25
                    }
                    let local = [cos(phase) * spread, sin(phase) * spread * 0.72, sin(phase * 1.37) * spread * 0.8]
                    var row: [Double] = []
                    for d in 0..<32 {
                        let signal = (0..<3).reduce(0.0) { $0 + (center[$1] + local[$1]) * scaffold.basis[$1][d] }
                        // Small hidden-feature noise; the fixture is deliberately
                        // engineered for visible clusters, never real model output.
                        row.append(signal + sin(Double(i + 1) * Double(d + 1) * 0.17) * 0.035)
                    }
                    vectors.append(row)
                }
                frames.append(TraceSnapshot(id: "\(layer)@\(step)", layer: layer, checkpoint: "step-\(step)",
                                            trainingStep: step, dtype: "float32", activations: vectors))
            }
        }
        return ActivationTrace(schemaVersion: 1, model: "Synthetic activations (no LLM run)", tokenizer: "synthetic-v1",
                               prompt: "Engineered synthetic clusters; repeated placeholder tokens, no language model capture", mode: "synthetic", tokens: tokens, snapshots: frames)
    }
}

// One reference transform fitted to ALL imported snapshots. Frame switching never
// refits it. These statistics must remain fixed for meaningful coordinate comparisons.
struct ReferenceNormalization {
    let means: [Double]
    let scales: [Double]
    init(trace: ActivationTrace, mode: Normalization) {
        let d = trace.dimensions
        let n = Double(trace.tokens.count * trace.snapshots.count)
        var means = Array(repeating: 0.0, count: d)
        for frame in trace.snapshots { for row in frame.activations { for c in 0..<d { means[c] += row[c] / n } } }
        var variance = Array(repeating: 0.0, count: d)
        if mode == .standardized {
            for frame in trace.snapshots { for row in frame.activations {
                for c in 0..<d { variance[c] += pow(row[c] - means[c], 2) / n }
            } }
        }
        self.means = means
        scales = variance.map { mode == .standardized && $0 > 1e-24 ? sqrt($0) : 1 }
    }
    func apply(_ row: [Double]) -> [Double] {
        row.indices.map { (row[$0] - means[$0]) / scales[$0] }
    }
}
