import Foundation
import simd

// FlowScope tour frames (see flowscope/tour.py): every block's token cloud, expressed in one shared
// principal basis of the residual stream. The tour math mirrors flowscope/tour.js exactly so the
// headset and the browser show the same views at the same tour time.

struct FlowFrame: Codable, Equatable {
    struct Layer: Codable, Equatable {
        let name: String
        let dim: Double     // participation ratio of the block's covariance (effective dimensionality)
        let scale: Double   // int8 -> coordinate
        let data: String    // base64 int8, N x C row-major, coordinates along the shared principal directions
    }
    let step: Int
    let C: Int
    let N: Int
    let T: Int
    let k: Int
    let layers: [Layer]
    let tokens: [Int]?
    var label: String?
    var vocab: String?  // token id -> character, when the run is character-level (lets the headset show text)
    // From checkpoint comparisons (flowscope.compare): one frame per checkpoint, in one shared basis.
    var name: String?              // e.g. "base", "multi-harness RL"; shown instead of the step
    var texts: [String]?           // each point's decoded token
    var groups: [Int]?             // each point's group, e.g. which harness the token came from
    var groupNames: [String]?
    var similarity: [Double]?      // per layer: linear CKA against the first checkpoint
    var change: [Double]?          // per layer: median relative change of each token's vector vs the first checkpoint
    var retained: Double?          // share of the full hidden size's spread kept by the C directions

    static let maxValues = 4_194_304

    func validate() throws {
        func require(_ ok: Bool, _ message: String) throws { if !ok { throw CSVError.invalid(message) } }
        try require(C >= 3 && N >= 2 && T >= 1 && !layers.isEmpty, "FlowScope frame needs ≥3 dims, ≥2 tokens and ≥1 layer.")
        try require(N * C * layers.count <= Self.maxValues, "FlowScope frame is too large.")
        try require(tokens == nil || tokens!.count == N, "FlowScope token ids don't match the point count.")
        try require(texts == nil || texts!.count == N, "FlowScope token texts don't match the point count.")
        try require(groups == nil || (groups!.count == N && groups!.allSatisfy { $0 >= 0 && $0 < (groupNames?.count ?? 0) }),
                    "FlowScope groups don't match the point count or the group names.")
        try require(similarity == nil || similarity!.count == layers.count, "FlowScope similarity needs one value per layer.")
        try require(change == nil || change!.count == layers.count, "FlowScope change needs one value per layer.")
    }
}

struct FlowCloud {
    let name: String
    let dim: Double
    let z: [Float]          // z[p * C + d]
    let variance: [Double]  // per principal direction; the data is centered
    let total: Double
    let N: Int
    let C: Int

    init(_ layer: FlowFrame.Layer, N: Int, C: Int) throws {
        guard let bytes = Data(base64Encoded: layer.data), bytes.count == N * C else {
            throw CSVError.invalid("FlowScope layer \(layer.name) has the wrong size.")
        }
        var z = [Float](repeating: 0, count: N * C)
        var v = [Double](repeating: 0, count: C)
        bytes.withUnsafeBytes { raw in
            for i in 0..<(N * C) {
                let value = Float(Double(Int8(bitPattern: raw[i])) * layer.scale)  // as tour.js: double product, one rounding
                z[i] = value
                v[i % C] += Double(value) * Double(value) / Double(N)
            }
        }
        name = layer.name; dim = layer.dim; self.z = z; variance = v; self.N = N; self.C = C
        total = max(v.reduce(0, +), 1e-12)
    }

    /// Variance inside the first n principal directions (or an explicit set of them).
    func inView(_ n: Int) -> Double { inView(dims: Array(0..<n)) }
    func inView(dims: [Int]) -> Double { dims.reduce(0) { $0 + variance[$1] } }

    func percentSeen(_ n: Int) -> Int { percentSeen(dims: Array(0..<n)) }
    func percentSeen(dims: [Int]) -> Int { Int((100 * inView(dims: dims) / total).rounded()) }

    /// Same zoom rule as tour.js: a typical 3D view of an n-dim cloud carries ~3/n of its variance, capped by
    /// the 95th-percentile radius so a collapsed cloud still fits when the window lines up with it.
    /// Returns the factor that maps tour coordinates to "1 = cloud radius".
    func fit(_ n: Int) -> Double { fit(dims: Array(0..<n)) }
    func fit(dims: [Int]) -> Double {
        var r = [Double](repeating: 0, count: N)
        for p in 0..<N {
            var s = 0.0
            for d in dims { let x = Double(z[p * C + d]); s += x * x }
            r[p] = s
        }
        r.sort()
        let r95 = sqrt(r[Int((0.95 * Double(N - 1)).rounded(.down))])
        return 1 / max(2.6 * sqrt(inView(dims: dims) / Double(dims.count)), r95 / 1.1, 1e-9)
    }

    /// Projected positions, in cloud-radius units (most points within ~1).
    func project(_ basis: [[Double]], n: Int, fit: Double, into out: inout [SIMD3<Float>]) {
        project(basis, dims: Array(0..<n), fit: fit, into: &out)
    }

    /// basis[a][k] is the weight of principal direction dims[k] on display axis a.
    func project(_ basis: [[Double]], dims: [Int], fit: Double, into out: inout [SIMD3<Float>]) {
        if out.count != N { out = [SIMD3<Float>](repeating: .zero, count: N) }
        let b0 = basis[0].map(Float.init), b1 = basis[1].map(Float.init), b2 = basis[2].map(Float.init)
        let f = Float(fit) * 0.44 / 0.5  // tour.js draws R = 0.44 * panel at fit; panel half-width = 0.5 panel
        z.withUnsafeBufferPointer { z in
            for p in 0..<N {
                var x: Float = 0, y: Float = 0, w: Float = 0
                let o = p * C
                for (k, d) in dims.enumerated() {
                    let v = z[o + d]
                    x += b0[k] * v; y += b1[k] * v; w += b2[k] * v
                }
                out[p] = SIMD3(x * f, y * f, w * f)
            }
        }
    }
}

// Asimov's torus method: rotate in every coordinate plane (i, j) of the toured subspace at once, at speeds
// whose ratios are irrational. The path never repeats and comes arbitrarily close to every 3D projection.
enum TorusTour {
    typealias Plane = (i: Int, j: Int, speed: Double)

    static func speeds(_ n: Int) -> [Plane] {
        var out: [Plane] = [], norm = 0.0, m = 2.0
        for i in 0..<n {
            for j in (i + 1)..<n {
                while sqrt(m).rounded() == sqrt(m) { m += 1 }
                let v = 0.25 + 0.75 * sqrt(m).truncatingRemainder(dividingBy: 1)  // fractional part: irrational
                out.append((i, j, v)); norm += v * v; m += 1
            }
        }
        let k = 0.5 * sqrt(3) / sqrt(norm)  // keep the visible motion calm whatever n is
        return out.map { ($0.i, $0.j, $0.speed * k * sqrt(Double(n))) }
    }

    /// Three orthonormal vectors in R^n; at t = 0 they are the first three principal directions.
    static func basis(_ n: Int, t: Double, planes: [Plane]) -> [[Double]] {
        var v = (0..<3).map { a in (0..<n).map { $0 == a ? 1.0 : 0.0 } }
        for plane in planes {
            let th = plane.speed * t, c = cos(th), s = sin(th)
            for a in 0..<3 {
                let x = v[a][plane.i], y = v[a][plane.j]
                v[a][plane.i] = c * x - s * y
                v[a][plane.j] = s * x + c * y
            }
        }
        return v
    }
}

enum FlowRecording {
    /// A FlowScope `--record` file (one frame per line), or a single frame.
    static func parse(_ data: Data) throws -> [FlowFrame] {
        guard data.count <= 64_000_000 else { throw CSVError.invalid("Recording exceeds 64 MB.") }
        let decoder = JSONDecoder()
        if let single = try? decoder.decode(FlowFrame.self, from: data) { try single.validate(); return [single] }
        let lines = String(decoding: data, as: UTF8.self).split(whereSeparator: \.isNewline)
        let frames = try lines.map { try decoder.decode(FlowFrame.self, from: Data($0.utf8)) }
        guard !frames.isEmpty else { throw CSVError.invalid("Recording has no frames.") }
        try frames.forEach { try $0.validate() }
        return frames
    }
}

enum FlowEvents {
    private struct Message: Decodable { let tour: FlowFrame? }

    /// One Server-Sent Events line from FlowScope's /events stream -> its tour frame, if any.
    static func frame(fromLine line: String) -> FlowFrame? {
        guard line.hasPrefix("data: ") else { return nil }
        let message = try? JSONDecoder().decode(Message.self, from: Data(line.dropFirst(6).utf8))
        guard let frame = message?.tour, (try? frame.validate()) != nil else { return nil }
        return frame
    }
}

enum FlowColors {
    static let viridis: [SIMD3<Float>] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]].map { $0 / 255 }

    static func viridis(_ t: Float) -> SIMD3<Float> {
        let x = min(1, max(0, t)) * Float(viridis.count - 1)
        let i = min(viridis.count - 2, Int(x)), f = x - Float(i)
        return viridis[i] + (viridis[i + 1] - viridis[i]) * f
    }
}

/// Which principal directions a view uses: the chosen X/Y/Z first, then the rest of the toured set. At tour
/// time 0 the window looks exactly along the chosen axes; starting the tour rotates through all of them.
func viewDims(axes: [Int], toured: Int, C: Int) -> [Int] {
    let axes = axes.filter { (0..<C).contains($0) }
    return axes + (0..<max(toured, 3)).filter { $0 < C && !axes.contains($0) }
}
