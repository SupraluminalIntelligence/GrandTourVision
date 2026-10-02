import Foundation
import simd

// The 3D window you step into: where it points inside the high-dimensional space (WindowFrame), and how the
// data sits in it (WindowView). Pure math, shared by the headset scene and the core tests.

/// The window's orientation inside the toured subspace: three orthonormal display axes (X, Y, Z) plus one hidden
/// direction that the slice depth slides along. Each vector has one weight per toured principal direction.
struct WindowFrame: Equatable {
    var vectors: [[Double]]
    var n: Int { vectors[0].count }

    static func identity(_ n: Int) -> WindowFrame {
        WindowFrame(vectors: (0..<min(4, n)).map { a in (0..<n).map { $0 == a ? 1.0 : 0.0 } })
    }

    /// The grand tour starting from this frame (Asimov's torus rotations). At t = 0 it is this frame.
    func toured(t: Double, planes: [TorusTour.Plane]) -> WindowFrame {
        var v = vectors
        for plane in planes {
            let th = plane.speed * t, c = cos(th), s = sin(th)
            for a in v.indices {
                let x = v[a][plane.i], y = v[a][plane.j]
                v[a][plane.i] = c * x - s * y
                v[a][plane.j] = s * x + c * y
            }
        }
        return WindowFrame(vectors: v)
    }

    /// Manual tour (Cook & Buja): tilt each display axis toward direction q by the given angles (radians).
    /// Each tilt happens in the plane of that axis and the hidden part of q, so the frame stays orthonormal and
    /// q swings into view. This is what changes the shape you see; pan and zoom never do.
    func tilted(toward q: Int, by angles: SIMD3<Double>) -> WindowFrame {
        var v = vectors
        for a in 0..<3 where abs(angles[a]) > 1e-9 {
            var u = [Double](repeating: 0, count: n); u[q] = 1
            for b in 0..<3 { let c = v[b][q]; for i in 0..<n { u[i] -= c * v[b][i] } }
            let norm = sqrt(u.reduce(0) { $0 + $1 * $1 })
            guard norm > 1e-6 else { continue }  // q is already fully in view
            let c = cos(angles[a]), s = sin(angles[a])
            for i in 0..<n { v[a][i] = c * v[a][i] + s * u[i] / norm }
        }
        if v.count > 3 { v[3] = Self.hiddenDirection(v[3], orthogonalTo: Array(v.prefix(3))) }
        return WindowFrame(vectors: v)
    }

    /// How much of direction q is in view (0...1), and where its spoke points in window space.
    func spoke(_ q: Int) -> SIMD3<Double> { SIMD3(vectors[0][q], vectors[1][q], vectors[2][q]) }
    func visibility(_ q: Int) -> Double { min(1, simd_length(spoke(q))) }

    /// The toured direction that dominates a display axis, and its weight.
    func dominant(axis a: Int) -> (q: Int, weight: Double) {
        let i = vectors[a].indices.max { abs(vectors[a][$0]) < abs(vectors[a][$1]) } ?? 0
        return (i, abs(vectors[a][i]))
    }

    private static func hiddenDirection(_ h: [Double], orthogonalTo axes: [[Double]]) -> [Double] {
        func orthogonalize(_ w: [Double]) -> [Double] {
            var w = w
            for b in axes { let d = zip(w, b).reduce(0) { $0 + $1.0 * $1.1 }; for i in w.indices { w[i] -= d * b[i] } }
            return w
        }
        var w = orthogonalize(h)
        if sqrt(w.reduce(0) { $0 + $1 * $1 }) < 1e-3 {  // fell into view: use the least visible direction instead
            var seen = [Double](repeating: 0, count: h.count)
            for b in axes { for i in b.indices { seen[i] += b[i] * b[i] } }
            let j = seen.indices.min { seen[$0] < seen[$1] } ?? 0
            w = orthogonalize(h.indices.map { $0 == j ? 1 : 0 })
        }
        let norm = sqrt(w.reduce(0) { $0 + $1 * $1 })
        return w.map { $0 / norm }
    }
}

enum WindowShape: String, CaseIterable { case cube = "Cube", sphere = "Sphere" }

/// One block projected through the window: positions in data units, plus how far each token sits from the 3D
/// slice (everything the projection throws away) and its coordinate along the hidden depth direction.
struct WindowSample {
    var base: [SIMD3<Float>] = []
    var residual: [Float] = []
    var hidden: [Float] = []
}

extension FlowCloud {
    func project(_ frame: WindowFrame, dims: [Int], into s: inout WindowSample) {
        if s.base.count != N { s = WindowSample(base: .init(repeating: .zero, count: N), residual: .init(repeating: 0, count: N), hidden: .init(repeating: 0, count: N)) }
        let v = frame.vectors.map { $0.map(Float.init) }, hasHidden = v.count > 3
        z.withUnsafeBufferPointer { z in
            for p in 0..<N {
                var x: Float = 0, y: Float = 0, w: Float = 0, h: Float = 0, n2: Float = 0
                let o = p * C
                for (k, d) in dims.enumerated() {
                    let e = z[o + d]
                    x += v[0][k] * e; y += v[1][k] * e; w += v[2][k] * e; n2 += e * e
                    if hasHidden { h += v[3][k] * e }
                }
                s.base[p] = SIMD3(x, y, w)
                s.hidden[p] = h
                s.residual[p] = sqrt(max(0, n2 - x * x - y * y - w * w - h * h))
            }
        }
    }
}

/// How the data sits in the window: you zoom and pan the data; the window itself never moves.
struct WindowView: Equatable {
    var center = SIMD3<Float>.zero  // data units: the point at the window's middle
    var zoom: Float = 1
    var fit: Float = 1              // data units -> window units at zoom 1 (the window spans -1...1)
    var shape = WindowShape.cube

    func local(_ base: SIMD3<Float>) -> SIMD3<Float> { (base - center) * fit * zoom }

    /// 1 on the boundary; inside < 1.
    func reach(_ p: SIMD3<Float>) -> Float {
        shape == .cube ? max(abs(p.x), abs(p.y), abs(p.z)) : simd_length(p)
    }

    /// A hand movement in window units moves the data with it.
    mutating func pan(byWindowUnits d: SIMD3<Float>) { center -= d / (fit * zoom) }

    /// Frame the view so 95% of tokens fall inside the window at zoom 1.
    static func framing(_ base: [SIMD3<Float>]) -> Float {
        guard !base.isEmpty else { return 1 }
        let r = base.map(simd_length).sorted()
        return 0.95 / max(r[Int((0.95 * Double(r.count - 1)).rounded(.down))], 1e-6)
    }
}

/// Distance of a token from a slice that has been slid `depth` along the hidden direction.
func sliceDistance(residual: Float, hidden: Float, depth: Float) -> Float {
    let off = hidden - depth
    return sqrt(max(0, residual * residual) + off * off)
}
