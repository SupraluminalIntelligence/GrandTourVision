import Foundation
import simd

// Physical placement in metres. No projection/normalization state belongs here.
struct ScenePlacement: Equatable {
    var center = SIMD3<Float>(0, 1.5, -3.5)
    var diameter: Float = 4
    var orientation = simd_quatf(angle: 0, axis: [0, 1, 0])
    static let minimumDiameter: Float = 1.5
    static let maximumDiameter: Float = 6
    mutating func move(to value: SIMD3<Float>) {
        guard value.x.isFinite, value.y.isFinite, value.z.isFinite else { return }
        center = SIMD3(max(-5, min(5, value.x)), max(0.5, min(4, value.y)), max(-8, min(-0.5, value.z)))
    }
    mutating func resize(to value: Float) {
        guard value.isFinite else { return }
        diameter = max(Self.minimumDiameter, min(Self.maximumDiameter, value))
    }
    mutating func rotate(to value: simd_quatf) {
        guard value.vector.x.isFinite, value.vector.y.isFinite, value.vector.z.isFinite,
              value.vector.w.isFinite, simd_length(value.vector) > 1e-6 else { return }
        orientation = simd_normalize(value)
    }
    mutating func reset() { self = ScenePlacement() }
    static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.center == rhs.center && lhs.diameter == rhs.diameter && lhs.orientation.vector == rhs.orientation.vector
    }
}
