import Foundation

// Columns are three orthonormal vectors in R^d. Ambient Givens rotations
// change the subspace while preserving all inner products exactly in theory.
struct Projection: Equatable {
    private(set) var basis: [[Double]]
    var dimensions: Int { basis[0].count }
    init(dimensions: Int, dense: Bool = false) {
        precondition(dimensions >= 3)
        basis = (0..<3).map { axis in (0..<dimensions).map { $0 == axis ? 1 : 0 } }
        if dense {
            // Deterministic Gaussian frame followed by modified Gram–Schmidt.
            var seed: UInt64 = 2026
            func uniform() -> Double {
                seed = seed &* 6364136223846793005 &+ 1
                return max(1e-12, Double(seed >> 11) / Double(UInt64.max >> 11))
            }
            for a in 0..<3 {
                var v = (0..<dimensions).map { _ in sqrt(-2 * log(uniform())) * cos(2 * .pi * uniform()) }
                for b in 0..<a {
                    let dot = zip(v, basis[b]).reduce(0) { $0 + $1.0 * $1.1 }
                    for i in 0..<dimensions { v[i] -= dot * basis[b][i] }
                }
                let norm = sqrt(v.reduce(0) { $0 + $1*$1 })
                basis[a] = v.map { $0 / norm }
            }
        }
    }
    mutating func rotate(_ i: Int, _ j: Int, radians: Double) {
        guard i != j, (0..<dimensions).contains(i), (0..<dimensions).contains(j) else { return }
        let c = cos(radians), s = sin(radians)
        for axis in 0..<3 {
            let a = basis[axis][i], b = basis[axis][j]
            basis[axis][i] = c * a - s * b
            basis[axis][j] = s * a + c * b
        }
    }
    func project(_ row: [Double]) -> [Double] {
        basis.map { zip($0, row).reduce(0) { $0 + $1.0 * $1.1 } }
    }
    mutating func advance(time: Double, delta: Double) {
        // A connected sparse set of ambient planes: O(D), rather than O(D²).
        // Smooth velocities preserve the basis without per-frame PCA/refitting.
        for i in 0..<(dimensions - 1) {
            rotate(i, i + 1, radians: delta * sin(time * 0.17 + Double(i + 1) * 1.618) * 0.25)
        }
        if dimensions > 3 {
            for i in 0..<(dimensions / 2) {
                rotate(i, i + dimensions / 2, radians: delta * cos(time * 0.13 + Double(i + 1) * 0.731) * 0.2)
            }
        }
        // Exploratory path only: no assertion of dense or finite complete coverage.
    }
}

enum Normalization: String, CaseIterable { case centered = "Center only", standardized = "Z-score" }
struct Dataset {
    var names: [String]
    var rows: [[Double]]
    var groups: [Int]
    var label: String
    func normalized(_ mode: Normalization) -> [[Double]] {
        let n = Double(rows.count)
        let means = names.indices.map { c in rows.reduce(0) { $0 + $1[c] } / n }
        let scales = names.indices.map { c -> Double in
            guard mode == .standardized else { return 1 }
            let v = rows.reduce(0) { $0 + pow($1[c] - means[c], 2) } / n
            return v > 1e-24 ? sqrt(v) : 1
        }
        return rows.map { row in names.indices.map { (row[$0] - means[$0]) / scales[$0] } }
    }
    static func synthetic() -> Dataset {
        var seed: UInt64 = 42
        func uniform() -> Double {
            seed = seed &* 6364136223846793005 &+ 1
            return Double(seed >> 11) / Double(UInt64.max >> 11)
        }
        let centers: [[Double]] = [[-2,-2,-2,2], [2,2,-2,-2], [-2,2,2,-2], [2,-2,2,2]]
        var rows: [[Double]] = [], groups: [Int] = []
        for g in 0..<4 {
            for _ in 0..<75 {
                rows.append(centers[g].map { $0 + (uniform()+uniform()+uniform()-1.5) * 0.6 })
                groups.append(g)
            }
        }
        return Dataset(names: ["A", "B", "C", "D"], rows: rows, groups: groups, label: "Synthetic 4D • four clusters")
    }
}

enum CSVError: LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

enum CSVImport {
    static let maxBytes = 2_000_000
    static let maxRows = 1000
    static let maxDimensions = 12
    static func parse(_ text: String) throws -> Dataset {
        guard text.utf8.count <= maxBytes else { throw CSVError.invalid("CSV exceeds 2 MB.") }
        var table: [[String]] = [], row: [String] = [], field = "", quoted = false, closed = false
        let chars = Array(text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")); var i = 0
        func endField() { row.append(field.trimmingCharacters(in: .whitespaces)); field = ""; closed = false }
        func endRow() { endField(); if row.contains(where: { !$0.isEmpty }) { table.append(row) }; row = [] }
        while i < chars.count {
            let c = chars[i]
            if quoted {
                if c == "\"" {
                    if i+1 < chars.count && chars[i+1] == "\"" { field.append("\""); i += 1 }
                    else { quoted = false; closed = true }
                } else { field.append(c) }
            } else if c == "\"" {
                guard field.isEmpty && !closed else { throw CSVError.invalid("Malformed quoted CSV field.") }
                quoted = true
            } else if c == "," { endField() }
            else if c == "\n" || c == "\r" {
                endRow()
                if c == "\r" && i+1 < chars.count && chars[i+1] == "\n" { i += 1 }
            } else {
                guard !closed || c == " " || c == "\t" else { throw CSVError.invalid("Unexpected text after closing quote.") }
                if !closed { field.append(c) }
            }
            i += 1
        }
        guard !quoted else { throw CSVError.invalid("Unclosed quoted CSV field.") }
        if !field.isEmpty || !row.isEmpty || closed { endRow() }
        guard table.count >= 3 else { throw CSVError.invalid("Use a header and at least two data rows.") }
        var names = table.removeFirst()
        names[0] = names[0].replacingOccurrences(of: "\u{FEFF}", with: "")
        guard names.count <= 64, table.count <= maxRows else { throw CSVError.invalid("Limit: 64 source columns and 1,000 rows.") }
        guard table.allSatisfy({ $0.count == names.count }) else { throw CSVError.invalid("Rows have different column counts.") }
        // Numeric columns must be complete and finite. Text/mixed/missing columns are excluded.
        let columns = names.indices.filter { c in table.allSatisfy { Double($0[c]).map { $0.isFinite && abs($0) <= 1e100 } ?? false } }
        guard (3...maxDimensions).contains(columns.count) else { throw CSVError.invalid("Need 3–12 complete finite numeric columns; text, missing and mixed columns are excluded.") }
        return Dataset(names: columns.map { names[$0].isEmpty ? "Column \($0+1)" : names[$0] }, rows: table.map { row in columns.map { Double(row[$0])! } }, groups: Array(repeating: 0, count: table.count), label: "CSV • \(table.count) rows • \(columns.count) numeric columns")
    }
}


// Session-only bookmarks belong to the current dataset. Physical plot gestures
// are intentionally excluded; restoring presents the projection in a neutral pose.
struct SavedViewpoint: Identifiable {
    let id = UUID()
    let name: String
    let projection: Projection
    let normalization: Normalization
    let time: Double
    let planeI: Int
    let planeJ: Int
    let manualAngle: Double
    let radius: Double
}
