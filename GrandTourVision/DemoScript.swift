import Foundation
import simd

/// A scripted walkthrough for recording a video: launch with `-GrandTourDemo`. It opens the gallery on its own,
/// hides the control panel, and plays the same moves on a healthy run and on the run without residual
/// connections, with captions. Normal launches never run it.
@MainActor
enum DemoScript {
    static let enabled = ProcessInfo.processInfo.arguments.contains("-GrandTourDemo")
    static let pointScale: Float = enabled ? 1.7 : 1  // bigger dots so they survive video compression
    static let block = 8      // a deep block: healthy keeps its spread, no-residual has collapsed
    static let step = 5       // the step-1000 snapshot in the bundled recordings

    static func run(_ m: GalleryModel) async {
        await pause(2)
        show(m, "healthy-12-layers")
        m.caption = "A 12-block GPT after 1,000 training steps.\nEach cloud: 256 tokens in the 64-dimensional residual stream."
        await pause(6)
        await explore(m, intro: "Step inside block 8: a 2.4 m window into its 64-D space")

        m.caption = "Now the same network without residual connections"
        m.focused = nil; m.resetView()
        await pause(1)
        show(m, "no-residual-12-layers")
        await pause(5)
        await explore(m, intro: "Same block, no residuals: the tokens collapse onto thin strands")

        m.caption = "Without the residual stream, depth squeezes 64 dimensions down to ~1.\nRank collapse, visible from step 0."
        m.focused = nil; m.resetView()
        await pause(6)
        m.caption = nil
    }

    /// The same moves on both runs: step in, tilt a hidden direction into view by hand, then let the tour roam.
    private static func explore(_ m: GalleryModel, intro: String) async {
        m.caption = intro
        m.focused = block
        await pause(3.5)
        m.caption = "Tilt the window toward a hidden direction (PC4)…"
        if let q = m.handleDirections.first(where: { $0.pc == 3 })?.q {
            await animate(3) { f in m.tilt(toward: q, by: SIMD3(0, f * .pi / 2, 0)) }
        }
        await pause(1.5)
        m.caption = "…or let the grand tour roam through all of them"
        m.setPlaying(true)
        await pause(7)
        m.setPlaying(false)
    }

    private static func show(_ m: GalleryModel, _ sample: String) {
        m.loadSample(sample)
        try? m.select(frame: step)
    }

    private static func pause(_ seconds: Double) async { try? await Task.sleep(for: .seconds(seconds)) }

    /// Spread a change over `seconds` at 30 steps per second; `step` gets the fraction of the total for each tick.
    private static func animate(_ seconds: Double, _ step: (Double) -> Void) async {
        let n = max(1, Int(seconds * 30))
        for _ in 0..<n { step(1 / Double(n)); await pause(1 / 30) }
    }
}
