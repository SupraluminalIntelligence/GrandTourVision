import XCTest

final class ImmersivePlotTests: XCTestCase {
    @MainActor
    func testRoomAnimationAndManipulationControls() throws {
        continueAfterFailure = false
        let app = openClassic()
        XCTAssertTrue(app.buttons["Synthetic trace"].waitForExistence(timeout: 30))
        app.buttons["Room demo (synthetic)"].tap()
        app.buttons["Enter room-scale plot"].tap()
        XCTAssertTrue(app.buttons["Pause animation"].waitForExistence(timeout: 30))
        XCTAssertTrue(app.staticTexts["room-summary"].label.contains("128 points"))
        XCTAssertTrue(app.staticTexts["room-summary"].label.contains("64 dimensions"))
        let before = app.staticTexts["room-time"].label
        app.buttons["Larger"].tap()
        XCTAssertTrue(app.staticTexts["room-span"].label.contains("4.8"))
        app.buttons["Move left"].tap()
        app.buttons["Rotate scene"].tap()
        XCTAssertTrue(app.buttons["Pause animation"].exists)
        app.buttons["Recenter scene"].tap()
        XCTAssertTrue(app.staticTexts["room-span"].label.contains("4.0"))
        app.buttons["Hide controls"].tap()
        XCTAssertTrue(app.buttons["Show room controls"].waitForExistence(timeout: 10))
        XCTAssertFalse(app.staticTexts["room-summary"].exists)
        captureStage("08-room-scale-animation", seconds: 16)
        app.buttons["Show room controls"].tap()
        XCTAssertTrue(app.staticTexts["room-time"].waitForExistence(timeout: 10))
        XCTAssertNotEqual(app.staticTexts["room-time"].label, before)
        app.buttons["Pause animation"].tap()
        XCTAssertTrue(app.buttons["Animate projection"].exists)
        app.switches["room-lock"].tap()
        XCTAssertEqual(app.switches["room-lock"].value as? String, "1")
        app.buttons["room-layer"].tap(); app.buttons["block.1"].tap()
        app.buttons["room-checkpoint"].tap(); app.buttons["step-100"].tap()
        XCTAssertTrue(app.staticTexts["room-snapshot"].label.contains("block.1@100"))
        app.buttons["Hide controls"].tap()
        captureStage("09-room-scale-locked-comparison", seconds: 12)
        app.buttons["Exit room"].tap()
        XCTAssertTrue(app.buttons["Enter room-scale plot"].waitForExistence(timeout: 20))
    }
    @MainActor
    func testRoomReentryAndManualProjection() throws {
        continueAfterFailure = false
        let app = openClassic()
        XCTAssertTrue(app.buttons["Public tiny GPT"].waitForExistence(timeout: 30))
        app.buttons["Public tiny GPT"].tap()
        for _ in 0..<2 {
            app.buttons["Enter room-scale plot"].tap()
            XCTAssertTrue(app.buttons["Pause animation"].waitForExistence(timeout: 30))
            XCTAssertEqual(app.buttons.matching(identifier: "Exit room").count, 1)
            app.buttons["Manual high-dimensional projection"].tap()
            XCTAssertTrue(app.buttons["Nudge projection"].waitForExistence(timeout: 10))
            app.buttons["Nudge projection"].tap()
            XCTAssertTrue(app.buttons["Animate projection"].exists)
            app.buttons["Animate projection"].tap()
            XCTAssertTrue(app.buttons["Pause animation"].exists)
            app.buttons["Exit room"].tap()
            XCTAssertTrue(app.buttons["Enter room-scale plot"].waitForExistence(timeout: 20))
        }
    }
    @MainActor
    func testLayerGallery() throws {
        continueAfterFailure = false
        let app = XCUIApplication(); app.launch()
        let status = app.staticTexts["gallery-status"]
        XCTAssertTrue(status.waitForExistence(timeout: 20))
        XCTAssertTrue(status.label.contains("6 snapshots"), status.label)
        app.buttons["enter-gallery"].tap()
        XCTAssertTrue(app.buttons["Exit gallery"].waitForExistence(timeout: 30))
        captureStage("20-gallery", seconds: 8)

        // step inside block 3: a fixed window; the data is framed so nearly every token fits
        app.buttons["focus-block"].tap(); app.buttons["b3"].tap()
        let inside = app.staticTexts["inside-count"]
        XCTAssertTrue(inside.waitForExistence(timeout: 10))
        captureStage("21-window-b3", seconds: 8)
        XCTAssertGreaterThan(count(inside), 230, inside.label)

        // zoom the data (not the window): tokens leave through the boundary
        app.buttons["Zoom in"].tap(); app.buttons["Zoom in"].tap()
        XCTAssertEqual(app.staticTexts["zoom-label"].label, "2.0×")
        captureStage("22-window-zoomed", seconds: 6)
        XCTAssertLessThan(count(inside), 230, inside.label)
        app.buttons["Re-center data"].tap()

        // tilt Y a quarter turn toward PC4: PC4 swings into view and the shape changes
        for _ in 0..<6 { app.buttons["tilt-y"].tap() }
        XCTAssertTrue(app.staticTexts["axis-summary"].label.contains("Y ≈ PC4"), app.staticTexts["axis-summary"].label)
        captureStage("23-window-tilted-pc4", seconds: 8)

        // slice: only tokens near this 3D slice of the 64-D space
        app.buttons["Slice"].tap()
        captureStage("24-window-slice", seconds: 6)
        XCTAssertLessThan(count(inside), 200, inside.label)
        app.buttons["Projection"].tap()

        app.buttons["tour-toggle"].tap()
        XCTAssertTrue(app.buttons["Stop tour"].waitForExistence(timeout: 5))
        captureStage("25-window-touring", seconds: 8)
        app.buttons["tour-toggle"].tap()

        app.buttons["Back to gallery"].tap()
        app.buttons["Exit gallery"].tap()
        XCTAssertTrue(app.buttons["enter-gallery"].waitForExistence(timeout: 20))
        // back home: switch to the healthy run
        let healthy = app.buttons["Healthy · 12 blocks (recorded)"]
        XCTAssertTrue(healthy.waitForExistence(timeout: 10))
        healthy.tap()
        XCTAssertTrue(app.staticTexts["gallery-status"].label.contains("6 snapshots"))
    }
    private func count(_ label: XCUIElement) -> Int { Int(label.label.split(separator: " ").first ?? "") ?? -1 }
    @MainActor private func openClassic() -> XCUIApplication {
        let app = XCUIApplication(); app.launch()
        XCTAssertTrue(app.buttons["open-classic"].waitForExistence(timeout: 30))
        app.buttons["open-classic"].tap()
        return app
    }
    @MainActor private func captureStage(_ name: String, seconds: Double) {
        print("GRANDTOUR_CAPTURE:\(name)")
        let hold = XCTestExpectation(description: "actual immersive framebuffer capture")
        hold.isInverted = true
        _ = XCTWaiter.wait(for: [hold], timeout: seconds)
    }
}
