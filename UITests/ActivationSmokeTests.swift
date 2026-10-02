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
        captureStage("10-gallery-still", seconds: 10)
        app.buttons["focus-block"].tap(); app.buttons["b3"].tap()
        captureStage("11-gallery-inside-b3", seconds: 10)
        app.buttons["Larger"].tap(); app.buttons["Larger"].tap()
        captureStage("12-gallery-inside-b3-zoomed", seconds: 8)
        app.buttons["Recenter"].tap()
        app.buttons["axis-0-next"].tap()  // PC2 and PC3 are on Y and Z, so X skips to PC4
        XCTAssertEqual(app.staticTexts["axis-0-label"].label, "X = PC4")
        captureStage("13-gallery-inside-b3-pc4", seconds: 8)
        app.buttons["tour-toggle"].tap()
        XCTAssertTrue(app.buttons["Stop tour"].waitForExistence(timeout: 5))
        captureStage("14-gallery-inside-b3-touring", seconds: 8)
        app.buttons["Back to gallery"].tap()
        app.buttons["1000"].tap()
        let healthy = app.buttons["Healthy · 12 blocks (recorded)"]
        XCTAssertTrue(healthy.waitForExistence(timeout: 10))
        healthy.tap()
        XCTAssertTrue(status.label.contains("6 snapshots"), status.label)
        captureStage("15-gallery-healthy", seconds: 8)
        app.buttons["Exit gallery"].tap()
        XCTAssertTrue(app.buttons["enter-gallery"].waitForExistence(timeout: 20))
    }
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
