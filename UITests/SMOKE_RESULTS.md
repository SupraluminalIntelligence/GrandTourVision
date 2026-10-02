# Native mixed-immersive tests — 2026-10-01

Final `ImmersivePlotTests` passed, two tests and zero failures, on Apple Vision Pro simulator visionOS 27.0 (24M362). Actual log: `ui-immersive-final.log`.

- `testRoomAnimationAndManipulationControls`: 57.546 s. Loads synthetic room demo (128 × 64), enters mixed immersion, changes physical scale/position/orientation while animation remains enabled, recenters, hides controls, verifies tour time advances, pauses/locks, switches block.1 / step-100, exits to data controls.
- `testRoomReentryAndManualProjection`: 30.845 s. Loads the public tiny-GPT trace, performs two room entry/exit cycles without duplicated exit controls, nudges high-dimensional projection (pauses tour), resumes animation and exits.

Physical placement operations share code with targeted drag/rotate/magnify gestures. The UI tests use accessible buttons and do not simulate actual headset hands. Screenshot/video holds use official simulator framebuffers; manual XCUIScreen screenshots on this runtime are 1×1 placeholders and are not delivered.

Simulator system accessibility/XPC warnings and diagnostic collection's inherited Command Line Tools path were nonfatal; test execution succeeded. All final builds use a per-command DEVELOPER_DIR override; no global Xcode setting was changed.
