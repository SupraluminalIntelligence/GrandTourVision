# Actual mixed-immersive simulator visualization

Unedited 3840×2160 framebuffers from Apple's visionOS 27.0 (24M362) Apple Vision Pro simulator, captured during native XCTest holds with official simctl. These are actual app output in the simulator room, not mockups or headset photos.

- **08-room-scale-animation.png**: one diffuse room-scale cloud, 128 explicitly synthetic vectors × 64 features; block.0 / step-0, moving orthonormal projection, colors by synthetic token position. NO MODEL RUN.
- **09-room-scale-locked-comparison.png**: same identities and shared coordinate/reference state, paused and locked, block.1 / step-100. It records the later locked projection; do not compare directly with the earlier animated frame as if the projection were identical.
- **08-room-scale-animation.mov**: 10-second actual H.264 framebuffer recording while the high-dimensional tour runs and the compact controls are hidden. Static simulator viewpoint; no simulated physical walking or hand-tracking footage.

Physical scale is 4 m on these captures, initially 3.5 m ahead at 1.5 m height. Actual point-cloud extent varies with the projection and can exceed the fixed display reference in unbounded immersion; points remain visible. Shorter axes avoid dominating the cloud. Legend and small console bar provide context; there is no multi-panel grid or bounded plot window.

Older 01–07 images show the superseded volume prototype. Final source ZIP excludes screenshots/video/reference image; previews are separate private Library items. Reproduce with ImmersivePlotTests and Scripts/capture_simulator_frames.py --immersive --video. Headset gestures, walking and performance remain outstanding validation.
