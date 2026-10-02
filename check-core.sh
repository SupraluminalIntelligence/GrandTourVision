#!/bin/zsh
set -eu
cd "${0:A:h}"
swiftc -module-cache-path .swift-cache GrandTourVision/Projection.swift GrandTourVision/ActivationTrace.swift GrandTourVision/ScenePlacement.swift GrandTourVision/TourModel.swift GrandTourVision/FlowTour.swift Tests/main.swift -o Tests/core-tests
Tests/core-tests
