import SwiftUI

@main
struct GrandTourApp: App {
    @State private var model = TourModel()
    @State private var gallery = GalleryModel()
    var body: some Scene {
        // First scene = the launch window: the gallery is where the app starts.
        WindowGroup(id: "gallery", for: String.self) { _ in
            GalleryControlsView(model: gallery, roomPlotOpen: model.immersiveActive)
        } defaultValue: { "gallery" }
        .defaultSize(width: 720, height: 860)
        .defaultWindowPlacement { _, _ in WindowPlacement(.utilityPanel) }  // low and close: the arc stays visible
        WindowGroup(id: "controls", for: String.self) { _ in
            ControlsView(model: model)
        } defaultValue: { "controls" }
        .defaultSize(width: 650, height: 760)
        WindowGroup(id: "room-controls", for: String.self) { _ in
            RoomControlsView(model: model)
        } defaultValue: { "room-controls" }
        .defaultSize(width: 560, height: 460)
        .windowResizability(.contentSize)
        .defaultWindowPlacement { _, _ in WindowPlacement(.utilityPanel) }
        ImmersiveSpace(id: "room-plot") {
            PlotView(model: model)
        }.immersionStyle(selection: .constant(.mixed), in: .mixed)
        WindowGroup(id: "gallery-panel", for: String.self) { _ in
            GalleryControlsView(model: gallery, roomPlotOpen: model.immersiveActive, compact: true)
        } defaultValue: { "gallery-panel" }
        .defaultSize(width: 620, height: 720)
        .defaultWindowPlacement { _, _ in WindowPlacement(.utilityPanel) }
        ImmersiveSpace(id: "layer-gallery") {
            GalleryView(model: gallery)
        }.immersionStyle(selection: .constant(.mixed), in: .mixed)
    }
}
