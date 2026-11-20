import QtQuick

Window {
    id: win
    property QtObject backend
    property bool demoMode: false
    color: "#000000"
    title: "lapis studio"
    minimumWidth: 720  // a half-screen tile fits (see main.cpp)
    minimumHeight: 600

    StudioApp {
        anchors.fill: parent
        backend: win.backend
        demoMode: win.demoMode
    }

    // Hide the OS mouse cursor while fullscreen in controller mode so only the controller cursor is visible.
    Item {
        anchors.fill: parent
        z: 999999
        enabled: win.visibility === Window.FullScreen && win.backend !== null && win.backend.navMode === "controller"
        HoverHandler {
            enabled: parent.enabled
            cursorShape: Qt.BlankCursor
        }
    }
}
