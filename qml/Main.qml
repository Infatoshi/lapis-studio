import QtQuick

Window {
    id: win
    property QtObject backend
    property bool demoMode: false
    color: "#000000"
    title: "lapis studio"
    minimumWidth: 1100
    minimumHeight: 700

    StudioApp {
        anchors.fill: parent
        backend: win.backend
        demoMode: win.demoMode
    }
}
