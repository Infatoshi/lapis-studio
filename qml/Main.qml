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
}
