import QtQuick

// The agent's full write-up, folded away until Space.
Rectangle {
    id: root
    objectName: "prosePanel"
    property string text: ""
    property string thinking: ""
    property bool shown: false
    color: Theme.card
    radius: 22
    opacity: shown ? 1 : 0
    visible: opacity > 0.005
    Behavior on opacity { NumberAnimation { duration: Theme.zoomMs; easing.type: Easing.OutCubic } }
    transform: Translate { y: root.shown ? 0 : 30; Behavior on y { NumberAnimation { duration: Theme.zoomMs; easing.type: Easing.OutCubic } } }

    function scrollBy(d) {
        flick.contentY = Math.max(0, Math.min(flick.contentHeight - flick.height, flick.contentY + d * 160));
    }
    function scrollToEnd(d) {  // vim g / G
        flick.contentY = d > 0 ? Math.max(0, flick.contentHeight - flick.height) : 0;
    }

    // Gamepad right stick: pixel deltas on an 8 ms tick, set directly (the key-step Behavior
    // is off while the stick moves, or every tick would restart a 160 ms animation).
    property bool padScrolling: false
    Timer { id: padIdle; interval: 120; onTriggered: root.padScrolling = false }
    function scrollPixels(px) {
        padScrolling = true;
        padIdle.restart();
        flick.contentY = Math.max(0, Math.min(flick.contentHeight - flick.height, flick.contentY + px));
    }
    Flickable {
        id: flick
        objectName: "proseFlick"
        anchors.fill: parent
        anchors.margins: 48
        contentHeight: col.height
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        Behavior on contentY {
            enabled: !root.padScrolling
            NumberAnimation { duration: 160 }
        }
        Column {
            id: col
            width: Math.min(1100, flick.width)
            x: (flick.width - width) / 2
            spacing: 26
            // Folded away, the panel lays out nothing: every live update used to re-parse and
            // re-wrap the whole markdown write-up behind the Stage.
            Text {
                width: parent.width
                visible: root.thinking.length > 0
                text: root.visible ? "✦ " + root.thinking : ""
                color: Theme.dim
                wrapMode: Text.Wrap
                font.family: Theme.mono
                font.pixelSize: Theme.px(16)
                lineHeight: 1.3
            }
            Text {
                width: parent.width
                text: !root.visible ? "" : (root.text.length > 0 ? root.text : "No write-up yet.")
                textFormat: Text.MarkdownText
                color: Theme.ink
                linkColor: Theme.gold
                wrapMode: Text.Wrap
                font.family: Theme.sans
                font.pixelSize: Theme.px(23)
                lineHeight: 1.32
            }
        }
    }
}
