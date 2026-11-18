import QtQuick

// Syntax-highlighted unified diff in Menlo.
Rectangle {
    id: root
    objectName: "diffView"
    property var artifact
    color: Theme.card
    radius: 20
    clip: true

    function scrollBy(d) {
        list.contentY = Math.max(0, Math.min(list.contentHeight - list.height, list.contentY + d * 34 * 4));
    }

    // Gamepad right stick: pixel deltas on an 8 ms tick, set directly (the key-step Behavior
    // is off while the stick moves, or every tick would restart a 160 ms animation).
    property bool padScrolling: false
    Timer { id: padIdle; interval: 120; onTriggered: root.padScrolling = false }
    function scrollPixels(px) {
        padScrolling = true;
        padIdle.restart();
        list.contentY = Math.max(0, Math.min(list.contentHeight - list.height, list.contentY + px));
    }
    ListView {
        id: list
        objectName: "diffList"
        anchors.fill: parent
        anchors.topMargin: 22
        anchors.bottomMargin: 22
        boundsBehavior: Flickable.StopAtBounds
        // A count, not the array: a live refresh of the same diff rebinds rows in place instead of
        // rebuilding every delegate and snapping the scroll position back to the top.
        readonly property var lines: root.artifact && root.artifact.lines ? root.artifact.lines : []
        model: lines.length
        Behavior on contentY {
            enabled: !root.padScrolling
            NumberAnimation { duration: 160; easing.type: Easing.OutCubic }
        }
        delegate: Rectangle {
            required property int index
            readonly property var modelData: list.lines[index] || ({ k: " ", html: "" })
            readonly property string k: modelData.k
            width: ListView.view.width
            height: Theme.px(k === "f" ? 46 : 36)
            color: k === "+" ? Theme.addBg : (k === "-" ? Theme.delBg : "transparent")
            Rectangle {
                width: 4
                height: parent.height
                visible: parent.k === "+" || parent.k === "-"
                color: parent.k === "+" ? Theme.gold : Theme.copper
            }
            Text {
                id: num
                x: 18
                width: 64
                anchors.verticalCenter: parent.verticalCenter
                horizontalAlignment: Text.AlignRight
                text: parent.modelData.no !== undefined ? parent.modelData.no : ""
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: Theme.px(17)
            }
            Text {
                x: 104
                width: 26
                anchors.verticalCenter: parent.verticalCenter
                text: parent.k === "+" || parent.k === "-" ? parent.k : ""
                color: parent.k === "+" ? Theme.gold : Theme.copper
                font.family: Theme.mono
                font.pixelSize: Theme.px(21)
            }
            Text {
                x: parent.k === "f" || parent.k === "@" ? 28 : 134
                width: parent.width - x - 24
                anchors.verticalCenter: parent.verticalCenter
                textFormat: Text.StyledText
                text: parent.modelData.html
                color: parent.k === "-" ? "#b8a58c" : Theme.text
                opacity: parent.k === "-" ? 0.82 : 1
                elide: Text.ElideRight
                font.family: Theme.mono
                font.pixelSize: Theme.px(parent.k === "f" ? 22 : 21)
                font.weight: parent.k === "f" ? Font.DemiBold : Font.Normal
            }
        }
    }
}
