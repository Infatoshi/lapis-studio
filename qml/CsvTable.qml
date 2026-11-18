import QtQuick

// CSV / markdown table. The header row stays pinned while the rows scroll in a clipped
// region below it, so no row ever renders underneath the header.
Rectangle {
    id: root
    objectName: "tableView"
    property var artifact
    readonly property var header: artifact ? artifact.header : []
    readonly property var rows: artifact ? artifact.rows : []
    readonly property int highlight: artifact && artifact.highlight !== undefined ? artifact.highlight : -1
    readonly property string highlightLabel: artifact && artifact.highlightLabel ? artifact.highlightLabel : ""
    color: Theme.card
    radius: 20
    clip: true

    // Column widths from the longest cell, stretched to the full width.
    readonly property var colW: {
        const n = header.length;
        if (n === 0) return [];
        const chars = [];
        for (let c = 0; c < n; ++c) {
            let m = String(header[c]).length;
            for (let r = 0; r < rows.length; ++r)
                m = Math.max(m, String(rows[r][c] || "").length);
            chars.push(Math.min(28, m) + 3);
        }
        const total = chars.reduce((a, b) => a + b, 0);
        const avail = width - 64;
        return chars.map(x => avail * x / total);
    }
    // Gamepad right stick: pixel deltas on an 8 ms tick, set directly (the key-step Behavior
    // is off while the stick moves, or every tick would restart a 150 ms animation).
    property bool padScrolling: false
    Timer { id: padIdle; interval: 120; onTriggered: root.padScrolling = false }
    function scrollPixels(px) {
        padScrolling = true;
        padIdle.restart();
        list.contentY = Math.max(0, Math.min(list.contentHeight - list.height, list.contentY + px));
    }
    function scrollBy(d) {
        list.contentY = Math.max(0, Math.min(list.contentHeight - list.height, list.contentY + d * 46 * 2));
    }

    Item {
        id: head
        x: 32
        width: parent.width - 64
        height: Theme.px(64)
        Row {
            anchors.verticalCenter: parent.verticalCenter
            Repeater {
                model: root.header.length
                Text {
                    required property int index
                    width: root.colW[index] || 100
                    text: root.header[index]
                    color: Theme.gold
                    elide: Text.ElideRight
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(19)
                    font.weight: Font.DemiBold
                    font.capitalization: Font.AllUppercase
                    font.letterSpacing: 0.8
                }
            }
        }
        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 1.5
            color: Theme.faint
        }
    }

    ListView {
        id: list
        objectName: "csvList"
        anchors.top: head.bottom
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.leftMargin: 32
        anchors.rightMargin: 32
        anchors.bottomMargin: 8
        clip: true
        boundsBehavior: Flickable.StopAtBounds
        model: root.rows.length  // a count, so a live refresh rebinds rows instead of rebuilding them
        Behavior on contentY {
            enabled: !root.padScrolling
            NumberAnimation { duration: 150; easing.type: Easing.OutCubic }
        }
        delegate: Rectangle {
            id: rowItem
            required property int index
            readonly property var modelData: root.rows[index] || []
            readonly property bool win: index === root.highlight
                || (root.highlightLabel.length > 0 && String(modelData[0]) === root.highlightLabel)
            width: ListView.view.width
            height: Theme.px(46)
            color: index % 2 ? "#100e0c" : "transparent"
            Row {
                anchors.verticalCenter: parent.verticalCenter
                Repeater {
                    model: root.header.length
                    Text {
                        required property int index
                        width: root.colW[index] || 100
                        text: rowItem.modelData[index] !== undefined ? rowItem.modelData[index] : ""
                        color: rowItem.win ? Theme.gold : Theme.text
                        elide: Text.ElideRight
                        font.family: Theme.mono
                        font.pixelSize: Theme.px(20)
                    }
                }
            }
        }
    }
}
