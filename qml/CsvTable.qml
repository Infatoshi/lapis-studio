import QtQuick

// CSV / markdown table. Uses clean sans typography, exact FontMetrics column sizing,
// word-wrapping so cells never truncate silently, and shrink-wraps to content centered on Stage.
Item {
    id: root
    objectName: "tableView"
    property var artifact
    readonly property var header: artifact ? artifact.header : []
    readonly property var rows: artifact ? artifact.rows : []
    readonly property int highlight: artifact && artifact.highlight !== undefined ? artifact.highlight : -1
    readonly property string highlightLabel: artifact && artifact.highlightLabel ? artifact.highlightLabel : ""

    FontMetrics {
        id: cellMetrics
        font.family: Theme.sans
        font.pixelSize: Theme.px(18)
    }
    FontMetrics {
        id: hdrMetrics
        font.family: Theme.mono
        font.pixelSize: Theme.px(15)
        font.weight: Font.DemiBold
        font.capitalization: Font.AllUppercase
        font.letterSpacing: 0.8
    }

    // Exact measured column widths; wide columns cap at 340 px and wrap instead of eliding.
    readonly property var colW: {
        const n = header.length;
        if (n === 0) return [];
        const maxCol = Theme.px(340);
        const minCol = Theme.px(72);
        const px = [];
        for (let c = 0; c < n; ++c) {
            let w = hdrMetrics.advanceWidth(String(header[c])) + Theme.px(28);
            for (let r = 0; r < rows.length; ++r) {
                const cell = String(rows[r][c] || "");
                w = Math.max(w, cellMetrics.advanceWidth(cell) + Theme.px(28));
            }
            px.push(Math.max(minCol, Math.min(maxCol, Math.ceil(w))));
        }
        const total = px.reduce((a, b) => a + b, 0);
        const avail = Math.max(320, root.width - 64);
        if (total <= avail)
            return px;
        // Scale proportionally down to fit available width (with a readable floor).
        return px.map(x => Math.max(Theme.px(68), Math.floor(avail * x / total)));
    }
    readonly property real innerW: colW.reduce((a, b) => a + b, 0)
    readonly property real tableW: colW.length === 0 ? Math.min(root.width, 420) : Math.min(root.width, innerW + 64)
    readonly property real tableH: Math.min(root.height, Math.max(Theme.px(140), Theme.px(56) + list.contentHeight + 24))

    // Gamepad right stick: pixel deltas on an 8 ms tick, set directly.
    property bool padScrolling: false
    Timer { id: padIdle; interval: 120; onTriggered: root.padScrolling = false }
    function scrollPixels(px) {
        padScrolling = true;
        padIdle.restart();
        list.contentY = Math.max(0, Math.min(Math.max(0, list.contentHeight - list.height), list.contentY + px));
    }
    function panPixels(dx, dy) {
        padScrolling = true;
        padIdle.restart();
        hFlick.contentX = Math.max(0, Math.min(Math.max(0, hFlick.contentWidth - hFlick.width), hFlick.contentX + dx));
        list.contentY = Math.max(0, Math.min(Math.max(0, list.contentHeight - list.height), list.contentY + dy));
    }
    function scrollBy(d) {
        list.contentY = Math.max(0, Math.min(Math.max(0, list.contentHeight - list.height), list.contentY + d * 48 * 2));
    }

    Rectangle {
        id: sheet
        anchors.centerIn: parent
        width: root.tableW
        height: root.tableH
        color: Theme.card
        radius: 20
        border.width: 1
        border.color: Theme.line
        clip: true

        Flickable {
            id: hFlick
            anchors.fill: parent
            anchors.leftMargin: 32
            anchors.rightMargin: 32
            anchors.bottomMargin: 8
            contentWidth: Math.max(width, root.innerW)
            contentHeight: height
            clip: true
            boundsBehavior: Flickable.StopAtBounds

            Item {
                id: head
                width: hFlick.contentWidth
                height: Theme.px(56)
                Row {
                    anchors.verticalCenter: parent.verticalCenter
                    Repeater {
                        model: root.header.length
                        Text {
                            required property int index
                            width: root.colW[index] || 100
                            rightPadding: Theme.px(16)
                            text: root.header[index]
                            color: Theme.gold
                            wrapMode: Text.WordWrap
                            font.family: Theme.mono
                            font.pixelSize: Theme.px(15)
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
                width: hFlick.contentWidth
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                model: root.rows.length
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
                    height: Math.max(Theme.px(44), cellRow.implicitHeight + Theme.px(16))
                    color: rowItem.win ? "#1a1308" : (index % 2 ? "#0f0d0b" : "transparent")
                    Row {
                        id: cellRow
                        anchors.verticalCenter: parent.verticalCenter
                        Repeater {
                            model: root.header.length
                            Text {
                                required property int index
                                width: root.colW[index] || 100
                                rightPadding: Theme.px(16)
                                text: rowItem.modelData[index] !== undefined ? rowItem.modelData[index] : ""
                                color: rowItem.win ? Theme.gold : Theme.text
                                wrapMode: Text.WordWrap
                                font.family: Theme.sans
                                font.pixelSize: Theme.px(18)
                                lineHeight: 1.25
                            }
                        }
                    }
                    Rectangle {
                        anchors.bottom: parent.bottom
                        width: parent.width
                        height: 1
                        color: "#191714"
                    }
                }
            }
        }
    }
}
