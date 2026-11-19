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
    // Hover-to-ask: one HoverHandler for the whole diff; the line comes from the pointer's y.
    property QtObject targets: null
    function plain(i) {
        const l = list.lines[i];
        if (!l)
            return "";
        const body = String(l.html || "").replace(/<[^>]*>/g, "").replace(/&lt;/g, "<").replace(/&gt;/g, ">")
            .replace(/&quot;/g, "\"").replace(/&#39;/g, "'").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&");
        const k = l.k === "+" || l.k === "-" ? l.k : " ";
        return (l.no !== undefined && l.no !== "" ? String(l.no) : "") + " " + k + " " + body;
    }
    function lineNo(i) {
        const l = list.lines[i];
        return l && l.no !== undefined && String(l.no).length > 0 ? String(l.no) : String(i + 1);
    }
    function reportHover(pos) {
        if (!targets)
            return;
        const lp = list.mapFromItem(root, pos.x, pos.y);
        const i = list.indexAt(lp.x + list.contentX, lp.y + list.contentY);
        const item = i >= 0 ? list.itemAtIndex(i) : null;
        if (!item) {
            clearHover();
            return;
        }
        const title = root.artifact && root.artifact.title ? root.artifact.title : "diff";
        const n = lineNo(i);
        const around = [];
        for (let j = Math.max(0, i - 4); j <= Math.min(list.lines.length - 1, i + 4); ++j)
            around.push((j === i ? "> " : "  ") + plain(j));
        const r = item.mapToItem(null, 0, 0, item.width, item.height);
        // path: the diff row (unique even when a - and + line share a file line number);
        // label: the file line number a human reads in the gutter.
        targets.setHoverTarget("diff_line", "", "", "line=" + (i + 1), title + " L" + n,
                               "Diff: " + title + "\n" + around.join("\n"), r.x, r.y, r.width, r.height);
    }
    function clearHover() {
        if (targets && targets.activeTarget.kind === "diff_line")
            targets.clearHoverTarget("diff_line");
    }
    HoverHandler {
        id: hover
        onPointChanged: if (hovered) root.reportHover(point.position)
        onHoveredChanged: if (!hovered) root.clearHover()
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
