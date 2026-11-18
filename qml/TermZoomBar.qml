import QtQuick

// A sparse row of hyperparameter / term pills under the Stage header ("regs: 128 ▾",
// "tile: 128x128 ▾", "KV cache ▾"). Clicking a pill, or Cmd+P then h/l (a/d, arrows), drops a
// compact card with the value across runs and a one-line plain-English meaning; Enter on it
// scopes the next quick question to that parameter.
//
// While the pointer is over the row, a card is open, or the keyboard is in the row, the pills
// are frozen: a background rescan that changes the params lands only once the user lets go, so
// nothing ever moves under his cursor.
Item {
    id: root
    objectName: "termZoomBar"
    property var params: []
    property bool active: false   // keyboard is in the row (Cmd+P)
    property int selected: 0
    property int expanded: -1     // index of the open card, -1 = none
    property real cardReveal: 1   // 0..1, the Stage drives it from its dim curve (crossfade, no overlap)
    property bool padHints: false // controller glyphs instead of keys (StudioApp decides)
    property var padGlyphs: ({})
    readonly property int cardGap: 20
    function padHint(action, word) {
        const g = padGlyphs[action];
        return g !== undefined && g.length > 0 ? g + " " + word : "";
    }
    // The open pill's rect in this item's coordinates (the focus cursor wraps it).
    function pillRect(i) {
        const it = i >= 0 && i < pillRep.count ? pillRep.itemAt(i) : null;
        return it ? Qt.rect(row.x + it.x, row.y + it.y, it.width, it.height) : Qt.rect(0, 0, 0, 0);
    }
    signal askAbout(var param)

    // What is drawn: a snapshot of params, refreshed only while nobody is interacting.
    property var shown: []
    readonly property bool interacting: active || expanded >= 0 || hover.hovered
    function refreshShown() {
        if (!interacting)
            shown = params ? params : [];
    }
    onParamsChanged: refreshShown()
    onInteractingChanged: refreshShown()
    Component.onCompleted: refreshShown()

    readonly property int count: shown.length
    visible: count > 0
    height: count > 0 ? 40 : 0
    implicitHeight: height

    function current() { return selected >= 0 && selected < count ? shown[selected] : null; }
    function open() {
        if (count === 0)
            return false;
        active = true;
        selected = Math.max(0, Math.min(selected, count - 1));
        expanded = selected;
        return true;
    }
    function close() {
        active = false;
        expanded = -1;
    }
    function step(d) {
        if (count === 0)
            return;
        selected = (selected + d + count) % count;
        expanded = selected;
    }
    function pillLabel(p) {
        return p.kind === "term" || !p.value ? p.name : p.name + ": " + p.value;
    }

    HoverHandler { id: hover }

    Row {
        id: row
        spacing: 10
        anchors.verticalCenter: parent.verticalCenter
        Repeater {
            id: pillRep
            model: root.count
            Rectangle {
                id: pill
                required property int index
                objectName: "paramPill"
                readonly property var p: root.shown[index]
                readonly property bool sel: root.active && index === root.selected
                readonly property bool open: index === root.expanded
                readonly property bool term: p && p.kind === "term"
                width: pillText.implicitWidth + 34
                height: 32
                radius: 16
                color: open ? Theme.goldA(0.14) : "transparent"
                border.width: sel || open ? 1.5 : 1
                border.color: sel || open ? Theme.gold : (term ? Theme.faint : Theme.line)
                Text {
                    id: pillText
                    objectName: "paramPillText"
                    x: 14
                    anchors.verticalCenter: parent.verticalCenter
                    text: pill.p ? root.pillLabel(pill.p) + "  " + (pill.open ? "▴" : "▾") : ""
                    color: pill.open ? Theme.gold : (pill.term ? Theme.dim : Theme.text)
                    font.family: pill.term ? Theme.sans : Theme.mono
                    font.pixelSize: Theme.px(15)
                }
                TapHandler {
                    onTapped: {
                        root.selected = pill.index;
                        root.expanded = root.expanded === pill.index ? -1 : pill.index;
                    }
                }
            }
        }
        Text {
            visible: root.count > 0 && !root.active && root.expanded < 0 && !root.padHints
            anchors.verticalCenter: parent.verticalCenter
            text: "⌘P"
            color: Theme.faint
            font.family: Theme.mono
            font.pixelSize: Theme.px(13)
        }
    }

    // The open card: drops over the artifact (it never pushes the Stage down).
    Rectangle {
        id: card
        objectName: "paramCard"
        readonly property var p: root.expanded >= 0 && root.expanded < root.count ? root.shown[root.expanded] : null
        readonly property Item anchorPill: root.expanded >= 0 && root.expanded < pillRep.count ? pillRep.itemAt(root.expanded) : null
        visible: p !== null
        z: 60
        x: anchorPill ? Math.max(0, Math.min(root.width - width, anchorPill.x)) : 0
        // 20 px under the row: the focus cursor wraps the open pill (ring pad 12 + halo 4 below a
        // pill that ends 4 px above the row's bottom), so no ring line ever crosses the card.
        y: root.height + root.cardGap
        width: Math.min(root.width, Math.max(420, cardCol.implicitWidth + 44))
        height: cardCol.implicitHeight + 32
        radius: 18
        color: "#0b0a08"  // opaque: the artifact under it never shows through the text
        border.width: 1.5
        border.color: Theme.gold
        opacity: root.cardReveal
        Column {
            id: cardCol
            x: 22
            y: 16
            width: card.width - 44
            spacing: 10
            Row {
                spacing: 12
                Text {
                    id: cardName
                    objectName: "paramCardName"
                    text: card.p ? card.p.name : ""
                    color: Theme.ink
                    font.family: Theme.sans
                    font.pixelSize: Theme.px(22)
                    font.weight: Font.DemiBold
                }
                Text {
                    anchors.baseline: cardName.baseline
                    visible: card.p && card.p.value.length > 0
                    text: card.p ? card.p.value : ""
                    color: Theme.gold
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(20)
                }
            }
            // Values across runs, oldest first; the latest in gold.
            Row {
                objectName: "paramRuns"
                visible: card.p && card.p.runs && card.p.runs.length > 1
                spacing: 8
                Text {
                    text: "runs"
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(14)
                    anchors.verticalCenter: parent.verticalCenter
                }
                Repeater {
                    model: card.p && card.p.runs ? card.p.runs.length : 0
                    Text {
                        required property int index
                        readonly property bool last: index === card.p.runs.length - 1
                        text: (index > 0 ? "→ " : "") + card.p.runs[index]
                        color: last ? Theme.gold : Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: Theme.px(16)
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }
            Text {
                objectName: "paramDetail"
                width: cardCol.width
                text: card.p ? card.p.detail : ""
                color: Theme.text
                wrapMode: Text.WordWrap
                font.family: Theme.sans
                font.pixelSize: Theme.px(17)
            }
            Text {
                objectName: "paramCardHints"
                text: root.padHints
                      ? [root.padHint("confirm", "ask about " + (card.p ? card.p.name : "")), "←→ next",
                         root.padHint("back", "close")].filter(t => t.length > 0).join("    ")
                      : root.active ? "⏎ ask about " + (card.p ? card.p.name : "") + "    ←→ next    esc close"
                                    : "click again to close    ⌘P keys"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: Theme.px(13)
            }
        }
        TapHandler { onTapped: if (card.p) root.askAbout(card.p) }
    }
}
