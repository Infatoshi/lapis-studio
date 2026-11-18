import QtQuick

// Morning Briefing: B (or R2 with a controller) on the Constellation, or on its own after an
// overnight / long-idle return. Only changed or active projects, two crisp lines each:
//   1  what happened, with concrete proof (a number, an artifact, new turns)
//   2  the single next action, doable by controller or voice (gold when it is a decision)
// The backend fills the local lines at once; qwen3.8-flash-next rewrites them when an
// OpenAI-compatible endpoint is configured. Up/Down pick a row, Enter / ✕ opens that project,
// Esc / B / ○ closes.
FocusScope {
    id: root
    objectName: "morningBriefing"
    property QtObject backend
    property bool open: false
    property int row: 0
    property bool padHints: false
    property var padGlyphs: ({})
    signal closed()
    signal openProject(int index)

    readonly property var rows: backend ? backend.briefing : []
    readonly property int count: Math.min(rows.length, 8)
    readonly property bool busy: backend ? backend.briefingBusy : false
    readonly property string source: backend ? backend.briefingSource : ""

    function show() {
        row = 0;
        open = true;
        if (backend)
            backend.requestBriefing();
        forceActiveFocus();
    }
    function hide() {
        if (!open)
            return;
        open = false;
        closed();
    }
    function move(d) {
        if (count > 0)
            row = (row + d + count) % count;
    }
    function confirm() {
        if (count === 0) {
            hide();
            return;
        }
        const r = rows[Math.max(0, Math.min(row, count - 1))];
        hide();
        openProject(r.index);
    }
    function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
    function glyph(action, word) {
        const g = padGlyphs[action];
        return g !== undefined && g.length > 0 ? g + " " + word : "";
    }

    visible: opacity > 0.01
    opacity: open ? 1 : 0
    Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
    enabled: open

    Keys.onPressed: (e) => {
        const ctrl = (e.modifiers & Qt.ControlModifier) !== 0;
        const k = e.key;
        if (ctrl) {
            return;  // Cmd +/-/0 zoom still reach the app
        } else if (k === Qt.Key_Escape || k === Qt.Key_B || k === Qt.Key_Q) {
            hide();
        } else if (k === Qt.Key_Down || k === Qt.Key_J || k === Qt.Key_S || k === Qt.Key_Tab) {
            move(1);
        } else if (k === Qt.Key_Up || k === Qt.Key_K || k === Qt.Key_W || k === Qt.Key_Backtab) {
            move(-1);
        } else if (k === Qt.Key_Return || k === Qt.Key_Enter || k === Qt.Key_E) {
            confirm();
        }
        e.accepted = true;
    }

    // OLED dim: the Constellation stays faintly visible behind the card.
    Rectangle {
        anchors.fill: parent
        color: "#000000"
        opacity: 0.88
        MouseArea { anchors.fill: parent; onClicked: root.hide(); onWheel: (w) => w.accepted = true }
    }

    Rectangle {
        id: card
        objectName: "briefingCard"
        anchors.centerIn: parent
        width: Math.min(parent.width - 120, 1040)
        height: content.height + 76
        radius: 28
        color: "#0b0a08"
        border.width: 1.5
        border.color: Qt.rgba(0.878, 0.624, 0.243, 0.55)
        MouseArea { anchors.fill: parent }  // clicks inside never close it

        Column {
            id: content
            x: 44
            y: 38
            width: parent.width - 88
            spacing: 14

            Item {
                width: parent.width
                height: 46
                Text {
                    text: "Morning briefing"
                    color: Theme.ink
                    font.family: Theme.sans
                    font.pixelSize: 30
                    font.weight: Font.DemiBold
                }
                Text {
                    objectName: "briefingSource"
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.topMargin: 10
                    text: root.busy ? "summarising…" : (root.source === "local" ? "latest turns" : root.source)
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 14
                }
            }

            Text {
                objectName: "briefingEmpty"
                visible: root.count === 0
                text: "Nothing changed while you were away."
                color: Theme.dim
                font.family: Theme.sans
                font.pixelSize: 20
            }

            Repeater {
                model: root.count
                Rectangle {
                    id: rowItem
                    required property int index
                    objectName: "briefingRow"
                    readonly property var r: root.rows[index]
                    readonly property bool sel: index === root.row
                    width: content.width
                    height: lines.implicitHeight + 24
                    radius: 16
                    color: sel ? Theme.cardHi : "transparent"
                    border.width: sel ? 1.5 : 0
                    border.color: Theme.gold
                    Column {
                        id: lines
                        x: 18
                        y: 12
                        width: parent.width - 36
                        spacing: 4
                        Text {
                            objectName: "briefingLine1"
                            width: parent.width
                            elide: Text.ElideRight
                            maximumLineCount: 1
                            textFormat: Text.StyledText
                            text: rowItem.r ? "<b>" + root.esc(rowItem.r.name) + "</b>&nbsp;&nbsp;" + root.esc(rowItem.r.line1) : ""
                            color: Theme.text
                            font.family: Theme.sans
                            font.pixelSize: 20
                        }
                        Text {
                            objectName: "briefingLine2"
                            width: parent.width
                            elide: Text.ElideRight
                            maximumLineCount: 1
                            text: "→ " + (rowItem.r ? rowItem.r.line2 : "")
                            color: rowItem.r && rowItem.r.blocker ? Theme.gold : Theme.dim
                            font.family: Theme.sans
                            font.pixelSize: 18
                        }
                    }
                    TapHandler {
                        onTapped: {
                            root.row = rowItem.index;
                            root.confirm();
                        }
                    }
                }
            }

            Text {
                objectName: "briefingHints"
                text: root.padHints
                      ? [root.glyph("confirm", "open"), root.glyph("back", "close"), "↕ move"].filter(t => t.length > 0).join("    ")
                      : "⏎ open    esc close    ↑↓ move"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 13
            }
        }
    }
}
