import QtQuick

// End-of-session review: Cmd+Shift+R, or the Review pill on the Constellation. Three short
// answers about the session across active projects:
//   Intent   what was set out to do vs what actually matched
//   Steers   the corrections the human gave along the way
//   Rules    1-3 personal taste rules worth keeping; Save Rule appends one to
//            ~/.lapis-next/session_reviews.jsonl and it joins the fast-model filter prompt
// The backend fills local lines at once; qwen3.8-flash-next rewrites them when an
// OpenAI-compatible endpoint is configured. Up/Down pick a rule, Enter / ✕ saves it,
// Esc / ○ closes.
FocusScope {
    id: root
    objectName: "sessionReview"
    property QtObject backend
    property bool open: false
    property int row: 0
    property bool padHints: false
    property var padGlyphs: ({})
    property var saved: []  // rules saved while this card is open
    signal closed()

    readonly property var refl: backend ? backend.reflection : ({})
    readonly property var rules: refl && refl.rules ? refl.rules : []
    readonly property int count: Math.min(rules.length, 3)
    readonly property bool busy: backend ? backend.reflectionBusy : false
    readonly property string source: backend ? backend.reflectionSource : ""
    readonly property var kept: backend ? backend.tasteRules : []

    function show() {
        row = 0;
        saved = [];
        open = true;
        if (backend)
            backend.runSessionReflection();
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
    function isSaved(rule) {
        return saved.indexOf(rule) >= 0 || kept.indexOf(rule) >= 0;
    }
    function saveRule(i) {
        const r = rules[i];
        if (!backend || !r || isSaved(r))
            return;
        backend.saveTasteRule(r, 1);
        saved = saved.concat([r]);
    }
    function confirm() {
        if (count === 0) {
            hide();
            return;
        }
        saveRule(Math.max(0, Math.min(row, count - 1)));
    }
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
            return;  // Cmd+Shift+R closes it, Cmd +/-/0 zoom: both reach the app
        } else if (k === Qt.Key_Escape || k === Qt.Key_Q) {
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

    component Caption: Text {
        color: Theme.faint
        font.family: Theme.mono
        font.pixelSize: 13
        font.letterSpacing: 1.6
        font.capitalization: Font.AllUppercase
    }

    Rectangle {
        id: card
        objectName: "reviewCard"
        anchors.centerIn: parent
        width: Math.min(parent.width - 120, 960)
        height: content.height + 76
        radius: 28
        color: "#0b0a08"
        border.width: 1.5
        border.color: Theme.goldA(0.55)
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
                    text: "Session review"
                    color: Theme.ink
                    font.family: Theme.sans
                    font.pixelSize: 30
                    font.weight: Font.DemiBold
                }
                Text {
                    objectName: "reviewSource"
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.topMargin: 10
                    text: root.busy ? "reflecting…" : (root.source === "local" ? "latest turns" : root.source)
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 14
                }
            }

            Caption { text: "Intent" }
            Text {
                objectName: "reviewIntent"
                width: parent.width
                text: root.refl && root.refl.intent ? root.refl.intent : ""
                textFormat: Text.PlainText
                wrapMode: Text.Wrap
                maximumLineCount: 4
                elide: Text.ElideRight
                color: Theme.text
                font.family: Theme.sans
                font.pixelSize: 19
                lineHeight: 1.3
            }

            Item { width: 1; height: 2 }
            Caption { text: "Steers" }
            Text {
                objectName: "reviewSteers"
                width: parent.width
                text: root.refl && root.refl.corrections ? root.refl.corrections : ""
                textFormat: Text.PlainText
                wrapMode: Text.Wrap
                maximumLineCount: 3
                elide: Text.ElideRight
                color: Theme.dim
                font.family: Theme.sans
                font.pixelSize: 18
                lineHeight: 1.3
            }

            Item { width: 1; height: 2 }
            Rectangle { width: parent.width; height: 1; color: Theme.line }
            Caption { text: "Rules to keep" }
            Text {
                objectName: "reviewNoRules"
                visible: root.count === 0
                text: "No new rules this session."
                color: Theme.dim
                font.family: Theme.sans
                font.pixelSize: 18
            }
            Repeater {
                model: root.count
                Rectangle {
                    id: ruleRow
                    required property int index
                    objectName: "reviewRule"
                    readonly property string rule: root.rules[index] || ""
                    readonly property bool sel: index === root.row
                    readonly property bool done: root.isSaved(rule)
                    width: content.width
                    height: Math.max(ruleText.implicitHeight, 32) + 22
                    radius: 16
                    color: sel ? Theme.cardHi : "transparent"
                    border.width: sel ? 1 : 0
                    border.color: Theme.goldA(0.55)
                    Text {
                        id: ruleText
                        objectName: "reviewRuleText"
                        x: 18
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width - savePill.width - 54
                        text: ruleRow.rule
                        textFormat: Text.PlainText
                        wrapMode: Text.Wrap
                        maximumLineCount: 2
                        elide: Text.ElideRight
                        color: Theme.ink
                        font.family: Theme.sans
                        font.pixelSize: 19
                    }
                    Rectangle {
                        id: savePill
                        objectName: "saveRulePill"
                        anchors.right: parent.right
                        anchors.rightMargin: 14
                        anchors.verticalCenter: parent.verticalCenter
                        width: saveRow.implicitWidth + 26
                        height: 30
                        radius: 15
                        color: "transparent"
                        border.width: 1
                        border.color: ruleRow.done ? Theme.goldA(0.55) : Theme.line
                        Row {
                            id: saveRow
                            anchors.centerIn: parent
                            spacing: 8
                            Text {
                                id: saveText
                                objectName: "saveRuleText"
                                anchors.verticalCenter: parent.verticalCenter
                                text: ruleRow.done ? "Saved" : "Save Rule"
                                color: ruleRow.done ? Theme.gold : (ruleRow.sel ? Theme.text : Theme.dim)
                                font.family: Theme.mono
                                font.pixelSize: 14
                            }
                            // The pad button that saves the selected rule, controller mode only.
                            Rectangle {
                                objectName: "saveRuleGlyph"
                                readonly property string g: root.padGlyphs["confirm"] || ""
                                anchors.verticalCenter: parent.verticalCenter
                                visible: root.padHints && ruleRow.sel && !ruleRow.done && g.length > 0
                                width: saveGlyphText.implicitWidth + 10
                                height: 18
                                radius: 5
                                color: "transparent"
                                border.width: 1
                                border.color: Theme.line
                                Text {
                                    id: saveGlyphText
                                    anchors.centerIn: parent
                                    text: parent.g
                                    color: Theme.dim
                                    font.family: Theme.mono
                                    font.pixelSize: 11
                                }
                            }
                        }
                        TapHandler {
                            onTapped: {
                                root.row = ruleRow.index;
                                root.saveRule(ruleRow.index);
                            }
                        }
                    }
                    TapHandler { onTapped: root.row = ruleRow.index }
                }
            }

            Text {
                objectName: "reviewHints"
                text: root.padHints
                      ? [root.glyph("confirm", "save"), root.glyph("back", "close"), "↕ move"].filter(t => t.length > 0).join("    ")
                      : "⏎ save    esc close    ↑↓ move"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 13
            }
        }
    }
}
