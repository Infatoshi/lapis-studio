import QtQuick

// Bottom bar: Enter asks (Sonnet, ~1 s), Opt+Enter thinks (trace collapses), Cmd+Enter steers the agent.
Item {
    id: root
    objectName: "composer"
    property QtObject backend
    property bool hasSession: false
    signal ask(string text)
    signal think(string text)
    signal steer(string text)
    signal escaped()

    readonly property bool hasQuick: backend && backend.quickQuestion.length > 0
    readonly property bool hasTrace: backend && backend.traceQuestion.length > 0
    readonly property bool streaming: hasTrace && backend.traceBusy

    implicitHeight: col.height
    height: col.height

    property string navMode: "vim"
    property bool pickerOpen: false
    // Closing the strip warms the chosen model's daemon (live only), so the next Enter is warm.
    onPickerOpenChanged: if (!pickerOpen && backend) backend.warmChannel()
    // The prompt Tab would accept: a warm model suggestion when one has landed, else the view's
    // heuristic one, else a follow-up to the last answer. Shown as gold ghost text.
    property string suggestion: ""
    readonly property bool inputFocused: input.activeFocus
    readonly property string inputText: input.text
    signal suggestionAccepted(string text)

    // Composer's own predictedAction (StudioApp uses it while the input has focus).
    readonly property var predictedAction: suggestion.length > 0 && input.text.length === 0
        ? { targetId: "composer", label: "", promptSuggestion: suggestion, actionKind: "prompt" }
        : null

    // A parameter pill scoped the next question ("regs: 128"); shown as a chip in the bar.
    property string scope: ""
    signal scopeCleared()

    function phaseLabel() {
        if (!backend)
            return "…";
        const ph = backend.quickPhase;
        const deep = backend.modelSpeed(backend.quickModel) === "Deep Think";
        if (ph === "sending")
            return "sent to " + backend.shortModelLabel(backend.quickModel) + " …";
        if (ph.startsWith("running"))
            return ph + " …";
        if (ph === "thinking")
            return (deep ? "deep thinking" : "thinking") + " …";
        return "…";
    }
    function providerLabel(p) {
        return p === "anthropic" ? "Anthropic" : p === "openai" ? "OpenAI" : p === "google" ? "Google" : "";
    }
    function focusInput(t) {
        input.forceActiveFocus();
        if (t.length > 0)
            input.insert(input.cursorPosition, t);
    }
    property string deletedBuffer: ""
    property bool highlighting: false
    property int highlightAnchor: 0
    property real restoreProgress: 0
    property bool backHolding: false

    // Esc without a key event (gamepad ○): save to undo buffer + clear text, then scope, then leave.
    function escapeInput() {
        if (pickerOpen) {
            pickerOpen = false;
        } else if (input.text.length > 0) {
            deletedBuffer = input.text;
            highlighting = false;
            input.text = "";
            if (backend) backend.recordAction("deleted prompt text");
        } else if (scope.length > 0) {
            scopeCleared();
        } else {
            escaped();
        }
    }
    function setBackHeld(down) {
        if (down && deletedBuffer.length > 0 && input.text.length === 0) {
            backHolding = true;
            restoreAnim.restart();
        } else {
            backHolding = false;
            restoreAnim.stop();
            restoreProgress = 0;
        }
    }
    function restoreDeleted() {
        if (deletedBuffer.length === 0)
            return;
        input.forceActiveFocus();
        input.text = deletedBuffer;
        input.cursorPosition = input.text.length;
        deletedBuffer = "";
        backHolding = false;
        restoreProgress = 0;
        if (backend) backend.recordAction("restored deleted prompt");
    }
    function cursorStep(dx) {
        const s = input.text;
        if (s.length === 0 || dx === 0)
            return;
        let pos = input.cursorPosition;
        if (dx < 0) {
            if (pos > 0) {
                --pos;
                while (pos > 0 && s[pos - 1] === ' ') --pos;
                while (pos > 0 && s[pos - 1] !== ' ') --pos;
            }
        } else {
            if (pos < s.length) {
                while (pos < s.length && s[pos] !== ' ') ++pos;
                while (pos < s.length && s[pos] === ' ') ++pos;
            }
        }
        input.cursorPosition = pos;
        if (highlighting)
            input.select(highlightAnchor, pos);
    }
    function toggleHighlight() {
        if (input.text.length === 0)
            return false;
        highlighting = !highlighting;
        if (highlighting) {
            highlightAnchor = input.cursorPosition;
            if (highlightAnchor === input.text.length && highlightAnchor > 0) {
                cursorStep(-1);
                highlightAnchor = input.text.length;
                input.select(highlightAnchor, input.cursorPosition);
            }
            if (backend) backend.recordAction("highlighted text in composer");
        } else {
            input.deselect();
        }
        return true;
    }
    function openPicker() {
        pickerOpen = true;
        input.forceActiveFocus();
    }
    // Tab: put the suggested prompt in the field (Enter then sends it with the chosen model).
    function acceptSuggestion(s) {
        const t = s !== undefined && s.length > 0 ? s : suggestion;
        input.forceActiveFocus();
        if (t.length === 0 || input.text.length > 0)
            return false;
        input.text = t;
        input.cursorPosition = t.length;
        suggestionAccepted(t);
        return true;
    }
    function send(mods) {
        const t = input.text;
        if (t.trim().length === 0)
            return;
        if (mods & Qt.ControlModifier) root.steer(t);
        else if (mods & Qt.AltModifier) root.think(t);
        else root.ask(t);
        input.text = "";
    }
    // The input bar's rect in this item's coordinates (the predictor ring wraps it).
    function barRect() {
        return Qt.rect(bar.x, col.y + bar.y, bar.width, bar.height);
    }

    Column {
        id: col
        width: parent.width
        spacing: 14

        // Quick answer
        Item {
            width: parent.width
            height: root.hasQuick ? 40 : 0
            visible: root.hasQuick
            Text {
                id: qq
                anchors.verticalCenter: parent.verticalCenter
                text: root.hasQuick ? root.backend.quickQuestion : ""
                color: Theme.dim
                font.family: Theme.sans
                font.pixelSize: Theme.px(21)
                width: Math.min(implicitWidth, 420)
                elide: Text.ElideRight
            }
            Text {
                id: arrow
                anchors.left: qq.right
                anchors.leftMargin: 16
                anchors.verticalCenter: parent.verticalCenter
                text: "→"
                color: Theme.gold
                font.family: Theme.sans
                font.pixelSize: Theme.px(22)
            }
            Text {
                objectName: "quickAnswer"
                anchors.left: arrow.right
                anchors.leftMargin: 16
                anchors.right: qsec.left
                anchors.rightMargin: 20
                anchors.verticalCenter: parent.verticalCenter
                readonly property bool waiting: root.hasQuick && root.backend.quickBusy && root.backend.quickAnswer.length === 0
                readonly property bool thinking: waiting && root.backend.quickThinking.length > 0
                // Before the first answer token: live thinking text (claude thinking_delta, codex
                // reasoning summaries, agy thinking when it sends any), newest words, dim #6e665e.
                // With no thinking text yet, the phase says what the model is doing. The first
                // answer token replaces it all.
                text: !root.hasQuick ? "" : thinking ? root.backend.quickThinking
                                      : waiting ? root.phaseLabel()
                                      : root.backend.quickAnswer
                color: waiting ? "#6e665e" : Theme.ink
                font.italic: waiting && !thinking
                elide: thinking ? Text.ElideLeft : Text.ElideRight
                font.family: Theme.sans
                font.pixelSize: Theme.px(23)
                font.weight: Font.Medium
            }
            Text {
                id: qsec
                objectName: "quickSeconds"
                anchors.right: parent.right
                anchors.verticalCenter: parent.verticalCenter
                // Live seconds while the model works (a 10 Hz tick), the final time once it is done.
                text: !root.hasQuick ? ""
                      : root.backend.quickBusy ? root.backend.quickElapsed.toFixed(1) + "s"
                      : root.backend.quickSeconds > 0
                        ? (root.backend.quickThinking.length > 0 || root.backend.quickThinkingTokens > 0 ? "✦ " : "")
                          + root.backend.quickSeconds.toFixed(1) + "s"
                      : ""
                color: root.hasQuick && root.backend.quickBusy ? Theme.dim : Theme.faint
                font.family: Theme.mono
                font.pixelSize: Theme.px(16)
            }
        }

        // Thinking trace: streams dim, then collapses to one line.
        Item {
            id: trace
            width: parent.width
            visible: root.hasTrace
            height: !root.hasTrace ? 0 : (root.streaming ? 168 : 82)
            Behavior on height { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }
            clip: true

            Text {
                id: tq
                y: 0
                text: root.hasTrace ? root.backend.traceQuestion : ""
                color: Theme.dim
                font.family: Theme.sans
                font.pixelSize: Theme.px(21)
                width: Math.min(implicitWidth, parent.width * 0.5)
                elide: Text.ElideRight
            }
            Text {
                objectName: "traceHeader"
                anchors.left: tq.right
                anchors.leftMargin: 18
                anchors.baseline: tq.baseline
                text: root.streaming ? "<think>" : "✦ Thought " + (root.hasTrace ? root.backend.traceSeconds.toFixed(1) : "0") + "s"
                color: root.streaming ? Theme.dim : Theme.copper
                font.family: Theme.mono
                font.pixelSize: Theme.px(17)
            }
            // Live trace, last lines only.
            Item {
                id: traceBody
                y: 38
                width: parent.width
                height: 120
                clip: true
                visible: root.streaming  // gone the moment it collapses, so it never overlaps the answer
                Text {
                    objectName: "traceText"
                    width: parent.width
                    y: Math.min(0, parent.height - implicitHeight)  // newest lines stay in view
                    text: root.hasTrace ? root.backend.traceThinking : ""
                    color: "#6e665e"
                    wrapMode: Text.Wrap
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(18)
                    lineHeight: 1.25
                }
            }
            Text {
                objectName: "traceAnswer"
                y: 40
                width: parent.width
                opacity: root.streaming ? 0 : 1
                Behavior on opacity { NumberAnimation { duration: 200 } }
                text: root.hasTrace ? root.backend.traceAnswer : ""
                color: Theme.ink
                elide: Text.ElideRight
                font.family: Theme.sans
                font.pixelSize: Theme.px(24)
                font.weight: Font.Medium
            }
        }

        // Inline Model & Effort strip (Cmd+M, or Up / k (vim) / w (wasd) in an empty composer).
        // Up/Down (w/s, k/j) cycle the model across all three providers; Left/Right (a/d, h/l)
        // cycle the efforts that model accepts; Enter sends with them; Esc closes the strip only.
        Rectangle {
            id: strip
            objectName: "modelStrip"
            visible: root.pickerOpen
            width: parent.width
            height: root.pickerOpen ? 66 : 0
            radius: 20
            color: "#000000"
            border.width: 1.5
            border.color: Theme.gold

            Text {
                id: provTag
                objectName: "pickerProvider"
                x: 28
                anchors.verticalCenter: parent.verticalCenter
                text: root.providerLabel(root.backend ? root.backend.chatProvider : "")
                color: Theme.dim
                font.family: Theme.sans
                font.pixelSize: Theme.px(15)
                font.weight: Font.Medium
                font.letterSpacing: 1.4
                font.capitalization: Font.AllUppercase
            }
            Text {
                id: modelName
                objectName: "pickerModel"
                anchors.left: provTag.right
                anchors.leftMargin: 16
                anchors.verticalCenter: parent.verticalCenter
                text: root.backend ? root.backend.shortModelLabel(root.backend.chatModel) : ""
                color: Theme.gold
                font.family: Theme.sans
                font.pixelSize: Theme.px(24)
                font.weight: Font.DemiBold
                TapHandler { onTapped: if (root.backend) root.backend.cycleModel(1) }
                WheelHandler {
                    onWheel: (ev) => { if (root.backend) root.backend.cycleModel(ev.angleDelta.y > 0 ? -1 : 1); }
                }
            }
            // Honest speed class: "Fast", "Deep Think" (gemini-4-hydrogen), "Teach", "Frontier".
            Rectangle {
                id: speedTag
                objectName: "pickerSpeed"
                readonly property string speed: root.backend ? root.backend.modelSpeed(root.backend.chatModel) : ""
                readonly property bool deep: speed === "Deep Think"
                visible: speed.length > 0
                anchors.left: modelName.right
                anchors.leftMargin: 12
                anchors.verticalCenter: parent.verticalCenter
                width: visible ? speedText.implicitWidth + 18 : 0
                height: 26
                radius: 13
                color: "transparent"
                border.width: 1
                border.color: deep ? Theme.copper : Theme.faint
                Text {
                    id: speedText
                    objectName: "pickerSpeedText"
                    anchors.centerIn: parent
                    text: parent.speed
                    color: parent.deep ? Theme.copper : Theme.dim
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(13)
                }
            }
            Text {
                id: modelKeys
                anchors.left: speedTag.right
                anchors.leftMargin: 14
                anchors.verticalCenter: parent.verticalCenter
                text: root.navMode === "vim" ? "k/j" : root.navMode === "wasd" ? "w/s" : "↑↓"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: Theme.px(14)
            }
            Row {
                id: effortRow
                anchors.left: modelKeys.right
                anchors.leftMargin: 34
                anchors.verticalCenter: parent.verticalCenter
                spacing: 8
                Repeater {
                    model: root.backend ? root.backend.chatEfforts : []
                    Rectangle {
                        required property string modelData
                        objectName: "effortChip"
                        readonly property bool current: root.backend && root.backend.chatEffort === modelData
                        width: chipText.implicitWidth + 24
                        height: 32
                        radius: 16
                        color: current ? Theme.gold : "transparent"
                        border.width: current ? 0 : 1
                        border.color: Theme.faint
                        Text {
                            id: chipText
                            anchors.centerIn: parent
                            text: parent.modelData
                            color: parent.current ? "#000000" : Theme.dim
                            font.family: Theme.mono
                            font.pixelSize: Theme.px(15)
                        }
                        TapHandler { onTapped: if (root.backend) root.backend.chatEffort = parent.modelData }
                    }
                }
            }
            Text {
                anchors.left: effortRow.right
                anchors.leftMargin: 14
                anchors.verticalCenter: parent.verticalCenter
                text: root.navMode === "vim" ? "h/l" : root.navMode === "wasd" ? "a/d" : "←→"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: Theme.px(14)
            }
            Text {
                anchors.right: parent.right
                anchors.rightMargin: 28
                anchors.verticalCenter: parent.verticalCenter
                text: "⏎ send   esc close"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: Theme.px(14)
            }
        }

        // Input bar
        Rectangle {
            id: bar
            objectName: "composerBar"
            width: parent.width
            height: 72
            radius: 22
            color: Theme.card
            border.width: input.activeFocus ? 1 : 0
            border.color: Qt.rgba(0.878, 0.624, 0.243, 0.35)

            // "about regs: 128  ×": the next Enter asks about this parameter.
            Rectangle {
                id: scopeChip
                objectName: "scopeChip"
                visible: root.scope.length > 0
                x: 22
                anchors.verticalCenter: parent.verticalCenter
                width: visible ? scopeText.implicitWidth + 26 : 0
                height: 34
                radius: 17
                color: Qt.rgba(0.878, 0.624, 0.243, 0.14)
                border.width: 1
                border.color: Theme.gold
                Text {
                    id: scopeText
                    objectName: "scopeChipText"
                    anchors.centerIn: parent
                    text: "about " + root.scope + "  ×"
                    color: Theme.gold
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(15)
                }
                TapHandler { onTapped: root.scopeCleared() }
            }
            Text {
                objectName: "composerGhost"
                anchors.left: input.left
                anchors.right: input.right
                anchors.verticalCenter: parent.verticalCenter
                visible: input.text.length === 0 && !input.inputMethodComposing
                text: root.suggestion.length > 0 ? root.suggestion : "Ask"
                color: root.suggestion.length > 0 ? Qt.rgba(0.878, 0.624, 0.243, 0.55) : Theme.faint
                elide: Text.ElideRight
                font.family: Theme.sans
                font.pixelSize: Theme.px(23)
            }
            TextInput {
                id: input
                objectName: "composerInput"
                x: scopeChip.visible ? scopeChip.x + scopeChip.width + 14 : 30
                anchors.verticalCenter: parent.verticalCenter
                width: modelChip.x - x - 24
                color: Theme.ink
                selectionColor: Theme.copper
                selectedTextColor: Theme.bg
                clip: true
                font.family: Theme.sans
                font.pixelSize: Theme.px(23)
                cursorDelegate: Caret {}
                activeFocusOnTab: false
                onActiveFocusChanged: if (!activeFocus) root.pickerOpen = false
                Keys.priority: Keys.BeforeItem
                Keys.onPressed: (e) => {
                    const ctrl = (e.modifiers & Qt.ControlModifier) !== 0;
                    const plain = (e.modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier | Qt.ShiftModifier)) === 0;
                    if (ctrl && e.key === Qt.Key_M) {
                        root.pickerOpen = !root.pickerOpen;
                        e.accepted = true;
                        return;
                    }
                    if (e.key === Qt.Key_Tab) {
                        root.acceptSuggestion();
                        e.accepted = true;
                        return;
                    }
                    if (root.pickerOpen && root.backend) {
                        const k = e.key;
                        if (k === Qt.Key_Up || (plain && (k === Qt.Key_W || k === Qt.Key_K))) {
                            root.backend.cycleModel(-1);
                        } else if (k === Qt.Key_Down || (plain && (k === Qt.Key_S || k === Qt.Key_J))) {
                            root.backend.cycleModel(1);
                        } else if (k === Qt.Key_Left || (plain && (k === Qt.Key_A || k === Qt.Key_H))) {
                            root.backend.cycleEffort(-1);
                        } else if (k === Qt.Key_Right || (plain && (k === Qt.Key_D || k === Qt.Key_L))) {
                            root.backend.cycleEffort(1);
                        } else if (k === Qt.Key_Escape) {
                            root.pickerOpen = false;  // the composer keeps focus
                        } else if (k === Qt.Key_Return || k === Qt.Key_Enter) {
                            root.pickerOpen = false;
                            root.send(e.modifiers);
                        } else {
                            return;  // anything else types as usual with the strip open
                        }
                        e.accepted = true;
                        return;
                    }
                    if (text.length === 0 && (e.key === Qt.Key_Up
                            || (plain && root.navMode === "vim" && e.key === Qt.Key_K)
                            || (plain && root.navMode === "wasd" && e.key === Qt.Key_W))) {
                        root.pickerOpen = true;
                        e.accepted = true;
                        return;
                    }
                    if (e.key === Qt.Key_Return || e.key === Qt.Key_Enter) {
                        root.send(e.modifiers);
                        e.accepted = true;
                    } else if (e.key === Qt.Key_Escape) {
                        if (text.length > 0) {
                            text = "";
                            e.accepted = true;
                        } else if (root.scope.length > 0) {
                            root.scopeCleared();  // first Esc drops the parameter scope
                            e.accepted = true;
                        } else {
                            root.escaped();
                            e.accepted = false;
                        }
                    } else if ((e.modifiers & Qt.ControlModifier) && e.key >= Qt.Key_1 && e.key <= Qt.Key_9) {
                        e.accepted = false;
                    }
                }
            }
            // The model the next Enter goes to; click (or Cmd+M) opens the strip.
            Rectangle {
                id: modelChip
                objectName: "modelChip"
                anchors.right: hints.left
                anchors.rightMargin: 22
                anchors.verticalCenter: parent.verticalCenter
                width: chipLabel.implicitWidth + 24
                height: 32
                radius: 16
                color: "transparent"
                border.width: 1
                border.color: root.pickerOpen ? Theme.gold : Theme.line
                Text {
                    id: chipLabel
                    objectName: "modelChipLabel"
                    anchors.centerIn: parent
                    text: root.backend ? root.backend.shortModelLabel(root.backend.chatModel) + " · " + root.backend.chatEffort : ""
                    color: root.pickerOpen ? Theme.gold : Theme.dim
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(14)
                }
                TapHandler {
                    onTapped: {
                        root.pickerOpen = !root.pickerOpen;
                        input.forceActiveFocus();
                    }
                }
            }
            Text {
                id: hints
                anchors.right: parent.right
                anchors.rightMargin: 30
                anchors.verticalCenter: parent.verticalCenter
                text: (root.suggestion.length > 0 && input.text.length === 0 ? "⇥ accept   " : "")
                      + (root.hasSession ? "⏎ ask   ⌥⏎ think   ⌘⏎ steer" : "⏎ ask   ⌥⏎ think")
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: Theme.px(16)
            }
        }
    }

    NumberAnimation {
        id: restoreAnim
        target: root
        property: "restoreProgress"
        from: 0
        to: 1
        duration: 450
        onFinished: if (root.backHolding) root.restoreDeleted()
    }

    // Hold-○-to-restore circular hint below the bar when text was just cleared.
    Rectangle {
        id: restorePill
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: col.bottom
        anchors.topMargin: 6
        width: restoreRow.implicitWidth + 24
        height: 24
        radius: 12
        color: "#110e09"
        border.width: 1
        border.color: root.backHolding ? Theme.gold : Theme.line
        visible: root.deletedBuffer.length > 0 && input.text.length === 0
        Row {
            id: restoreRow
            anchors.centerIn: parent
            spacing: 8
            Canvas {
                width: 14; height: 14
                anchors.verticalCenter: parent.verticalCenter
                property real p: root.restoreProgress
                onPChanged: requestPaint()
                onPaint: {
                    const ctx = getContext("2d");
                    ctx.clearRect(0, 0, width, height);
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = "#3a352c";
                    ctx.beginPath();
                    ctx.arc(7, 7, 5, 0, Math.PI * 2);
                    ctx.stroke();
                    if (p > 0.01) {
                        ctx.strokeStyle = "#e09f3e";
                        ctx.beginPath();
                        ctx.arc(7, 7, 5, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * p);
                        ctx.stroke();
                    }
                }
            }
            Text {
                text: root.backHolding ? "Restoring…" : "Hold ○ to restore deleted text"
                color: root.backHolding ? Theme.gold : Theme.dim
                font.family: Theme.mono
                font.pixelSize: 11
            }
        }
        TapHandler { onTapped: root.restoreDeleted() }
    }
}
