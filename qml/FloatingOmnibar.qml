import QtQuick

// Prompt-from-anywhere: a floating ask card anchored under the hovered target, independent of
// the Stage and the Composer. Cmd+K, "/" (no text field focused), middle-click or Shift+Click on
// a hovered target open it. Chips show the hovered target, or every pinned one (Tab / "+ Pin");
// Enter streams an answer from the fast model through backend.askStreamWithSlice with those
// targets as the Tier 3 slice. The header drags; "Pin card" detaches the card into a persistent
// answer card (any number stream at once); Esc closes and cancels the card's stream.
Item {
    id: root
    objectName: "floatingOmnibar"
    property QtObject backend
    readonly property QtObject targets: backend ? backend.targets : null
    // Fallbacks when no chip names a session / artifact (StudioApp binds them).
    property string projectKey: ""
    property int artifactIndex: -1
    property bool mouseMode: false
    property bool suppressed: false  // a modal overlay is up
    // Controller mode: Ask / + Pin / × carry the pad button that does the same (✕ / R1 / ○).
    property bool padHints: false
    property var padGlyphs: ({})
    // L2 is held and Phonon is listening (StudioApp binds it to gamepad.voiceActive).
    property bool voiceListening: false
    onSuppressedChanged: if (suppressed && open) close()  // a modal over it takes the pad and keys

    property bool open: false
    property int cardSeq: 0
    property string cardId: ""
    readonly property string streamId: cardId.length > 0 ? "omni:" + cardId : ""
    // The hovered target when the card opened (the pointer leaves it to reach the card).
    property var snapshot: []
    readonly property var chips: targets && targets.pinnedCount > 0 ? targets.pinnedTargets : snapshot
    readonly property bool canPin: {
        if (!targets)
            return false;
        const n = targets.pinnedCount;  // re-evaluate on every pin / unpin
        if (targets.hasActiveTarget)
            return !targets.isPinned(targets.activeTarget);
        return snapshot.length > 0 && n === 0;
    }
    property string question: ""
    property string answer: ""
    property string thinking: ""
    property string error: ""
    property bool busy: false
    // What the last send dispatched (tests and the snapshot read these).
    property string lastSlice: ""
    property string lastProject: ""
    property int lastArtifact: -1
    property int sends: 0
    signal closed()

    readonly property real margin: 16
    function clampX(x) { return Math.max(margin, Math.min(width - card.width - margin, x)); }
    function clampY(y) { return Math.max(margin, Math.min(height - card.height - margin, y)); }

    // Opens the card under the active target (x, y + h + 10), else at (px, py), else top centre.
    function openAt(px, py) {
        if (!backend || suppressed)
            return;
        const t = targets && targets.hasActiveTarget ? targets.activeTarget : null;
        snapshot = t ? [t] : [];
        if (!open || busy) {
            // A fresh card (a still-streaming one keeps going only if pinned; otherwise restart).
            if (busy && streamId.length > 0)
                backend.cancelStream(streamId);
            cardSeq += 1;
            cardId = String(cardSeq);
            question = "";
            answer = "";
            thinking = "";
            error = "";
            busy = false;
        }
        open = true;
        const at = anchorFor(t, px, py);
        card.x = clampX(at.x);
        card.y = clampY(at.y);
        input.forceActiveFocus();
    }
    // L2 (Phonon push-to-talk): a fresh centred card when closed; when already open (a previous
    // turn's question and answer still showing) just take focus back, so the next dictation
    // lands in the prompt and the multi-turn context stays on screen.
    function focusForVoice() {
        if (suppressed || !backend)
            return;
        if (!open) {
            // Snapshot the target (its chip names what the question is about), but sit centred
            // upper third like a command bar rather than off to the side of the island.
            openAt(-1, -1);
            card.x = clampX(Math.round((width - card.width) / 2));
            card.y = clampY(Math.max(margin, Math.round(height * 0.16)));
            return;
        }
        if (targets && targets.hasActiveTarget && targets.pinnedCount === 0)
            snapshot = [targets.activeTarget];
        card.x = clampX(card.x);
        card.y = clampY(card.y);
        input.forceActiveFocus();
        input.cursorPosition = input.text.length;
    }
    // Right beside the highlighted target: its right side when the card fits there, else its
    // left, else under it, else over it, else tucked inside its bottom-right corner (a target as
    // big as the window). No target: the point that was clicked, else upper centre (Raycast).
    function anchorFor(t, px, py) {
        const gap = 14;
        const cw = card.width, ch = Math.max(card.height, 160);
        if (t && t.w > 0 && t.h > 0) {
            const top = Math.max(margin, Math.min(height - ch - margin, t.y));
            if (t.x + t.w + gap + cw <= width - margin)
                return { x: t.x + t.w + gap, y: top };
            if (t.x - gap - cw >= margin)
                return { x: t.x - gap - cw, y: top };
            if (t.y + t.h + gap + ch <= height - margin)
                return { x: t.x, y: t.y + t.h + gap };
            if (t.y - gap - ch >= margin)
                return { x: t.x, y: t.y - gap - ch };
            return { x: t.x + t.w - cw - 24, y: t.y + t.h - ch - 24 };
        }
        if (px !== undefined && px >= 0)
            return { x: px, y: py + 10 };
        return { x: (width - cw) / 2, y: Math.max(margin, Math.round(height * 0.16)) };
    }
    // Send what is typed (✕ / R2 on the pad, Enter on the keyboard).
    function submit() { ask(input.text); }
    // A model's [[ACTION:..]] tags (anywhere in the reply) ran in the backend; they never show as
    // text. A tag between two runs of prose becomes a paragraph break (a CUA follow-up's reply
    // reads as its own paragraph); streaming, a tag still arriving at the tail is cut so it never
    // flashes. [[IGNORE]] shows nothing. Mirrors answerDisplayText() in backend.cpp.
    function displayAnswer(a, streaming) {
        let s = String(a);
        if (s.indexOf("[[IGNORE]]") >= 0)
            return "";
        if (streaming !== false) {
            const open = s.lastIndexOf("[[");
            if (open >= 0 && s.indexOf("]]", open) < 0)
                s = s.substring(0, open);
            else if (s.endsWith("["))
                s = s.substring(0, s.length - 1);
        }
        const tag = /\[\[ACTION:[^\]]*\]\]/g;
        let out = "", last = 0, m;
        while ((m = tag.exec(s)) !== null) {
            out += s.substring(last, m.index);
            last = m.index + m[0].length;
            const head = out.replace(/[ \t]+$/, "");
            const next = last < s.length ? s.charAt(last) : "\n";
            const space = /\s/.test(next);
            if (head.length > 0 && !head.endsWith("\n") && !space && next !== "[")
                out = head + "\n\n";
            else if (space || /[!-\/:-@\[-`{-~]/.test(next))
                out = head;
        }
        out += s.substring(last);
        return out.replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
    }
    function toggle() {
        if (open)
            close();
        else
            openAt(-1, -1);
    }
    function close() {
        if (!open)
            return;
        if (busy && streamId.length > 0)
            backend.cancelStream(streamId);
        busy = false;
        open = false;
        input.text = "";
        closed();
    }
    function pinActive() {
        if (!targets)
            return;
        if (targets.hasActiveTarget)
            targets.pinActiveTarget();
        else if (snapshot.length > 0)
            targets.pinTarget(snapshot[0]);
    }
    function removeChip(i) {
        if (targets && targets.pinnedCount > 0)
            targets.unpinTarget(i);
        else
            snapshot = [];
    }
    function ask(text) {
        const q = String(text).trim();
        if (!backend || q.length === 0 || streamId.length === 0)
            return;
        const list = chips;
        const slice = !targets ? "" : (targets.pinnedCount > 0 ? targets.buildCombinedSlice(4000)
                                                                : targets.buildSliceFor(snapshot, 4000));
        const first = list.length > 0 ? list[0] : null;
        const pk = first && first.sessionId ? first.sessionId : projectKey;
        const aid = first ? parseInt(first.artifactId) : NaN;
        const ai = !isNaN(aid) ? aid : artifactIndex;
        question = q;
        answer = "";
        thinking = "";
        error = "";
        busy = true;
        lastSlice = slice;
        lastProject = pk;
        lastArtifact = ai;
        sends += 1;
        input.text = "";
        backend.askStreamWithSlice(streamId, pk, ai, slice, q);  // may finish synchronously (mock)
    }
    // Detach this card into a persistent answer card; its stream keeps flowing into it.
    function pinCard() {
        if (!open || (question.length === 0 && answer.length === 0))
            return;
        const names = [];
        for (let i = 0; i < chips.length; ++i)
            names.push(chips[i].label || chips[i].kind);
        pinnedCards.append({ streamId: streamId, question: question, answer: answer, thinking: thinking,
                             busy: busy, error: error, chipsLabel: names.join(" · "), px: card.x, py: card.y });
        busy = false;   // the pinned card owns the stream now: closing must not cancel it
        cardId = "";
        open = false;
        input.text = "";
        closed();
    }
    function closePinned(i) {
        if (i < 0 || i >= pinnedCards.count)
            return;
        const c = pinnedCards.get(i);
        if (c.busy)
            backend.cancelStream(c.streamId);
        pinnedCards.remove(i);
    }
    function pinnedIndex(id) {
        for (let i = 0; i < pinnedCards.count; ++i)
            if (pinnedCards.get(i).streamId === id)
                return i;
        return -1;
    }
    readonly property int pinnedCardCount: pinnedCards.count
    ListModel { id: pinnedCards }

    Connections {
        target: root.backend
        function onStreamDelta(id, text, think) {
            if (id === root.streamId && root.streamId.length > 0) {
                root.answer += text;
                root.thinking += think;
                return;
            }
            const i = root.pinnedIndex(id);
            if (i >= 0) {
                const c = pinnedCards.get(i);
                pinnedCards.setProperty(i, "answer", c.answer + text);
                pinnedCards.setProperty(i, "thinking", c.thinking + think);
            }
        }
        function onStreamFinished(id, err) {
            if (id === root.streamId && root.streamId.length > 0) {
                root.busy = false;
                root.error = err;
                if (root.open)
                    input.forceActiveFocus();  // never strand keyboard / Phonon focus after a turn
                return;
            }
            const i = root.pinnedIndex(id);
            if (i >= 0) {
                pinnedCards.setProperty(i, "busy", false);
                pinnedCards.setProperty(i, "error", err);
            }
        }
    }

    // --- Hover pill: what the pointer is on, and how to ask about it (mouse mode only) ---------
    Rectangle {
        id: pill
        objectName: "hoverPill"
        readonly property var t: root.targets && root.targets.hasActiveTarget ? root.targets.activeTarget : null
        visible: root.mouseMode && !root.open && !root.suppressed && t !== null
        width: pillRow.implicitWidth + 22
        height: 28
        radius: 14
        color: "#0d0c0b"
        border.width: 1
        border.color: Theme.goldA(0.35)
        // Small targets (a cell, a line, a point): beside it, right if it fits, else left, so it
        // never covers the next row. Large ones (an island, a turn): just below its left edge.
        readonly property bool beside: t !== null && t.h <= 64
        readonly property bool fitsRight: t !== null && t.x + t.w + 10 + width <= root.width - root.margin
        x: !t ? 0 : Math.max(root.margin, Math.min(root.width - width - root.margin,
                     beside ? (fitsRight ? t.x + t.w + 10 : t.x - width - 10) : t.x))
        y: !t ? 0 : Math.max(root.margin, Math.min(root.height - height - root.margin,
                     beside ? t.y + (t.h - height) / 2 : t.y + t.h + 6))
        Row {
            id: pillRow
            anchors.centerIn: parent
            spacing: 10
            Text {
                text: pill.t ? pill.t.label : ""
                width: Math.min(implicitWidth, 300)
                elide: Text.ElideRight
                color: Theme.ink
                font.family: Theme.mono
                font.pixelSize: 13
            }
            Text {
                text: "⌘K ask · Tab pin"
                color: Theme.dim
                font.family: Theme.mono
                font.pixelSize: 13
            }
        }
    }

    // --- While the card is open, a thin ring marks every target its chips refer to --------------
    Repeater {
        model: root.open ? root.chips.length : 0
        Rectangle {
            objectName: "targetRing"
            required property int index
            readonly property var t: root.chips[index] || ({ x: 0, y: 0, w: 0, h: 0 })
            visible: t.w > 0 && t.h > 0
            // Wider than the target on the left (cell text starts at its edge), inset vertically
            // so rings on adjacent rows never touch.
            x: t.x - 10
            y: t.y + 3
            width: t.w + 10
            height: Math.max(8, t.h - 6)
            radius: 6
            color: "transparent"
            border.width: 1.5
            border.color: Theme.goldA(0.6)
        }
    }

    // --- Pinned answer cards ---------------------------------------------------------------------
    Repeater {
        model: pinnedCards
        delegate: Rectangle {
            id: pc
            objectName: "pinnedAskCard"
            required property int index
            required property string streamId
            required property string question
            required property string answer
            required property string thinking
            required property bool busy
            required property string error
            required property string chipsLabel
            required property real px
            required property real py
            x: px
            y: py
            width: Math.min(460, root.width - 2 * root.margin)
            height: pcCol.implicitHeight + 28
            radius: 16
            color: Theme.card
            border.width: 1
            border.color: Theme.goldA(0.25)
            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                acceptedButtons: Qt.AllButtons
            }
            Column {
                id: pcCol
                x: 16
                y: 12
                width: parent.width - 32
                spacing: 8
                Item {
                    width: parent.width
                    height: 22
                    DragHandler {
                        target: pc
                        xAxis.minimum: root.margin
                        xAxis.maximum: root.width - pc.width - root.margin
                        yAxis.minimum: root.margin
                        yAxis.maximum: root.height - pc.height - root.margin
                    }
                    Text {
                        anchors.left: parent.left
                        anchors.right: pcClose.left
                        anchors.rightMargin: 8
                        anchors.verticalCenter: parent.verticalCenter
                        text: pc.chipsLabel.length > 0 ? pc.chipsLabel : "Pinned answer"
                        elide: Text.ElideRight
                        color: Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 13
                    }
                    Text {
                        id: pcClose
                        objectName: "pinnedAskClose"
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        text: "×"
                        color: Theme.dim
                        font.family: Theme.sans
                        font.pixelSize: 20
                        Click { onClicked: root.closePinned(pc.index) }
                    }
                }
                AnswerBody {
                    width: parent.width
                    backend: root.backend
                    question: pc.question
                    answer: pc.answer
                    thinking: pc.thinking
                    busy: pc.busy
                    error: pc.error
                    maxH: 240
                }
            }
        }
    }

    // --- The live ask card -------------------------------------------------------------------------
    // A command bar: a quiet header (Lapis, Pin, one close), the target chips, the prompt with
    // its pad hints inside it, and the answer drawer under a hairline.
    Rectangle {
        id: card
        objectName: "omnibarCard"
        visible: root.open
        width: Math.min(620, root.width - 2 * root.margin)
        height: col.implicitHeight + 32
        radius: 16
        color: Theme.card
        border.width: 1
        border.color: Theme.goldA(root.voiceListening ? 0.75 : 0.38)
        onHeightChanged: if (root.open) y = root.clampY(y)
        Behavior on border.color { ColorAnimation { duration: 140 } }

        // Swallow clicks and hover on the card so they never reach the lens underneath.
        MouseArea {
            anchors.fill: parent
            hoverEnabled: true
            acceptedButtons: Qt.AllButtons
        }
        // The inner top highlight: a hairline of light along the card's upper edge.
        Rectangle {
            x: 14
            y: 1
            width: parent.width - 28
            height: 1
            color: Theme.goldA(0.08)
        }

        Column {
            id: col
            x: 18
            y: 16
            width: parent.width - 36
            spacing: 12

            // Header: drag handle, title, Pin, one close control.
            Item {
                objectName: "omnibarHeader"
                width: parent.width
                height: 24
                DragHandler {
                    id: drag
                    target: card
                    xAxis.minimum: root.margin
                    xAxis.maximum: root.width - card.width - root.margin
                    yAxis.minimum: root.margin
                    yAxis.maximum: root.height - card.height - root.margin
                }
                Row {
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 9
                    Rectangle {
                        width: 7; height: 7; radius: 3.5
                        anchors.verticalCenter: parent.verticalCenter
                        color: Theme.gold
                    }
                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        text: "Lapis"
                        color: Theme.ink
                        font.family: Theme.sans
                        font.pixelSize: Theme.px(14)
                        font.weight: Font.DemiBold
                    }
                }
                Row {
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 16
                    Text {
                        objectName: "omnibarPinCard"
                        anchors.verticalCenter: parent.verticalCenter
                        text: "Pin"
                        visible: root.question.length > 0
                        color: Theme.gold
                        font.family: Theme.mono
                        font.pixelSize: 12
                        Click { onClicked: root.pinCard() }
                    }
                    // One close: the pad's back button when it is bound, else a quiet ×.
                    PadBadge {
                        id: closeBadge
                        objectName: "omnibarClose"
                        anchors.verticalCenter: parent.verticalCenter
                        action: "back"
                        Click { onClicked: root.close() }
                    }
                    Text {
                        objectName: "omnibarClose"
                        anchors.verticalCenter: parent.verticalCenter
                        visible: !closeBadge.visible
                        text: "×"
                        color: Theme.dim
                        font.family: Theme.sans
                        font.pixelSize: 19
                        Click { onClicked: root.close() }
                    }
                }
            }

            // Target chips: the hovered target, or every pinned one.
            Flow {
                objectName: "omnibarChips"
                width: parent.width
                spacing: 8
                visible: root.chips.length > 0 || root.canPin
                Repeater {
                    model: root.chips.length
                    Rectangle {
                        objectName: "omnibarChip"
                        required property int index
                        readonly property var t: root.chips[index] || ({ label: "", kind: "" })
                        readonly property bool pinned: root.targets && root.targets.pinnedCount > 0
                        height: 26
                        width: chipRow.implicitWidth + 20
                        radius: 13
                        color: pinned ? Theme.goldA(0.10) : Theme.cardHi
                        border.width: 1
                        border.color: pinned ? Theme.goldA(0.6) : Theme.line
                        Row {
                            id: chipRow
                            anchors.centerIn: parent
                            spacing: 7
                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: String(parent.parent.t.kind).replace("_", " ")
                                color: Theme.dim
                                font.family: Theme.mono
                                font.pixelSize: 12
                            }
                            Text {
                                objectName: "omnibarChipLabel"
                                anchors.verticalCenter: parent.verticalCenter
                                text: parent.parent.t.label || parent.parent.t.path || ""
                                width: Math.min(implicitWidth, 240)
                                elide: Text.ElideRight
                                color: Theme.text
                                font.family: Theme.mono
                                font.pixelSize: 12
                            }
                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                text: "×"
                                color: Theme.dim
                                font.family: Theme.sans
                                font.pixelSize: 14
                                Click { onClicked: root.removeChip(index) }
                            }
                        }
                    }
                }
                Rectangle {
                    objectName: "omnibarPinChip"
                    visible: root.canPin
                    height: 26
                    width: pinRow.implicitWidth + 20
                    radius: 13
                    color: "transparent"
                    border.width: 1
                    border.color: Theme.line
                    Row {
                        id: pinRow
                        anchors.centerIn: parent
                        spacing: 7
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            text: "+ Pin"
                            color: Theme.dim
                            font.family: Theme.mono
                            font.pixelSize: 12
                        }
                        PadBadge {
                            id: pinBadge
                            anchors.verticalCenter: parent.verticalCenter
                            action: "accept_prediction"
                            size: 15
                        }
                        Text {
                            anchors.verticalCenter: parent.verticalCenter
                            visible: !pinBadge.visible
                            text: "⇥"
                            color: Theme.faint
                            font.family: Theme.mono
                            font.pixelSize: 12
                        }
                    }
                    Click { onClicked: root.pinActive() }
                }
            }

            // The prompt: an inset field with its live hint on the right.
            Rectangle {
                width: parent.width
                height: Theme.px(46)
                radius: 11
                color: "#050504"
                border.width: 1
                border.color: root.voiceListening ? Theme.gold : (input.activeFocus ? Theme.goldA(0.5) : Theme.line)
                // "● Listening…" while L2 is held, else the pad keys: [L2] voice · [R2] send.
                Row {
                    id: voiceHint
                    objectName: "omnibarVoiceHint"
                    readonly property string voiceKey: root.padGlyphs["voice_hold"] || "L2"
                    readonly property string sendKey: root.padGlyphs["send"] || "R2"
                    readonly property bool typed: input.text.length > 0
                    // What the hint says, as text (tests and accessibility read it).
                    readonly property string text: root.voiceListening ? "● Listening…"
                                                 : (typed ? sendKey + " send" : voiceKey + " voice · " + sendKey + " send")
                    anchors.right: parent.right
                    anchors.rightMargin: 14
                    anchors.verticalCenter: parent.verticalCenter
                    visible: root.voiceListening || root.padHints
                    spacing: 6
                    Rectangle {
                        id: listenDot
                        visible: root.voiceListening
                        width: 7; height: 7; radius: 3.5
                        anchors.verticalCenter: parent.verticalCenter
                        color: Theme.gold
                        SequentialAnimation on opacity {
                            running: root.voiceListening
                            loops: Animation.Infinite
                            onStopped: listenDot.opacity = 1
                            NumberAnimation { from: 1; to: 0.35; duration: 600; easing.type: Easing.InOutSine }
                            NumberAnimation { from: 0.35; to: 1; duration: 600; easing.type: Easing.InOutSine }
                        }
                    }
                    Text {
                        visible: root.voiceListening
                        anchors.verticalCenter: parent.verticalCenter
                        text: "Listening…"
                        color: Theme.gold
                        font.family: Theme.mono
                        font.pixelSize: 12
                    }
                    PadIcon {
                        visible: !root.voiceListening && !voiceHint.typed
                        anchors.verticalCenter: parent.verticalCenter
                        button: voiceHint.voiceKey
                        size: 17
                    }
                    Text {
                        visible: !root.voiceListening && !voiceHint.typed
                        anchors.verticalCenter: parent.verticalCenter
                        text: "voice"
                        color: Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 12
                    }
                    Item {
                        visible: !root.voiceListening && !voiceHint.typed
                        width: 6
                        height: 1
                    }
                    PadIcon {
                        visible: !root.voiceListening
                        anchors.verticalCenter: parent.verticalCenter
                        button: voiceHint.sendKey
                        size: 17
                        active: voiceHint.typed
                    }
                    Text {
                        visible: !root.voiceListening
                        anchors.verticalCenter: parent.verticalCenter
                        text: "send"
                        color: voiceHint.typed ? Theme.ink : Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 12
                    }
                }
                TextInput {
                    id: input
                    objectName: "omnibarInput"
                    anchors.fill: parent
                    anchors.leftMargin: 16
                    anchors.rightMargin: voiceHint.visible ? voiceHint.implicitWidth + 28 : 16
                    verticalAlignment: TextInput.AlignVCenter
                    clip: true
                    color: Theme.ink
                    selectionColor: Theme.copper
                    font.family: Theme.sans
                    font.pixelSize: Theme.px(18)
                    cursorDelegate: Caret {}
                    Keys.priority: Keys.BeforeItem
                    Keys.onPressed: (e) => {
                        if (e.key === Qt.Key_Return || e.key === Qt.Key_Enter) {
                            root.ask(text);
                            e.accepted = true;
                        } else if (e.key === Qt.Key_Escape) {
                            root.close();
                            e.accepted = true;
                        } else if (e.key === Qt.Key_Tab) {
                            root.pinActive();
                            e.accepted = true;
                        } else if (e.key === Qt.Key_Backspace && text.length === 0 && root.targets
                                   && root.targets.pinnedCount > 0) {
                            root.targets.unpinTarget(root.targets.pinnedCount - 1);
                            e.accepted = true;
                        } else if (e.key === Qt.Key_Up || e.key === Qt.Key_Down || e.key === Qt.Key_Backtab) {
                            e.accepted = true;  // never move the lens underneath while typing
                        }
                    }
                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        visible: input.text.length === 0
                        text: root.chips.length > 1 ? "Ask about these " + root.chips.length + " targets…"
                                                    : (root.chips.length === 1 ? "Ask about this…" : "Ask anything…")
                        color: Theme.muted
                        font: input.font
                    }
                }
            }

            // The answer drawer, under a hairline.
            Column {
                width: parent.width
                spacing: 12
                visible: root.question.length > 0
                Rectangle {
                    width: parent.width
                    height: 1
                    color: Theme.line
                }
                AnswerBody {
                    objectName: "omnibarAnswerBody"
                    width: parent.width
                    backend: root.backend
                    question: root.question
                    answer: root.answer
                    thinking: root.thinking
                    busy: root.busy
                    error: root.error
                    maxH: 300
                }
            }

            // Keyboard / mouse only, and only before the first ask.
            Text {
                visible: !root.padHints && root.question.length === 0
                text: "⏎ ask · ⇥ pin · esc"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 12
            }
        }
    }

    // The bound pad button for an action, as its vector icon (controller mode only).
    component PadBadge: PadIcon {
        id: badge
        property string action: ""
        readonly property string glyph: root.padGlyphs[action] || ""
        objectName: "omnibarPadBadge"
        visible: root.padHints && glyph.length > 0
        button: glyph
        size: 18
    }
    // A small button's hit area (a little larger than its glyph).
    component Click: MouseArea {
        anchors.fill: parent
        anchors.margins: -6
        cursorShape: Qt.PointingHandCursor
    }

    // Question, the streaming answer (scrolls, follows the tail) and a quiet thinking indicator.
    component AnswerBody: Column {
        id: ab
        property QtObject backend
        property string question
        property string answer
        property string thinking
        property bool busy: false
        property string error
        property real maxH: 300
        // The reply as shown: harness tags removed, follow-up turns as their own paragraphs.
        readonly property string shownAnswer: root.displayAnswer(answer, busy)
        spacing: 8
        Text {
            width: ab.width
            text: "› " + ab.question
            wrapMode: Text.Wrap
            maximumLineCount: 3
            elide: Text.ElideRight
            color: Theme.gold
            font.family: Theme.sans
            font.pixelSize: Theme.px(14)
            font.weight: Font.Medium
        }
        Row {
            objectName: "omnibarThinking"
            visible: ab.busy && ab.shownAnswer.length === 0
            spacing: 8
            Rectangle {
                id: thinkDot
                width: 7; height: 7; radius: 3.5
                anchors.verticalCenter: parent.verticalCenter
                color: Theme.copper
                SequentialAnimation on opacity {
                    running: thinkDot.visible
                    loops: Animation.Infinite
                    NumberAnimation { from: 1; to: 0.25; duration: 520; easing.type: Easing.InOutSine }
                    NumberAnimation { from: 0.25; to: 1; duration: 520; easing.type: Easing.InOutSine }
                }
            }
            Text {
                text: ab.thinking.length > 0 ? "thinking · " + Math.max(1, Math.round(ab.thinking.length / 4)) + " tok"
                                             : "waiting for first token"
                color: Theme.dim
                font.family: Theme.mono
                font.pixelSize: 13
            }
        }
        Flickable {
            id: answerFlick
            width: ab.width
            height: Math.min(ab.maxH, answerText.implicitHeight)
            visible: ab.shownAnswer.length > 0
            contentHeight: answerText.implicitHeight
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            onContentHeightChanged: if (ab.busy) contentY = Math.max(0, contentHeight - height)
            // Markdown rendered as rich text (code blocks highlighted on a dark block, inline code
            // in mono); line spacing comes from the HTML so code lines stay tight.
            Text {
                id: answerText
                objectName: "omnibarAnswer"
                width: answerFlick.width
                text: ab.backend ? ab.backend.markdownHtml(ab.shownAnswer, Theme.px(14)) : ab.shownAnswer
                textFormat: ab.backend ? Text.RichText : Text.PlainText
                wrapMode: Text.Wrap
                color: Theme.text
                linkColor: Theme.gold
                font.family: Theme.sans
                font.pixelSize: Theme.px(16)
                onLinkActivated: link => Qt.openUrlExternally(link)
            }
        }
        Text {
            visible: ab.error.length > 0
            width: ab.width
            text: ab.error
            wrapMode: Text.Wrap
            color: Theme.copper
            font.family: Theme.mono
            font.pixelSize: 13
        }
    }
}
