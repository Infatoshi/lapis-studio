import QtQuick

// Fortnite-style DualShock 4 / Gamepad Mapping screen.
// Left: interactive DualShock 4 schematic highlighting the selected input.
// Right: remappable slots. Left/Right cycles action; Enter/Cross enters live button capture.
FocusScope {
    id: root
    objectName: "controllerMapOverlay"
    property QtObject backend
    property bool open: false
    property int row: 0
    signal closed()

    readonly property QtObject gamepad: backend ? backend.gamepad : null
    readonly property var slots: gamepad ? gamepad.bindingSlots : []
    readonly property var actions: gamepad ? gamepad.availableActions : []
    readonly property bool capturing: gamepad ? gamepad.captureMode : false
    readonly property string selectedId: row >= 0 && row < slots.length ? slots[row].id : "l2"
    readonly property string selectedActionLabel: row >= 0 && row < slots.length ? slots[row].actionLabel : ""

    function show() {
        row = 0;
        open = true;
        if (gamepad)
            gamepad.captureMode = false;
        forceActiveFocus();
    }

    function hide() {
        if (!open)
            return;
        if (gamepad)
            gamepad.captureMode = false;
        open = false;
        closed();
    }

    function cycleAction(delta) {
        if (!backend || !gamepad || slots.length === 0 || actions.length === 0)
            return;
        const slot = slots[row];
        let idx = 0;
        for (let i = 0; i < actions.length; ++i) {
            if (actions[i].id === slot.action) {
                idx = i;
                break;
            }
        }
        const nextIdx = (idx + delta + actions.length) % actions.length;
        backend.setGamepadBinding(slot.id, actions[nextIdx].id);
    }

    function startCapture() {
        if (!gamepad || slots.length === 0)
            return;
        gamepad.captureMode = true;
    }

    function selectSlotById(id) {
        for (let i = 0; i < slots.length; ++i) {
            if (slots[i].id === id) {
                row = i;
                return;
            }
        }
    }

    Connections {
        target: root.gamepad
        function onRawInputCaptured(inputId) {
            if (!root.open || !root.backend || root.slots.length === 0)
                return;
            const currentSlot = root.slots[root.row];
            const targetAction = currentSlot.action;
            const oldActionOnCaptured = root.gamepad.actionFor(inputId);
            // Swap cleanly if the pressed input already held a different action.
            root.backend.setGamepadBinding(inputId, targetAction);
            if (inputId !== currentSlot.id)
                root.backend.setGamepadBinding(currentSlot.id, oldActionOnCaptured);
            root.selectSlotById(inputId);
        }
    }

    visible: opacity > 0.01
    opacity: open ? 1 : 0
    Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
    enabled: open

    Keys.onPressed: (e) => {
        if (capturing) {
            if (e.key === Qt.Key_Escape && gamepad)
                gamepad.captureMode = false;
            e.accepted = true;
            return;
        }
        const k = e.key;
        if (k === Qt.Key_Escape || k === Qt.Key_Q) {
            hide();
        } else if (k === Qt.Key_Down || k === Qt.Key_J || k === Qt.Key_S || k === Qt.Key_Tab) {
            if (slots.length > 0) row = (row + 1) % slots.length;
        } else if (k === Qt.Key_Up || k === Qt.Key_K || k === Qt.Key_W || k === Qt.Key_Backtab) {
            if (slots.length > 0) row = (row - 1 + slots.length) % slots.length;
        } else if (k === Qt.Key_Right || k === Qt.Key_L || k === Qt.Key_D) {
            cycleAction(1);
        } else if (k === Qt.Key_Left || k === Qt.Key_H || k === Qt.Key_A) {
            cycleAction(-1);
        } else if (k === Qt.Key_Return || k === Qt.Key_Enter) {
            startCapture();
        } else if (k === Qt.Key_R) {
            if (backend) backend.resetGamepadBindings();
        }
        e.accepted = true;
    }

    Rectangle {
        anchors.fill: parent
        color: "#000000"
        opacity: 0.92
        MouseArea { anchors.fill: parent; onClicked: root.hide(); onWheel: (w) => w.accepted = true }
    }

    Rectangle {
        anchors.centerIn: parent
        width: 1080
        height: 700
        radius: 26
        color: "#0b0a08"
        border.width: 1
        border.color: Theme.line
        MouseArea { anchors.fill: parent }

        Column {
            anchors.fill: parent
            anchors.margins: 32
            spacing: 20

            // Header
            Item {
                width: parent.width
                height: 44
                Text {
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    text: root.capturing
                          ? "Press any controller button, trigger, or stick click…"
                          : "Controller Mapping · DualShock 4"
                    color: root.capturing ? Theme.gold : Theme.ink
                    font.family: Theme.sans
                    font.pixelSize: 24
                    font.weight: Font.DemiBold
                }
                Row {
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 12
                    Rectangle {
                        width: 148; height: 34; radius: 17
                        color: "#15120d"
                        border.width: 1; border.color: Theme.line
                        Text {
                            anchors.centerIn: parent
                            text: "Reset Defaults (R)"
                            color: Theme.dim
                            font.family: Theme.mono; font.pixelSize: 12
                        }
                        TapHandler { onTapped: if (root.backend) root.backend.resetGamepadBindings() }
                    }
                    Rectangle {
                        width: 82; height: 34; radius: 17
                        color: "#15120d"
                        border.width: 1; border.color: Theme.line
                        Text {
                            anchors.centerIn: parent
                            text: "Done ○"
                            color: Theme.ink
                            font.family: Theme.mono; font.pixelSize: 12
                        }
                        TapHandler { onTapped: root.hide() }
                    }
                }
            }

            // Body: Left DualShock 4 diagram + Right bindings list
            Row {
                width: parent.width
                height: parent.height - 64
                spacing: 28

                // Left: DualShock 4 visual layout
                Rectangle {
                    width: 460
                    height: parent.height
                    radius: 20
                    color: "#060605"
                    border.width: 1
                    border.color: Theme.line

                    // Selected slot readout at top of diagram
                    Column {
                        anchors.top: parent.top
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: 20
                        spacing: 4
                        Text {
                            text: root.gamepad ? root.gamepad.inputLabel(root.selectedId) : ""
                            color: Theme.gold
                            font.family: Theme.sans; font.pixelSize: 18; font.weight: Font.DemiBold
                        }
                        Text {
                            text: root.selectedActionLabel
                            color: Theme.ink
                            font.family: Theme.mono; font.pixelSize: 14
                        }
                    }

                    // Interactive DualShock 4 silhouette
                    Item {
                        anchors.centerIn: parent
                        anchors.verticalCenterOffset: 24
                        width: 390
                        height: 290

                        // Main body
                        Rectangle {
                            x: 35; y: 58; width: 320; height: 146; radius: 52
                            color: "#12110f"; border.width: 1; border.color: Theme.line
                        }
                        // Left & right grips
                        Rectangle {
                            x: 24; y: 126; width: 92; height: 142; radius: 42; rotation: 16
                            color: "#12110f"; border.width: 1; border.color: Theme.line
                        }
                        Rectangle {
                            x: 274; y: 126; width: 92; height: 142; radius: 42; rotation: -16
                            color: "#12110f"; border.width: 1; border.color: Theme.line
                        }

                        // Reusable button node helper via Repeater
                        Repeater {
                            model: [
                                { id: "l2", label: "L2", x: 54,  y: 4,   w: 66, h: 24, r: 8 },
                                { id: "r2", label: "R2", x: 270, y: 4,   w: 66, h: 24, r: 8 },
                                { id: "l1", label: "L1", x: 54,  y: 34,  w: 66, h: 20, r: 6 },
                                { id: "r1", label: "R1", x: 270, y: 34,  w: 66, h: 20, r: 6 },
                                { id: "options", label: "PAD", x: 132, y: 68, w: 126, h: 52, r: 10 },
                                { id: "menu", label: "☰", x: 264, y: 74, w: 26, h: 20, r: 10 },
                                { id: "y", label: "△", x: 298, y: 90,  w: 32, h: 32, r: 16 },
                                { id: "b", label: "○", x: 332, y: 122, w: 32, h: 32, r: 16 },
                                { id: "a", label: "✕", x: 298, y: 154, w: 32, h: 32, r: 16 },
                                { id: "x", label: "□", x: 264, y: 122, w: 32, h: 32, r: 16 },
                                { id: "l3", label: "L3", x: 112, y: 156, w: 54, h: 54, r: 27 },
                                { id: "r3", label: "R3", x: 224, y: 156, w: 54, h: 54, r: 27 }
                            ]
                            delegate: Rectangle {
                                required property var modelData
                                readonly property bool active: root.selectedId === modelData.id
                                x: modelData.x; y: modelData.y
                                width: modelData.w; height: modelData.h; radius: modelData.r
                                color: active ? "#241b0b" : "#0b0a09"
                                border.width: active ? 2 : 1
                                border.color: active ? Theme.gold : Theme.line
                                Text {
                                    anchors.centerIn: parent
                                    text: modelData.label
                                    color: active ? Theme.gold : Theme.dim
                                    font.family: Theme.mono
                                    font.pixelSize: 12
                                    font.weight: active ? Font.DemiBold : Font.Normal
                                }
                                TapHandler { onTapped: root.selectSlotById(modelData.id) }
                            }
                        }

                        // D-pad (fixed navigation)
                        Rectangle {
                            x: 56; y: 106; width: 52; height: 52; radius: 14
                            color: "#0b0a09"; border.width: 1; border.color: Theme.line
                            Text { anchors.centerIn: parent; text: "✥"; color: Theme.faint; font.pixelSize: 18 }
                        }
                    }

                    // Bottom stick legend
                    Text {
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: 20
                        text: "Left Stick / D-pad: Navigate   ·   Right Stick ↕: Smooth Scroll\nEnter / ✕: Listen for button   ·   ← →: Cycle action"
                        color: Theme.faint
                        font.family: Theme.mono
                        font.pixelSize: 11
                        lineHeight: 1.4
                    }
                }

                // Right: Remappable slots table
                Column {
                    width: parent.width - 488
                    height: parent.height
                    spacing: 8

                    Repeater {
                        model: root.slots
                        delegate: Rectangle {
                            required property var modelData
                            required property int index
                            readonly property bool current: root.row === index
                            width: parent.width
                            height: 41
                            radius: 10
                            color: current ? "#15110a" : "#080706"
                            border.width: current ? 1.5 : 1
                            border.color: current ? (root.capturing ? Theme.gold : Theme.gold) : Theme.line

                            Text {
                                anchors.left: parent.left; anchors.leftMargin: 16
                                anchors.verticalCenter: parent.verticalCenter
                                text: modelData.label
                                color: current ? Theme.gold : Theme.ink
                                font.family: Theme.sans; font.pixelSize: 14; font.weight: current ? Font.DemiBold : Font.Normal
                            }

                            Row {
                                anchors.right: parent.right; anchors.rightMargin: 14
                                anchors.verticalCenter: parent.verticalCenter
                                spacing: 10

                                Text {
                                    text: "‹"
                                    color: current ? Theme.gold : Theme.faint
                                    font.family: Theme.mono; font.pixelSize: 16
                                    TapHandler { onTapped: { root.row = index; root.cycleAction(-1); } }
                                }
                                Text {
                                    width: 200
                                    horizontalAlignment: Text.AlignHCenter
                                    text: (current && root.capturing) ? "Press button…" : modelData.actionLabel
                                    color: (current && root.capturing) ? Theme.gold : Theme.ink
                                    font.family: Theme.mono; font.pixelSize: 13
                                }
                                Text {
                                    text: "›"
                                    color: current ? Theme.gold : Theme.faint
                                    font.family: Theme.mono; font.pixelSize: 16
                                    TapHandler { onTapped: { root.row = index; root.cycleAction(1); } }
                                }
                            }

                            TapHandler { onTapped: root.row = index }
                            TapHandler {
                                acceptedButtons: Qt.LeftButton
                                onDoubleTapped: { root.row = index; root.startCapture(); }
                            }
                        }
                    }
                }
            }
        }
    }
}
