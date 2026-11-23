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

                        // Each input as its vector keycap, placed where it sits on the pad
                        // (cx, cy: centre; s: icon height). The selected one glows gold.
                        Repeater {
                            model: [
                                { id: "l2", cx: 84,  cy: 16,  s: 30 },
                                { id: "r2", cx: 306, cy: 16,  s: 30 },
                                { id: "l1", cx: 84,  cy: 46,  s: 22 },
                                { id: "r1", cx: 306, cy: 46,  s: 22 },
                                { id: "options", cx: 195, cy: 94, s: 52 },
                                { id: "menu", cx: 277, cy: 82, s: 24 },
                                { id: "y", cx: 314, cy: 106, s: 30 },
                                { id: "b", cx: 346, cy: 138, s: 30 },
                                { id: "a", cx: 314, cy: 170, s: 30 },
                                { id: "x", cx: 282, cy: 138, s: 30 },
                                { id: "l3", cx: 139, cy: 183, s: 50 },
                                { id: "r3", cx: 251, cy: 183, s: 50 }
                            ]
                            delegate: Item {
                                required property var modelData
                                readonly property bool active: root.selectedId === modelData.id
                                x: modelData.cx - width / 2
                                y: modelData.cy - height / 2
                                width: nodeIcon.width
                                height: nodeIcon.height
                                Rectangle {
                                    anchors.centerIn: parent
                                    visible: parent.active
                                    width: parent.width + 10
                                    height: parent.height + 10
                                    radius: Math.min(width, height) / 2
                                    color: Theme.goldA(0.10)
                                    border.width: 1
                                    border.color: Theme.goldA(0.35)
                                }
                                PadIcon {
                                    id: nodeIcon
                                    button: modelData.id
                                    size: modelData.s
                                    active: parent.active
                                }
                                TapHandler { onTapped: root.selectSlotById(modelData.id) }
                            }
                        }

                        // D-pad (fixed navigation)
                        PadIcon {
                            x: 82 - width / 2; y: 132 - height / 2
                            button: "dpad"
                            size: 50
                        }
                    }

                    // Bottom stick legend
                    Column {
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.margins: 20
                        spacing: 10
                        Row {
                            spacing: 22
                            LegendItem { icons: ["lstick", "dpad"]; label: "Navigate" }
                            LegendItem { icons: ["rstick"]; label: "Smooth scroll" }
                        }
                        Row {
                            spacing: 22
                            LegendItem { icons: ["a"]; label: "Listen for button" }
                            LegendItem { icons: ["dpad_lr"]; label: "Cycle action" }
                        }
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

                            Item {
                                id: slotIcon
                                anchors.left: parent.left; anchors.leftMargin: 10
                                anchors.verticalCenter: parent.verticalCenter
                                width: 36
                                height: 24
                                PadIcon {
                                    anchors.centerIn: parent
                                    button: modelData.id
                                    size: 22
                                    active: current
                                }
                            }
                            Text {
                                anchors.left: slotIcon.right; anchors.leftMargin: 10
                                anchors.verticalCenter: parent.verticalCenter
                                // The icon shows the glyph: "✕ Cross" reads "Cross", "Options ☰" "Options".
                                text: String(modelData.label).replace(/^[✕○□△]\s*/, "").replace(/\s*☰$/, "")
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

    // A legend entry: one or more pad icons, then what they do.
    component LegendItem: Row {
        property var icons: []
        property string label: ""
        spacing: 6
        Repeater {
            model: parent.icons
            PadIcon {
                required property string modelData
                anchors.verticalCenter: parent.verticalCenter
                button: modelData
                size: 18
            }
        }
        Text {
            anchors.verticalCenter: parent.verticalCenter
            leftPadding: 2
            text: parent.label
            color: Theme.dim
            font.family: Theme.mono
            font.pixelSize: 12
        }
    }
}
