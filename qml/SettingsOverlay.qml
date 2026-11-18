import QtQuick

// Cmd+, : a centred OLED card. Mode (Neovim / Gaming / Mouse), UI scale, default model and
// effort, model suggestions, and the active keybinds. j/k (w/s, Up/Down) pick a row, h/l (a/d,
// Left/Right) change it, Esc or Cmd+, closes. Everything is clickable too.
FocusScope {
    id: root
    objectName: "settingsOverlay"
    property QtObject backend
    property bool open: false
    property int row: 0
    signal closed()
    signal openControllerMap()

    readonly property string mode: backend ? backend.navMode : "controller"
    readonly property var modes: [
        { id: "controller", name: "Controller", tag: "Recommended" },
        { id: "vim", name: "Neovim", tag: "" },
        { id: "wasd", name: "Gaming", tag: "" },
        { id: "mouse", name: "Mouse", tag: "" }
    ]
    readonly property int rowCount: 6
    readonly property var keybinds: mode === "controller" ? [
            ["✕", "confirm"], ["○", "back"], ["△", "tab"], ["□", "prose"],
            ["L1 R1", "slide"], ["L2 R2", "zoom"], ["☰", "settings"], ["pad", "new session"]]
        : mode === "vim" ? [
            ["h j k l", "move"], ["g  G", "first / last"], ["⏎", "zoom in"], ["esc", "back"],
            ["space", "write-up"], ["⇥ tab", "accept prediction"], ["⌘M", "model & effort"], ["⌘N", "new session"]]
        : mode === "wasd" ? [
            ["w a s d", "move"], ["e  ⏎", "interact"], ["q  esc", "back"], ["space", "write-up"],
            ["⇥ tab", "accept prediction"], ["⌘M", "model & effort"], ["⌘N", "new session"], ["⌘,", "settings"]]
        : [
            ["click", "open"], ["wheel", "zoom islands"], ["⌘ wheel", "zoom anywhere"], ["esc", "back"],
            ["⇥ tab", "accept prediction"], ["⌘M", "model & effort"], ["⌘N", "new session"], ["⌘,", "settings"]]

    readonly property QtObject gamepad: backend ? backend.gamepad : null
    readonly property bool padConnected: gamepad ? gamepad.connected : false
    readonly property string padName: gamepad && gamepad.controllerName.length > 0 ? gamepad.controllerName : "Gamepad"
    readonly property var padBinds: [
        ["L2", "voice"], ["R2", "send"], ["✕", "confirm"], ["○", "back"],
        ["△", "tab"], ["□", "Write-up"], ["L1 R1", "slide"], ["R3", "zoom +"],
        ["L3", "zoom 1x"], ["L-Stick", "nav"], ["R-Stick", "scroll"], ["☰", "settings"]]

    function show() {
        row = 0;
        open = true;
        forceActiveFocus();
    }
    function hide() {
        if (!open)
            return;
        open = false;
        closed();
    }
    function change(d) {
        if (!backend)
            return;
        switch (row) {
        case 0: {
            const ids = modes.map(m => m.id);
            const i = Math.max(0, ids.indexOf(mode));
            backend.navMode = ids[Math.max(0, Math.min(ids.length - 1, i + d))];
            break;
        }
        case 1: backend.zoomBy(d > 0 ? 1.1 : 1 / 1.1); break;
        case 2: backend.cycleModel(d); break;
        case 3: backend.cycleEffort(d); break;
        case 4: backend.modelSuggestions = d > 0; break;
        case 5: openControllerMap(); break;
        }
    }

    visible: opacity > 0.01
    opacity: open ? 1 : 0
    Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
    enabled: open

    Keys.onPressed: (e) => {
        const ctrl = (e.modifiers & Qt.ControlModifier) !== 0;
        const k = e.key;
        if (k === Qt.Key_Escape || (ctrl && k === Qt.Key_Comma) || (!ctrl && k === Qt.Key_Q)) {
            hide();
        } else if (ctrl) {
            return;  // Cmd +/-/0 zoom still reach the app
        } else if (k === Qt.Key_Down || k === Qt.Key_J || k === Qt.Key_S || k === Qt.Key_Tab) {
            row = (row + 1) % rowCount;
        } else if (k === Qt.Key_Up || k === Qt.Key_K || k === Qt.Key_W || k === Qt.Key_Backtab) {
            row = (row - 1 + rowCount) % rowCount;
        } else if (k === Qt.Key_Right || k === Qt.Key_L || k === Qt.Key_D) {
            change(1);
        } else if (k === Qt.Key_Left || k === Qt.Key_H || k === Qt.Key_A) {
            change(-1);
        } else if (k === Qt.Key_Return || k === Qt.Key_Enter) {
            if (row === 5)
                openControllerMap();
            else
                hide();
        } else if (k === Qt.Key_C) {
            openControllerMap();
        }
        e.accepted = true;
    }

    // OLED dim: the app stays faintly visible behind the card.
    Rectangle {
        anchors.fill: parent
        color: "#000000"
        opacity: 0.88
        MouseArea { anchors.fill: parent; onClicked: root.hide(); onWheel: (w) => w.accepted = true }
    }

    Rectangle {
        id: card
        objectName: "settingsCard"
        anchors.centerIn: parent
        width: 900
        height: content.height + 76
        radius: 28
        color: "#000000"
        border.width: 1.5
        border.color: Qt.rgba(0.878, 0.624, 0.243, 0.55)
        MouseArea { anchors.fill: parent }  // clicks inside never close it

        Column {
            id: content
            x: 44
            y: 38
            width: parent.width - 88
            spacing: 6

            Item {
                width: parent.width
                height: 54
                Text {
                    text: "Settings"
                    color: Theme.ink
                    font.family: Theme.sans
                    font.pixelSize: 30
                    font.weight: Font.DemiBold
                }
                Text {
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.topMargin: 10
                    text: "⌘,  esc"
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 15
                }
            }

            // 0. Mode
            SettingRow {
                index: 0
                caption: "Mode"
                Row {
                    spacing: 12
                    anchors.verticalCenter: parent.verticalCenter
                    Repeater {
                        model: root.modes
                        Rectangle {
                            required property var modelData
                            objectName: "settingsMode"
                            readonly property bool current: root.mode === modelData.id
                            width: modeName.implicitWidth + (modeTag.visible ? modeTag.implicitWidth + 14 : 0) + 36
                            height: 40
                            radius: 20
                            color: current ? Qt.rgba(0.878, 0.624, 0.243, 0.14) : "transparent"
                            border.width: 1.5
                            border.color: current ? Theme.gold : Theme.line
                            Row {
                                anchors.centerIn: parent
                                spacing: 14
                                Text {
                                    id: modeName
                                    text: modelData.name
                                    color: parent.parent.current ? Theme.gold : Theme.dim
                                    font.family: Theme.sans
                                    font.pixelSize: 18
                                    font.weight: Font.DemiBold
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                                Text {
                                    id: modeTag
                                    visible: modelData.tag.length > 0
                                    text: modelData.tag
                                    color: Theme.copper
                                    font.family: Theme.mono
                                    font.pixelSize: 12
                                    anchors.verticalCenter: parent.verticalCenter
                                }
                            }
                            TapHandler { onTapped: { root.row = 0; if (root.backend) root.backend.navMode = modelData.id; } }
                        }
                    }
                }
            }

            // 1. UI scale
            SettingRow {
                index: 1
                caption: "UI Scale"
                Row {
                    spacing: 18
                    anchors.verticalCenter: parent.verticalCenter
                    Text {
                        objectName: "settingsScale"
                        width: 64
                        text: root.backend ? root.backend.uiScale.toFixed(2) + "x" : "1.00x"
                        color: Theme.gold
                        font.family: Theme.mono
                        font.pixelSize: 18
                        anchors.verticalCenter: parent.verticalCenter
                    }
                    Rectangle {
                        width: 200
                        height: 6
                        radius: 3
                        color: Theme.track
                        anchors.verticalCenter: parent.verticalCenter
                        Rectangle {
                            height: parent.height
                            radius: 3
                            color: Theme.gold
                            width: parent.width * ((root.backend ? root.backend.uiScale : 1) - 0.75) / (1.60 - 0.75)
                        }
                    }
                    Text {
                        text: "0.75x – 1.60x · mouse wheel / ⌘ scroll"
                        color: Theme.dim
                        font.family: Theme.sans
                        font.pixelSize: 15
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }

            // 2. Default model
            SettingRow {
                index: 2
                caption: "Default Model"
                Row {
                    spacing: 16
                    anchors.verticalCenter: parent.verticalCenter
                    Text {
                        objectName: "settingsModel"
                        text: root.backend ? root.backend.shortModelLabel(root.backend.chatModel) + "  ·  "
                                             + root.backend.modelSpeed(root.backend.chatModel) : ""
                        color: Theme.ink
                        font.family: Theme.sans
                        font.pixelSize: 19
                        font.weight: Font.DemiBold
                        anchors.verticalCenter: parent.verticalCenter
                        TapHandler { onTapped: { root.row = 2; if (root.backend) root.backend.cycleModel(1); } }
                    }
                    Text {
                        text: root.backend ? (root.backend.chatProvider === "anthropic" ? "Anthropic · claude stream-json"
                              : root.backend.chatProvider === "openai" ? "OpenAI · codex app-server"
                              : "Google · agy stream-json") : ""
                        color: Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 14
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }

            // 3. Default effort
            SettingRow {
                index: 3
                caption: "Default Effort"
                Row {
                    spacing: 8
                    anchors.verticalCenter: parent.verticalCenter
                    Repeater {
                        model: root.backend ? root.backend.chatEfforts : []
                        Rectangle {
                            required property string modelData
                            readonly property bool current: root.backend && root.backend.chatEffort === modelData
                            width: effText.implicitWidth + 24
                            height: 32
                            radius: 16
                            color: current ? Theme.gold : "transparent"
                            border.width: current ? 0 : 1
                            border.color: Theme.faint
                            Text {
                                id: effText
                                anchors.centerIn: parent
                                text: parent.modelData
                                color: parent.current ? "#000000" : Theme.dim
                                font.family: Theme.mono
                                font.pixelSize: 15
                            }
                            TapHandler { onTapped: { root.row = 3; if (root.backend) root.backend.chatEffort = parent.modelData; } }
                        }
                    }
                }
            }

            // 4. Model suggestions for the Tab predictor
            SettingRow {
                index: 4
                caption: "Model Hints"
                Row {
                    spacing: 16
                    anchors.verticalCenter: parent.verticalCenter
                    Rectangle {
                        width: 52
                        height: 28
                        radius: 14
                        color: root.backend && root.backend.modelSuggestions ? Theme.gold : Theme.track
                        anchors.verticalCenter: parent.verticalCenter
                        Rectangle {
                            width: 22; height: 22; radius: 11
                            y: 3
                            x: root.backend && root.backend.modelSuggestions ? 27 : 3
                            color: "#000000"
                        }
                        TapHandler { onTapped: { root.row = 4; if (root.backend) root.backend.modelSuggestions = !root.backend.modelSuggestions; } }
                    }
                    Text {
                        text: root.backend && root.backend.modelSuggestions ? "Tab prompts come from fast model, warm" : "Tab prompts from local heuristics only"
                        color: Theme.dim
                        font.family: Theme.sans
                        font.pixelSize: 15
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }

            SettingRow {
                index: 5
                caption: "Controller"
                Row {
                    spacing: 14
                    Rectangle {
                        width: ctrlMapLbl.implicitWidth + 28
                        height: 36
                        radius: 18
                        color: root.row === 5 ? "#1b1408" : "#11100e"
                        border.width: 1
                        border.color: root.row === 5 ? Theme.gold : Theme.line
                        Text {
                            id: ctrlMapLbl
                            anchors.centerIn: parent
                            text: "Controller Mapping  →"
                            color: root.row === 5 ? Theme.gold : Theme.ink
                            font.family: Theme.sans
                            font.pixelSize: 15
                            font.weight: Font.Medium
                        }
                        TapHandler { onTapped: { root.row = 5; root.openControllerMap(); } }
                    }
                    Text {
                        text: "Remap DualShock 4 buttons, triggers, and sticks (C)"
                        color: Theme.dim
                        font.family: Theme.sans
                        font.pixelSize: 14
                        anchors.verticalCenter: parent.verticalCenter
                    }
                }
            }

            Item { width: 1; height: 14 }
            Rectangle { width: parent.width; height: 1; color: Theme.line }
            Item { width: 1; height: 10 }

            Text {
                text: "Keybinds"
                color: Theme.dim
                font.family: Theme.sans
                font.pixelSize: 14
                font.weight: Font.Medium
                font.letterSpacing: 1.6
                font.capitalization: Font.AllUppercase
            }
            Grid {
                objectName: "settingsKeybinds"
                columns: 2
                columnSpacing: 40
                rowSpacing: 10
                topPadding: 8
                Repeater {
                    model: root.keybinds
                    Row {
                        required property var modelData
                        spacing: 16
                        width: 366
                        Text {
                            width: 110
                            text: modelData[0]
                            color: Theme.gold
                            font.family: Theme.mono
                            font.pixelSize: 16
                        }
                        Text {
                            text: modelData[1]
                            color: Theme.text
                            font.family: Theme.sans
                            font.pixelSize: 16
                        }
                    }
                }
            }

            // PS4 Controller: only while a pad is connected (the window enables it; offscreen never).
            Item { width: 1; height: 14; visible: root.padConnected }
            Item {
                width: parent.width
                height: 24
                visible: root.padConnected
                Text {
                    objectName: "settingsPadHeader"
                    anchors.left: parent.left
                    anchors.verticalCenter: parent.verticalCenter
                    text: "PS4 Controller  ·  " + root.padName
                    color: Theme.dim
                    font.family: Theme.sans
                    font.pixelSize: 14
                    font.weight: Font.Medium
                    font.letterSpacing: 1.6
                    font.capitalization: Font.AllUppercase
                }
                Rectangle {
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    width: mapBtnText.implicitWidth + 20
                    height: 24
                    radius: 12
                    color: "#15120d"
                    border.width: 1
                    border.color: Theme.line
                    Text {
                        id: mapBtnText
                        anchors.centerIn: parent
                        text: "Customize Mapping (C)"
                        color: Theme.gold
                        font.family: Theme.mono
                        font.pixelSize: 11
                    }
                    TapHandler { onTapped: root.openControllerMap() }
                }
            }
            Grid {
                objectName: "settingsPadBinds"
                visible: root.padConnected
                columns: 4
                columnSpacing: 20
                rowSpacing: 10
                topPadding: 8
                Repeater {
                    model: root.padBinds
                    Row {
                        required property var modelData
                        spacing: 10
                        width: 188
                        Text {
                            width: 62
                            text: modelData[0]
                            color: Theme.gold
                            font.family: Theme.mono
                            font.pixelSize: 16
                        }
                        Text {
                            text: modelData[1]
                            color: Theme.text
                            font.family: Theme.sans
                            font.pixelSize: 16
                        }
                    }
                }
            }
        }
    }

    component SettingRow: Item {
        id: sr
        property int index: 0
        property string caption: ""
        default property alias content: body.data
        width: parent ? parent.width : 0
        height: 62
        Rectangle {
            anchors.fill: parent
            anchors.leftMargin: -18
            anchors.rightMargin: -18
            radius: 14
            color: root.row === sr.index ? Theme.cardHi : "transparent"
        }
        Rectangle {
            x: -18
            width: 3
            height: 26
            radius: 1.5
            anchors.verticalCenter: parent.verticalCenter
            color: Theme.gold
            visible: root.row === sr.index
        }
        Text {
            width: 170
            anchors.verticalCenter: parent.verticalCenter
            text: sr.caption
            color: root.row === sr.index ? Theme.ink : Theme.dim
            font.family: Theme.sans
            font.pixelSize: 16
            font.weight: Font.Medium
        }
        Item {
            id: body
            x: 180
            width: parent.width - 180
            height: parent.height
        }
    }
}
