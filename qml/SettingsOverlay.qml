import QtQuick

// Cmd+, : a centred OLED card. Rows run most-used first: Model (every model visible at once),
// Default Effort, Controller Mapping, Ambient Voice, Navigation Mode, Theme, UI Scale, Code Diffs,
// Model Suggestions, Jarvis Bar; then the active keybinds. j/k (w/s, Up/Down) pick a row, h/l (a/d,
// Left/Right) change it, Esc or Cmd+, closes. Everything is clickable too.
FocusScope {
    id: root
    objectName: "settingsOverlay"
    property QtObject backend
    property bool open: false
    property int row: 0
    signal closed()
    signal openControllerMap()

    // Row indices, most frequently used first. StudioApp opens on modelRow for "model settings".
    readonly property int modelRow: 0
    readonly property int effortRow: 1
    readonly property int controllerRow: 2
    readonly property int ambientRow: 3
    readonly property int navRow: 4
    readonly property int themeRow: 5
    readonly property int scaleRow: 6
    readonly property int diffsRow: 7
    readonly property int suggestRow: 8
    readonly property int jarvisRow: 9  // last, so no older row index moves
    readonly property int rowCount: 10

    readonly property string mode: backend ? backend.navMode : "controller"
    readonly property var modes: [
        { id: "controller", name: "Controller", tag: "Recommended" },
        { id: "vim", name: "Neovim", tag: "" },
        { id: "wasd", name: "Gaming", tag: "" },
        { id: "mouse", name: "Mouse", tag: "" }
    ]
    readonly property var themes: [
        { id: "gold", name: "Gold" },
        { id: "lapis", name: "Lapis Blue" }
    ]
    readonly property var jarvisModes: [
        { id: "overlay", name: "Unified Overlay" },
        { id: "split", name: "Split Studio / Desktop" }
    ]
    readonly property var models: backend ? backend.models : []
    function providerName(p) {
        return p === "anthropic" ? "Anthropic" : p === "openai" ? "OpenAI" : p === "google" ? "Google"
             : p === "openrouter" ? "OpenRouter" : p === "openai-compat" ? "Local vLLM" : p;
    }
    readonly property var keybinds: mode === "controller" ? [
            ["✕", "confirm"], ["○", "back"], ["△", "tab"], ["□", "prose"],
            ["L1 R1", "slide"], ["L2 R2", "zoom"], ["☰", "settings"], ["pad", "new session"]]
        : mode === "vim" ? [
            ["h j k l", "move"], ["g  G", "first / last"], ["⏎", "zoom in"], ["esc", "back"],
            ["space", "write-up"], ["⇥ tab", "accept prediction"], ["⇧⌘M", "model & effort"], ["⌘N", "new session"]]
        : mode === "wasd" ? [
            ["w a s d", "move"], ["e  ⏎", "interact"], ["q  esc", "back"], ["space", "write-up"],
            ["⇥ tab", "accept prediction"], ["⇧⌘M", "model & effort"], ["⌘N", "new session"], ["⌘,", "settings"]]
        : [
            ["click", "open"], ["wheel", "zoom islands"], ["⌘ wheel", "zoom anywhere"], ["esc", "back"],
            ["⇥ tab", "accept prediction"], ["⇧⌘M", "model & effort"], ["⌘N", "new session"], ["⌘,", "settings"]]

    readonly property QtObject gamepad: backend ? backend.gamepad : null
    readonly property bool padConnected: gamepad ? gamepad.connected : false
    readonly property string padName: gamepad && gamepad.controllerName.length > 0 ? gamepad.controllerName : "Gamepad"
    // Live from the pad's bindings (defaults shown here), so a remap shows up at once; an action
    // with no button bound is left out.
    readonly property var padBinds: {
        const g = gamepad;
        if (g)
            g.bindingSlots;  // re-evaluate on bindingsChanged
        const rows = [["voice_hold", "L2", "voice"], ["send", "R2", "send"],
                      ["accept_prediction", "R1", "Tab / Full Send"], ["confirm", "✕", "select"],
                      ["back", "○", "back"], ["secondary", "□", "write-up"],
                      ["step_prev", "L1", "previous view"], ["step_next", "△", "next view"],
                      ["zoom_in", "R3", "zoom +"], ["zoom_reset", "L3", "zoom 1x"],
                      ["toggle_settings", "☰", "settings"]];
        const out = [];
        for (const r of rows) {
            const glyph = g ? g.glyphFor(r[0]) : r[1];
            if (glyph.length > 0)
                out.push([glyph, r[2]]);
        }
        return out.concat([["lstick", "nav"], ["rstick", "scroll"]]);
    }

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
        case modelRow: backend.cycleModel(d); break;
        case effortRow: backend.cycleEffort(d); break;
        case controllerRow: openControllerMap(); break;
        case ambientRow: backend.ambientVoiceEnabled = d > 0; break;
        case navRow: {
            const ids = modes.map(m => m.id);
            const i = Math.max(0, ids.indexOf(mode));
            backend.navMode = ids[Math.max(0, Math.min(ids.length - 1, i + d))];
            break;
        }
        case themeRow: backend.theme = d > 0 ? "lapis" : "gold"; break;
        case scaleRow: backend.zoomBy(d > 0 ? 1.1 : 1 / 1.1); break;
        case diffsRow: backend.showCodeDiffs = d > 0; break;
        case suggestRow: backend.modelSuggestions = d > 0; break;
        case jarvisRow: backend.jarvisBarMode = d > 0 ? "split" : "overlay"; break;
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
            if (row === controllerRow)
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
        width: 1000
        height: content.height + 68
        radius: 28
        color: "#000000"
        border.width: 1.5
        border.color: Theme.goldA(0.55)
        MouseArea { anchors.fill: parent }  // clicks inside never close it

        Column {
            id: content
            x: 44
            y: 32
            width: parent.width - 88
            spacing: 4

            Item {
                width: parent.width
                height: 48
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

            // 0. Model: every model in the lineup, visible and clickable at once.
            SettingRow {
                index: root.modelRow
                caption: "Model"
                stacked: true
                height: 30 + modelGrid.height + 14
                Grid {
                    id: modelGrid
                    objectName: "settingsModelGrid"
                    columns: 5
                    columnSpacing: 10
                    rowSpacing: 10
                    readonly property real chipW: (width - columnSpacing * (columns - 1)) / columns
                    width: parent.width
                    Repeater {
                        model: root.models
                        Rectangle {
                            required property var modelData
                            objectName: "settingsModelChip"
                            readonly property bool current: root.backend && root.backend.chatModel === modelData.id
                            width: modelGrid.chipW
                            height: 54
                            radius: 14
                            color: current ? Theme.goldA(0.14) : "transparent"
                            border.width: 1.5
                            border.color: current ? Theme.gold : Theme.line
                            Column {
                                anchors.centerIn: parent
                                spacing: 3
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: modelData.label
                                    color: parent.parent.current ? Theme.gold : Theme.ink
                                    font.family: Theme.sans
                                    font.pixelSize: 15
                                    font.weight: Font.DemiBold
                                }
                                Text {
                                    anchors.horizontalCenter: parent.horizontalCenter
                                    text: modelData.speed + " · " + root.providerName(modelData.provider)
                                    color: Theme.dim
                                    font.family: Theme.mono
                                    font.pixelSize: 11
                                }
                            }
                            TapHandler {
                                onTapped: { root.row = root.modelRow; if (root.backend) root.backend.chatModel = modelData.id; }
                            }
                        }
                    }
                }
            }

            // 1. Default effort (the chips the selected model accepts)
            SettingRow {
                index: root.effortRow
                caption: "Default Effort"
                Row {
                    spacing: 8
                    anchors.verticalCenter: parent.verticalCenter
                    Repeater {
                        model: root.backend ? root.backend.chatEfforts : []
                        Rectangle {
                            required property string modelData
                            objectName: "settingsEffort"
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
                            TapHandler { onTapped: { root.row = root.effortRow; if (root.backend) root.backend.chatEffort = parent.modelData; } }
                        }
                    }
                }
            }

            // 2. Controller mapping
            SettingRow {
                index: root.controllerRow
                caption: "Controller"
                Row {
                    spacing: 14
                    anchors.verticalCenter: parent.verticalCenter
                    Rectangle {
                        width: ctrlMapLbl.implicitWidth + 28
                        height: 36
                        radius: 18
                        color: root.row === root.controllerRow ? "#1b1408" : "#11100e"
                        border.width: 1
                        border.color: root.row === root.controllerRow ? Theme.gold : Theme.line
                        Text {
                            id: ctrlMapLbl
                            anchors.centerIn: parent
                            text: "Controller Mapping  →"
                            color: root.row === root.controllerRow ? Theme.gold : Theme.ink
                            font.family: Theme.sans
                            font.pixelSize: 15
                            font.weight: Font.Medium
                        }
                        TapHandler { onTapped: { root.row = root.controllerRow; root.openControllerMap(); } }
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

            // 3. Ambient voice
            SettingRow {
                index: root.ambientRow
                caption: "Ambient Voice"
                OnOffPills {
                    objectName: "settingsAmbient"
                    rowIndex: root.ambientRow
                    on: root.backend ? root.backend.ambientVoiceEnabled : false
                    firstOn: true
                    note: on ? "Spoken lines become Studio actions" : "Voice only when held"
                    onPicked: (v) => { if (root.backend) root.backend.ambientVoiceEnabled = v; }
                }
            }

            // 4. Navigation mode
            SettingRow {
                index: root.navRow
                caption: "Navigation Mode"
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
                            color: current ? Theme.goldA(0.14) : "transparent"
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
                            TapHandler { onTapped: { root.row = root.navRow; if (root.backend) root.backend.navMode = modelData.id; } }
                        }
                    }
                }
            }

            // 5. Accent theme
            SettingRow {
                index: root.themeRow
                caption: "Theme"
                Row {
                    spacing: 12
                    anchors.verticalCenter: parent.verticalCenter
                    Repeater {
                        model: root.themes
                        Rectangle {
                            required property var modelData
                            objectName: "settingsTheme"
                            readonly property bool current: (root.backend ? root.backend.theme : "gold") === modelData.id
                            // No colour swatch: gold mode must stay free of blue pixels, and the
                            // selected pill already shows its accent.
                            width: themeName.implicitWidth + 36
                            height: 40
                            radius: 20
                            color: current ? Theme.goldA(0.14) : "transparent"
                            border.width: 1.5
                            border.color: current ? Theme.gold : Theme.line
                            Text {
                                id: themeName
                                anchors.centerIn: parent
                                text: parent.modelData.name
                                color: parent.current ? Theme.gold : Theme.dim
                                font.family: Theme.sans
                                font.pixelSize: 18
                                font.weight: Font.DemiBold
                            }
                            TapHandler { onTapped: { root.row = root.themeRow; if (root.backend) root.backend.theme = modelData.id; } }
                        }
                    }
                }
            }

            // 6. UI scale
            SettingRow {
                index: root.scaleRow
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

            // 7. Code diffs on the Stage (visual-first: off by default; the Stage's Code pill too)
            SettingRow {
                index: root.diffsRow
                caption: "Code Diffs"
                OnOffPills {
                    objectName: "settingsCodeDiffs"
                    rowIndex: root.diffsRow
                    on: root.backend ? root.backend.showCodeDiffs : false
                    firstOn: false
                    note: on ? "Diffs follow the visuals on the Stage" : "Stage shows results only; diffs on request"
                    onPicked: (v) => { if (root.backend) root.backend.showCodeDiffs = v; }
                }
            }

            // 8. Model suggestions for the Tab predictor
            SettingRow {
                index: root.suggestRow
                caption: "Model Suggestions"
                OnOffPills {
                    objectName: "settingsSuggestions"
                    rowIndex: root.suggestRow
                    on: root.backend ? root.backend.modelSuggestions : false
                    firstOn: true
                    note: on ? "Tab prompts come from fast model, warm" : "Tab prompts from local heuristics only"
                    onPicked: (v) => { if (root.backend) root.backend.modelSuggestions = v; }
                }
            }

            // 9. Jarvis Bar: L2 opens the desktop overlay everywhere (Studio too), or split
            // (the in-Studio ask card inside Studio, the overlay over other apps).
            SettingRow {
                index: root.jarvisRow
                caption: "Jarvis Bar"
                Row {
                    spacing: 12
                    anchors.verticalCenter: parent.verticalCenter
                    Repeater {
                        model: root.jarvisModes
                        Rectangle {
                            required property var modelData
                            objectName: "settingsJarvisBar"
                            readonly property bool current: (root.backend ? root.backend.jarvisBarMode : "overlay") === modelData.id
                            width: jarvisName.implicitWidth + 36
                            height: 40
                            radius: 20
                            color: current ? Theme.goldA(0.14) : "transparent"
                            border.width: 1.5
                            border.color: current ? Theme.gold : Theme.line
                            Text {
                                id: jarvisName
                                anchors.centerIn: parent
                                text: parent.modelData.name
                                color: parent.current ? Theme.gold : Theme.dim
                                font.family: Theme.sans
                                font.pixelSize: 18
                                font.weight: Font.DemiBold
                            }
                            TapHandler { onTapped: { root.row = root.jarvisRow; if (root.backend) root.backend.jarvisBarMode = modelData.id; } }
                        }
                    }
                }
            }

            Item { width: 1; height: 12 }
            Rectangle { width: parent.width; height: 1; color: Theme.line }
            Item { width: 1; height: 8 }

            // With a pad connected in Controller mode, the live PS4 section below replaces the
            // generic glyph list (same buttons), so the card still fits a 1080 p window.
            readonly property bool showKeybinds: !(root.padConnected && root.mode === "controller")
            Text {
                visible: content.showKeybinds
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
                visible: content.showKeybinds
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
                        // Controller mode: the buttons as vector keycaps ("L1 R1" is two).
                        Row {
                            visible: root.mode === "controller"
                            width: 110
                            spacing: 6
                            anchors.verticalCenter: parent.verticalCenter
                            Repeater {
                                model: root.mode === "controller" ? String(modelData[0]).split(" ") : []
                                PadIcon {
                                    required property string modelData
                                    anchors.verticalCenter: parent.verticalCenter
                                    button: modelData
                                    size: 22
                                }
                            }
                        }
                        Text {
                            visible: root.mode !== "controller"
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
            Item { width: 1; height: 14; visible: root.padConnected && content.showKeybinds }
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
                        Item {
                            width: 44
                            height: 22
                            anchors.verticalCenter: parent.verticalCenter
                            PadIcon {
                                id: bindIcon
                                anchors.verticalCenter: parent.verticalCenter
                                button: modelData[0]
                                size: 22
                            }
                            Text {
                                visible: bindIcon.kind.length === 0
                                anchors.verticalCenter: parent.verticalCenter
                                text: modelData[0]
                                color: Theme.gold
                                font.family: Theme.mono
                                font.pixelSize: 16
                            }
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

    // Two pills (On / Off, in the order the row reads best) and a one-line note.
    component OnOffPills: Row {
        id: oo
        property int rowIndex: 0
        property bool on: false
        property bool firstOn: true
        property string note: ""
        signal picked(bool value)
        spacing: 12
        anchors.verticalCenter: parent ? parent.verticalCenter : undefined
        Repeater {
            model: oo.firstOn ? [true, false] : [false, true]
            Rectangle {
                required property bool modelData
                readonly property bool current: oo.on === modelData
                width: 72
                height: 34
                radius: 17
                color: current ? Theme.goldA(0.14) : "transparent"
                border.width: 1.5
                border.color: current ? Theme.gold : Theme.line
                Text {
                    anchors.centerIn: parent
                    text: parent.modelData ? "On" : "Off"
                    color: parent.current ? Theme.gold : Theme.dim
                    font.family: Theme.sans
                    font.pixelSize: 16
                    font.weight: Font.DemiBold
                }
                TapHandler { onTapped: { root.row = oo.rowIndex; oo.picked(parent.modelData); } }
            }
        }
        Text {
            text: oo.note
            color: Theme.dim
            font.family: Theme.sans
            font.pixelSize: 15
            anchors.verticalCenter: parent.verticalCenter
        }
    }

    component SettingRow: Item {
        id: sr
        property int index: 0
        property string caption: ""
        property bool stacked: false  // caption above a full-width body (the Model grid)
        default property alias content: body.data
        width: parent ? parent.width : 0
        height: 56
        Rectangle {
            anchors.fill: parent
            anchors.leftMargin: -18
            anchors.rightMargin: -18
            radius: 14
            color: root.row === sr.index ? Theme.cardHi : "transparent"
        }
        Rectangle {
            x: -18
            y: sr.stacked ? 8 : (sr.height - height) / 2
            width: 3
            height: 26
            radius: 1.5
            color: Theme.gold
            visible: root.row === sr.index
        }
        Text {
            width: 170
            y: sr.stacked ? 10 : (sr.height - height) / 2
            text: sr.caption
            color: root.row === sr.index ? Theme.ink : Theme.dim
            font.family: Theme.sans
            font.pixelSize: 16
            font.weight: Font.Medium
        }
        Item {
            id: body
            x: sr.stacked ? 0 : 180
            y: sr.stacked ? 36 : 0
            width: parent.width - x
            height: sr.stacked ? parent.height - y : parent.height
        }
    }
}
