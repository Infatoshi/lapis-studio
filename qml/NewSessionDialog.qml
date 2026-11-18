import QtQuick

// Cmd+N / Cmd+T: folder, harness, model and the opening goal. Enter starts it.
Item {
    id: root
    objectName: "newSessionDialog"
    property QtObject backend
    property bool open: false
    signal closed()

    readonly property var harnesses: ["claude", "codex", "agy"]
    property int harnessIdx: 0
    property int modelIdx: 0
    property int folderIdx: 0
    readonly property string harness: harnesses[harnessIdx]
    readonly property var models: backend ? backend.harnessModels[harness] : []
    readonly property string model: models && models.length > 0 ? models[Math.min(modelIdx, models.length - 1)] : ""
    readonly property var folders: backend ? backend.folders : []

    visible: opacity > 0.005
    opacity: open ? 1 : 0
    // Opens at zoom pace; closes fast so it never ghosts over the island zooming in behind it.
    property int fadeMs: Theme.zoomMs
    Behavior on opacity { NumberAnimation { duration: root.fadeMs; easing.type: Easing.OutCubic } }

    function show() {
        goal.text = "";
        folder.text = backend ? backend.defaultFolder : "~/dev";
        folderIdx = Math.max(0, folders.indexOf(folder.text));
        fadeMs = Theme.zoomMs;
        open = true;
        goal.forceActiveFocus();
    }
    function hide() {
        fadeMs = 70;
        open = false;
        root.closed();
    }
    function start() {
        if (goal.text.trim().length === 0)
            return;
        const g = goal.text;
        fadeMs = 70;
        open = false;
        backend.startSession(folder.text, harness, model, g);
        root.closed();
    }
    function cycleFolder(d) {
        if (folders.length === 0) return;
        folderIdx = (folderIdx + d + folders.length) % folders.length;
        folder.text = folders[folderIdx];
    }
    function cycleHarness(d) { harnessIdx = (harnessIdx + d + 3) % 3; modelIdx = 0; }
    function cycleModel(d) { const n = Math.max(1, models.length); modelIdx = (modelIdx + d + n) % n; }
    function handle(e) {
        const alt = (e.modifiers & Qt.AltModifier) !== 0;
        if (e.key === Qt.Key_Escape) { hide(); e.accepted = true; }
        else if (e.key === Qt.Key_Return || e.key === Qt.Key_Enter) { start(); e.accepted = true; }
        else if (alt && e.key === Qt.Key_Right) { harnessIdx = (harnessIdx + 1) % 3; modelIdx = 0; e.accepted = true; }
        else if (alt && e.key === Qt.Key_Left) { harnessIdx = (harnessIdx + 2) % 3; modelIdx = 0; e.accepted = true; }
        else if (alt && e.key === Qt.Key_Down) { modelIdx = (modelIdx + 1) % Math.max(1, models.length); e.accepted = true; }
        else if (alt && e.key === Qt.Key_Up) { modelIdx = (modelIdx + models.length - 1) % Math.max(1, models.length); e.accepted = true; }
        else if (e.key === Qt.Key_Down) { cycleFolder(1); e.accepted = true; }
        else if (e.key === Qt.Key_Up) { cycleFolder(-1); e.accepted = true; }
        else if (e.key === Qt.Key_Tab || e.key === Qt.Key_Backtab) {
            if (goal.activeFocus) folder.forceActiveFocus(); else goal.forceActiveFocus();
            e.accepted = true;
        }
    }

    Rectangle {
        anchors.fill: parent
        color: Qt.rgba(0, 0, 0, 0.78)
        MouseArea { anchors.fill: parent; onClicked: root.hide() }
    }

    Rectangle {
        id: card
        width: Math.min(1080, parent.width - 160)
        height: 330
        anchors.centerIn: parent
        radius: 28
        color: Theme.card
        scale: root.open ? 1 : 0.96
        Behavior on scale { NumberAnimation { duration: Theme.zoomMs; easing.type: Easing.OutCubic } }
        MouseArea { anchors.fill: parent }

        Text {
            x: 48
            y: 40
            text: "New session"
            color: Theme.dim
            font.family: Theme.sans
            font.pixelSize: 21
            font.weight: Font.DemiBold
            font.letterSpacing: 1.4
            font.capitalization: Font.AllUppercase
        }

        Item {
            x: 48
            y: 92
            width: parent.width - 96
            height: 64
            Text {
                anchors.verticalCenter: parent.verticalCenter
                visible: goal.text.length === 0 && !goal.inputMethodComposing
                text: "Goal"
                color: Theme.faint
                font.family: Theme.sans
                font.pixelSize: 40
                font.weight: Font.Medium
            }
            TextInput {
                id: goal
                objectName: "goalInput"
                anchors.fill: parent
                verticalAlignment: TextInput.AlignVCenter
                color: Theme.ink
                selectionColor: Theme.copper
                clip: true
                font.family: Theme.sans
                font.pixelSize: 40
                font.weight: Font.Medium
                cursorDelegate: Caret {}
                Keys.priority: Keys.BeforeItem
                Keys.onPressed: (e) => root.handle(e)
            }
        }

        Rectangle {
            x: 48
            y: 172
            width: parent.width - 96
            height: 1
            color: Theme.line
        }

        Row {
            x: 48
            y: 204
            spacing: 14
            height: 46

            Rectangle {
                height: 46
                width: folder.contentWidth + 40
                radius: 23
                color: "transparent"
                border.width: 1.5
                border.color: folder.activeFocus ? Theme.gold : Theme.faint
                TextInput {
                    id: folder
                    objectName: "folderInput"
                    anchors.centerIn: parent
                    width: Math.max(contentWidth, 40)
                    color: Theme.ink
                    font.family: Theme.mono
                    font.pixelSize: 20
                    cursorDelegate: Caret {}
                    Keys.priority: Keys.BeforeItem
                    Keys.onPressed: (e) => root.handle(e)
                }
            }
            Repeater {
                model: root.harnesses
                Rectangle {
                    required property var modelData
                    required property int index
                    readonly property bool on: index === root.harnessIdx
                    height: 46
                    width: hl.implicitWidth + 40
                    radius: 23
                    color: on ? Qt.rgba(0.878, 0.624, 0.243, 0.14) : "transparent"
                    border.width: 1.5
                    border.color: on ? Theme.gold : Theme.line
                    Text {
                        id: hl
                        anchors.centerIn: parent
                        text: parent.modelData
                        color: parent.on ? Theme.gold : Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 20
                    }
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: { root.harnessIdx = parent.index; root.modelIdx = 0; goal.forceActiveFocus(); }
                    }
                }
            }
            Rectangle {
                height: 46
                width: ml.implicitWidth + 40
                radius: 23
                color: "transparent"
                border.width: 1.5
                border.color: Theme.copper
                Text {
                    id: ml
                    anchors.centerIn: parent
                    text: root.backend ? root.backend.shortModel(root.model) : root.model
                    color: Theme.ink
                    font.family: Theme.mono
                    font.pixelSize: 20
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    onClicked: { root.modelIdx = (root.modelIdx + 1) % Math.max(1, root.models.length); goal.forceActiveFocus(); }
                }
            }
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 48
            y: 276
            text: "↑↓ folder   ⌥←→ harness   ⌥↑↓ model   ⏎ start"
            color: Theme.faint
            font.family: Theme.mono
            font.pixelSize: 15
        }
    }
}
