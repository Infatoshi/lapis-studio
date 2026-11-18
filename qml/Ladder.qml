import QtQuick

// Lens 2: four sparse rungs. Goal, assumptions, arms, winner.
Item {
    id: root
    property var project
    property QtObject backend

    readonly property var ladder: project ? project.ladder : null
    readonly property var assumptions: ladder ? ladder.assumptions : []
    readonly property var arms: ladder ? ladder.arms : []
    readonly property string winner: ladder ? ladder.winner : ""
    readonly property int reached: !ladder ? 0 : (winner.length > 0 ? 4 : (arms.length > 0 ? 3 : (assumptions.length > 0 ? 2 : 1)))

    readonly property real colW: Math.min(1180, width - 160)
    readonly property real rungH: Math.min(190, (height - 40) / 4)

    // Tab predictor, a small state machine over the rung reached: a finished ladder with
    // artifacts opens the Stage; otherwise it suggests the question that moves the current rung.
    readonly property var predictedAction: {
        if (!project)
            return null;
        const arts = project.artifacts ? project.artifacts.length : 0;
        if (reached >= 4 || (arts > 0 && reached >= 3 && arms.length === 0))
            return arts > 0
                ? { targetId: "rung:4", label: "Open Stage", promptSuggestion: "", actionKind: "stage" }
                : { targetId: "rung:4", label: "Ask why it won", promptSuggestion: "Why did " + winner.split(" ·")[0] + " win?", actionKind: "prompt" };
        if (reached === 3)
            return { targetId: "rung:3", label: "Ask which arm leads", promptSuggestion: "Which arm is leading so far?", actionKind: "prompt" };
        if (reached === 2)
            return { targetId: "rung:2", label: "Ask what matters", promptSuggestion: "Which assumption matters most?", actionKind: "prompt" };
        return { targetId: "rung:1", label: "Ask for the plan", promptSuggestion: "What is the plan for this goal?", actionKind: "prompt" };
    }
    // The rung's numbered marker (a 44 px circle), in this item's coordinates.
    function targetRect(id) {
        if (!id.startsWith("rung:"))
            return Qt.rect(0, 0, 0, 0);
        const i = parseInt(id.slice(5)) - 1;
        return Qt.rect(rungs.x, rungs.y + i * rungH + 8, 44, 44);
    }

    // Spine joining the rung markers.
    Rectangle {
        x: (root.width - root.colW) / 2 + 21
        y: rungs.y + 30
        width: 2
        height: root.rungH * 3
        color: Theme.line
    }
    Rectangle {
        x: (root.width - root.colW) / 2 + 21
        y: rungs.y + 30
        width: 2
        height: root.rungH * Math.max(0, root.reached - 1)
        color: Theme.copper
        Behavior on height { NumberAnimation { duration: 420; easing.type: Easing.OutCubic } }
    }

    Column {
        id: rungs
        x: (root.width - root.colW) / 2
        y: Math.max(0, (root.height - root.rungH * 4) / 2)
        width: root.colW

        // 1. Goal
        Rung {
            number: 1
            caption: "Goal"
            done: root.reached >= 1
            height: root.rungH
            current: root.reached ===1
            Text {
                objectName: "rungText"
                width: parent.width
                text: root.ladder ? root.ladder.goal : ""
                color: Theme.ink
                elide: Text.ElideRight
                font.family: Theme.sans
                font.pixelSize: Theme.px(38)
                font.weight: Font.DemiBold
            }
        }

        // 2. Assumptions: pills, click to edit in place (dictation works in the field).
        Rung {
            number: 2
            caption: "Assumptions"
            done: root.reached >= 2
            height: root.rungH
            current: root.reached ===2
            Row {
                spacing: 16
                Text {
                    objectName: "rungText"
                    visible: root.assumptions.length === 0
                    text: "—"
                    color: Theme.faint
                    font.family: Theme.sans
                    font.pixelSize: Theme.px(32)
                }
                Repeater {
                    // A count, not the array: every publish hands QML a fresh array, which would
                    // rebuild each pill (replaying its pop-in and dropping an open editor).
                    model: root.assumptions.length
                    Rectangle {
                        id: pill
                        required property int index
                        readonly property var modelData: root.assumptions[index] || ({ text: "", pinned: false })
                        property bool editing: false
                        objectName: "assumptionPill"
                        height: 54
                        width: (editing ? editor.contentWidth : label.implicitWidth) + 48
                        radius: 27
                        color: modelData.pinned ? Theme.goldA(0.12) : "transparent"
                        border.width: 1.5
                        border.color: modelData.pinned ? Theme.gold : Theme.copper
                        opacity: 0
                        scale: 0.85
                        Component.onCompleted: { opacity = 1; scale = 1; }
                        Behavior on opacity { NumberAnimation { duration: 220 } }
                        Behavior on scale { NumberAnimation { duration: 260; easing.type: Easing.OutBack } }
                        Text {
                            id: label
                            objectName: "rungText"
                            anchors.centerIn: parent
                            visible: !pill.editing
                            text: pill.modelData.text
                            color: pill.modelData.pinned ? Theme.gold : Theme.ink
                            font.family: Theme.mono
                            font.pixelSize: Theme.px(24)
                        }
                        TextInput {
                            id: editor
                            anchors.centerIn: parent
                            visible: pill.editing
                            color: Theme.gold
                            selectionColor: Theme.copper
                            font.family: Theme.mono
                            font.pixelSize: Theme.px(24)
                            cursorDelegate: Caret {}
                            Keys.priority: Keys.BeforeItem
                            Keys.onPressed: (e) => {
                                if (e.key === Qt.Key_Return || e.key === Qt.Key_Enter) {
                                    root.backend.editAssumption(root.project.key, pill.index, text);
                                    pill.editing = false;
                                    e.accepted = true;
                                } else if (e.key === Qt.Key_Escape) {
                                    pill.editing = false;
                                    e.accepted = true;
                                }
                            }
                            onActiveFocusChanged: if (!activeFocus) pill.editing = false
                        }
                        MouseArea {
                            anchors.fill: parent
                            enabled: !pill.editing
                            cursorShape: Qt.IBeamCursor
                            acceptedButtons: Qt.LeftButton | Qt.RightButton
                            onClicked: (m) => {
                                if (m.button === Qt.RightButton) {
                                    root.backend.togglePin(root.project.key, pill.index);
                                    return;
                                }
                                editor.text = pill.modelData.text;
                                pill.editing = true;
                                editor.forceActiveFocus();
                                editor.selectAll();
                            }
                        }
                    }
                }
            }
        }

        // 3. Arms: competing variants run empirically; bars fill as they finish.
        Rung {
            number: 3
            caption: "Arms"
            done: root.reached >= 3
            height: root.rungH
            current: root.reached ===3
            Column {
                spacing: 16
                Text {
                    objectName: "rungText"
                    text: root.ladder && root.ladder.armsLabel ? root.ladder.armsLabel : "—"
                    color: root.arms.length > 0 ? Theme.ink : Theme.faint
                    font.family: Theme.sans
                    font.pixelSize: Theme.px(30)
                    font.weight: Font.Medium
                }
                Row {
                    id: armRow
                    spacing: 28
                    visible: root.arms.length > 0
                    readonly property real armW: Math.min(230, (root.colW - 76 - spacing * (root.arms.length - 1))
                                                          / Math.max(1, root.arms.length))
                    Repeater {
                        model: root.arms.length  // stable bars; values rebind in place
                        Column {
                            required property int index
                            readonly property var modelData: root.arms[index] || ({ name: "", value: "", progress: 0, best: false })
                            spacing: 8
                            width: armRow.armW
                            Rectangle {
                                width: armRow.armW
                                height: 12
                                radius: 6
                                color: Theme.track
                                Rectangle {
                                    width: parent.width * Math.max(0, Math.min(1, modelData.progress))
                                    height: parent.height
                                    radius: 6
                                    color: modelData.best ? Theme.gold : Theme.copper
                                }
                            }
                            Item {
                                width: armRow.armW
                                height: 22
                                Text {
                                    id: armValue
                                    anchors.right: parent.right
                                    text: modelData.value ? modelData.value : ""
                                    color: modelData.best ? Theme.gold : Theme.dim
                                    font.family: Theme.mono
                                    font.pixelSize: Theme.px(17)
                                }
                                Text {
                                    anchors.left: parent.left
                                    anchors.right: armValue.left
                                    anchors.rightMargin: armValue.text.length > 0 ? 10 : 0
                                    text: modelData.name
                                    elide: Text.ElideRight
                                    color: modelData.best ? Theme.gold : Theme.dim
                                    font.family: Theme.mono
                                    font.pixelSize: Theme.px(17)
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. Winner
        Rung {
            number: 4
            caption: "Winner"
            done: root.reached >= 4
            height: root.rungH
            current: root.reached ===4
            Text {
                objectName: "rungText"
                text: root.winner.length > 0 ? root.winner : "—"
                color: root.winner.length > 0 ? Theme.gold : Theme.faint
                font.family: Theme.sans
                font.pixelSize: Theme.px(44)
                font.weight: Font.DemiBold
                scale: root.winner.length > 0 ? 1 : 0.96
                transformOrigin: Item.Left
                Behavior on scale { NumberAnimation { duration: 260; easing.type: Easing.OutBack } }
            }
        }
    }

    component Rung: Item {
        id: rung
        property int number: 1
        property string caption: ""
        property bool done: false
        property bool current: false
        default property alias content: body.data
        objectName: "rung"
        width: parent ? parent.width : 0
        opacity: done ? 1 : 0.32
        Behavior on opacity { NumberAnimation { duration: 260 } }

        Rectangle {
            width: 44; height: 44; radius: 22
            y: 8
            color: rung.done ? Theme.card : Theme.bg
            border.width: 1.5
            border.color: rung.current ? Theme.gold : (rung.done ? Theme.copper : Theme.faint)
            Text {
                anchors.centerIn: parent
                text: String(rung.number)
                color: rung.current ? Theme.gold : Theme.dim
                font.family: Theme.mono
                font.pixelSize: Theme.px(18)
            }
        }
        Text {
            x: 76
            y: 0
            text: rung.caption
            color: Theme.dim
            font.family: Theme.sans
            font.pixelSize: Theme.px(17)
            font.weight: Font.Medium
            font.letterSpacing: 1.6
            font.capitalization: Font.AllUppercase
        }
        Item {
            id: body
            x: 76
            y: 34
            width: parent.width - 76
            height: parent.height - 34
        }
    }
}
