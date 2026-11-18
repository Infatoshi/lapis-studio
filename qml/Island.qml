import QtQuick

// One project island: name, pulse dot, one sparkbar and one headline metric.
Item {
    id: root
    property var proj
    property int number: 1
    property bool selected: false
    property real recede: 0        // 0..1 while the Constellation zooms into a sibling
    property bool morphing: false  // this island is the shared element: the zoom layer stands in for it
    property real pulse: -1        // the Constellation's shared halo clock (0..1); -1 = no halo
    signal clicked()

    readonly property string jobName: proj && proj.jobName ? proj.jobName : ""
    readonly property int fresh: proj && proj.fresh ? proj.fresh : 0

    readonly property bool active: proj ? proj.active : false
    readonly property var spark: proj && proj.spark ? proj.spark : []
    readonly property var dots: proj && proj.dots ? proj.dots : []
    readonly property int sparkBest: proj && proj.sparkBest !== undefined ? proj.sparkBest : -1

    // Siblings recede (0.94, transparent) while the chosen island grows into the next lens.
    scale: 1 - 0.06 * recede
    property real born: 0
    Component.onCompleted: born = 1
    Behavior on born { NumberAnimation { duration: 260; easing.type: Easing.OutCubic } }
    opacity: morphing ? 0 : born * (1 - recede)

    // Focus and hover lift on a spring, so arrow-key travel across islands feels physical.
    readonly property real lift: mouse.pressed ? 0.97 : (selected ? 1.03 : (mouse.containsMouse ? 1.015 : 1))

    Item {
        id: body
        objectName: "islandBody"
        anchors.fill: parent
        scale: root.lift
        Behavior on scale { SpringAnimation { spring: Theme.springK; damping: Theme.springDamping; epsilon: 0.0005 } }

        Rectangle {
            anchors.fill: parent
            radius: 26
            color: mouse.containsMouse || root.selected ? Theme.cardHi : Theme.card
        }

        Text {
            x: 28
            y: 24
            text: root.number <= 9 ? String(root.number) : ""
            color: Theme.faint
            font.family: Theme.mono
            font.pixelSize: Theme.px(16)
        }

        // Active pulse.
        Item {
            x: parent.width - 40
            y: 26
            width: 14
            height: 14
            Rectangle {
                id: halo
                objectName: "islandHalo"
                anchors.centerIn: parent
                width: 14; height: 14; radius: 7
                color: Theme.gold
                // Driven by the Constellation's one shared pulse clock: no per-island animations.
                scale: 1 + 1.4 * Math.max(0, root.pulse)
                opacity: 0.45 * (1 - Math.max(0, root.pulse))
                visible: root.active && root.pulse >= 0 && root.pulse < 0.999
            }
            Rectangle {
                anchors.centerIn: parent
                width: 10; height: 10; radius: 5
                color: root.active ? Theme.gold : (root.proj && root.proj.status === "finished" ? Theme.copper : Theme.faint)
            }
        }

        Text {
            id: name
            x: 28
            y: 56
            width: parent.width - 56
            text: root.proj ? root.proj.name : ""
            color: Theme.ink
            elide: Text.ElideRight
            font.family: Theme.sans
            font.pixelSize: Theme.px(31)
            font.weight: Font.DemiBold
        }

        // Plain-English job name ("Sweeping sm120 FP4 tile sizes") under a cryptic folder name
        // or a push that has run over 15 minutes.
        Text {
            objectName: "islandJob"
            x: 28
            y: name.y + name.height + 2
            width: parent.width - 56
            visible: root.jobName.length > 0
            text: root.jobName
            color: Theme.text
            opacity: 0.78
            elide: Text.ElideRight
            font.family: Theme.sans
            font.pixelSize: Theme.px(18)
        }

        // "+3": turns the agent wrote while this island was off screen. Sits left of the pulse
        // dot and never moves anything; opening the island clears it.
        Rectangle {
            objectName: "freshBadge"
            visible: root.fresh > 0
            anchors.right: parent.right
            anchors.rightMargin: 56
            y: 21
            width: freshText.implicitWidth + 16
            height: 24
            radius: 12
            color: "transparent"
            border.width: 1
            border.color: Qt.rgba(0.878, 0.624, 0.243, 0.55)
            Text {
                id: freshText
                objectName: "freshBadgeText"
                anchors.centerIn: parent
                text: "+" + root.fresh
                color: Theme.gold
                font.family: Theme.mono
                font.pixelSize: Theme.px(14)
            }
        }

        // Sparkbar (or one dot per session when there is nothing to chart yet).
        Row {
            id: bars
            x: 28
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 30
            spacing: 5
            height: 40
            visible: root.spark.length > 0
            Repeater {
                // A count, not the array: a republished array would rebuild every bar.
                model: root.spark.length
                Rectangle {
                    required property int index
                    width: 9
                    radius: 2
                    height: Math.max(4, 40 * (root.spark[index] || 0))
                    anchors.bottom: parent.bottom
                    readonly property bool win: index === root.sparkBest
                    color: win ? Theme.gold : Theme.copper
                    opacity: win ? 1 : 0.7
                }
            }
        }
        Row {
            x: 28
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 34
            spacing: 7
            visible: root.spark.length === 0
            Repeater {
                model: Math.min(root.dots.length, 10)
                Rectangle {
                    required property int index
                    width: 9; height: 9; radius: 4.5
                    color: root.dots[index] >= 3 ? Theme.gold : (root.dots[index] === 2 ? Theme.copper : Theme.faint)
                }
            }
        }

        Text {
            anchors.right: parent.right
            anchors.rightMargin: 28
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 28
            width: parent.width - (bars.visible ? bars.width + 76 : 120)
            horizontalAlignment: Text.AlignRight
            elide: Text.ElideLeft
            text: root.proj ? root.proj.headline : ""
            color: Theme.gold
            font.family: Theme.mono
            font.pixelSize: Theme.px(20)
        }

    }

    MouseArea {
        id: mouse
        anchors.fill: parent
        hoverEnabled: true
        cursorShape: Qt.PointingHandCursor
        onClicked: root.clicked()
    }
}
