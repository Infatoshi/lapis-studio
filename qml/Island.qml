import QtQuick

// One project island: name, then its 1-to-2-turn state in plain English (what the agent is on,
// and its latest step or, in gold, the decision it waits on). No uptime, no sparkbar; the metric
// shows only when pinned. An island that wants attention radiates two staggered gold waves.
Item {
    id: root
    property var proj
    property int number: 1
    property bool selected: false
    property real recede: 0        // 0..1 while the Constellation zooms into a sibling
    property bool morphing: false  // this island is the shared element: the zoom layer stands in for it
    property real pulse: -1        // the Constellation's shared wave clock (0..1, linear); -1 = off
    signal clicked()

    readonly property int fresh: proj && proj.fresh ? proj.fresh : 0
    readonly property bool active: proj ? proj.active === true : false
    readonly property bool blocker: proj ? proj.blocker === true : false
    readonly property string nowLine: proj && proj.nowLine ? proj.nowLine : ""
    readonly property string lastLine: proj && proj.lastLine ? proj.lastLine : ""
    readonly property bool metricPinned: proj ? proj.metricPinned === true : false
    readonly property string metric: proj && proj.headline ? proj.headline : ""
    // Needs a look: waiting on an OK, turns landed while away, or live work.
    readonly property bool attention: proj ? (proj.status === "needs-ok" || fresh > 0 || active) : false

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

        // Radiating border waves: two rounded rects, half a beat apart, grow out of the card's
        // edge and fade 0.35 -> 0. Geometry never changes (fixed card-sized rects under a Scale
        // transform), so the border is tessellated once; each frame moves two matrices and two
        // opacities, driven by the Constellation's one shared clock (no per-island animations).
        Repeater {
            model: root.attention && root.pulse >= 0 ? 2 : 0
            Rectangle {
                required property int index
                objectName: "islandWave"
                readonly property real t: (root.pulse + index * 0.5) % 1
                readonly property real reach: 3 + 22 * (1 - (1 - t) * (1 - t))  // px out from the edge, OutQuad
                width: body.width
                height: body.height
                radius: 26
                color: "transparent"
                border.width: 1.5
                border.color: Theme.gold
                opacity: 0.35 * (1 - t)
                antialiasing: true
                transform: Scale {
                    origin.x: body.width / 2
                    origin.y: body.height / 2
                    xScale: 1 + 2 * reach / Math.max(1, body.width)
                    yScale: 1 + 2 * reach / Math.max(1, body.height)
                }
            }
        }

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

        // Top right: the pinned metric (only when pinned) and "+3", turns the agent wrote while
        // this island was off screen. Opening the island clears the badge.
        Row {
            anchors.right: parent.right
            anchors.rightMargin: 28
            y: 21
            spacing: 10
            Text {
                objectName: "islandMetric"
                visible: root.metricPinned && root.metric.length > 0
                anchors.verticalCenter: parent.verticalCenter
                text: root.metric
                color: Theme.gold
                font.family: Theme.mono
                font.pixelSize: Theme.px(16)
            }
            Rectangle {
                objectName: "freshBadge"
                visible: root.fresh > 0
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

        // Line 1: what the agent is on ("Sweeping sm120 FP4 tile sizes"), from the job or goal.
        Text {
            objectName: "islandJob"
            x: 28
            y: name.y + name.height + 2
            width: parent.width - 56
            visible: root.nowLine.length > 0
            text: root.nowLine
            color: Theme.text
            opacity: 0.85
            elide: Text.ElideRight
            maximumLineCount: 1
            font.family: Theme.sans
            font.pixelSize: Theme.px(18)
        }

        // Line 2: the latest completed step, or the decision the agent waits on (gold).
        Text {
            objectName: "islandLast"
            x: 28
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 26
            width: parent.width - 56
            visible: root.lastLine.length > 0
            text: root.lastLine
            color: root.blocker ? Theme.gold : Theme.dim
            elide: Text.ElideRight
            maximumLineCount: 1
            font.family: Theme.sans
            font.pixelSize: Theme.px(17)
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
