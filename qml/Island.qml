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
    property bool draggable: false // the Constellation is settled: press-and-move drags the card
    signal clicked()
    // Pointer travel since the press, in the parent's coordinates. dragMoved fires per move
    // while the card rides a Translate (no relayout); dragEnded fires once, on release.
    signal dragMoved(real dx, real dy)
    signal dragEnded(real dx, real dy)
    signal dragCanceled()
    property real dragDx: 0
    property real dragDy: 0
    readonly property bool dragging: mouse.moved
    transform: Translate { x: root.dragDx; y: root.dragDy }
    z: dragging ? 10 : 0

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
    readonly property real lift: dragging ? 1.04 : (mouse.pressed ? 0.97 : (selected ? 1.03 : (mouse.containsMouse ? 1.015 : 1)))

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
                border.color: Theme.goldA(0.55)
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
            // Active blinking dot.
            Item {
                width: 14
                height: 14
                anchors.verticalCenter: parent.verticalCenter
                Rectangle {
                    id: halo
                    objectName: "islandHalo"
                    anchors.centerIn: parent
                    width: 14; height: 14; radius: 7
                    color: Theme.gold
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
        }

        Text {
            id: name
            x: 28
            y: 56
            width: parent.width - 56
            text: root.proj ? root.proj.name : ""
            color: Theme.ink
            elide: Text.ElideRight
            // Tab islands are named by a phrase ("Add support to PS4 controller"): shrink before eliding.
            fontSizeMode: Text.HorizontalFit
            minimumPixelSize: Theme.px(22)
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
        cursorShape: moved ? Qt.ClosedHandCursor : Qt.PointingHandCursor
        preventStealing: true
        property point pressAt
        property bool moved: false
        property bool wasDragged: false  // clicked fires after released: a drag never opens the card
        function endDrag(commit) {
            wasDragged = moved;
            if (moved)
                commit ? root.dragEnded(root.dragDx, root.dragDy) : root.dragCanceled();
            moved = false;
            root.dragDx = 0;
            root.dragDy = 0;
        }
        onPressed: (m) => {
            pressAt = mapToItem(root.parent, m.x, m.y);
            moved = false;
            wasDragged = false;
        }
        onPositionChanged: (m) => {
            if (!pressed || !root.draggable)
                return;
            const p = mapToItem(root.parent, m.x, m.y);
            const dx = p.x - pressAt.x, dy = p.y - pressAt.y;
            if (!moved && Math.hypot(dx, dy) < 6)
                return;  // a click with a shaky hand stays a click
            moved = true;
            root.dragDx = dx;
            root.dragDy = dy;
            root.dragMoved(dx, dy);
        }
        onReleased: endDrag(true)
        onCanceled: endDrag(false)
        onClicked: if (!wasDragged) root.clicked()
        // Hover-to-ask: the island reports itself as a session target (window coordinates).
        onContainsMouseChanged: root.reportHover()
    }
    property QtObject targets: null
    function reportHover() {
        if (!targets || !proj)
            return;
        if (mouse.containsMouse) {
            const r = root.mapToItem(null, 0, 0, root.width, root.height);
            const job = proj.jobName ? proj.jobName : "";
            const state = proj.stateLine ? proj.stateLine : "";
            targets.setHoverTarget("session", proj.key, "", "", proj.name,
                                   job.length > 0 && state.length > 0 ? job + " — " + state : job + state,
                                   r.x, r.y, r.width, r.height);
        } else if (targets.activeTarget.kind === "session" && targets.activeTarget.sessionId === proj.key) {
            targets.clearHoverTarget("session");
        }
    }
}
