import QtQuick

// The Tab next-action predictor: a warm-gold ring around the predicted target plus a
// "label  ⇥ Tab" pill. The pulse runs only while the ring is shown and the user is idle, so it
// costs nothing while hidden or while keys are flying.
Item {
    id: root
    objectName: "predictorRing"
    property rect target: Qt.rect(0, 0, 0, 0)  // in this item's coordinates
    property string label: ""
    property bool round: false   // a circle (pager dot, rung marker) instead of a rounded card
    property real cardRadius: 26
    property bool shown: false
    property bool idle: true
    property string badgeSide: "top"  // "top" | "bottom" | "left" | "right"

    readonly property bool active: shown && target.width > 0 && target.height > 0
    readonly property bool live: active && idle

    // What is drawn. It follows the inputs only while the ring is live; while it fades out it
    // keeps the old target and label, so a new prediction never flashes over a view in transition.
    // One snapshot handler instead of five Binding objects: while keys fly (not idle) nothing
    // here is evaluated at all; the snapshot is taken only when the ring is live.
    property rect sTarget: Qt.rect(0, 0, 0, 0)
    property string sLabel: ""
    property bool sRound: false
    property real sRadius: 26
    property string sSide: "top"
    function snap() {
        if (!live)
            return;
        sTarget = target;
        sLabel = label;
        sRound = round;
        sRadius = cardRadius;
        sSide = badgeSide;
    }
    onLiveChanged: snap()
    onTargetChanged: snap()
    onLabelChanged: snap()
    onRoundChanged: snap()
    onCardRadiusChanged: snap()
    onBadgeSideChanged: snap()

    readonly property real pad: sRound ? 10 : 9
    // The centre glides from target to target on the focus ring's 120 ms OutExpo hop, and only
    // while the ring is live: a hidden ring jumps straight to its next snapshot.
    property real cx: sTarget.x + sTarget.width / 2
    property real cy: sTarget.y + sTarget.height / 2
    Behavior on cx { enabled: root.live && root.visible; NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutExpo } }
    Behavior on cy { enabled: root.live && root.visible; NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutExpo } }
    readonly property real rw: sRound ? Math.max(sTarget.width, sTarget.height) + pad * 2 : sTarget.width + pad * 2
    readonly property real rh: sRound ? rw : sTarget.height + pad * 2

    opacity: live ? 1 : 0
    visible: opacity > 0.01
    Behavior on opacity { NumberAnimation { duration: 220; easing.type: Easing.OutCubic } }

    // 0..1 breathing phase: drives the halo's spread and fade, 1.6 s per breath.
    property real phase: 0
    SequentialAnimation on phase {
        running: root.active && root.idle && root.visible
        loops: Animation.Infinite
        NumberAnimation { from: 0; to: 1; duration: 1600; easing.type: Easing.InOutSine }
        NumberAnimation { from: 1; to: 0; duration: 1600; easing.type: Easing.InOutSine }
    }

    // Halo: a second ring that breathes outward.
    Rectangle {
        id: halo
        x: root.cx - width / 2
        y: root.cy - height / 2
        width: root.rw + 10 + 8 * root.phase
        height: root.rh + 10 + 8 * root.phase
        radius: root.sRound ? width / 2 : root.sRadius + 10
        color: "transparent"
        border.width: 2
        border.color: Qt.rgba(0.878, 0.624, 0.243, 0.10 + 0.22 * (1 - root.phase))
        antialiasing: true
    }
    Rectangle {
        id: ring
        objectName: "predictorRingShape"
        x: root.cx - width / 2
        y: root.cy - height / 2
        width: root.rw
        height: root.rh
        radius: root.sRound ? width / 2 : root.sRadius + 6
        color: "transparent"
        border.width: 2
        border.color: Qt.rgba(0.878, 0.624, 0.243, 0.70 + 0.30 * root.phase)  // #e09f3e
        antialiasing: true
    }

    // "Open Stage  ⇥ Tab"
    Rectangle {
        id: badge
        objectName: "tabBadge"
        visible: root.sLabel.length > 0
        readonly property real bx: root.sSide === "right" ? ring.x + ring.width + 14
                                   : root.sSide === "left" ? ring.x - width - 14
                                   : root.cx - width / 2
        readonly property real by: root.sSide === "right" || root.sSide === "left" ? root.cy - height / 2
                                   : root.sSide === "bottom" ? ring.y + ring.height + 12
                                   : ring.y - height / 2
        x: Math.max(8, Math.min(root.width - width - 8, bx))
        y: Math.max(8, Math.min(root.height - height - 8, by))
        width: badgeRow.implicitWidth + 26
        height: 34
        radius: 17
        color: "#000000"
        border.width: 1.5
        border.color: Theme.gold
        Row {
            id: badgeRow
            anchors.centerIn: parent
            spacing: 12
            Text {
                objectName: "tabBadgeLabel"
                text: root.sLabel
                color: Theme.ink
                font.family: Theme.sans
                font.pixelSize: 16
                font.weight: Font.DemiBold
                anchors.verticalCenter: parent.verticalCenter
            }
            Rectangle {
                width: tabKey.implicitWidth + 14
                height: 22
                radius: 6
                color: Qt.rgba(0.878, 0.624, 0.243, 0.16)
                anchors.verticalCenter: parent.verticalCenter
                Text {
                    id: tabKey
                    anchors.centerIn: parent
                    text: "⇥ Tab"
                    color: Theme.gold
                    font.family: Theme.mono
                    font.pixelSize: 13
                }
            }
        }
    }
}
