import QtQuick

// The Constellation's free pad cursor: a small gold reticle centred on (x, y) of its parent.
// It only draws; Constellation integrates its motion (left stick, 240 Hz) and hit-tests it.
// Over a card it opens up and softens (the aim-assist slowdown reads as a looser ring); while
// ✕ holds a card it tightens into a solid grip dot.
Item {
    id: root
    objectName: "mapCursor"
    property real cx: 0
    property real cy: 0
    property bool overCard: false
    property bool grabbing: false
    property bool marquee: false
    x: cx - width / 2
    y: cy - height / 2
    width: 44
    height: 44

    // Outer ring: wide and faint over empty space, tight and brighter over a card.
    Rectangle {
        id: ring
        anchors.centerIn: parent
        property real d: root.grabbing ? 18 : (root.overCard ? 34 : 24)
        width: d
        height: d
        radius: d / 2
        color: "transparent"
        border.width: 1.5
        border.color: Theme.goldA(root.grabbing ? 0.95 : (root.overCard ? 0.75 : 0.55))
        antialiasing: true
        Behavior on d { SpringAnimation { spring: Theme.springK; damping: Theme.springDamping; epsilon: 0.05 } }
    }
    // Soft halo so it reads on both the black field and a card.
    Rectangle {
        anchors.centerIn: parent
        width: ring.d + 10
        height: width
        radius: width / 2
        color: "transparent"
        border.width: 4
        border.color: Theme.goldA(0.10)
        antialiasing: true
    }
    // Centre dot: the precise hot spot.
    Rectangle {
        anchors.centerIn: parent
        width: root.grabbing ? 8 : 4
        height: width
        radius: width / 2
        color: Theme.gold
        antialiasing: true
    }
}
