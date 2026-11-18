import QtQuick

// The element-aware focus cursor: one warm-gold (#e09f3e) ring that springs to whatever the
// controller or keyboard is on (a Constellation island, a Stage artifact or term pill, the
// composer's selection), opens up a few pixels once it lands, and carries a compact action chip
// ("✕ Expand", "E Prompt").
//
// Motion: a critically damped spring (omega 36 rad/s, zeta 1) on x, y, width and height,
// integrated in fixed 1/240 s sub-steps whatever the display rate, so it never overshoots and
// lands the same on a 60 Hz or 120 Hz panel. The integrator runs only while the ring is moving;
// at rest it costs nothing.
Item {
    id: root
    objectName: "focusCursor"
    property rect target: Qt.rect(0, 0, 0, 0)  // in this item's coordinates
    property bool shown: false
    property real cornerRadius: 26
    property bool round: false        // fully rounded ends (pills, small targets)
    property string hint: ""          // "✕ Expand"; empty = no chip
    property string hintSide: "bottom"  // "bottom" | "right"

    readonly property bool active: shown && target.width > 0 && target.height > 0

    // Spring state (centre and size) and velocities.
    property real cx: 0
    property real cy: 0
    property real cw: 0
    property real ch: 0
    property real vx: 0
    property real vy: 0
    property real vw: 0
    property real vh: 0
    property bool settled: true
    readonly property real omega: 36
    readonly property real stepS: 1 / 240

    function snap() {
        cx = target.x + target.width / 2;
        cy = target.y + target.height / 2;
        cw = target.width;
        ch = target.height;
        vx = vy = vw = vh = 0;
        settled = true;
    }
    function retarget() {
        if (!active)
            return;
        if (opacity < 0.05 || cw <= 0) {
            snap();  // appearing: start on the element, never fly in from a stale spot
            return;
        }
        settled = false;
    }
    onTargetChanged: retarget()
    onActiveChanged: retarget()

    function integrate(dt) {
        const tx = target.x + target.width / 2, ty = target.y + target.height / 2;
        const tw = target.width, th = target.height;
        const k = omega * omega, c = 2 * omega;  // critical damping
        let t = Math.min(dt, 1 / 20);  // a stalled frame never jumps a whole hop
        let x = cx, y = cy, w = cw, h = ch, ax = vx, ay = vy, aw = vw, ah = vh;
        while (t > 1e-6) {
            const s = Math.min(stepS, t);
            ax += (k * (tx - x) - c * ax) * s; x += ax * s;
            ay += (k * (ty - y) - c * ay) * s; y += ay * s;
            aw += (k * (tw - w) - c * aw) * s; w += aw * s;
            ah += (k * (th - h) - c * ah) * s; h += ah * s;
            t -= s;
        }
        const near = Math.abs(tx - x) < 0.3 && Math.abs(ty - y) < 0.3 && Math.abs(tw - w) < 0.3 && Math.abs(th - h) < 0.3
                     && Math.abs(ax) + Math.abs(ay) + Math.abs(aw) + Math.abs(ah) < 4;
        if (near) {
            snap();
            return;
        }
        cx = x; cy = y; cw = w; ch = h;
        vx = ax; vy = ay; vw = aw; vh = ah;
    }
    FrameAnimation {
        running: root.active && !root.settled && root.visible
        onTriggered: root.integrate(frameTime)
    }

    // Lands, then opens up a few pixels: "this is the one".
    property real grow: settled && active ? 1 : 0
    Behavior on grow { NumberAnimation { duration: 260; easing.type: Easing.OutCubic } }
    readonly property real pad: 7 + 5 * grow
    readonly property real rw: cw + pad * 2
    readonly property real rh: ch + pad * 2

    opacity: active ? 1 : 0
    visible: opacity > 0.01
    Behavior on opacity { NumberAnimation { duration: 180; easing.type: Easing.OutCubic } }

    Rectangle {
        id: ring
        objectName: "focusCursorRing"
        x: root.cx - width / 2
        y: root.cy - height / 2
        width: root.rw
        height: root.rh
        radius: root.round ? height / 2 : Math.min(height / 2, root.cornerRadius + root.pad)
        color: "transparent"
        border.width: 2
        border.color: Theme.gold
        antialiasing: true
    }
    // A faint outer halo that widens with the landing.
    Rectangle {
        x: ring.x - 4 * root.grow
        y: ring.y - 4 * root.grow
        width: ring.width + 8 * root.grow
        height: ring.height + 8 * root.grow
        radius: root.round ? height / 2 : ring.radius + 4 * root.grow
        color: "transparent"
        border.width: 1
        border.color: Qt.rgba(0.878, 0.624, 0.243, 0.22 * root.grow)
        antialiasing: true
        visible: root.grow > 0.02
    }

    // "✕ Expand": the action this element takes, in the current input's glyph.
    Rectangle {
        id: chip
        objectName: "focusCursorHint"
        visible: root.hint.length > 0
        readonly property real bx: root.hintSide === "right" ? ring.x + ring.width + 12 : root.cx - width / 2
        readonly property real by: root.hintSide === "right" ? ring.y : ring.y + ring.height + 8
        x: Math.max(8, Math.min(root.width - width - 8, bx))
        y: Math.max(8, Math.min(root.height - height - 8, by))
        width: hintText.implicitWidth + 20
        height: 26
        radius: 13
        color: "#000000"
        border.width: 1
        border.color: Qt.rgba(0.878, 0.624, 0.243, 0.6)
        opacity: 0.4 + 0.6 * root.grow
        Text {
            id: hintText
            objectName: "focusCursorHintText"
            anchors.centerIn: parent
            text: root.hint
            color: Theme.gold
            font.family: Theme.mono
            font.pixelSize: 13
        }
    }
}
