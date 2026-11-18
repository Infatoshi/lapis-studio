import QtQuick

// Lens 1: spacious OLED project islands, joined by faint constellation lines.
Item {
    id: root
    property var projects: []
    property int selectedIndex: 0
    property real recede: 0       // 0..1: the zoom into a project (siblings shrink to 0.94 and fade)
    property int focusIndex: -1   // the island that is the shared element of that zoom
    property var visited: []      // project keys zoomed into this session (StudioApp keeps it)
    property bool cursorActive: false  // the app-wide focus cursor is wrapping the selected island
    signal enter(int index)
    signal zoomBy(real factor)
    signal newSession()

    // Tab predictor: which island to open next. Needs-attention first, then live work not yet
    // looked at, then anything with artifacts not yet looked at, else the focused island.
    // No projects at all: start one.
    readonly property var predictedAction: {
        if (count === 0)
            return { targetId: "newSession", label: "New Session", promptSuggestion: "", actionKind: "newSession", index: -1 };
        let pick = -1;
        const fresh = (p) => visited.indexOf(p.key) < 0;
        for (let i = 0; i < count && pick < 0; ++i)
            if (projects[i].status === "needs-ok" && fresh(projects[i])) pick = i;
        for (let i = 0; i < count && pick < 0; ++i)
            if (projects[i].active && fresh(projects[i])) pick = i;
        for (let i = 0; i < count && pick < 0; ++i)
            if (projects[i].artifacts.length > 0 && fresh(projects[i])) pick = i;
        if (pick < 0)
            pick = Math.max(0, Math.min(selectedIndex, count - 1));
        const stage = projects[pick].artifacts.length > 0;
        return { targetId: "island:" + pick, label: stage ? "Open Stage" : "Open Ladder", promptSuggestion: "",
                 actionKind: "open", index: pick };
    }
    // Where the predicted target sits, in this item's coordinates.
    function targetRect(id) {
        if (id.startsWith("island:"))
            return islandRect(parseInt(id.slice(7)));
        if (id === "newSession")
            return Qt.rect(newHint.x, newHint.y, newHint.width, newHint.height);
        return Qt.rect(0, 0, 0, 0);
    }

    // Nothing to show yet: one quiet "+" in the middle, the target of the New Session prediction.
    Rectangle {
        id: newHint
        visible: root.count === 0
        anchors.centerIn: parent
        width: 64; height: 64; radius: 32
        color: Theme.card
        Text {
            anchors.centerIn: parent
            text: "+"
            color: Theme.gold
            font.family: Theme.sans
            font.pixelSize: 30
        }
        TapHandler { onTapped: root.newSession() }
    }

    readonly property int count: Math.min(projects.length, 12)
    readonly property int columns: count <= 4 ? 2 : (count <= 9 ? 3 : 4)
    readonly property int rows: Math.max(1, Math.ceil(count / columns))
    readonly property real cellW: Math.min(560, (width - 200) / columns)
    readonly property real cellH: Math.min(330, (height - 240) / rows)
    // Islands grow with the UI zoom until their cell is full.
    readonly property real islandW: Math.min(400 * Theme.fontScale, cellW - 90)
    readonly property real islandH: Math.min(206 * Theme.fontScale, cellH - 70)

    function islandCenter(i) {
        const c = i % columns;
        const r = Math.floor(i / columns);
        const inRow = Math.min(columns, count - r * columns);
        const ox = (width - cellW * inRow) / 2;
        const oy = 40 + (height - 80 - cellH * rows) / 2;  // no title row: centred
        const jx = Math.sin(i * 2.3 + 0.7) * 26 + (r % 2 ? 34 : -20);
        const jy = Math.cos(i * 1.7 + 0.4) * 20;
        return Qt.point(ox + c * cellW + cellW / 2 + jx, oy + r * cellH + cellH / 2 + jy);
    }

    // Geometry of island i (ix, iy, iw, ih): where the shared-element zoom starts and ends.
    function islandRect(i) {
        const c = islandCenter(i);
        return Qt.rect(c.x - islandW / 2, c.y - islandH / 2, islandW, islandH);
    }

    readonly property var edges: {
        const out = [];
        for (let i = 0; i < count; ++i) {
            if ((i % columns) < columns - 1 && i + 1 < count)
                out.push([i, i + 1]);
            if (i + columns < count)
                out.push([i, i + columns]);
        }
        return out;
    }

    // Nothing scrolls on the Constellation, so a plain wheel or two-finger scroll zooms the UI.
    // (Cmd+wheel and pinch zoom anywhere; StudioApp handles those.)
    WheelHandler {
        objectName: "constellationWheel"
        enabled: root.recede < 0.01
        acceptedModifiers: Qt.NoModifier
        onWheel: (ev) => {
            const dy = ev.angleDelta.y !== 0 ? ev.angleDelta.y : ev.pixelDelta.y;
            if (dy !== 0)
                root.zoomBy(Math.pow(1.0012, dy));
        }
    }

    Repeater {
        model: root.edges
        Rectangle {
            required property var modelData
            readonly property point a: root.islandCenter(modelData[0])
            readonly property point b: root.islandCenter(modelData[1])
            x: a.x
            y: a.y
            width: Math.hypot(b.x - a.x, b.y - a.y)
            height: 1.5
            color: Theme.line
            opacity: 1 - root.recede
            antialiasing: true
            transformOrigin: Item.TopLeft
            rotation: Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI
        }
    }

    Repeater {
        model: root.count
        Island {
            required property int index
            objectName: "island"
            proj: root.projects[index]
            number: index + 1
            selected: index === root.selectedIndex
            recede: index === root.focusIndex ? 0 : root.recede
            morphing: index === root.focusIndex && root.recede > 0.001
            pulse: root.pulse
            width: root.islandW
            height: root.islandH
            readonly property point c: root.islandCenter(index)
            x: c.x - width / 2
            y: c.y - height / 2
            onClicked: root.enter(index)
        }
    }

    // Keyboard focus ring. Every island has the same size, so the ring's size is a constant
    // (islandW + 18 by islandH + 18) and only x / y move: a 120 ms OutExpo glide that lands
    // without overshoot. A hop retargets mid-flight from wherever the ring is, so key repeat
    // reads as one continuous motion. (It used to spring width and height too, underdamped at
    // 4.5 / 0.36: ~700 ms of wobble and a re-tessellated border on every frame.)
    Rectangle {
        id: ring
        objectName: "focusRing"
        readonly property rect target: root.islandRect(Math.max(0, Math.min(root.selectedIndex, root.count - 1)))
        visible: root.count > 0
        x: target.x - 9
        y: target.y - 9
        width: root.islandW + 18
        height: root.islandH + 18
        radius: 34
        color: "transparent"
        border.width: 1.5
        border.color: Qt.rgba(0.878, 0.624, 0.243, 0.5)
        // StudioApp's focus cursor takes over whenever it is on screen (one ring, not two).
        opacity: root.cursorActive ? 0 : 1 - root.recede
        antialiasing: true
        Behavior on x { enabled: root.recede < 0.01; NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutExpo } }
        Behavior on y { enabled: root.recede < 0.01; NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutExpo } }
    }

    // One shared, linear wave clock for every island that wants attention (each island derives
    // its two staggered waves from it). It runs only while such an island exists and the
    // Constellation is on screen and settled; at rest it costs nothing.
    readonly property bool anyAttention: {
        for (let i = 0; i < count; ++i) {
            const p = projects[i];
            if (p && (p.active || p.status === "needs-ok" || (p.fresh || 0) > 0))
                return true;
        }
        return false;
    }
    property real pulse: 0
    NumberAnimation on pulse {
        running: root.anyAttention && root.visible && root.recede < 0.01
        loops: Animation.Infinite
        from: 0
        to: 1
        duration: 2400
    }
}
