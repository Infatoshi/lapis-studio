import QtQuick

// Lens 1: spacious OLED project islands, joined by faint constellation lines.
// Islands can be dragged anywhere (the position persists per project); the top bar filters to
// the sessions that need the human (All · Needs Me) and opens the session review.
Item {
    id: root
    property var projects: []
    property int selectedIndex: 0  // a project index (into projects), not a slot
    property real recede: 0       // 0..1: the zoom into a project (siblings shrink to 0.94 and fade)
    property int focusIndex: -1   // the island that is the shared element of that zoom
    property var visited: []      // project keys zoomed into this session (StudioApp keeps it)
    property bool cursorActive: false  // the app-wide focus cursor is wrapping the selected island
    property QtObject targets: null    // backend.targets: islands report hover as session targets
    property string filter: "all"      // "all" | "needs": only the islands waiting on the human
    property var cardPositions: ({})   // {projectKey: {x, y}}: card centre as a fraction of this item
    signal enter(int index)
    signal zoomBy(real factor)
    signal newSession()
    signal cardMoved(string key, real x, real y)
    signal filterPicked(string filter)
    signal resetLayout()
    signal reviewRequested()
    signal musicRequested()
    property string musicLabel: "♪"     // "♪ night drive" while a track plays
    property bool musicPlaying: false

    // Which projects have a slot on screen, in order (project indices). The filter never
    // reorders: islands keep their relative order, so muscle memory survives toggling it.
    readonly property var slots: {
        const out = [];
        for (let i = 0; i < projects.length && out.length < 12; ++i)
            if (filter !== "needs" || projects[i].needsMe === true)
                out.push(i);
        return out;
    }
    readonly property int count: slots.length
    function slotOf(i) { return slots.indexOf(i); }
    readonly property int needsCount: {
        let n = 0;
        for (let i = 0; i < projects.length; ++i)
            if (projects[i].needsMe === true) ++n;
        return n;
    }
    // Custom positions that apply to a project on screen (the reset action shows only then).
    readonly property bool hasCustomLayout: {
        for (let i = 0; i < projects.length; ++i)
            if (cardPositions[projects[i].key] !== undefined) return true;
        return false;
    }
    // Keyboard travel in slot order: d = +-1 steps (wrapping), +-columns moves a row (clamped).
    function stepSelection(i, d, wrap) {
        if (count === 0)
            return i;
        const s = Math.max(0, slotOf(i));
        const t = wrap ? (s + d + count) % count : Math.max(0, Math.min(count - 1, s + d));
        return slots[t];
    }
    function slotProject(s) { return s >= 0 && s < count ? slots[s] : -1; }

    // Tab predictor: which island to open next. Needs-attention first, then live work not yet
    // looked at, then anything with artifacts not yet looked at, else the focused island.
    // No projects at all: start one. A filter that leaves nothing on screen predicts nothing.
    readonly property var predictedAction: {
        if (projects.length === 0)
            return { targetId: "newSession", label: "New Session", promptSuggestion: "", actionKind: "newSession", index: -1 };
        if (count === 0)
            return null;
        let pick = -1;
        const fresh = (p) => visited.indexOf(p.key) < 0;
        for (let s = 0; s < count && pick < 0; ++s)
            if (projects[slots[s]].status === "needs-ok" && fresh(projects[slots[s]])) pick = slots[s];
        for (let s = 0; s < count && pick < 0; ++s)
            if (projects[slots[s]].active && fresh(projects[slots[s]])) pick = slots[s];
        for (let s = 0; s < count && pick < 0; ++s)
            if (projects[slots[s]].artifacts.length > 0 && fresh(projects[slots[s]])) pick = slots[s];
        if (pick < 0)
            pick = slotOf(selectedIndex) >= 0 ? selectedIndex : slots[0];
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
        visible: root.projects.length === 0
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
    // The Needs Me filter with nothing waiting: one calm line, no empty grid.
    Text {
        objectName: "needsEmpty"
        visible: root.projects.length > 0 && root.count === 0
        anchors.centerIn: parent
        text: "Nothing needs you."
        color: Theme.dim
        opacity: 1 - root.recede
        font.family: Theme.sans
        font.pixelSize: Theme.px(22)
    }

    readonly property int columns: count <= 4 ? 2 : (count <= 9 ? 3 : 4)
    readonly property int rows: Math.max(1, Math.ceil(count / columns))
    readonly property real cellW: Math.min(560, (width - 200) / columns)
    readonly property real cellH: Math.min(330, (height - 240) / rows)
    // Islands grow with the UI zoom until their cell is full.
    readonly property real islandW: Math.min(400 * Theme.fontScale, cellW - 90)
    readonly property real islandH: Math.min(206 * Theme.fontScale, cellH - 70)
    readonly property real topBarH: 64  // dragged cards never slide under the filter bar

    // Auto-layout centre of slot s.
    function slotCenter(s) {
        const c = s % columns;
        const r = Math.floor(s / columns);
        const inRow = Math.min(columns, count - r * columns);
        const ox = (width - cellW * inRow) / 2;
        const oy = 40 + (height - 80 - cellH * rows) / 2;  // no title row: centred
        const jx = Math.sin(s * 2.3 + 0.7) * 26 + (r % 2 ? 34 : -20);
        const jy = Math.cos(s * 1.7 + 0.4) * 20;
        return Qt.point(ox + c * cellW + cellW / 2 + jx, oy + r * cellH + cellH / 2 + jy);
    }
    // A centre kept fully on screen, clear of the top bar and the bottom pills.
    function clampCenter(x, y) {
        const hw = islandW / 2 + 16, hh = islandH / 2;
        return Qt.point(Math.max(hw, Math.min(width - hw, x)),
                        Math.max(topBarH + hh + 8, Math.min(height - hh - 48, y)));
    }
    // Resting centre of project i: its saved position, else its auto-layout slot.
    function islandCenter(i) {
        const s = slotOf(i);
        if (s < 0)
            return Qt.point(width / 2, height / 2);
        const p = projects[i];
        const pos = p ? cardPositions[p.key] : undefined;
        if (pos !== undefined && width > 0 && height > 0)
            return clampCenter(pos.x * width, pos.y * height);
        return slotCenter(s);
    }

    // Drag state: the island under the pointer moves by a transform only (no relayout); the
    // lines follow it, and the position is committed once, on release.
    property int dragIndex: -1
    property real dragDx: 0
    property real dragDy: 0
    readonly property bool dragging: dragIndex >= 0
    function liveCenter(i) {
        const c = islandCenter(i);
        return i === dragIndex ? Qt.point(c.x + dragDx, c.y + dragDy) : c;
    }

    // Geometry of island i (ix, iy, iw, ih): where the shared-element zoom starts and ends.
    function islandRect(i) {
        const c = liveCenter(i);
        return Qt.rect(c.x - islandW / 2, c.y - islandH / 2, islandW, islandH);
    }

    readonly property var edges: {
        const out = [];
        for (let s = 0; s < count; ++s) {
            if ((s % columns) < columns - 1 && s + 1 < count)
                out.push([slots[s], slots[s + 1]]);
            if (s + columns < count)
                out.push([slots[s], slots[s + columns]]);
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
            readonly property point a: root.liveCenter(modelData[0])
            readonly property point b: root.liveCenter(modelData[1])
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

    // One delegate per slot (model is a count, so a publish never rebuilds the islands).
    Repeater {
        model: root.count
        Island {
            id: isl
            required property int index
            readonly property int projIndex: root.slots[index] !== undefined ? root.slots[index] : -1
            objectName: "island"
            proj: projIndex >= 0 ? root.projects[projIndex] : null
            number: index + 1
            selected: projIndex === root.selectedIndex
            recede: projIndex === root.focusIndex ? 0 : root.recede
            morphing: projIndex === root.focusIndex && root.recede > 0.001
            pulse: root.pulse
            targets: root.targets
            draggable: root.recede < 0.01
            width: root.islandW
            height: root.islandH
            readonly property point c: projIndex >= 0 ? root.islandCenter(projIndex) : Qt.point(0, 0)
            x: c.x - width / 2
            y: c.y - height / 2
            onClicked: root.enter(projIndex)
            onDragMoved: (dx, dy) => {
                root.dragIndex = projIndex;
                root.dragDx = dx;
                root.dragDy = dy;
            }
            onDragEnded: (dx, dy) => {
                const at = root.clampCenter(c.x + dx, c.y + dy);
                root.dragIndex = -1;
                root.dragDx = 0;
                root.dragDy = 0;
                if (proj && root.width > 0 && root.height > 0)
                    root.cardMoved(proj.key, at.x / root.width, at.y / root.height);
            }
            onDragCanceled: {
                root.dragIndex = -1;
                root.dragDx = 0;
                root.dragDy = 0;
            }
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
        readonly property int target_i: root.slotOf(root.selectedIndex) >= 0 ? root.selectedIndex : root.slotProject(0)
        readonly property rect target: root.islandRect(target_i)
        visible: root.count > 0 && !root.dragging
        x: target.x - 9
        y: target.y - 9
        width: root.islandW + 18
        height: root.islandH + 18
        radius: 34
        color: "transparent"
        border.width: 1.5
        border.color: Theme.goldA(0.5)
        // StudioApp's focus cursor takes over whenever it is on screen (one ring, not two).
        opacity: root.cursorActive ? 0 : 1 - root.recede
        antialiasing: true
        Behavior on x { enabled: root.recede < 0.01; NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutExpo } }
        Behavior on y { enabled: root.recede < 0.01; NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutExpo } }
    }

    // Top bar: All · Needs Me (N) filter, centred; layout reset (only with custom positions) and
    // the session review, top right. Hairline pills, no fill, gold only for the active choice.
    component BarPill: Rectangle {
        id: bp
        property string label: ""
        property bool on: false
        signal tapped()
        width: bpText.implicitWidth + 28
        height: 30
        radius: 15
        color: bpHover.hovered ? Theme.card : "transparent"
        border.width: 1
        border.color: on ? Theme.goldA(0.55) : Theme.line
        Text {
            id: bpText
            objectName: "barPillText"
            anchors.centerIn: parent
            text: bp.label
            color: bp.on ? Theme.gold : (bpHover.hovered ? Theme.text : Theme.dim)
            font.family: Theme.mono
            font.pixelSize: 14
        }
        HoverHandler { id: bpHover; cursorShape: Qt.PointingHandCursor }
        TapHandler { onTapped: bp.tapped() }
    }
    Item {
        id: topBar
        objectName: "constellationTopBar"
        width: parent.width
        height: root.topBarH
        visible: root.projects.length > 0 && opacity > 0.01
        opacity: 1 - root.recede
        Row {
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            spacing: 10
            BarPill {
                objectName: "filterAll"
                label: "All"
                on: root.filter !== "needs"
                onTapped: root.filterPicked("all")
            }
            BarPill {
                objectName: "filterNeedsMe"
                label: "Needs Me (" + root.needsCount + ")"
                on: root.filter === "needs"
                onTapped: root.filterPicked(root.filter === "needs" ? "all" : "needs")
            }
        }
        Row {
            anchors.right: parent.right
            anchors.rightMargin: 24
            anchors.verticalCenter: parent.verticalCenter
            spacing: 10
            BarPill {
                objectName: "resetLayoutPill"
                visible: root.hasCustomLayout
                label: "Reset layout"
                onTapped: root.resetLayout()
            }
            BarPill {
                objectName: "reviewPill"
                label: "Review"
                onTapped: root.reviewRequested()
            }
            BarPill {
                objectName: "musicPill"
                label: root.musicLabel
                on: root.musicPlaying
                onTapped: root.musicRequested()
            }
        }
    }

    // One shared, linear wave clock for every island that wants attention (each island derives
    // its two staggered waves from it). It runs only while such an island exists and the
    // Constellation is on screen and settled; at rest it costs nothing.
    readonly property bool anyAttention: {
        for (let s = 0; s < count; ++s) {
            const p = projects[slots[s]];
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
