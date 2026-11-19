import QtQuick
import QtQuick.Shapes

// Lens 1: spacious OLED project islands, grouped into category clusters (Work · Projects · Apps ·
// Quick Qs) with a quiet micro-label above each and faint constellation lines inside it.
// Islands can be dragged or flicked anywhere (the settled position persists per project); a
// press-drag on empty space draws a rubber-band selection that pins every card it covers as an
// ask target. With a pad connected, the left stick drives a free 240 Hz cursor over the map
// (✕ held on a card grabs it, on empty space draws the marquee). The bell filters to the
// sessions that need the human.
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
    readonly property bool needsMeOnly: filter === "needs"
    property var cardPositions: ({})   // {projectKey: {x, y}}: card centre as a fraction of this item
    property var clusterOrder: ["Work", "Projects", "Apps", "Quick Qs"]
    property string mapSort: "category"  // inside a cluster: "category" (project order) | "recent" | "activity"
    // What the layout actually uses. A change to clusterOrder / mapSort turns the glide on first
    // and only then applies, so every card springs to its new place instead of jumping.
    property var appliedOrder: []
    property string appliedSort: "category"
    property bool layoutReady: false
    // A plain JS copy: a list property read back can stay a live view of the C++ list.
    function copyOrder(o) {
        const out = [];
        for (let i = 0; o && i < o.length; ++i)
            out.push(String(o[i]));
        return out;
    }
    Component.onCompleted: {
        appliedOrder = copyOrder(clusterOrder);
        appliedSort = mapSort;
        layoutReady = true;
    }
    onClusterOrderChanged: {
        const next = copyOrder(clusterOrder);
        if (next.join("|") === appliedOrder.join("|"))
            return;
        if (layoutReady)
            animateRelayout();
        appliedOrder = next;
    }
    onMapSortChanged: {
        if (mapSort === appliedSort)
            return;
        if (layoutReady)
            animateRelayout();
        appliedSort = mapSort;
    }
    property bool freeCursor: false      // a pad is connected: the left stick drives the free cursor
    property var selectedKeys: []        // the rubber-band multi-selection (project keys)
    signal enter(int index)
    signal zoomBy(real factor)
    signal newSession()
    signal cardMoved(string key, real x, real y)
    signal filterPicked(string filter)
    signal resetLayout()
    signal reviewRequested()
    signal musicRequested()
    signal cursorPicked(int index)       // the free cursor came to rest over island `index`
    property string musicLabel: "♪"     // "♪ night drive" while a track plays
    property bool musicPlaying: false
    signal ambientRequested()
    property bool ambientOn: false      // ambient voice listening (gold while on)
    property bool ambientBusy: false
    // Controller mode: pills carry the pad button bound to them (StudioApp binds these).
    property bool padHints: false
    property var padGlyphs: ({})

    // --- Clusters ------------------------------------------------------------------------------
    // The backend names each project's cluster ("cluster"); a project map without one (an old
    // fixture) falls back on its key.
    readonly property var defaultOrder: ["Work", "Projects", "Apps", "Quick Qs"]
    function clusterOf(p) {
        if (!p)
            return "Projects";
        if (p.cluster)
            return p.cluster;
        return p.key && p.key.startsWith("~tab:") ? "Quick Qs" : "Projects";
    }
    // Activity rank: waiting on the human, then live work, then turns landed while away.
    function activityRank(p) {
        if (p.needsMe === true || p.status === "needs-ok") return 0;
        if (p.active) return 1;
        if ((p.fresh || 0) > 0) return 2;
        return 3;
    }

    // Which projects have a slot on screen, in order (project indices): cluster by cluster in
    // clusterOrder, the map sort inside each. The filter never reorders: islands keep their
    // relative order, so muscle memory survives toggling it.
    readonly property var slots: {
        const vis = [];
        for (let i = 0; i < projects.length && vis.length < 12; ++i)
            if (filter !== "needs" || projects[i].needsMe === true)
                vis.push(i);
        const groups = {};
        const names = (appliedOrder && appliedOrder.length > 0 ? appliedOrder : defaultOrder).slice();
        for (const i of vis) {
            const c = clusterOf(projects[i]);
            if (!groups[c]) groups[c] = [];
            groups[c].push(i);
            if (names.indexOf(c) < 0) names.push(c);
        }
        const out = [];
        for (const c of names) {
            const g = groups[c];
            if (!g) continue;
            if (appliedSort === "recent")
                g.sort((a, b) => ((projects[b].mtime || 0) - (projects[a].mtime || 0)) || (a - b));
            else if (appliedSort === "activity")
                g.sort((a, b) => (activityRank(projects[a]) - activityRank(projects[b])) || (a - b));
            for (const i of g) out.push(i);
        }
        return out;
    }
    readonly property int count: slots.length
    function slotOf(i) { return slots.indexOf(i); }
    // Consecutive runs of slots that share a cluster: [{name, start, members: [project index]}].
    readonly property var clusters: {
        const out = [];
        for (let s = 0; s < count; ++s) {
            const c = clusterOf(projects[slots[s]]);
            if (out.length === 0 || out[out.length - 1].name !== c)
                out.push({ name: c, start: s, members: [] });
            out[out.length - 1].members.push(slots[s]);
        }
        return out;
    }
    readonly property bool clustered: clusters.length > 1
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
    // Arrow / d-pad travel. One cluster: the grid order (wrapping across, clamped down). Several:
    // spatial, the nearest island whose centre lies that way (straight ahead beats sideways).
    function navStep(i, dx, dy) {
        if (count === 0)
            return i;
        if (!clustered)
            return dx !== 0 ? stepSelection(i, dx, true) : stepSelection(i, dy * columns, false);
        const from = slotOf(i) >= 0 ? liveCenter(i) : liveCenter(slots[0]);
        let best = -1, bestScore = 1e9;
        for (let s = 0; s < count; ++s) {
            const j = slots[s];
            if (j === i) continue;
            const c = liveCenter(j);
            const along = (c.x - from.x) * dx + (c.y - from.y) * dy;
            if (along <= 4) continue;
            const across = Math.abs((c.x - from.x) * dy) + Math.abs((c.y - from.y) * dx);
            const score = along + 2.2 * across;
            if (score < bestScore) { bestScore = score; best = j; }
        }
        return best >= 0 ? best : (slotOf(i) >= 0 ? i : slots[0]);
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

    // --- Layout ----------------------------------------------------------------------------------
    // One cluster: the original jittered grid. Several: a two-column grid of cluster cells, each a
    // compact sub-grid under its label; one card size for the whole map, sized so the busiest
    // cluster fits its cell (so no two cards, and no two clusters, ever overlap).
    readonly property int columns: count <= 4 ? 2 : (count <= 9 ? 3 : 4)
    readonly property int rows: Math.max(1, Math.ceil(count / columns))
    readonly property real cellW: Math.min(560, (width - 200) / columns)
    readonly property real cellH: Math.min(330, (height - 240) / rows)
    readonly property real topBarH: 64  // dragged cards never slide under the filter bar
    readonly property real clusterLabelH: 34
    readonly property real clusterGapX: 34
    readonly property real clusterGapY: 28

    // Auto-layout centre of slot s (single-cluster grid).
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

    readonly property var layout: {
        const fs = Theme.fontScale;
        const centers = {};
        const edges = [];
        if (!clustered) {
            for (let s = 0; s < count; ++s) {
                centers[slots[s]] = slotCenter(s);
                if ((s % columns) < columns - 1 && s + 1 < count)
                    edges.push([slots[s], slots[s + 1]]);
                if (s + columns < count)
                    edges.push([slots[s], slots[s + columns]]);
            }
            return { w: Math.min(400 * fs, cellW - 90), h: Math.min(206 * fs, cellH - 70), centers: centers,
                     labels: [], edges: edges };
        }
        const n = clusters.length, ccols = 2, crows = Math.ceil(n / ccols);
        const mx = 56, top = topBarH + 12, bottom = 60, pad = 18;
        const aw = Math.max(1, width - 2 * mx), ah = Math.max(1, height - top - bottom);
        const rw = aw / ccols, rh = ah / crows;
        const lh = clusterLabelH, gx = clusterGapX, gy = clusterGapY;
        let scale = fs;
        const shapes = [];
        for (const cl of clusters) {
            const k = cl.members.length;
            let best = null;
            for (let sc = 1; sc <= Math.min(k, 4); ++sc) {
                const sr = Math.ceil(k / sc);
                const fit = Math.min((rw - 2 * pad - (sc - 1) * gx) / sc / 400,
                                     (rh - 2 * pad - lh - (sr - 1) * gy) / sr / 206);
                if (!best || fit > best.fit + 1e-9)
                    best = { sc: sc, sr: sr, fit: fit };
            }
            shapes.push(best);
            scale = Math.min(scale, best.fit);
        }
        scale = Math.max(scale, 0.25);
        const w = 400 * scale, h = 206 * scale;
        const labels = [];
        for (let ci = 0; ci < n; ++ci) {
            const cl = clusters[ci], sh = shapes[ci], k = cl.members.length;
            const row = Math.floor(ci / ccols), col = ci % ccols;
            const inRow = Math.min(ccols, n - row * ccols);
            const rx = mx + (aw - inRow * rw) / 2 + col * rw, ry = top + row * rh;
            const bw = sh.sc * w + (sh.sc - 1) * gx, bh = lh + sh.sr * h + (sh.sr - 1) * gy;
            const bx = rx + (rw - bw) / 2, by = ry + (rh - bh) / 2;
            labels.push({ name: cl.name, x: bx + 6, y: by, w: bw, count: k });
            for (let j = 0; j < k; ++j) {
                const r = Math.floor(j / sh.sc), c = j % sh.sc;
                const rowCount = Math.min(sh.sc, k - r * sh.sc);
                const off = (sh.sc - rowCount) * (w + gx) / 2;
                centers[cl.members[j]] = Qt.point(bx + off + c * (w + gx) + w / 2, by + lh + r * (h + gy) + h / 2);
                if (c < rowCount - 1)
                    edges.push([cl.members[j], cl.members[j + 1]]);
                if (j + sh.sc < k)
                    edges.push([cl.members[j], cl.members[j + sh.sc]]);
            }
        }
        return { w: w, h: h, centers: centers, labels: labels, edges: edges };
    }
    // Islands grow with the UI zoom until their cell is full.
    readonly property real islandW: layout.w
    readonly property real islandH: layout.h

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
        const c = layout.centers[i];
        return c !== undefined ? c : Qt.point(width / 2, height / 2);
    }

    // A layout reset (Cmd+0, the Reset pill, a re-sort) wins over any card still in flight.
    onCardPositionsChanged: {
        if (flightCount > 0 && Object.keys(cardPositions).length === 0)
            stopFlights(sim.flights.map((f) => f.i));
    }

    // Mouse drag state: the island under the pointer (and, when it is part of the selection, the
    // rest of the selection) moves by a transform only (no relayout); the lines follow it, and
    // the position is committed once it settles after release.
    property int dragIndex: -1
    property var dragGroup: []     // project indices riding the mouse drag (dragIndex included)
    property real dragDx: 0
    property real dragDy: 0
    property var dragSamples: []   // [{t, dx, dy}]: the release velocity of a mouse flick
    readonly property bool dragging: dragIndex >= 0 || grabMode === "card"
    // Pad grabs and flicks: absolute centres that override the resting layout while they ride.
    property var live: ({})
    function liveCenter(i) {
        const l = live[i];
        if (l !== undefined)
            return Qt.point(l.x, l.y);
        const c = islandCenter(i);
        return dragGroup.indexOf(i) >= 0 ? Qt.point(c.x + dragDx, c.y + dragDy) : c;
    }

    // Geometry of island i (ix, iy, iw, ih): where the shared-element zoom starts and ends.
    function islandRect(i) {
        const c = liveCenter(i);
        return Qt.rect(c.x - islandW / 2, c.y - islandH / 2, islandW, islandH);
    }
    // The topmost island under (x, y), or -1.
    function islandAt(x, y) {
        for (let s = count - 1; s >= 0; --s) {
            const r = islandRect(slots[s]);
            if (x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height)
                return slots[s];
        }
        return -1;
    }

    readonly property var edges: layout.edges

    // --- Re-sort glide -------------------------------------------------------------------------
    // A sort_map / move_cluster / layout reset re-flows every card: for a moment their x / y
    // ride a soft underdamped spring (a small pop past the target, then rest).
    property bool gliding: false
    function animateRelayout() {
        gliding = true;
        glideTimer.restart();
    }
    Timer { id: glideTimer; interval: 1100; onTriggered: root.gliding = false }

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

    // Empty space: press-drag draws the rubber band; a plain click clears the selection.
    MouseArea {
        id: field
        objectName: "constellationField"
        anchors.fill: parent
        enabled: root.recede < 0.01 && root.count > 0
        acceptedButtons: Qt.LeftButton
        property point at
        property bool banding: false
        onPressed: (m) => {
            at = Qt.point(m.x, m.y);
            banding = false;
        }
        onPositionChanged: (m) => {
            if (!pressed)
                return;
            if (!banding && Math.hypot(m.x - at.x, m.y - at.y) < 6)
                return;
            if (!banding) {
                banding = true;
                root.beginMarquee(at.x, at.y);
            }
            root.updateMarquee(m.x, m.y);
        }
        onReleased: (m) => {
            if (banding)
                root.endMarquee();
            else
                root.clearSelection();
            banding = false;
        }
        onCanceled: {
            if (banding)
                root.cancelMarquee();
            banding = false;
        }
    }

    // Cluster micro-labels: uppercase, tracked, low contrast, above each cluster's cards.
    Repeater {
        model: root.layout.labels
        Text {
            required property var modelData
            objectName: "clusterLabel"
            x: modelData.x
            y: modelData.y + 4
            text: modelData.name.toUpperCase()
            color: Theme.muted
            opacity: 1 - root.recede
            font.family: Theme.mono
            font.pixelSize: Theme.px(12)
            font.letterSpacing: 2.4
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
            height: 1
            color: Theme.line
            opacity: 1 - root.recede
            antialiasing: true
            transformOrigin: Item.TopLeft
            rotation: Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI
        }
    }

    // One delegate per project (model is a count, so a publish never rebuilds the islands), so a
    // re-sort moves each card to its new slot instead of swapping contents between slots. A
    // project the filter (or the 12-card cap) leaves out has no slot and hides.
    Repeater {
        model: root.projects.length
        Island {
            id: isl
            required property int index
            readonly property int projIndex: index
            readonly property int slot: root.slotOf(index)
            readonly property string key: proj ? proj.key : ""
            objectName: "island"
            visible: slot >= 0
            proj: root.projects[index] !== undefined ? root.projects[index] : null
            number: slot + 1
            showNumber: !root.padHints
            selected: projIndex === root.selectedIndex
            recede: projIndex === root.focusIndex ? 0 : root.recede
            morphing: projIndex === root.focusIndex && root.recede > 0.001
            pulse: root.pulse
            targets: root.targets
            draggable: root.recede < 0.01
            cursorOver: root.cursorVisible && projIndex === root.hoverIndex && root.grabMode !== "marquee"
            grabbed: root.grabMode === "card" && root.grabGroup.indexOf(projIndex) >= 0
            multiSelected: key.length > 0 && root.selectedKeys.indexOf(key) >= 0
            marqueeHit: root.marqueeActive && root.marqueeHits.indexOf(projIndex) >= 0
            groupDx: projIndex !== root.dragIndex && root.dragGroup.indexOf(projIndex) >= 0 ? root.dragDx : 0
            groupDy: projIndex !== root.dragIndex && root.dragGroup.indexOf(projIndex) >= 0 ? root.dragDy : 0
            glide: root.gliding
            width: root.islandW
            height: root.islandH
            readonly property point c: root.live[projIndex] !== undefined ? Qt.point(root.live[projIndex].x, root.live[projIndex].y)
                                                                         : root.islandCenter(projIndex)
            x: c.x - width / 2
            y: c.y - height / 2
            onClicked: root.enter(projIndex)
            onDragMoved: (dx, dy) => root.mouseDragMoved(projIndex, dx, dy)
            onDragEnded: (dx, dy) => root.mouseDragEnded(projIndex, dx, dy)
            onDragCanceled: root.mouseDragReset()
        }
    }

    // Keyboard focus ring. Every island has the same size, so the ring's size is a constant
    // (islandW + 18 by islandH + 18) and only x / y move: a 120 ms OutExpo glide that lands
    // without overshoot. A hop retargets mid-flight from wherever the ring is, so key repeat
    // reads as one continuous motion.
    Rectangle {
        id: ring
        objectName: "focusRing"
        readonly property int target_i: root.slotOf(root.selectedIndex) >= 0 ? root.selectedIndex : root.slotProject(0)
        readonly property rect target: root.islandRect(target_i)
        visible: root.count > 0 && !root.dragging && root.flightCount === 0
        x: target.x - 9
        y: target.y - 9
        width: root.islandW + 18
        height: root.islandH + 18
        radius: 34
        color: "transparent"
        border.width: 1.5
        border.color: Theme.goldA(0.5)
        // StudioApp's focus cursor takes over whenever it is on screen (one ring, not two); the
        // free pad cursor's own hover highlight stands in for it while that is on.
        opacity: root.cursorActive || root.cursorVisible ? 0 : 1 - root.recede
        antialiasing: true
        Behavior on x { enabled: root.recede < 0.01; NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutExpo } }
        Behavior on y { enabled: root.recede < 0.01; NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutExpo } }
    }

    // --- Rubber-band multi-select ------------------------------------------------------------
    property bool marqueeActive: false
    property point marqueeFrom
    property point marqueeTo
    readonly property rect marqueeRect: Qt.rect(Math.min(marqueeFrom.x, marqueeTo.x), Math.min(marqueeFrom.y, marqueeTo.y),
                                                Math.abs(marqueeTo.x - marqueeFrom.x), Math.abs(marqueeTo.y - marqueeFrom.y))
    readonly property var marqueeHits: {
        if (!marqueeActive)
            return [];
        const m = marqueeRect, out = [];
        for (let s = 0; s < count; ++s) {
            const r = islandRect(slots[s]);
            if (r.x < m.x + m.width && r.x + r.width > m.x && r.y < m.y + m.height && r.y + r.height > m.y)
                out.push(slots[s]);
        }
        return out;
    }
    function beginMarquee(x, y) {
        marqueeFrom = Qt.point(x, y);
        marqueeTo = Qt.point(x, y);
        marqueeActive = true;
    }
    function updateMarquee(x, y) { marqueeTo = Qt.point(Math.max(0, Math.min(width, x)), Math.max(0, Math.min(height, y))); }
    function endMarquee() {
        const keys = [];
        for (const i of marqueeHits)
            keys.push(projects[i].key);
        marqueeActive = false;
        setSelection(keys);
    }
    function cancelMarquee() { marqueeActive = false; }
    Rectangle {
        objectName: "marqueeRect"
        visible: root.marqueeActive && (root.marqueeRect.width > 1 || root.marqueeRect.height > 1)
        x: root.marqueeRect.x
        y: root.marqueeRect.y
        width: root.marqueeRect.width
        height: root.marqueeRect.height
        color: Theme.goldA(0.05)
        border.width: 1
        border.color: Theme.goldA(0.55)
        antialiasing: true
        z: 20
    }

    // The selection is pinned into backend.targets (session targets), so one ask covers it all;
    // clearing it unpins exactly those session pins and leaves any other pins alone.
    function targetFor(i) {
        const p = projects[i];
        const r = islandRect(i);
        const g = root.mapToItem(null, r.x, r.y, r.width, r.height);
        const job = p.jobName ? p.jobName : "";
        const state = p.stateLine ? p.stateLine : "";
        return { kind: "session", sessionId: p.key, artifactId: "", path: "", label: p.name,
                 contextText: job.length > 0 && state.length > 0 ? job + " — " + state : job + state,
                 x: g.x, y: g.y, w: g.width, h: g.height };
    }
    function setSelection(keys) {
        if (targets) {
            const pins = targets.pinnedTargets;
            for (let k = pins.length - 1; k >= 0; --k)
                if (pins[k].kind === "session" && selectedKeys.indexOf(pins[k].sessionId) >= 0 && keys.indexOf(pins[k].sessionId) < 0)
                    targets.unpinTarget(k);
        }
        selectedKeys = keys;
        if (targets)
            for (let s = 0; s < count; ++s)
                if (keys.indexOf(projects[slots[s]].key) >= 0)
                    targets.pinTarget(targetFor(slots[s]));
    }
    function clearSelection() {
        if (selectedKeys.length > 0)
            setSelection([]);
    }
    readonly property bool hasSelection: selectedKeys.length > 0
    // Project indices on screen that are selected (a drag of any of them moves them all).
    function groupFor(i) {
        const p = projects[i];
        if (!p || selectedKeys.indexOf(p.key) < 0)
            return [i];
        const out = [];
        for (let s = 0; s < count; ++s)
            if (selectedKeys.indexOf(projects[slots[s]].key) >= 0)
                out.push(slots[s]);
        return out;
    }

    // --- Mouse drag + flick ---------------------------------------------------------------------
    function mouseDragMoved(i, dx, dy) {
        if (dragIndex !== i) {
            dragIndex = i;
            dragGroup = groupFor(i);
            dragSamples = [];
            stopFlights(dragGroup);
        }
        dragDx = dx;
        dragDy = dy;
        const now = Date.now();
        const s = dragSamples.filter((e) => now - e.t <= 120);
        s.push({ t: now, dx: dx, dy: dy });
        dragSamples = s;
    }
    // Release velocity: the pointer's travel over its last ~70 ms. A pause before letting go
    // (or a burst too short to time) is a plain drop.
    function releaseVelocity(samples, now) {
        const s = samples.filter((e) => now - e.t <= 70);
        if (s.length < 2 || now - s[s.length - 1].t > 40)
            return Qt.point(0, 0);
        const a = s[0], b = s[s.length - 1];
        const dt = (b.t - a.t) / 1000;
        if (dt < 0.010)
            return Qt.point(0, 0);
        let vx = (b.dx - a.dx) / dt, vy = (b.dy - a.dy) / dt;
        const sp = Math.hypot(vx, vy);
        if (sp > maxFlick) { vx *= maxFlick / sp; vy *= maxFlick / sp; }
        return Qt.point(vx, vy);
    }
    function mouseDragEnded(i, dx, dy) {
        const v = releaseVelocity(dragSamples, Date.now());
        const group = dragGroup.length > 0 ? dragGroup : [i];
        const starts = [];
        for (const j of group) {
            const c = islandCenter(j);
            starts.push({ i: j, x: c.x + dx, y: c.y + dy });
        }
        mouseDragReset();
        launch(starts, v.x, v.y);
    }
    function mouseDragReset() {
        dragIndex = -1;
        dragGroup = [];
        dragDx = 0;
        dragDy = 0;
        dragSamples = [];
    }

    // --- Physics: free cursor + flicked cards, fixed 240 Hz steps ---------------------------------
    readonly property real physicsHz: 240
    readonly property real friction: 5.0        // 1/s: exponential velocity decay of a flicked card
    readonly property real bounce: 0.38         // velocity kept (reversed) off a window margin
    readonly property real maxFlick: 4200       // px/s
    readonly property real cursorSpeed: Math.max(900, width * 1.05)  // px/s at full stick
    readonly property real aimAssist: 0.65      // stick speed over a card (35% magnetic slowdown)
    property real stickX: 0                     // shaped left stick (-1..1, y down), from the pad
    property real stickY: 0
    readonly property bool stickActive: stickX !== 0 || stickY !== 0
    property real cursorX: width / 2
    property real cursorY: height / 2
    property real cursorVx: 0                   // px/s, the cursor's current velocity (the flick)
    property real cursorVy: 0
    property bool cursorGliding: false
    property point glideTo
    property bool cursorPlaced: false
    property var sim: ({ acc: 0, flights: [] })  // integrator state, mutated in place (not bound)
    property int flightCount: 0
    readonly property bool cursorVisible: freeCursor && count > 0 && recede < 0.01
    readonly property int hoverIndex: cursorVisible ? islandAt(cursorX, cursorY) : -1
    property bool cursorDriving: false          // the selection change came from the cursor itself
    onHoverIndexChanged: {
        // A glide onto a hopped-to card passes over others: only the stick picks what it crosses.
        if (hoverIndex >= 0 && grabMode === "" && !cursorGliding) {
            cursorDriving = true;
            cursorPicked(hoverIndex);
            cursorDriving = false;
        }
    }
    // Discrete nav (d-pad, keys, L1 / △) moves the selection: the cursor glides onto that card.
    onSelectedIndexChanged: if (!cursorDriving) placeCursorOnSelection(false)
    onCursorVisibleChanged: if (cursorVisible) placeCursorOnSelection(!cursorPlaced)
    function placeCursorOnSelection(jump) {
        if (!freeCursor || count === 0 || grabMode !== "")
            return;
        const i = slotOf(selectedIndex) >= 0 ? selectedIndex : slots[0];
        const c = islandCenter(i);
        if (jump) {
            cursorX = c.x;
            cursorY = c.y;
            cursorGliding = false;
        } else {
            glideTo = c;
            cursorGliding = true;
        }
        cursorPlaced = true;
    }
    function setStick(x, y) {
        stickX = x;
        stickY = y;
        if (x !== 0 || y !== 0)
            cursorGliding = false;
    }
    readonly property bool ticking: visible && (stickActive || cursorGliding || flightCount > 0 || grabMode === "card")
    FrameAnimation {
        running: root.ticking
        onTriggered: root.advance(frameTime)
    }
    // Advance the simulation by dt seconds in fixed 1/240 s steps (tests call it directly).
    function advance(dt) {
        const h = 1 / physicsHz;
        sim.acc = Math.min(sim.acc + Math.max(0, Math.min(dt, 0.25)), 0.25);
        let cx = cursorX, cy = cursorY, vx = 0, vy = 0;
        let steps = 0;
        while (sim.acc >= h) {
            sim.acc -= h;
            ++steps;
            if (freeCursor && (stickX !== 0 || stickY !== 0)) {
                const over = grabMode === "" && islandAt(cx, cy) >= 0;
                const sp = cursorSpeed * (over ? aimAssist : 1);
                vx = stickX * sp;
                vy = stickY * sp;
                cx = Math.max(6, Math.min(width - 6, cx + vx * h));
                cy = Math.max(6, Math.min(height - 6, cy + vy * h));
            } else if (cursorGliding) {
                const k = 1 - Math.exp(-h * 24);
                cx += (glideTo.x - cx) * k;
                cy += (glideTo.y - cy) * k;
                if (Math.hypot(glideTo.x - cx, glideTo.y - cy) < 0.4) {
                    cx = glideTo.x;
                    cy = glideTo.y;
                    cursorGliding = false;
                }
            }
            stepFlights(h);
        }
        if (steps === 0)
            return;
        cursorVx = vx;
        cursorVy = vy;
        cursorX = cx;
        cursorY = cy;
        publishSim();
    }
    function bounds() {
        const lo = clampCenter(-1e6, -1e6), hi = clampCenter(1e6, 1e6);
        return { x0: lo.x, y0: lo.y, x1: hi.x, y1: hi.y };
    }
    // One 1/240 s step of every flicked card: drift, exponential friction, a soft bounce off the
    // margins, and a gentle push out of any card it overlaps. A card that has been slow and
    // clear for 60 ms lands: its fraction is committed (and persisted) once.
    function stepFlights(h) {
        const fl = sim.flights;
        if (fl.length === 0)
            return;
        const b = bounds();
        const damp = Math.exp(-friction * h);
        const w = islandW + 14, ht = islandH + 14;
        for (const f of fl) {
            f.x += f.vx * h;
            f.y += f.vy * h;
            f.vx *= damp;
            f.vy *= damp;
            if (f.x < b.x0) { f.x = b.x0; f.vx = Math.abs(f.vx) * bounce; }
            if (f.x > b.x1) { f.x = b.x1; f.vx = -Math.abs(f.vx) * bounce; }
            if (f.y < b.y0) { f.y = b.y0; f.vy = Math.abs(f.vy) * bounce; }
            if (f.y > b.y1) { f.y = b.y1; f.vy = -Math.abs(f.vy) * bounce; }
            let pushed = false;
            for (let s = 0; s < count; ++s) {
                const j = slots[s];
                if (j === f.i)
                    continue;
                let o = null;
                for (const g of fl)
                    if (g.i === j) o = g;
                const c = o ? Qt.point(o.x, o.y) : islandCenter(j);
                const dx = f.x - c.x, dy = f.y - c.y;
                const ox = w - Math.abs(dx), oy = ht - Math.abs(dy);
                if (ox <= 0 || oy <= 0)
                    continue;
                pushed = true;
                // Out along the shallower overlap, a few percent per step: a soft nudge, no snap.
                if (ox < oy)
                    f.x += (dx >= 0 ? 1 : -1) * Math.min(ox, Math.max(0.6, ox * 0.06));
                else
                    f.y += (dy >= 0 ? 1 : -1) * Math.min(oy, Math.max(0.6, oy * 0.06));
            }
            f.x = Math.max(b.x0, Math.min(b.x1, f.x));
            f.y = Math.max(b.y0, Math.min(b.y1, f.y));
            f.age += h;
            const slow = Math.hypot(f.vx, f.vy) < 6;
            f.rest = slow && !pushed ? f.rest + h : 0;
            // A card boxed in with no free spot stops pushing after 3 s and lands where it is.
            // A still drop onto open space (h = 0, the launch probe) lands at once.
            if (f.rest >= 0.06 || f.age > 3 || (h === 0 && slow && !pushed))
                f.done = true;
        }
    }
    // Once per frame: hand the integrator's positions to the bindings and commit landed cards.
    function publishSim() {
        const out = {};
        if (grabMode === "card")
            for (const g of grab.group)
                out[g.i] = { x: g.x + cursorX - grab.sx, y: g.y + cursorY - grab.sy };
        const keep = [];
        const landed = [];
        for (const f of sim.flights) {
            if (f.done)
                landed.push(f);
            else {
                keep.push(f);
                out[f.i] = { x: f.x, y: f.y };
            }
        }
        sim.flights = keep;
        flightCount = keep.length;
        live = out;
        if (width > 0 && height > 0)
            for (const f of landed)
                if (projects[f.i] && projects[f.i].key === f.key)
                    cardMoved(f.key, f.x / width, f.y / height);
    }
    // Start cards flying from absolute centres with a shared velocity (0, 0: a drop that only
    // needs to settle, i.e. clear an overlap).
    function launch(starts, vx, vy) {
        const ids = starts.map((s) => s.i);
        sim.flights = sim.flights.filter((f) => ids.indexOf(f.i) < 0);
        const b = bounds();
        for (const s of starts) {
            const p = projects[s.i];
            if (!p)
                continue;
            sim.flights.push({ i: s.i, key: p.key, x: Math.max(b.x0, Math.min(b.x1, s.x)), y: Math.max(b.y0, Math.min(b.y1, s.y)),
                               vx: vx, vy: vy, age: 0, rest: 0, done: false });
        }
        flightCount = sim.flights.length;
        // A drop with no speed onto open space lands this frame (no wait for the next tick).
        stepFlights(0);
        publishSim();
    }
    function stopFlights(ids) {
        sim.flights = sim.flights.filter((f) => ids.indexOf(f.i) < 0);
        flightCount = sim.flights.length;
        const out = Object.assign({}, live);
        for (const i of ids) delete out[i];
        live = out;
    }

    // --- Pad grab (✕ held) ------------------------------------------------------------------------
    // ✕ down on a card grabs it (and the whole selection when it is part of one); ✕ down on
    // empty space starts the marquee. A quick tap (< 220 ms, < 12 px) opens the card instead.
    property string grabMode: ""   // "" | "card" | "marquee"
    property var grab: ({})
    readonly property var grabGroup: grabMode === "card" && grab.group ? grab.group.map((g) => g.i) : []
    function padGrab(down) {
        if (down) {
            if (count === 0 || recede > 0.01)
                return;
            const hit = islandAt(cursorX, cursorY);
            cursorGliding = false;
            if (hit >= 0) {
                const members = groupFor(hit);
                stopFlights(members);
                const group = members.map((i) => {
                    const c = liveCenter(i);
                    return { i: i, x: c.x, y: c.y };
                });
                grab = { t0: Date.now(), sx: cursorX, sy: cursorY, hit: hit, group: group, far: 0 };
                grabMode = "card";
            } else {
                grab = { t0: Date.now(), sx: cursorX, sy: cursorY, hit: -1, group: [], far: 0 };
                grabMode = "marquee";
                beginMarquee(cursorX, cursorY);
            }
            return;
        }
        if (grabMode === "")
            return;
        const mode = grabMode;
        const far = Math.max(grab.far, Math.hypot(cursorX - grab.sx, cursorY - grab.sy));
        const tap = Date.now() - grab.t0 < 220 && far < 12;
        if (mode === "card") {
            const starts = grab.group.map((g) => ({ i: g.i, x: g.x + cursorX - grab.sx, y: g.y + cursorY - grab.sy }));
            const hit = grab.hit;
            grabMode = "";
            grab = ({});
            if (tap) {
                live = ({});
                enter(hit);
            } else {
                launch(starts, cursorVx, cursorVy);
            }
        } else {
            grabMode = "";
            grab = ({});
            if (tap) {
                cancelMarquee();
                // Empty space: a tap clears the selection; with none it opens the focused card.
                if (hasSelection)
                    clearSelection();
                else if (count > 0)
                    enter(slotOf(selectedIndex) >= 0 ? selectedIndex : slots[0]);
            } else {
                endMarquee();
            }
        }
    }
    onCursorXChanged: trackGrab()
    onCursorYChanged: trackGrab()
    function trackGrab() {
        if (grabMode === "")
            return;
        grab.far = Math.max(grab.far || 0, Math.hypot(cursorX - grab.sx, cursorY - grab.sy));
        if (grabMode === "marquee")
            updateMarquee(cursorX, cursorY);
    }

    MapCursor {
        cx: root.cursorX
        cy: root.cursorY
        visible: root.cursorVisible
        overCard: root.hoverIndex >= 0
        grabbing: root.grabMode === "card"
        marquee: root.grabMode === "marquee"
        z: 30
    }

    // Top bar: the Needs Me bell, centred; layout reset (only with custom positions) and the
    // session review, top right. Hairline pills, no fill, gold only for the active choice.
    component BarPill: Rectangle {
        id: bp
        property string label: ""
        property string glyph: ""  // pad button badge, shown in controller mode only
        property bool on: false
        signal tapped()
        width: bpRow.implicitWidth + 28
        height: 30
        radius: 15
        color: bpHover.hovered ? Theme.card : "transparent"
        border.width: 1
        border.color: on ? Theme.goldA(0.55) : Theme.line
        Row {
            id: bpRow
            anchors.centerIn: parent
            spacing: 8
            Text {
                id: bpText
                objectName: "barPillText"
                anchors.verticalCenter: parent.verticalCenter
                text: bp.label
                color: bp.on ? Theme.gold : (bpHover.hovered ? Theme.text : Theme.dim)
                font.family: Theme.mono
                font.pixelSize: 14
            }
            Rectangle {
                objectName: "barPillGlyph"
                anchors.verticalCenter: parent.verticalCenter
                visible: root.padHints && bp.glyph.length > 0
                width: bpGlyph.implicitWidth + 10
                height: 18
                radius: 5
                color: "transparent"
                border.width: 1
                border.color: Theme.line
                Text {
                    id: bpGlyph
                    anchors.centerIn: parent
                    text: bp.glyph
                    color: Theme.dim
                    font.family: Theme.mono
                    font.pixelSize: 11
                }
            }
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
        // Needs Me: one bell. Gold with a count badge while anything waits; a bare muted bell at
        // zero; a gold ring while the map is filtered to it. Cmd+1 / Cmd+2 still pick All / Needs.
        Rectangle {
            id: bell
            objectName: "needsBell"
            readonly property bool on: root.needsMeOnly
            readonly property int n: root.needsCount
            anchors.horizontalCenter: parent.horizontalCenter
            anchors.verticalCenter: parent.verticalCenter
            width: 44
            height: 30
            radius: 15
            color: bellHover.hovered ? Theme.card : "transparent"
            border.width: 1
            border.color: on ? Theme.goldA(0.55) : Theme.line
            Shape {
                objectName: "needsBellIcon"
                anchors.centerIn: parent
                width: 16
                height: 18
                preferredRendererType: Shape.CurveRenderer
                ShapePath {
                    strokeWidth: 1.4
                    strokeColor: bell.on || bell.n > 0 ? Theme.gold : (bellHover.hovered ? Theme.text : Theme.dim)
                    fillColor: "transparent"
                    capStyle: ShapePath.RoundCap
                    joinStyle: ShapePath.RoundJoin
                    // Dome, flared lip, and the clapper below it.
                    startX: 2; startY: 13.5
                    PathLine { x: 3.4; y: 11.6 }
                    PathLine { x: 3.4; y: 7.4 }
                    PathArc { x: 12.6; y: 7.4; radiusX: 4.6; radiusY: 4.8 }
                    PathLine { x: 12.6; y: 11.6 }
                    PathLine { x: 14; y: 13.5 }
                    PathLine { x: 2; y: 13.5 }
                    PathMove { x: 6.4; y: 15.6 }
                    PathArc { x: 9.6; y: 15.6; radiusX: 1.6; radiusY: 1.4; direction: PathArc.Counterclockwise }
                    PathMove { x: 8; y: 2.6 }
                    PathLine { x: 8; y: 1.2 }
                }
            }
            Rectangle {
                objectName: "needsBellBadge"
                visible: bell.n > 0
                x: parent.width - width / 2 - 6
                y: -5
                width: Math.max(16, badgeText.implicitWidth + 8)
                height: 16
                radius: 8
                color: Theme.gold
                Text {
                    id: badgeText
                    objectName: "needsBellCount"
                    anchors.centerIn: parent
                    text: String(bell.n)
                    color: Theme.bg
                    font.family: Theme.mono
                    font.pixelSize: 10
                    font.weight: Font.DemiBold
                }
            }
            HoverHandler { id: bellHover; cursorShape: Qt.PointingHandCursor }
            TapHandler { onTapped: root.filterPicked(root.needsMeOnly ? "all" : "needs") }
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
                objectName: "ambientPill"
                label: root.ambientBusy ? "Ambient ·" : "Ambient"
                on: root.ambientOn
                onTapped: root.ambientRequested()
            }
            BarPill {
                objectName: "reviewPill"
                label: "Review"
                glyph: root.padGlyphs["toggle_review"] || ""
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
