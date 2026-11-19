import QtQuick

// Music: Cmd+M, or the ♪ pill in the top bar. The curated slowmix rotation (tracks played in the
// last 30 days, most-listened first; All adds the archive), and only the controls in real use:
// Play / Pause, Loop, Speed, Reverb, Bass. Picking a track applies its learned favorite speed and
// reverb. Esc closes the card; the music keeps playing.
//   ↑↓ (j/k, w/s) pick   ⏎ play it   space play / pause   ←→ speed   esc close
// DualShock 4 (L1+R1 anywhere opens it; StudioApp routes the pad here while it is open):
//   ↕ pick  ✕ play  □ pause  ◄► / L1 / R1 speed  △ reverb  L2 loop  R2 bass  ○ close
FocusScope {
    id: root
    objectName: "slowmixOverlay"
    property QtObject backend
    property bool open: false
    property int row: 0
    // Controller mode: the footer speaks DualShock glyphs (StudioApp binds these).
    property bool padHints: false
    signal closed()

    readonly property QtObject player: backend ? backend.slowmix : null
    readonly property var tracks: player ? player.tracks : []
    readonly property int count: tracks.length
    readonly property bool playing: player ? player.playing : false
    readonly property string currentPath: player ? player.currentPath : ""

    readonly property var speeds: [0.85, 0.90, 1.0, 1.10, 1.25, 1.6]
    readonly property var reverbs: [["off", "Off"], ["hall", "Hall"], ["heavy", "Heavy"]]
    readonly property var basses: [[0, "Flat"], [2, "Warm"], [4, "Punch"]]

    function show() {
        if (player)
            player.refresh();
        row = 0;
        for (let i = 0; i < count; ++i)
            if (tracks[i].path === currentPath) {
                row = i;
                break;
            }
        open = true;
        forceActiveFocus();
        list.positionViewAtIndex(row, ListView.Contain);
    }
    function hide() {
        if (!open)
            return;
        open = false;
        closed();
    }
    function toggle() { open ? hide() : show(); }
    function move(d) {
        if (count > 0) {
            row = (row + d + count) % count;
            list.positionViewAtIndex(row, ListView.Contain);
        }
    }
    function playRow(i) {
        if (player && i >= 0 && i < count) {
            row = i;
            player.selectTrack(tracks[i].path);
        }
    }
    function stepSpeed(d) {
        if (!player)
            return;
        let i = 0, best = 1e9;
        for (let k = 0; k < speeds.length; ++k) {
            const dist = Math.abs(speeds[k] - player.speed);
            if (dist < best) { best = dist; i = k; }
        }
        player.setSpeed(speeds[Math.max(0, Math.min(speeds.length - 1, i + d))]);
    }
    function togglePlay() { if (player) player.togglePlay(); }
    function toggleLoop() { if (player) player.setLooping(!player.looping); }
    // Off -> Hall -> Heavy -> Off.
    function cycleReverb() {
        if (!player)
            return;
        let i = 0;
        for (let k = 0; k < reverbs.length; ++k)
            if (reverbs[k][0] === player.reverb) { i = k; break; }
        player.setReverb(reverbs[(i + 1) % reverbs.length][0]);
    }
    // Flat -> Warm -> Punch -> Flat.
    function cycleBass() {
        if (!player)
            return;
        let i = 0, best = 1e9;
        for (let k = 0; k < basses.length; ++k) {
            const dist = Math.abs(basses[k][0] - player.bass);
            if (dist < best) { best = dist; i = k; }
        }
        player.setBass(basses[(i + 1) % basses.length][0]);
    }
    function speedLabel(s) {
        if (Math.abs(s - 1.0) < 0.001) return "1.0x";
        if (Math.abs(s - 1.6) < 0.001) return "1.6x";
        return s.toFixed(2) + "x";
    }
    function minutesLabel(m) {
        const mins = Math.round(m);
        if (mins < 60)
            return mins + "m";
        const h = Math.floor(mins / 60), r = mins % 60;
        return r === 0 ? h + "h" : h + "h " + r + "m";
    }

    visible: opacity > 0.01
    opacity: open ? 1 : 0
    Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
    enabled: open

    Keys.onPressed: (e) => {
        const ctrl = (e.modifiers & Qt.ControlModifier) !== 0;
        const k = e.key;
        if (ctrl) {
            return;  // Cmd+M closes it, Cmd +/-/0 zoom: both reach the app
        } else if (k === Qt.Key_Escape || k === Qt.Key_Q) {
            hide();
        } else if (k === Qt.Key_Down || k === Qt.Key_J || k === Qt.Key_S || k === Qt.Key_Tab) {
            move(1);
        } else if (k === Qt.Key_Up || k === Qt.Key_K || k === Qt.Key_W || k === Qt.Key_Backtab) {
            move(-1);
        } else if (k === Qt.Key_Return || k === Qt.Key_Enter || k === Qt.Key_E) {
            playRow(row);
        } else if (k === Qt.Key_Space) {
            if (player) player.togglePlay();
        } else if (k === Qt.Key_Left || k === Qt.Key_H || k === Qt.Key_A) {
            stepSpeed(-1);
        } else if (k === Qt.Key_Right || k === Qt.Key_L || k === Qt.Key_D) {
            stepSpeed(1);
        }
        e.accepted = true;
    }

    Rectangle {
        anchors.fill: parent
        color: "#000000"
        opacity: 0.88
        MouseArea { anchors.fill: parent; onClicked: root.hide(); onWheel: (w) => w.accepted = true }
    }

    component Caption: Text {
        color: Theme.faint
        font.family: Theme.mono
        font.pixelSize: 13
        font.letterSpacing: 1.6
        font.capitalization: Font.AllUppercase
    }
    // Hairline chip: gold border + gold text while on.
    component Chip: Rectangle {
        id: chip
        property string label
        property bool on: false
        signal tapped()
        width: chipText.implicitWidth + 28
        height: 32
        radius: 16
        color: chipHover.hovered ? Theme.card : "transparent"
        border.width: 1
        border.color: on ? Theme.goldA(0.6) : Theme.line
        Text {
            id: chipText
            anchors.centerIn: parent
            text: chip.label
            color: chip.on ? Theme.gold : (chipHover.hovered ? Theme.text : Theme.dim)
            font.family: Theme.mono
            font.pixelSize: 14
        }
        HoverHandler { id: chipHover; cursorShape: Qt.PointingHandCursor }
        TapHandler { onTapped: chip.tapped() }
    }

    Rectangle {
        id: card
        objectName: "slowmixCard"
        anchors.centerIn: parent
        width: Math.min(parent.width - 120, 820)
        height: content.height + 76
        radius: 28
        color: "#0b0a08"
        border.width: 1.5
        border.color: Theme.goldA(0.55)
        MouseArea { anchors.fill: parent; onWheel: (w) => w.accepted = true }

        Column {
            id: content
            x: 44
            y: 38
            width: parent.width - 88
            spacing: 14

            Item {
                width: parent.width
                height: 46
                Text {
                    text: "Music"
                    color: Theme.ink
                    font.family: Theme.sans
                    font.pixelSize: 30
                    font.weight: Font.DemiBold
                }
                Text {
                    objectName: "slowmixSource"
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.topMargin: 10
                    text: root.player && !root.player.available ? "slowmix not built"
                          : root.player && root.player.showArchive ? "all songs" : "last 30 days"
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 14
                }
            }

            // Now playing: name, preset, and a hairline progress bar while the player reports one.
            Item {
                objectName: "slowmixNowPlaying"
                width: parent.width
                height: 30
                visible: root.currentPath.length > 0
                Text {
                    id: nowGlyph
                    anchors.verticalCenter: parent.verticalCenter
                    text: "♪"
                    color: root.playing ? Theme.gold : Theme.dim
                    font.pixelSize: 20
                }
                Text {
                    objectName: "slowmixNowText"
                    anchors.left: nowGlyph.right
                    anchors.leftMargin: 12
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    text: root.player ? root.player.currentTrack + (root.playing ? "" : "  ·  paused") : ""
                    color: root.playing ? Theme.ink : Theme.dim
                    elide: Text.ElideRight
                    font.family: Theme.sans
                    font.pixelSize: 19
                }
                Rectangle {
                    anchors.bottom: parent.bottom
                    width: parent.width
                    height: 1
                    color: Theme.line
                    visible: root.player && root.player.duration > 0
                    Rectangle {
                        height: 1
                        width: root.player && root.player.duration > 0
                               ? parent.width * Math.min(1, root.player.position / root.player.duration) : 0
                        color: Theme.goldA(0.7)
                    }
                }
            }

            Text {
                objectName: "slowmixEmpty"
                visible: root.count === 0
                text: root.player && root.player.showArchive ? "No songs in the clips folder."
                                                            : "Nothing played in the last 30 days. All shows every song."
                color: Theme.dim
                font.family: Theme.sans
                font.pixelSize: 18
            }

            ListView {
                id: list
                objectName: "slowmixList"
                width: parent.width
                height: Math.max(0, Math.min(root.count, 7) * 52 - 4)
                spacing: 4
                visible: root.count > 0
                clip: true
                model: root.count
                boundsBehavior: Flickable.StopAtBounds
                delegate: Rectangle {
                    id: trackRow
                    required property int index
                    objectName: "slowmixTrack"
                    readonly property var t: root.tracks[index] || ({})
                    readonly property bool sel: index === root.row
                    readonly property bool current: t.path === root.currentPath
                    width: list.width
                    height: 48
                    radius: 16
                    color: sel ? Theme.cardHi : "transparent"
                    border.width: sel ? 1 : 0
                    border.color: Theme.goldA(0.55)
                    Text {
                        id: glyph
                        x: 16
                        anchors.verticalCenter: parent.verticalCenter
                        text: "♪"
                        visible: trackRow.current
                        color: root.playing ? Theme.gold : Theme.dim
                        font.pixelSize: 17
                    }
                    Text {
                        objectName: "slowmixTrackName"
                        x: 42
                        anchors.verticalCenter: parent.verticalCenter
                        width: parent.width - x - presetChip.width - minutes.width - 52
                        text: trackRow.t.name || ""
                        elide: Text.ElideRight
                        color: trackRow.current ? Theme.gold : Theme.ink
                        font.family: Theme.sans
                        font.pixelSize: 19
                    }
                    Text {
                        id: minutes
                        objectName: "slowmixTrackMinutes"
                        anchors.right: presetChip.left
                        anchors.rightMargin: 16
                        anchors.verticalCenter: parent.verticalCenter
                        text: (trackRow.t.recentMinutes || 0) > 0 ? root.minutesLabel(trackRow.t.recentMinutes) : "–"
                        color: Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 14
                    }
                    Rectangle {
                        id: presetChip
                        objectName: "slowmixTrackPreset"
                        anchors.right: parent.right
                        anchors.rightMargin: 14
                        anchors.verticalCenter: parent.verticalCenter
                        width: 128
                        height: 26
                        radius: 13
                        color: "transparent"
                        border.width: 1
                        border.color: Theme.line
                        Text {
                            anchors.centerIn: parent
                            text: root.speedLabel(trackRow.t.favoriteSpeed || 1.0) + " · " + (trackRow.t.favoriteReverb || "hall")
                            color: Theme.dim
                            font.family: Theme.mono
                            font.pixelSize: 13
                        }
                    }
                    TapHandler {
                        onTapped: root.row = trackRow.index
                        onDoubleTapped: root.playRow(trackRow.index)
                    }
                }
            }

            Item { width: 1; height: 2 }
            Rectangle { width: parent.width; height: 1; color: Theme.line }

            Row {
                spacing: 10
                Chip {
                    objectName: "slowmixPlay"
                    label: root.playing ? "Pause" : "Play"
                    on: root.playing
                    onTapped: if (root.player) root.player.togglePlay()
                }
                Chip {
                    objectName: "slowmixLoop"
                    label: "Loop"
                    on: root.player ? root.player.looping : false
                    onTapped: if (root.player) root.player.setLooping(!root.player.looping)
                }
            }

            Grid {
                columns: 2
                columnSpacing: 18
                rowSpacing: 12
                verticalItemAlignment: Grid.AlignVCenter
                Caption { text: "Speed"; width: 86 }
                Row {
                    spacing: 8
                    Repeater {
                        model: root.speeds
                        Chip {
                            required property var modelData
                            objectName: "slowmixSpeed"
                            label: root.speedLabel(modelData)
                            on: root.player ? Math.abs(root.player.speed - modelData) < 0.001 : false
                            onTapped: if (root.player) root.player.setSpeed(modelData)
                        }
                    }
                }
                Caption { text: "Reverb"; width: 86 }
                Row {
                    spacing: 8
                    Repeater {
                        model: root.reverbs
                        Chip {
                            required property var modelData
                            objectName: "slowmixReverb"
                            label: modelData[1]
                            on: root.player ? root.player.reverb === modelData[0] : false
                            onTapped: if (root.player) root.player.setReverb(modelData[0])
                        }
                    }
                }
                Caption { text: "Bass"; width: 86 }
                Row {
                    spacing: 8
                    Repeater {
                        model: root.basses
                        Chip {
                            required property var modelData
                            objectName: "slowmixBass"
                            label: modelData[1]
                            on: root.player ? Math.abs(root.player.bass - modelData[0]) < 0.01 : false
                            onTapped: if (root.player) root.player.setBass(modelData[0])
                        }
                    }
                }
            }

            Item {
                width: parent.width
                height: 30
                Text {
                    objectName: "slowmixHints"
                    anchors.verticalCenter: parent.verticalCenter
                    text: root.padHints ? "✕ play · □ pause · ◄► speed · △ reverb · ○ close"
                                        : "⏎ play    space pause    ←→ speed    esc close"
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 13
                }
                // Archive toggle, deliberately quiet: stale and never-played songs live behind it.
                Text {
                    objectName: "slowmixAll"
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    readonly property bool on: root.player ? root.player.showArchive : false
                    text: "All"
                    color: on ? Theme.gold : (allHover.hovered ? Theme.text : Theme.faint)
                    font.family: Theme.mono
                    font.pixelSize: 13
                    HoverHandler { id: allHover; cursorShape: Qt.PointingHandCursor }
                    TapHandler {
                        onTapped: {
                            if (!root.player)
                                return;
                            root.player.setShowArchive(!root.player.showArchive);
                            root.row = 0;
                        }
                    }
                }
            }
        }
    }
}
