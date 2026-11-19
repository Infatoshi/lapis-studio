import QtQuick

// Music: Cmd+M, or the ♪ pill in the top bar. Two tabs in the header, Slowmix · Spotify (click,
// Tab, or R3 on the pad). Starting one source pauses the other (the backend enforces it).
//
// Slowmix: the curated rotation (tracks played in the last 30 days, most-listened first; All adds
// the archive), and only the controls in real use: Play / Pause, Loop, Speed, Reverb, Bass,
// Volume. Picking a track applies its learned favorite speed and reverb.
//   ↑↓ (j/k, w/s) pick   ⏎ play it   space play / pause   ←→ speed   ⇥ Spotify   esc close
// Spotify (spotify_player): now playing, Play / Pause, Prev, Next, Volume and a search-and-play box.
//   space play / pause   ←→ prev / next   ↑↓ volume   / or ⏎ search   ⇥ Slowmix   esc close
// DualShock 4 (L1+R1 anywhere opens it; StudioApp routes the pad here while it is open):
//   Slowmix  ↕ pick  ✕ play  □ pause  ◄► / L1 / R1 speed  △ reverb  L2 loop  R2 bass  ○ close
//   Spotify  □ play / pause  L1 prev  R1 next  ↕ volume  ✕ play / pause  ○ close     R3 switches tab
// The Esc / ○ close leaves the music playing.
FocusScope {
    id: root
    objectName: "slowmixOverlay"
    property QtObject backend
    property bool open: false
    property int row: 0
    // "slowmix" | "spotify"
    property string tab: "slowmix"
    // Controller mode: the footer and the chips speak DualShock glyphs (StudioApp binds these).
    property bool padHints: false
    signal closed()

    readonly property QtObject player: backend ? backend.slowmix : null
    readonly property QtObject spotify: backend ? backend.spotify : null
    readonly property var tracks: player ? player.tracks : []
    readonly property int count: tracks.length
    readonly property bool playing: player ? player.playing : false
    readonly property string currentPath: player ? player.currentPath : ""
    readonly property bool spotifyPlaying: spotify ? spotify.playing : false

    readonly property var speeds: [0.85, 0.90, 1.0, 1.10, 1.25, 1.6]
    readonly property var reverbs: [["off", "Off"], ["hall", "Hall"], ["heavy", "Heavy"]]
    readonly property var basses: [[0, "Flat"], [2, "Warm"], [4, "Punch"]]
    readonly property var volumes: [20, 40, 60, 80, 100]

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
        takeKeys();
        list.positionViewAtIndex(row, ListView.Contain);
    }
    function showTab(t) {
        tab = t === "spotify" ? "spotify" : "slowmix";
        if (!open)
            show();
    }
    function hide() {
        if (!open)
            return;
        open = false;
        closed();
    }
    function toggle() { open ? hide() : show(); }
    // The card is a FocusScope: focusing it restores its focused child, so the search box has to
    // give up focus first or keys would keep typing into it.
    function takeKeys() {
        search.focus = false;
        forceActiveFocus();
    }
    function switchTab() {
        tab = tab === "slowmix" ? "spotify" : "slowmix";
        takeKeys();
    }
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
    // Spotify volume one chip up / down (20..100); unknown volume starts from 60.
    function stepSpotifyVolume(d) {
        if (!spotify)
            return;
        const cur = spotify.volume >= 0 ? spotify.volume : 60;
        let i = 0, best = 1e9;
        for (let k = 0; k < volumes.length; ++k) {
            const dist = Math.abs(volumes[k] - cur);
            if (dist < best) { best = dist; i = k; }
        }
        spotify.setVolume(volumes[Math.max(0, Math.min(volumes.length - 1, i + d))]);
    }
    function spotifyToggle() { if (spotify) spotify.togglePlay(); }
    function spotifyNext() { if (spotify) spotify.nextTrack(); }
    function spotifyPrev() { if (spotify) spotify.prevTrack(); }
    function focusSearch() { search.forceActiveFocus(); }
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

    // spotify_player is polled only while its tab is on screen.
    Binding {
        target: root.spotify
        when: root.spotify !== null
        property: "polling"
        value: root.open && root.tab === "spotify"
        restoreMode: Binding.RestoreNone
    }

    visible: opacity > 0.01
    opacity: open ? 1 : 0
    Behavior on opacity { NumberAnimation { duration: 160; easing.type: Easing.OutCubic } }
    enabled: open

    Keys.onPressed: (e) => {
        const ctrl = (e.modifiers & Qt.ControlModifier) !== 0;
        const k = e.key;
        if (ctrl)
            return;  // Cmd+M closes it, Cmd +/-/0 zoom: both reach the app
        if (k === Qt.Key_Escape || k === Qt.Key_Q) {
            hide();
        } else if (k === Qt.Key_Tab || k === Qt.Key_Backtab) {
            switchTab();
        } else if (tab === "spotify") {
            if (k === Qt.Key_Space)
                spotifyToggle();
            else if (k === Qt.Key_Left || k === Qt.Key_H || k === Qt.Key_A)
                spotifyPrev();
            else if (k === Qt.Key_Right || k === Qt.Key_L || k === Qt.Key_D)
                spotifyNext();
            else if (k === Qt.Key_Up || k === Qt.Key_K || k === Qt.Key_W)
                stepSpotifyVolume(1);
            else if (k === Qt.Key_Down || k === Qt.Key_J || k === Qt.Key_S)
                stepSpotifyVolume(-1);
            else if (k === Qt.Key_Return || k === Qt.Key_Enter || k === Qt.Key_Slash || k === Qt.Key_E)
                focusSearch();
        } else if (k === Qt.Key_Down || k === Qt.Key_J || k === Qt.Key_S) {
            move(1);
        } else if (k === Qt.Key_Up || k === Qt.Key_K || k === Qt.Key_W) {
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
    // Controller glyph badge ("□", "L1", "R3"): hairline, mono, only in controller mode.
    component PadBadge: Rectangle {
        id: badge
        property string glyph: ""
        objectName: "padBadge"
        visible: root.padHints && glyph.length > 0
        width: badgeText.implicitWidth + 10
        height: 18
        radius: 5
        color: "transparent"
        border.width: 1
        border.color: Theme.line
        Text {
            id: badgeText
            anchors.centerIn: parent
            text: badge.glyph
            color: Theme.dim
            font.family: Theme.mono
            font.pixelSize: 11
        }
    }
    // Hairline chip: gold border + gold text while on. `glyph` adds a pad badge in controller mode.
    component Chip: Rectangle {
        id: chip
        property string label
        property string glyph: ""
        property bool on: false
        signal tapped()
        width: chipRow.implicitWidth + 28
        height: 32
        radius: 16
        color: chipHover.hovered ? Theme.card : "transparent"
        border.width: 1
        border.color: on ? Theme.goldA(0.6) : Theme.line
        Row {
            id: chipRow
            anchors.centerIn: parent
            spacing: 8
            Text {
                anchors.verticalCenter: parent.verticalCenter
                text: chip.label
                color: chip.on ? Theme.gold : (chipHover.hovered ? Theme.text : Theme.dim)
                font.family: Theme.mono
                font.pixelSize: 14
            }
            PadBadge {
                anchors.verticalCenter: parent.verticalCenter
                glyph: chip.glyph
            }
        }
        HoverHandler { id: chipHover; cursorShape: Qt.PointingHandCursor }
        TapHandler { onTapped: chip.tapped() }
    }
    // Header tab: ink while active, faint otherwise.
    component TabLabel: Text {
        id: tabLabel
        property string key
        readonly property bool on: root.tab === key
        color: on ? Theme.ink : (tabHover.hovered ? Theme.text : Theme.faint)
        font.family: Theme.sans
        font.pixelSize: 30
        font.weight: on ? Font.DemiBold : Font.Normal
        HoverHandler { id: tabHover; cursorShape: Qt.PointingHandCursor }
        TapHandler { onTapped: { root.tab = tabLabel.key; root.takeKeys(); } }
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
                Row {
                    objectName: "musicTabs"
                    spacing: 14
                    TabLabel { objectName: "tabSlowmix"; key: "slowmix"; text: "Slowmix" }
                    Text {
                        anchors.verticalCenter: parent.verticalCenter
                        text: "·"
                        color: Theme.faint
                        font.family: Theme.sans
                        font.pixelSize: 24
                    }
                    TabLabel { objectName: "tabSpotify"; key: "spotify"; text: "Spotify" }
                    PadBadge {
                        anchors.verticalCenter: parent.verticalCenter
                        glyph: "R3"
                    }
                }
                Text {
                    objectName: "slowmixSource"
                    anchors.right: parent.right
                    anchors.top: parent.top
                    anchors.topMargin: 10
                    text: root.tab === "spotify"
                          ? (root.spotify && !root.spotify.available ? "spotify_player not found" : "spotify_player")
                          : root.player && !root.player.available ? "slowmix not built"
                          : root.player && root.player.showArchive ? "all songs" : "last 30 days"
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 14
                }
            }

            // --- Slowmix ------------------------------------------------------------------------
            Column {
                id: slowmixView
                objectName: "slowmixView"
                width: parent.width
                spacing: 14
                visible: root.tab === "slowmix"

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
                        glyph: "□"
                        on: root.playing
                        onTapped: if (root.player) root.player.togglePlay()
                    }
                    Chip {
                        objectName: "slowmixLoop"
                        label: "Loop"
                        glyph: "L2"
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
                    Caption { text: "Volume"; width: 86 }
                    Row {
                        spacing: 8
                        Repeater {
                            model: root.volumes
                            Chip {
                                required property var modelData
                                objectName: "slowmixVolume"
                                label: modelData + "%"
                                on: root.player ? Math.abs(root.player.volume * 100 - modelData) < 1 : false
                                onTapped: if (root.player) root.player.setVolume(modelData / 100)
                            }
                        }
                    }
                }
            }

            // --- Spotify ------------------------------------------------------------------------
            Column {
                id: spotifyView
                objectName: "spotifyView"
                width: parent.width
                spacing: 14
                visible: root.tab === "spotify"

                Item {
                    objectName: "spotifyNowPlaying"
                    width: parent.width
                    height: 56
                    Text {
                        id: spGlyph
                        anchors.verticalCenter: parent.verticalCenter
                        text: "♪"
                        color: root.spotifyPlaying ? Theme.gold : Theme.dim
                        font.pixelSize: 20
                    }
                    Column {
                        anchors.left: spGlyph.right
                        anchors.leftMargin: 12
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        spacing: 4
                        Text {
                            objectName: "spotifyTrack"
                            width: parent.width
                            text: root.spotify && root.spotify.track.length > 0
                                  ? root.spotify.track + (root.spotifyPlaying ? "" : "  ·  paused")
                                  : "Nothing playing"
                            color: root.spotifyPlaying ? Theme.ink : Theme.dim
                            elide: Text.ElideRight
                            font.family: Theme.sans
                            font.pixelSize: 19
                        }
                        Text {
                            objectName: "spotifyArtist"
                            width: parent.width
                            visible: text.length > 0
                            text: root.spotify ? root.spotify.artist : ""
                            color: Theme.dim
                            elide: Text.ElideRight
                            font.family: Theme.sans
                            font.pixelSize: 16
                        }
                    }
                }

                Text {
                    objectName: "spotifyError"
                    width: parent.width
                    visible: text.length > 0
                    text: root.spotify ? root.spotify.error : ""
                    color: Theme.copper
                    wrapMode: Text.Wrap
                    font.family: Theme.mono
                    font.pixelSize: 13
                }

                Rectangle { width: parent.width; height: 1; color: Theme.line }

                Row {
                    spacing: 10
                    Chip {
                        objectName: "spotifyPrev"
                        label: "Prev"
                        glyph: "L1"
                        onTapped: root.spotifyPrev()
                    }
                    Chip {
                        objectName: "spotifyPlay"
                        label: root.spotifyPlaying ? "Pause" : "Play"
                        glyph: "□"
                        on: root.spotifyPlaying
                        onTapped: root.spotifyToggle()
                    }
                    Chip {
                        objectName: "spotifyNext"
                        label: "Next"
                        glyph: "R1"
                        onTapped: root.spotifyNext()
                    }
                }

                Grid {
                    columns: 2
                    columnSpacing: 18
                    rowSpacing: 12
                    verticalItemAlignment: Grid.AlignVCenter
                    Caption { text: "Volume"; width: 86 }
                    Row {
                        spacing: 8
                        Repeater {
                            model: root.volumes
                            Chip {
                                required property var modelData
                                objectName: "spotifyVolume"
                                label: modelData + "%"
                                on: root.spotify ? Math.abs(root.spotify.volume - modelData) < 1 : false
                                onTapped: if (root.spotify) root.spotify.setVolume(modelData)
                            }
                        }
                    }
                    Caption { text: "Search"; width: 86 }
                    Rectangle {
                        width: spotifyView.width - 86 - 18
                        height: 36
                        radius: 18
                        color: "transparent"
                        border.width: 1
                        border.color: search.activeFocus ? Theme.goldA(0.55) : Theme.line
                        TextInput {
                            id: search
                            objectName: "spotifySearch"
                            anchors.fill: parent
                            anchors.leftMargin: 16
                            anchors.rightMargin: 16
                            verticalAlignment: TextInput.AlignVCenter
                            color: Theme.ink
                            selectionColor: Theme.goldA(0.35)
                            font.family: Theme.sans
                            font.pixelSize: 17
                            clip: true
                            onAccepted: {
                                if (root.spotify && text.trim().length > 0)
                                    root.spotify.searchAndPlay(text.trim());
                                text = "";
                                Qt.callLater(root.takeKeys);  // not inside the input's own key event
                            }
                            Keys.onEscapePressed: Qt.callLater(root.takeKeys)
                            Keys.onTabPressed: root.switchTab()
                            Text {
                                anchors.verticalCenter: parent.verticalCenter
                                visible: search.text.length === 0
                                text: "song or artist, ⏎ to play"
                                color: Theme.faint
                                font: search.font
                            }
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
                    text: root.tab === "spotify"
                          ? (root.padHints ? "□ play · L1 prev · R1 next · ↕ volume · R3 slowmix · ○ close"
                                           : "space play    ←→ prev / next    ↑↓ volume    / search    ⇥ slowmix")
                          : (root.padHints ? "✕ play · □ pause · ◄► speed · △ reverb · ○ close"
                                           : "⏎ play    space pause    ←→ speed    esc close")
                    color: Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 13
                }
                // Archive toggle, deliberately quiet: stale and never-played songs live behind it.
                Text {
                    objectName: "slowmixAll"
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    visible: root.tab === "slowmix"
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
