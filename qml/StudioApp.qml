import QtQuick

// Three lenses over the same projects: 0 Constellation, 1 Ladder, 2 Stage.
FocusScope {
    id: app
    objectName: "studioApp"
    property QtObject backend
    property bool demoMode: false

    property int lens: 0
    property string projectKey: ""
    property int artifactIndex: 0
    property bool proseOpen: false
    property int selectedIsland: 0
    // The docked composer stays collapsed unless it is in use (see the Composer below).
    property bool composerDocked: false
    readonly property bool composerShown: composerDocked || answers.inputFocused || answers.pickerOpen
                                          || answers.hasQuick || answers.hasTrace
    // Constellation filter: "all" | "needs" (only sessions waiting on the human). Cmd+1 / Cmd+2.
    property string islandFilter: "all"
    function setIslandFilter(f) {
        islandFilter = f === "needs" ? "needs" : "all";
        // The ring never waits on an island the filter just hid.
        if (constellation.count > 0 && constellation.slotOf(selectedIsland) < 0)
            selectedIsland = constellation.slots[0];
    }
    // Code diffs on the Stage are opt-in (the Code pill or Settings). Turning them on lands on
    // the first diff, so the click shows what it asked for.
    function toggleCodeDiffs() {
        if (!backend)
            return;
        const on = !backend.showCodeDiffs;
        backend.showCodeDiffs = on;
        if (on && lens === 2) {
            for (let i = 0; i < artifacts.length; ++i)
                if (artifacts[i].kind === "diff") {
                    stage.direction = i > artifactIndex ? 1 : -1;
                    artifactIndex = i;
                    break;
                }
        } else if (artifactIndex >= artifacts.length) {
            artifactIndex = Math.max(0, artifacts.length - 1);
        }
    }

    // Music (slowmix): "♪ night drive" while a track plays, a bare "♪" otherwise.
    readonly property QtObject player: backend ? backend.slowmix : null
    readonly property QtObject spotify: backend ? backend.spotify : null
    readonly property bool spotifyPlaying: spotify ? spotify.playing : false
    readonly property bool musicPlaying: (player ? player.playing : false) || spotifyPlaying
    // Idle in controller mode it names the chord that opens it: "♪ L1+R1".
    readonly property string musicLabel: player && player.playing && player.currentTrack.length > 0 ? "♪ " + player.currentTrack
                                         : spotifyPlaying && spotify.track.length > 0 ? "♪ " + spotify.track
                                         : (padHints ? "♪ L1+R1" : "♪")
    // Ambient voice (Cmd+Shift+V or the Ambient pill): Phonon "You:" lines become intents.
    readonly property bool ambientOn: backend ? backend.ambientVoiceEnabled : false
    function toggleAmbient() { if (backend) backend.toggleAmbientVoice(); }
    // L1 + R1 from anywhere: whatever else is up steps aside for the music card.
    function padToggleMusic() {
        poke();
        if (onboarding.open || controllerMap.open)
            return;
        if (music.open) {
            music.hide();
            return;
        }
        if (omnibar.open) omnibar.close();
        if (settings.open) settings.hide();
        if (briefing.open) briefing.hide();
        if (review.open) review.hide();
        if (dialog.open) dialog.hide();
        music.show();
    }
    function toggleMusic() {
        if (music.open)
            music.hide();
        else if (!settings.open && !briefing.open && !review.open && !controllerMap.open && !dialog.open && !onboarding.open)
            music.show();
    }

    // UI zoom (0.75..1.60) and input mode ("vim" | "wasd" | "mouse"), persisted by the backend.
    // There is no arrows mode; arrow keys are a silent fallback in every mode.
    readonly property real uiScale: backend ? backend.uiScale : 1.0
    readonly property string navMode: backend ? backend.navMode : "controller"
    // Hovered / pinned semantic targets; viewers push into it, the floating ask card reads it.
    readonly property QtObject targets: backend ? backend.targets : null
    Binding {
        target: app.targets
        when: app.targets !== null
        property: "scopeSessionId"
        value: app.lens > 0 ? app.projectKey : ""
    }
    Binding {
        target: app.targets
        when: app.targets !== null
        property: "scopeArtifactId"
        value: app.lens === 2 ? String(app.artifactIndex) : ""
    }

    // --- Tab next-action predictor -------------------------------------------------------------
    property var visited: []        // project keys zoomed into this session
    property var seenArtifacts: []  // artifact indices shown for the current project
    property bool idle: true        // no key for 600 ms: the ring may breathe
    property int predictorAccepts: 0
    property string lastAccepted: ""
    // An animation, not a Timer: it runs on the animation clock, so the demo recorder and the
    // bench (which step that clock frame by frame) see the ring appear deterministically.
    // (A NumberAnimation, not a PauseAnimation: Qt times a lone pause with a wall-clock timer.)
    property real idleT: 0
    SequentialAnimation {
        id: idleTimer
        NumberAnimation { target: app; property: "idleT"; from: 0; to: 1; duration: 600 }
        ScriptAction { script: app.idle = true }
    }
    function poke() {
        lastInputMs = Date.now();
        idle = false;
        idleTimer.restart();
    }
    function markSeen() {
        if (lens === 2 && seenArtifacts.indexOf(artifactIndex) < 0)
            seenArtifacts = seenArtifacts.concat([artifactIndex]);
    }
    onArtifactIndexChanged: markSeen()
    onLensChanged: {
        if (targets)
            targets.clearHoverTarget();  // the lens that reported it is fading out
        if (lens === 0)
            seenArtifacts = [];  // re-entering a project starts a fresh tour
        markSeen();
        if (lens > 0 && projectKey.length > 0 && visited.indexOf(projectKey) < 0)
            visited = visited.concat([projectKey]);
    }
    onProjectKeyChanged: {
        seenArtifacts = [];
        paramScope = null;
    }

    // The view in charge supplies the prediction: the composer while it has focus, else the lens.
    readonly property var predictedAction: {
        if (settings.open || onboarding.open || dialog.open || briefing.open || omnibar.open || review.open || music.open)
            return null;
        if (answers.inputFocused)
            return answers.predictedAction;
        if (lens === 0)
            return constellation.predictedAction;
        if (lens === 1)
            return ladder.predictedAction;
        return stage.predictedAction;
    }
    // The prompt Tab puts in the composer: a warm model suggestion for this exact artifact when
    // one has landed, a follow-up right after an answer, else the lens's own heuristic prompt.
    readonly property string promptSuggestion: {
        if (!project)
            return "";
        const key = projectKey + ":" + artifactIndex;
        if (backend && backend.suggestion.length > 0 && backend.suggestionKey === key)
            return backend.suggestion;
        if (backend && backend.quickAnswer.length > 0 && !backend.quickBusy)
            return "What should we try next?";
        const p = lens === 1 ? ladder.predictedAction : (lens === 2 ? stage.predictedAction : null);
        if (p && p.actionKind === "prompt")
            return p.promptSuggestion;
        return lens === 2 ? stage.promptFor(artifact) : "";
    }
    // Ask the warm model for a suggestion once the user settles on an artifact (opt-in, cheap).
    Timer {
        id: suggestTimer
        interval: 900
        onTriggered: if (app.backend && app.lens === 2) app.backend.requestSuggestion(app.projectKey, app.artifactIndex)
    }
    onPromptSuggestionChanged: suggestTimer.restart()

    // Where the predicted target is on screen (re-evaluated when anything that moves it changes).
    readonly property rect predictedRect: {
        const p = predictedAction;
        const deps = [zoom, width, height, uiScale, artifactIndex, lens, answers.pickerOpen, answers.height,
                      constellation.width, constellation.height, constellation.scrollY, constellation.restCenters,
                      selectedIsland, stage.slideP, stage.opacity, ladder.opacity];
        if (!p || deps.length === 0)
            return Qt.rect(0, 0, 0, 0);
        let r = Qt.rect(0, 0, 0, 0);
        let item = null;
        if (p.targetId === "composer") {
            r = answers.barRect();
            item = answers;
        } else if (lens === 0) {
            r = constellation.targetRect(p.targetId);
            item = constellation;
        } else if (p.targetId.startsWith("rung:")) {
            r = ladder.targetRect(p.targetId);
            item = ladder;
        } else if (p.targetId.startsWith("pager:")) {
            r = stage.targetRect(p.targetId);
            item = stage;
        }
        if (!item || r.width <= 0)
            return Qt.rect(0, 0, 0, 0);
        return item.mapToItem(app, r.x, r.y, r.width, r.height);
    }

    function acceptPrediction() {
        const p = predictedAction;
        if (!p)
            return false;
        if (p.actionKind === "prompt")
            return answers.acceptSuggestion(p.promptSuggestion);  // counted by onSuggestionAccepted
        lastAccepted = p.label;
        predictorAccepts += 1;
        switch (p.actionKind) {
        case "open": enterProject(p.index); break;
        case "stage": goLens(2); break;
        case "ladder": goLens(1); break;
        case "artifact":
            stage.direction = p.index > artifactIndex ? 1 : -1;
            artifactIndex = p.index;
            proseOpen = false;
            break;
        case "newSession": dialog.show(); break;
        }
        return true;
    }
    Binding {
        target: Theme
        property: "fontScale"
        value: app.uiScale
    }
    Binding {
        target: Theme
        property: "theme"
        value: backend ? backend.theme : "gold"
    }
    function zoomBy(f) { if (backend) backend.zoomBy(f); }
    function setZoom(s) { if (backend) backend.uiScale = s; }

    // A nav letter in the active keymap, as the arrow key it stands for (0 when it is not one).
    // Arrow keys keep working in every mode; mouse mode has no nav letters at all.
    function navArrow(e) {
        if (e.modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier | Qt.ShiftModifier))
            return 0;
        if (navMode === "vim") {
            switch (e.key) {
            case Qt.Key_H: return Qt.Key_Left;
            case Qt.Key_L: return Qt.Key_Right;
            case Qt.Key_J: return Qt.Key_Down;
            case Qt.Key_K: return Qt.Key_Up;
            }
        } else if (navMode === "wasd") {
            switch (e.key) {
            case Qt.Key_A: return Qt.Key_Left;
            case Qt.Key_D: return Qt.Key_Right;
            case Qt.Key_S: return Qt.Key_Down;
            case Qt.Key_W: return Qt.Key_Up;
            }
        }
        return 0;
    }

    readonly property var projects: backend ? backend.projects : []
    readonly property var project: {
        for (let i = 0; i < projects.length; ++i)
            if (projects[i].key === projectKey)
                return projects[i];
        return null;
    }
    readonly property var artifacts: project ? project.artifacts : []
    // The session's readable conversation ([{role, text}], oldest first), shown on the Ladder.
    readonly property var turns: project && project.turns ? project.turns : []
    readonly property var artifact: artifacts.length > 0 ? artifacts[Math.min(artifactIndex, artifacts.length - 1)] : null

    // Shared-element zoom. The island of the current project (ix, iy, iw, ih) grows into the
    // full-bleed Ladder/Stage while its siblings recede, and Esc shrinks it back into that island.
    readonly property int projectIdx: {
        for (let i = 0; i < projects.length; ++i)
            if (projects[i].key === projectKey)
                return i;
        return -1;
    }
    readonly property rect islandRect: projectIdx >= 0 && constellation.slotOf(projectIdx) >= 0
        ? constellation.islandRect(projectIdx)
        : Qt.rect(width / 2 - 200, height / 2 - 103, 400, 206)
    property real zoom: lens === 0 ? 0 : 1
    Behavior on zoom { NumberAnimation { duration: Theme.zoomMs; easing.type: Easing.OutQuint } }

    focus: true

    // Keyboard home: a focus sink inside this scope, so keys bubble to the handler below
    // and no hidden text field keeps focus after the dialog or composer lets go.
    Item {
        id: sink
        focus: true
    }
    function takeFocus() {
        sink.forceActiveFocus();
    }

    function goLens(l) {
        if (l > 0 && !project) {
            if (projects.length === 0)
                return;
            projectKey = projects[Math.min(selectedIsland, projects.length - 1)].key;
        }
        if (l === 0) {
            proseOpen = false;
            if (projectIdx >= 0)
                selectedIsland = projectIdx;  // the ring waits on the island we shrink back into
        }
        lens = l;
    }

    function enterProject(i) {
        if (i < 0 || i >= projects.length)
            return;
        selectedIsland = i;
        if (projectKey !== projects[i].key) {
            projectKey = projects[i].key;
            artifactIndex = 0;
            backend.clearAnswers();
        }
        lens = projects[i].artifacts.length > 0 ? 2 : 1;
    }

    function stepArtifact(d) {
        if (artifacts.length === 0)
            return;
        stage.direction = d;
        artifactIndex = (artifactIndex + d + artifacts.length) % artifacts.length;
        proseOpen = false;
    }

    // A parameter pill's Enter scopes the next quick question to it: "regs (128; runs 96 -> 168 -> 128)".
    property var paramScope: null  // cleared with the project (onProjectKeyChanged above)
    function scopeLabel(p) {
        if (!p)
            return "";
        let s = p.name;
        if (p.value && p.value.length > 0)
            s += " = " + p.value;
        if (p.runs && p.runs.length > 1)
            s += " (runs: " + p.runs.join(", ") + ")";
        return s;
    }
    function scopeToParam(p) {
        if (!p)
            return;
        paramScope = p;
        stage.terms.close();
        answers.focusInput("");
    }
    function ask(text) {
        if (!project)
            return;
        const q = paramScope ? "About " + scopeLabel(paramScope) + ": " + text : text;
        paramScope = null;
        backend.ask(projectKey, artifactIndex, q);
    }

    // The project on screen: its new turns never raise a +N badge, and opening it clears one.
    Binding {
        target: app.backend
        when: app.backend !== null
        property: "focusedProject"
        value: app.lens > 0 ? app.projectKey : ""
    }
    function think(text) { if (project) backend.think(projectKey, artifactIndex, text); }
    function steer(text) { if (project) backend.steer(projectKey, text); }

    // --- Studio UI actions ("can you open the slowmix tab", a model's [[ACTION:..]] tag) ---------
    // The backend matched the words, ran what it owns (playback, theme, diffs, layout) and put
    // the one-line confirmation in the answer strip; this side opens and filters the views.
    property string lastUiAction: ""
    property int uiActions: 0
    function clearOverlaysFor(keep) {
        if (omnibar.open) omnibar.close();
        if (keep !== "music" && music.open) music.hide();
        if (keep !== "settings" && settings.open) settings.hide();
        if (keep !== "briefing" && briefing.open) briefing.hide();
        if (keep !== "review" && review.open) review.hide();
        if (dialog.open) dialog.hide();
    }
    function runUiAction(action, arg) {
        lastUiAction = arg && arg.length > 0 ? action + ":" + arg : action;
        uiActions += 1;
        if (onboarding.open || controllerMap.open)
            return;
        switch (action) {
        case "open_music": clearOverlaysFor("music"); music.showTab("slowmix"); break;
        case "open_spotify": clearOverlaysFor("music"); music.showTab("spotify"); break;
        case "go_home":
            if (music.open) music.hide();
            takeFocus();
            goLens(0);
            break;
        case "focus_project":
        case "open_stage":
        case "open_ladder": {
            const i = backend ? backend.projectIndex(arg) : -1;
            if (i < 0)
                break;
            if (music.open) music.hide();
            takeFocus();
            if (action === "focus_project") {
                goLens(0);
                selectedIsland = i;
            } else {
                enterProject(i);
                goLens(action === "open_stage" && projects[i].artifacts.length > 0 ? 2 : 1);
            }
            break;
        }
        case "close_music": if (music.open) music.hide(); break;
        case "open_review": clearOverlaysFor("review"); if (!review.open) review.show(); break;
        case "open_settings": clearOverlaysFor("settings"); if (!settings.open) settings.show(); break;
        case "open_briefing":
            clearOverlaysFor("briefing");
            takeFocus();
            goLens(0);
            if (!briefing.open) briefing.show();
            break;
        case "filter_needs_me":
        case "filter_all":
            if (music.open) music.hide();
            takeFocus();
            goLens(0);
            setIslandFilter(action === "filter_needs_me" ? "needs" : "all");
            break;
        case "reset_layout":
            takeFocus();
            goLens(0);
            break;
        case "sort_map":
        case "move_cluster":
            // The backend already re-sorted (and reset dragged cards): the islands glide there.
            if (music.open) music.hide();
            takeFocus();
            goLens(0);
            constellation.animateRelayout();
            break;
        case "show_diffs":
            // The backend flipped the pref; land on the first diff, or keep the index in range.
            if (arg === "1" && lens === 2) {
                for (let i = 0; i < artifacts.length; ++i)
                    if (artifacts[i].kind === "diff") {
                        stage.direction = i > artifactIndex ? 1 : -1;
                        artifactIndex = i;
                        break;
                    }
            } else if (artifactIndex >= artifacts.length) {
                artifactIndex = Math.max(0, artifacts.length - 1);
            }
            break;
        }
    }

    // What is on screen, for the model's Tier 3 "Studio live state:" (re-pushed on every change).
    readonly property var uiStateMap: {
        const overlays = [];
        if (music.open) overlays.push(music.tab === "spotify" ? "Music (Spotify)" : "Music (Slowmix)");
        if (settings.open) overlays.push("Settings");
        if (briefing.open) overlays.push("Morning Briefing");
        if (review.open) overlays.push("Session Review");
        if (dialog.open) overlays.push("New Session");
        if (controllerMap.open) overlays.push("Controller Map");
        if (omnibar.open) overlays.push("Ask card");
        if (proseOpen && lens > 0) overlays.push("Write-up");
        const t = targets && targets.hasActiveTarget ? targets.activeTarget : null;
        return {
            view: lens === 0 ? "Constellation" : (lens === 1 ? "Ladder" : "Stage"),
            project: project ? project.name : "",
            artifact: lens === 2 && artifact ? (artifact.title || artifact.kind || "") : "",
            filter: islandFilter,
            highlighted: t ? (t.kind + ": " + t.label) : "",
            overlays: overlays
        };
    }
    onUiStateMapChanged: if (backend) backend.setUiState(uiStateMap)
    Component.onCompleted: {
        if (backend)
            backend.setUiState(uiStateMap);
        restoreViewState();  // projects may already be published (no projectsChanged will follow)
    }

    Connections {
        target: app.backend
        function onUiActionRequested(action, arg) { app.runUiAction(action, arg); }
        function onSessionStarted(key) {
            app.projectKey = key;
            app.artifactIndex = 0;
            app.backend.clearAnswers();
            const i = app.backend.projectIndex(key);
            app.selectedIsland = Math.max(0, i);
            app.lens = 1;
        }
    }

    // --- Persisted view state (prefs "viewState") ----------------------------------------------
    // Restored once, on the first projects publish, so a restart lands where you were; a saved
    // project that no longer exists falls back cleanly to the Constellation. Every later change
    // goes to the backend, which debounces the prefs write (and only the live window has a file).
    property bool viewRestored: false
    readonly property var viewStateMap: ({
        lens: lens === 0 ? "constellation" : (lens === 1 ? "ladder" : "stage"),
        focusedProject: projectKey,
        artifactIndex: artifactIndex,
        needsMeOnly: islandFilter === "needs"
    })
    onViewStateMapChanged: if (viewRestored && backend) backend.setViewState(viewStateMap)
    function restoreViewState() {
        if (viewRestored || !backend || projects.length === 0)
            return;
        viewRestored = true;
        const v = backend.viewState;
        if (!v || !v.lens)
            return;
        setIslandFilter(v.needsMeOnly ? "needs" : "all");
        const i = backend.projectIndex(v.focusedProject || "");
        if (i < 0) {
            lens = 0;
            return;
        }
        selectedIsland = i;
        projectKey = projects[i].key;
        if (v.lens === "constellation")
            return;
        const arts = projects[i].artifacts || [];
        if (v.lens === "stage" && arts.length > 0) {
            artifactIndex = Math.max(0, Math.min(v.artifactIndex || 0, arts.length - 1));
            lens = 2;
        } else {
            artifactIndex = 0;
            lens = 1;
        }
    }
    onProjectsChanged: restoreViewState()

    Keys.onPressed: (e) => {
        const ctrl = (e.modifiers & Qt.ControlModifier) !== 0;
        // Cmd +/-/0 zoom and Cmd+K keymap cycle work everywhere, even from the composer or dialog
        // (text fields pass unhandled Cmd keys up to here).
        if (ctrl && (e.key === Qt.Key_Equal || e.key === Qt.Key_Plus)) {
            zoomBy(1.1);
            e.accepted = true;
            return;
        }
        if (ctrl && (e.key === Qt.Key_Minus || e.key === Qt.Key_Underscore)) {
            zoomBy(1 / 1.1);
            e.accepted = true;
            return;
        }
        if (ctrl && e.key === Qt.Key_0) {
            setZoom(1.0);
            // On the Constellation it also snaps dragged cards back to the auto-layout.
            if (lens === 0 && backend && !dialog.open && !onboarding.open)
                backend.resetCardPositions();
            e.accepted = true;
            return;
        }
        // Cmd+K: the floating ask card (prompt from anywhere). Cmd+Shift+K: cycle the keymap.
        if (ctrl && e.key === Qt.Key_K) {
            if ((e.modifiers & Qt.ShiftModifier) !== 0) {
                if (backend)
                    backend.cycleNavMode();
            } else {
                omnibar.toggle();
            }
            e.accepted = true;
            return;
        }
        poke();
        if (dialog.open || onboarding.open)
            return;
        if (ctrl && e.key === Qt.Key_Comma) {
            settings.open ? settings.hide() : settings.show();
            e.accepted = true;
            return;
        }
        // Cmd+Shift+R: the end-of-session review (intent vs outcome, steers, taste rules).
        if (ctrl && (e.modifiers & Qt.ShiftModifier) !== 0 && e.key === Qt.Key_R) {
            if (review.open) {
                review.hide();
            } else if (!settings.open && !briefing.open && !controllerMap.open) {
                review.show();
            }
            e.accepted = true;
            return;
        }
        // Cmd+Shift+V: ambient voice on / off (Phonon "You:" lines become intents).
        if (ctrl && (e.modifiers & Qt.ShiftModifier) !== 0 && e.key === Qt.Key_V) {
            toggleAmbient();
            e.accepted = true;
            return;
        }
        // Cmd+M: the music card (slowmix). Esc closes it; the music keeps playing.
        if (ctrl && e.key === Qt.Key_M && (e.modifiers & Qt.ShiftModifier) === 0) {
            if (music.open)
                music.hide();
            else if (!settings.open && !briefing.open && !review.open && !controllerMap.open)
                music.show();
            e.accepted = true;
            return;
        }
        if (settings.open || briefing.open || review.open || music.open)
            return;
        // Cmd+Shift+M: model & effort.
        if (ctrl && e.key === Qt.Key_M) {
            if (lens > 0) {
                answers.openPicker();
            } else {
                settings.show();
                settings.row = settings.modelRow;
            }
            e.accepted = true;
            return;
        }
        // Cmd+P: into the Stage's hyperparameter / term pills; h/l (a/d, arrows) move, each pill
        // drops its card; Enter asks about it; Esc (q) leaves the row.
        if (ctrl && e.key === Qt.Key_P && lens === 2) {
            if (stage.terms.active)
                stage.terms.close();
            else
                stage.terms.open();
            e.accepted = true;
            return;
        }
        if (lens === 2 && stage.terms.active && !ctrl) {
            const nv = navArrow(e);
            const plainK = (e.modifiers & (Qt.AltModifier | Qt.MetaModifier | Qt.ShiftModifier)) === 0;
            let k2 = nv !== 0 ? nv : e.key;
            if (navMode === "wasd" && plainK && e.key === Qt.Key_E) k2 = Qt.Key_Return;
            if (navMode === "wasd" && plainK && e.key === Qt.Key_Q) k2 = Qt.Key_Escape;
            if (k2 === Qt.Key_Right || k2 === Qt.Key_Down || k2 === Qt.Key_Tab) {
                stage.terms.step(1);
            } else if (k2 === Qt.Key_Left || k2 === Qt.Key_Up || k2 === Qt.Key_Backtab) {
                stage.terms.step(-1);
            } else if (k2 === Qt.Key_Return || k2 === Qt.Key_Enter) {
                scopeToParam(stage.terms.current());
            } else if (k2 === Qt.Key_Escape) {
                stage.terms.close();
            } else {
                stage.terms.close();  // anything else leaves the row and acts as usual
                return;
            }
            e.accepted = true;
            return;
        }
        // "/" with no text field focused opens the floating ask card.
        if (!ctrl && e.text === "/") {
            omnibar.openAt(-1, -1);
            e.accepted = true;
            return;
        }
        if (e.key === Qt.Key_Tab && !ctrl) {
            // Mouse mode, pointer on a target: Tab pins it for a multi-target ask.
            if (navMode === "mouse" && targets && targets.hasActiveTarget)
                targets.pinActiveTarget();
            else
                acceptPrediction();
            e.accepted = true;
            return;
        }
        if (ctrl && (e.key === Qt.Key_N || e.key === Qt.Key_T)) {
            dialog.show();
            e.accepted = true;
            return;
        }
        // Cmd+1 / Cmd+2 on the Constellation: All / Needs Me. Elsewhere Cmd+1..3 pick the lens
        // (Cmd+1 also clears the filter, so it always comes home to every island).
        if (ctrl && lens === 0 && (e.key === Qt.Key_1 || e.key === Qt.Key_2)) {
            app.takeFocus();
            setIslandFilter(e.key === Qt.Key_2 ? "needs" : "all");
            e.accepted = true;
            return;
        }
        if (ctrl && e.key >= Qt.Key_1 && e.key <= Qt.Key_3) {
            app.takeFocus();
            if (e.key === Qt.Key_1)
                setIslandFilter("all");
            goLens(e.key - Qt.Key_1);
            e.accepted = true;
            return;
        }
        // This handler only sees keys when no text field has focus, so nav letters are safe here.
        // Gaming mode: e interacts (Enter), q goes back (Esc). Neovim mode: g / G first and last.
        const plainKey = (e.modifiers & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier | Qt.ShiftModifier)) === 0;
        let action = e.key;
        if (navMode === "wasd" && plainKey && e.key === Qt.Key_E)
            action = Qt.Key_Return;
        else if (navMode === "wasd" && plainKey && e.key === Qt.Key_Q)
            action = Qt.Key_Escape;
        if (action === Qt.Key_Escape) {
            app.takeFocus();
            if (omnibar.open)
                omnibar.close();
            else if (proseOpen)
                proseOpen = false;
            else if (lens > 0)
                goLens(0);
            else if (constellation.hasSelection)
                constellation.clearSelection();
            e.accepted = true;
            return;
        }
        if (ctrl)
            return;
        // B on the Constellation: the Morning Briefing (changed / active projects, two lines each).
        if (lens === 0 && plainKey && e.key === Qt.Key_B) {
            briefing.show();
            e.accepted = true;
            return;
        }
        if (navMode === "vim" && e.key === Qt.Key_G && !(e.modifiers & (Qt.AltModifier | Qt.MetaModifier))) {
            const last = (e.modifiers & Qt.ShiftModifier) !== 0;
            if (lens === 0 && constellation.count > 0) {
                selectedIsland = constellation.slots[last ? constellation.count - 1 : 0];
            } else if (proseOpen) {
                prose.scrollToEnd(last ? 1 : -1);
            } else if (lens === 2 && artifacts.length > 0) {
                const to = last ? artifacts.length - 1 : 0;
                if (to !== artifactIndex) {
                    stage.direction = to > artifactIndex ? 1 : -1;
                    artifactIndex = to;
                }
            }
            e.accepted = true;
            return;
        }
        const nav = navArrow(e);
        const k = nav !== 0 ? nav : action;
        if (lens === 0) {
            // Islands travel in slot order (the number on each card), so the Needs Me filter
            // and dragged cards keep 1..9 and the arrows meaning what is on screen.
            const n = constellation.count;
            if (k >= Qt.Key_1 && k <= Qt.Key_9) {
                const pi = constellation.slotProject(k - Qt.Key_1);
                if (pi >= 0)
                    enterProject(pi);
                e.accepted = true;
            } else if ((k === Qt.Key_Return || k === Qt.Key_Enter) && n > 0) {
                enterProject(constellation.slotOf(selectedIsland) >= 0 ? selectedIsland : constellation.slots[0]);
                e.accepted = true;
            } else if (k === Qt.Key_Right) {
                selectedIsland = constellation.navStep(selectedIsland, 1, 0);
                e.accepted = true;
            } else if (k === Qt.Key_Backtab) {
                selectedIsland = constellation.stepSelection(selectedIsland, -1, true);
                e.accepted = true;
            } else if (k === Qt.Key_Left) {
                selectedIsland = constellation.navStep(selectedIsland, -1, 0);
                e.accepted = true;
            } else if (k === Qt.Key_Down) {
                selectedIsland = constellation.navStep(selectedIsland, 0, 1);
                e.accepted = true;
            } else if (k === Qt.Key_Up) {
                selectedIsland = constellation.navStep(selectedIsland, 0, -1);
                e.accepted = true;
            }
            return;
        }
        if (k === Qt.Key_Space) {
            proseOpen = !proseOpen;
            e.accepted = true;
        } else if (k === Qt.Key_Return || k === Qt.Key_Enter) {
            // Enter zooms one level further: Ladder -> Stage -> the composer.
            if (lens === 1 && artifacts.length > 0)
                goLens(2);
            else
                answers.focusInput("");
            e.accepted = true;
        } else if (lens === 2 && (k === Qt.Key_Right || k === Qt.Key_BracketRight)) {
            stepArtifact(1);
            e.accepted = true;
        } else if (lens === 2 && (k === Qt.Key_Left || k === Qt.Key_BracketLeft)) {
            stepArtifact(-1);
            e.accepted = true;
        } else if (nav !== 0) {
            // A nav letter with nothing to do on this lens (h/l on the Ladder) never starts a question.
            if (k === Qt.Key_Down) lensScrollBy(1);
            else if (k === Qt.Key_Up) lensScrollBy(-1);
            e.accepted = true;
        } else if (k === Qt.Key_Down || k === Qt.Key_PageDown) {
            lensScrollBy(1);
            e.accepted = true;
        } else if (k === Qt.Key_Up || k === Qt.Key_PageUp) {
            lensScrollBy(-1);
            e.accepted = true;
        } else if (e.text.length === 1 && e.text > " " && !(e.modifiers & Qt.AltModifier)) {
            composer.focusInput(e.text);
            e.accepted = true;
        }
    }

    // --- PS4 DualShock 4 ------------------------------------------------------------------------
    // backend.gamepad debounces the pad in C++ (stick / d-pad repeat, trigger hysteresis, 8 ms
    // right-stick scroll) and emits one signal per action; these route it to whatever is in charge,
    // topmost first: onboarding, new-session dialog, settings, model picker, composer, term pills,
    // then the lens. Defaults: ✕ select · ○ back · R1 Tab / Full Send · □ prose / picker ·
    // L1 previous view · △ next view · L2 voice · R2 send.
    readonly property QtObject gamepad: backend ? backend.gamepad : null
    readonly property bool padConnected: gamepad ? gamepad.connected : false
    // On-screen hints speak controller (R1 Tab, ✕ Select) in Controller mode or whenever a pad
    // is connected; keyboard glyphs (⇥, ⏎) otherwise.
    readonly property bool padHints: navMode === "controller" || padConnected
    // action id -> bound glyph ("accept_prediction" -> "R1"), live across remaps. An unbound
    // action has no entry, so its hint is left out rather than naming the wrong button.
    readonly property var padGlyphs: {
        if (!gamepad)
            return { voice_hold: "L2", send: "R2", accept_prediction: "R1", confirm: "✕", back: "○",
                     secondary: "□", step_prev: "L1", step_next: "△" };
        const out = {};
        const slots = gamepad.bindingSlots;
        for (let i = slots.length - 1; i >= 0; --i)  // first bound input wins
            out[slots[i].action] = slots[i].glyph;
        return out;
    }
    function padKey(action) {
        const g = padGlyphs[action];
        return g !== undefined ? g : "";
    }
    // The key that confirms / interacts in the current input mode: ✕ (pad), E (wasd), ⏎ (else).
    readonly property string confirmKey: padHints ? padKey("confirm") : (navMode === "wasd" ? "E" : "⏎")
    function padNav(dx, dy) {
        poke();
        if (omnibar.open)
            return;  // the ask card is a text field: the stick never moves the lens under it
        if (controllerMap.open) {
            if (dy !== 0 && controllerMap.slots.length > 0)
                controllerMap.row = (controllerMap.row + dy + controllerMap.slots.length) % controllerMap.slots.length;
            else if (dx !== 0)
                controllerMap.cycleAction(dx);
        } else if (briefing.open) {
            briefing.move(dy !== 0 ? dy : dx);
        } else if (review.open) {
            review.move(dy !== 0 ? dy : dx);
        } else if (music.open) {
            if (dy !== 0) music.move(dy); else music.stepSpeed(dx);
        } else if (onboarding.open) {
            onboarding.choice = Math.max(0, Math.min(onboarding.cards.length - 1, onboarding.choice + dx + dy));
        } else if (dialog.open) {
            if (dx !== 0) dialog.cycleHarness(dx); else dialog.cycleFolder(dy);
        } else if (settings.open) {
            if (dy !== 0) settings.row = (settings.row + dy + settings.rowCount) % settings.rowCount;
            else settings.change(dx);
        } else if (answers.pickerOpen && backend) {
            if (dy !== 0) backend.cycleModel(dy); else backend.cycleEffort(dx);
        } else if (answers.inputFocused) {
            if (dx !== 0) answers.cursorStep(dx);
            else if (dy < 0) answers.openPicker();          // Up from the composer: the model strip
            else if (dy > 0) takeFocus();               // Down: back to the lens, text kept
        } else if (lens === 2 && stage.terms.active) {
            stage.terms.step(dx + dy);
        } else if (lens === 0) {
            if (constellation.count === 0) return;
            selectedIsland = constellation.navStep(selectedIsland, dx, dy);
        } else if (dx !== 0) {
            if (lens === 2) stepArtifact(dx);
        } else {
            lensScrollBy(dy);
        }
    }
    // Scroll whatever reads on this lens: the prose panel when open, else the Stage, else the
    // Ladder's conversation.
    function lensScrollBy(d) {
        if (proseOpen && lens > 0) prose.scrollBy(d);
        else if (lens === 2) stage.scrollBy(d);
        else if (lens === 1) ladder.scrollBy(d);
    }
    function lensScrollPixels(px) {
        if (proseOpen && lens > 0) prose.scrollPixels(px);
        else if (lens === 2) stage.scrollPixels(px);
        else if (lens === 1) ladder.scrollPixels(px);
    }
    // The MacBook-wide Spotlight bar is up: Backend routes R2 / ✕ / ○ / L2 to it alone.
    readonly property bool spotlightOwnsPad: backend ? backend.spotlightOpen : false
    function padConfirm() {
        poke();
        if (spotlightOwnsPad) return;
        if (omnibar.open) omnibar.submit();
        else if (controllerMap.open) controllerMap.startCapture();
        else if (onboarding.open) onboarding.confirm();
        else if (briefing.open) briefing.confirm();
        else if (review.open) review.confirm();
        else if (music.open) music.playRow(music.row);
        else if (dialog.open) dialog.start();
        else if (settings.open) settings.hide();
        else if (answers.pickerOpen) answers.pickerOpen = false;  // the composer keeps focus
        else if (answers.inputFocused) { if (!answers.toggleHighlight()) answers.send(0); }
        else if (lens === 2 && stage.terms.active) scopeToParam(stage.terms.current());
        else if (lens === 0 && mapCursorOwnsPad) return;  // ✕ is hold-to-grab here (onConfirmHeld)
        else if (lens === 0) { if (constellation.count > 0) enterProject(constellation.slotOf(selectedIsland) >= 0 ? selectedIsland : constellation.slots[0]); }
        else if (lens === 1 && artifacts.length > 0) goLens(2);
        else omnibar.openAt(-1, -1);  // ✕ Prompt: the ask card, beside what the cursor is on
    }
    function padBack() {
        poke();
        if (spotlightOwnsPad) return;
        if (onboarding.open) return;  // a mode has to be picked once
        if (omnibar.open) omnibar.close();
        else if (controllerMap.open) controllerMap.hide();
        else if (briefing.open) briefing.hide();
        else if (review.open) review.hide();
        else if (music.open) music.hide();
        else if (dialog.open) dialog.hide();
        else if (settings.open) settings.hide();
        else if (answers.pickerOpen) answers.pickerOpen = false;
        else if (answers.inputFocused) answers.escapeInput();
        else if (lens === 2 && stage.terms.active) stage.terms.close();
        else if (proseOpen) proseOpen = false;
        else if (lens > 0) { takeFocus(); goLens(0); }
        else if (constellation.hasSelection) constellation.clearSelection();
    }
    function padAcceptPrediction() {
        poke();
        if (onboarding.open || dialog.open || settings.open)
            return;
        if (omnibar.open) {
            omnibar.pinActive();  // R1 is the pad's Tab: pin the hovered target, like ⇥
            return;
        }
        acceptPrediction();
    }
    function padSecondary() {
        poke();
        if (onboarding.open || dialog.open || briefing.open || omnibar.open)
            return;
        if (settings.open) {
            settings.row = settings.modelRow;
        } else if (answers.inputFocused) {
            answers.pickerOpen ? (answers.pickerOpen = false) : answers.openPicker();
        } else if (lens === 0) {
            settings.show();
            settings.row = settings.modelRow;
        } else {
            proseOpen = !proseOpen;
        }
    }
    function padStep(d) {
        poke();
        if (omnibar.open)
            return;
        if (onboarding.open) {
            onboarding.choice = Math.max(0, Math.min(onboarding.cards.length - 1, onboarding.choice + d));
        } else if (briefing.open) {
            briefing.move(d);
        } else if (dialog.open) {
            dialog.cycleModel(d);
        } else if (settings.open) {
            settings.row = (settings.row + d + settings.rowCount) % settings.rowCount;
        } else if (answers.pickerOpen && backend) {
            backend.cycleModel(d);
        } else if (review.open) {
            review.move(d);
        } else if (music.open) {
            music.move(d);
        } else if (lens === 0) {
            if (constellation.count > 0) selectedIsland = constellation.stepSelection(selectedIsland, d, true);
        } else if (lens === 1) {
            if (d > 0 && artifacts.length > 0) goLens(2);
            else if (d < 0) goLens(0);
        } else {
            takeFocus();
            stepArtifact(d);
        }
    }
    function padZoom(d) {
        poke();
        if (backend) backend.uiScale = Math.round((uiScale + d) * 100) / 100;
    }
    function padSettings() {
        poke();
        if (onboarding.open || dialog.open)
            return;
        settings.open ? settings.hide() : settings.show();
    }
    function padNewSession() {
        poke();
        if (onboarding.open || settings.open)
            return;
        dialog.open ? dialog.hide() : dialog.show();
    }
    function padScroll(dy) {
        if (onboarding.open || dialog.open || settings.open)
            return;
        lensScrollPixels(dy);
    }
    // L2: dictate into the ask card, opened beside the highlighted target (into the composer
    // instead when it already has focus). Already open (a previous turn on screen): focus goes
    // back to its prompt, so every hold dictates into it, turn after turn.
    function padVoiceHold(active) {
        poke();
        if (spotlightOwnsPad) return;
        if (active && !onboarding.open && !dialog.open && !settings.open && !controllerMap.open
                && !briefing.open && !review.open && !answers.inputFocused)
            omnibar.focusForVoice();
    }
    function padSend() {
        poke();
        if (spotlightOwnsPad) return;
        if (omnibar.open) omnibar.submit();
        else if (controllerMap.open) controllerMap.startCapture();
        else if (onboarding.open) onboarding.confirm();
        else if (dialog.open) dialog.start();
        else if (briefing.open) briefing.hide();
        else if (answers.inputFocused) answers.send(0);
        else if (lens > 0) answers.focusInput("");
        else if (!settings.open) briefing.show();  // the Constellation has nothing to send: Morning Briefing
    }
    // On the settled Constellation with nothing over it, the left stick drives the free map
    // cursor (analog, 240 Hz) and ✕ is hold-to-grab; everywhere else the stick navigates in steps.
    readonly property bool mapCursorOwnsPad: padConnected && lens === 0 && zoom < 0.01 && !omnibar.open
                                             && !controllerMap.open && !briefing.open && !review.open && !music.open
                                             && !onboarding.open && !dialog.open && !settings.open
                                             && !answers.inputFocused && !answers.pickerOpen
    onMapCursorOwnsPadChanged: if (!mapCursorOwnsPad) constellation.setStick(0, 0)
    Binding {
        target: app.gamepad
        when: app.gamepad !== null
        property: "analogCursor"
        value: app.mapCursorOwnsPad
    }
    // While the Music card is open it owns the pad: ↕ pick · ✕ play · □ pause · ◄► / L1 / R1
    // speed · △ reverb · L2 loop · R2 bass · ○ close (music keeps playing).
    Binding {
        target: app.gamepad
        when: app.gamepad !== null
        property: "musicMode"
        value: music.open
    }
    function musicPad(kind, v) {
        poke();
        // Spotify tab: □ / ✕ play-pause · L1 prev · R1 next · ↕ volume · ◄► prev / next · ○ close.
        if (music.tab === "spotify") {
            switch (kind) {
            case "nav": if (v.dy !== 0) music.stepSpotifyVolume(-v.dy); else if (v.dx < 0) music.spotifyPrev(); else music.spotifyNext(); break;
            case "confirm":
            case "secondary": music.spotifyToggle(); break;
            case "nextTab": music.spotifyNext(); break;  // R1 (accept_prediction)
            case "step": if (v < 0) music.spotifyPrev(); break;  // L1; △ does nothing here
            case "back": music.hide(); break;
            }
            return;
        }
        switch (kind) {
        case "nav": if (v.dy !== 0) music.move(v.dy); else music.stepSpeed(v.dx); break;
        case "confirm": music.playRow(music.row); break;
        case "secondary": music.togglePlay(); break;
        case "prevTab": music.stepSpeed(-1); break;
        case "nextTab": music.stepSpeed(1); break;
        case "step": if (v < 0) music.stepSpeed(-1); else music.cycleReverb(); break;
        case "voice": if (v) music.toggleLoop(); break;
        case "send": music.cycleBass(); break;
        case "back": music.hide(); break;
        }
    }
    Connections {
        target: app.gamepad
        function onNav(dx, dy) { if (music.open) app.musicPad("nav", { dx: dx, dy: dy }); else app.padNav(dx, dy); }
        function onConfirm() { if (music.open) app.musicPad("confirm"); else app.padConfirm(); }
        function onConfirmHeld(down) {
            // A release always reaches a grab in progress, even if something opened meanwhile.
            if (app.mapCursorOwnsPad || (!down && constellation.grabMode !== "")) {
                app.poke();
                constellation.padGrab(down);
            }
        }
        function onCursorMove(vx, vy) {
            if (app.mapCursorOwnsPad) {
                app.poke();
                constellation.setStick(vx, vy);
            } else {
                constellation.setStick(0, 0);
            }
        }
        function onBack() { if (music.open) app.musicPad("back"); else app.padBack(); }
        function onBackHeld(down) { if (!music.open) answers.setBackHeld(down); }
        function onAcceptPrediction() { if (music.open) app.musicPad("nextTab"); else app.padAcceptPrediction(); }
        function onSecondary() { if (music.open) app.musicPad("secondary"); else app.padSecondary(); }
        function onStep(d) { if (music.open) app.musicPad("step", d); else app.padStep(d); }
        function onZoomStep(d) { if (!music.open) app.padZoom(d); }  // R3 switches the music tab instead
        function onR3Pressed() { if (music.open) { app.poke(); music.switchTab(); } }
        function onZoomReset() { if (app.backend) app.backend.resetZoom(); }
        function onToggleSettings() { if (!music.open) app.padSettings(); }
        function onToggleNewSession() { if (!music.open) app.padNewSession(); }
        function onToggleMusic() { app.padToggleMusic(); }
        function onToggleReview() {
            app.poke();
            if (review.open)
                review.hide();
            else if (!music.open && !onboarding.open && !dialog.open && !settings.open && !controllerMap.open && !briefing.open)
                review.show();
        }
        function onToggleBriefing() {
            app.poke();
            if (!music.open && !onboarding.open && !dialog.open && !settings.open && !controllerMap.open)
                briefing.open ? briefing.hide() : briefing.show();
        }
        function onScrollDelta(dy) { if (!music.open) app.padScroll(dy); }
        function onTouchpadPan(dx, dy) {
            if (music.open || onboarding.open || dialog.open || settings.open || controllerMap.open)
                return;
            if (app.proseOpen && app.lens > 0) prose.scrollPixels(dy);
            else if (app.lens === 2) stage.panPixels(dx, dy);
            else if (app.lens === 1) ladder.scrollPixels(dy);
        }
        function onHorizontalInspect(dir) {
            app.poke();
            if (music.open || onboarding.open || dialog.open || settings.open || controllerMap.open)
                return;
            if (app.lens === 2) stage.panPixels(dir * 96, 0);
        }
        function onVoiceHold(active) { if (music.open) app.musicPad("voice", active); else app.padVoiceHold(active); }
        function onSendPressed() { if (music.open) app.musicPad("send"); else app.padSend(); }
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.bg
    }

    // Lens 1: constellation of project islands.
    Constellation {
        id: constellation
        objectName: "constellation"
        anchors.fill: parent
        projects: app.projects
        selectedIndex: app.selectedIsland
        recede: app.zoom
        focusIndex: app.projectIdx
        visible: app.zoom < 0.999
        visited: app.visited
        cursorActive: app.lens === 0 && focusCursor.active
        targets: app.lens === 0 ? app.targets : null
        filter: app.islandFilter
        cardPositions: app.backend ? app.backend.cardPositions : ({})
        clusterOrder: app.backend ? app.backend.clusterOrder : ["Work", "Projects", "Apps", "Quick Qs"]
        mapSort: app.backend ? app.backend.mapSort : "category"
        freeCursor: app.padConnected && app.lens === 0
        onCursorPicked: (i) => { app.poke(); app.selectedIsland = i; }
        onEnter: (i) => app.enterProject(i)
        onZoomBy: (f) => app.zoomBy(f)
        onNewSession: dialog.show()
        onCardMoved: (key, x, y) => { if (app.backend) app.backend.setCardPosition(key, x, y); }
        onFilterPicked: (f) => app.setIslandFilter(f)
        onResetLayout: if (app.backend) app.backend.resetCardPositions()
        onReviewRequested: review.show()
        musicLabel: app.musicLabel
        musicPlaying: app.musicPlaying
        onMusicRequested: app.toggleMusic()
        ambientOn: app.ambientOn
        ambientBusy: app.backend ? app.backend.ambientBusy : false
        onAmbientRequested: app.toggleAmbient()
        padHints: app.padHints
        padGlyphs: app.padGlyphs
    }

    // The shared element: a card that starts as the island and becomes the whole window.
    Item {
        id: morph
        objectName: "zoomMorph"
        readonly property real t: app.zoom
        readonly property rect from: app.islandRect
        x: from.x * (1 - t)
        y: from.y * (1 - t)
        width: from.width + (app.width - from.width) * t
        height: from.height + (app.height - from.height) * t
        visible: t > 0.001
        clip: t < 0.999

        Rectangle {
            anchors.fill: parent
            radius: 26 * (1 - morph.t)
            color: Qt.rgba(0.051 * (1 - morph.t), 0.047 * (1 - morph.t), 0.043 * (1 - morph.t), 1)
        }
        // The island itself rides the first fifth of the zoom, then hands over to the lens.
        Island {
            enabled: false
            proj: app.project
            number: constellation.slotOf(app.projectIdx) + 1
            width: morph.from.width
            height: morph.from.height
            visible: opacity > 0.005
            opacity: Math.max(0, 1 - morph.t / 0.2)
        }

        // Lenses 2 and 3 live in one layer, laid out at window size and scaled into the card.
        Item {
            id: inner
            width: app.width
            height: app.height
            transformOrigin: Item.TopLeft
            scale: morph.width / Math.max(1, app.width)
            y: (morph.height - app.height * scale) / 2
            opacity: Math.min(1, Math.max(0, (morph.t - 0.3) / 0.45))
            visible: opacity > 0.005

            // Header: project name, status dot, lens dots.
            Item {
                id: header
                x: 56
                y: 36
                width: parent.width - 112
                height: 36
                Rectangle {
                    id: statusDot
                    width: 10; height: 10; radius: 5
                    anchors.verticalCenter: parent.verticalCenter
                    color: app.project && app.project.active ? Theme.gold : Theme.faint
                }
                Text {
                    id: headerName
                    anchors.left: statusDot.right
                    anchors.leftMargin: 14
                    anchors.verticalCenter: parent.verticalCenter
                    text: app.project ? app.project.name : ""
                    color: Theme.dim
                    font.family: Theme.sans
                    font.pixelSize: Theme.px(20)
                    font.weight: Font.Medium
                    font.letterSpacing: 0.4
                }
                // What the agent is doing, in plain English, next to the (often cryptic) folder name.
                Text {
                    objectName: "headerJob"
                    anchors.left: headerName.right
                    anchors.leftMargin: 14
                    anchors.right: codePill.visible ? codePill.left : headerAmbient.left
                    anchors.rightMargin: 24
                    anchors.verticalCenter: parent.verticalCenter
                    visible: text.length > 0
                    text: app.project && app.project.jobName ? "·  " + app.project.jobName : ""
                    color: Theme.text
                    elide: Text.ElideRight
                    font.family: Theme.sans
                    font.pixelSize: Theme.px(20)
                }
                // Stage only, and only when this project has diffs: code stays out of the deck
                // until asked for. Gold hairline while on.
                Rectangle {
                    id: codePill
                    objectName: "codePill"
                    readonly property bool on: app.backend ? app.backend.showCodeDiffs : false
                    visible: app.lens === 2 && app.project !== null && (app.project.diffCount || 0) > 0
                    anchors.right: headerAmbient.left
                    anchors.rightMargin: 12
                    anchors.verticalCenter: parent.verticalCenter
                    width: codeText.implicitWidth + 26
                    height: 28
                    radius: 14
                    color: codeHover.hovered ? Theme.card : "transparent"
                    border.width: 1
                    border.color: on ? Theme.goldA(0.55) : Theme.line
                    Text {
                        id: codeText
                        anchors.centerIn: parent
                        text: "Code"
                        color: codePill.on ? Theme.gold : Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 14
                    }
                    HoverHandler { id: codeHover; cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: app.toggleCodeDiffs() }
                }
                // Ambient voice: the same pill as the Constellation's top bar (Cmd+Shift+V).
                Rectangle {
                    id: headerAmbient
                    objectName: "headerAmbientPill"
                    anchors.right: headerMusic.left
                    anchors.rightMargin: 12
                    anchors.verticalCenter: parent.verticalCenter
                    width: headerAmbientText.implicitWidth + 26
                    height: 28
                    radius: 14
                    color: headerAmbientHover.hovered ? Theme.card : "transparent"
                    border.width: 1
                    border.color: app.ambientOn ? Theme.goldA(0.55) : Theme.line
                    Text {
                        id: headerAmbientText
                        anchors.centerIn: parent
                        text: app.backend && app.backend.ambientBusy ? "Ambient ·" : "Ambient"
                        color: app.ambientOn ? Theme.gold : Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 14
                    }
                    HoverHandler { id: headerAmbientHover; cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: app.toggleAmbient() }
                }
                // Music: the same quiet ♪ pill as the Constellation's top bar.
                Rectangle {
                    id: headerMusic
                    objectName: "headerMusicPill"
                    anchors.right: lensDots.left
                    anchors.rightMargin: 22
                    anchors.verticalCenter: parent.verticalCenter
                    width: Math.min(headerMusicText.implicitWidth, 260) + 26
                    height: 28
                    radius: 14
                    color: headerMusicHover.hovered ? Theme.card : "transparent"
                    border.width: 1
                    border.color: app.musicPlaying ? Theme.goldA(0.55) : Theme.line
                    Text {
                        id: headerMusicText
                        anchors.centerIn: parent
                        width: Math.min(implicitWidth, 260)
                        elide: Text.ElideRight
                        text: app.musicLabel
                        color: app.musicPlaying ? Theme.gold : Theme.dim
                        font.family: Theme.mono
                        font.pixelSize: 14
                    }
                    HoverHandler { id: headerMusicHover; cursorShape: Qt.PointingHandCursor }
                    TapHandler { onTapped: app.toggleMusic() }
                }
                Row {
                    id: lensDots
                    anchors.right: parent.right
                    anchors.verticalCenter: parent.verticalCenter
                    spacing: 10
                    Repeater {
                        model: 3
                        Rectangle {
                            required property int index
                            width: index === app.lens ? 22 : 8
                            height: 8
                            radius: 4
                            color: index === app.lens ? Theme.gold : Theme.faint
                            Behavior on width { NumberAnimation { duration: Theme.zoomMs } }
                        }
                    }
                }
            }

            Ladder {
                id: ladder
                anchors.top: header.bottom
                anchors.topMargin: 24
                anchors.bottom: app.composerShown ? answers.top : parent.bottom
                anchors.bottomMargin: app.composerShown ? 16 : 28
                anchors.left: parent.left
                anchors.right: parent.right
                project: app.project
                backend: app.backend
                opacity: app.lens === 1 ? 1 : 0
                visible: opacity > 0.005
                scale: 0.97 + 0.03 * opacity
                Behavior on opacity { NumberAnimation { duration: Theme.zoomMs; easing.type: Easing.OutQuint } }
            }

            Stage {
                id: stage
                anchors.top: header.bottom
                anchors.topMargin: 18
                anchors.bottom: app.composerShown ? answers.top : parent.bottom
                anchors.bottomMargin: app.composerShown ? 18 : 30
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.leftMargin: 56
                anchors.rightMargin: 56
                artifact: app.artifact
                artifacts: app.artifacts
                artifactIndex: app.artifactIndex
                artifactCount: app.artifacts.length
                seen: app.seenArtifacts
                live: app.backend ? app.backend.live : false
                backend: app.backend
                params: app.project && app.project.params ? app.project.params : []
                padHints: app.padHints
                padGlyphs: app.padGlyphs
                onAskAbout: (p) => app.scopeToParam(p)
                opacity: app.lens === 2 ? 1 : 0
                visible: opacity > 0.005
                scale: 0.97 + 0.03 * opacity
                animate: app.lens === 2 && app.zoom > 0.99 && opacity > 0.99
                Behavior on opacity { NumberAnimation { duration: Theme.zoomMs; easing.type: Easing.OutQuint } }
                onPick: (i) => {
                    if (i === app.artifactIndex)
                        return;
                    stage.direction = i > app.artifactIndex ? 1 : -1;
                    app.artifactIndex = i;
                    app.proseOpen = false;
                }
            }

            ProsePanel {
                id: prose
                anchors.fill: stage
                text: app.project ? app.project.prose : ""
                thinking: app.project ? app.project.thinking : ""
                shown: app.proseOpen && app.lens > 0
            }

            // Docked composer: collapsed below the window edge by default, so charts, tables and
            // the Ladder get the full height (prompts go through the ask card at the target).
            // It slides in while it has focus (Enter, typing on a lens, Cmd+Shift+M), while it
            // holds an answer, or while app.composerDocked pins it open.
            Composer {
                id: answers
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.leftMargin: 56
                anchors.rightMargin: 56
                anchors.bottomMargin: app.composerShown ? 32 : -(height + 48)
                backend: app.backend
                hasSession: app.project !== null && app.project.sessionCount > 0
                navMode: app.navMode
                padHints: app.padHints
                padGlyphs: app.padGlyphs
                suggestion: app.paramScope ? "" : app.promptSuggestion
                scope: app.paramScope ? app.paramScope.name + (app.paramScope.value ? ": " + app.paramScope.value : "") : ""
                onScopeCleared: app.paramScope = null
                onSuggestionAccepted: (t) => { app.lastAccepted = t; app.predictorAccepts += 1; }
                onAsk: (t) => app.ask(t)
                onThink: (t) => app.think(t)
                onSteer: (t) => app.steer(t)
                onEscaped: app.takeFocus()
            }
        }
    }
    property alias composer: answers

    // Cmd+wheel zooms anywhere (scroll views underneath ignore it); a trackpad pinch zooms too.
    Item {
        anchors.fill: parent
        WheelHandler {
            objectName: "zoomWheel"
            acceptedModifiers: Qt.ControlModifier
            onWheel: (ev) => {
                const dy = ev.angleDelta.y !== 0 ? ev.angleDelta.y : ev.pixelDelta.y;
                if (dy !== 0)
                    app.zoomBy(Math.pow(1.0012, dy));
            }
        }
        PinchHandler {
            id: pinch
            target: null
            property real base: 1.0
            onActiveChanged: if (active) base = app.uiScale
            onActiveScaleChanged: if (active) app.setZoom(base * activeScale)
        }
    }

    // Scale pill: shows "1.15x" for 1.2 s after a change, then fades.
    Rectangle {
        id: scalePill
        objectName: "scalePill"
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 6  // below the composer (its bottom edge sits 32 px up)
        width: scaleText.implicitWidth + 24
        height: scaleText.implicitHeight + 6
        radius: height / 2
        color: Theme.cardHi
        border.width: 1
        border.color: Theme.line
        opacity: 0
        visible: opacity > 0.01
        Text {
            id: scaleText
            objectName: "scalePillText"
            anchors.centerIn: parent
            text: app.uiScale.toFixed(2) + "x"
            color: Theme.ink
            font.family: Theme.mono
            font.pixelSize: 14
        }
        Behavior on opacity { NumberAnimation { duration: 260; easing.type: Easing.OutCubic } }
        Timer {
            id: scaleHide
            interval: 1200
            onTriggered: scalePill.opacity = 0
        }
        Connections {
            target: app
            function onUiScaleChanged() {
                scalePill.opacity = 1;
                scaleHide.restart();
            }
        }
    }

    // Tab predictor ring + "⇥ Tab" (or "R1") badge, over whichever lens is settled on screen.
    PredictorRing {
        id: predictor
        z: 15
        anchors.fill: parent
        readonly property var p: app.predictedAction
        readonly property string tid: p ? p.targetId : ""
        showRing: !(app.lens === 0 && focusCursor.active && tid === "island:" + app.selectedIsland)
        target: app.predictedRect
        label: p ? p.label : ""
        round: tid.startsWith("pager:") || tid.startsWith("rung:") || tid === "newSession"
        cardRadius: tid === "composer" ? 22 : 26
        badgeSide: tid.startsWith("pager:") ? "right" : tid.startsWith("rung:") ? "left"
                   : tid === "newSession" ? "bottom" : tid.startsWith("island:") ? "topTuck" : "top"
        idle: app.idle
        // One input at a time: the bound button alone with a controller ("R1"), "⇥ Tab" on keys.
        // An unbound accept action shows no key chip rather than a keyboard glyph.
        keyHint: app.padHints ? app.padKey("accept_prediction") : "⇥ Tab"
        visible: opacity > 0.01 && !(app.lens === 2 && stage.cardOpen)
        shown: p !== null && (app.zoom < 0.001 || app.zoom > 0.999) && !(app.lens === 2 && app.stageSliding)
               && !(app.lens === 2 && (stage.cardOpen || stage.terms.active))
               && !app.proseOpen && (app.lens !== 1 || ladder.opacity > 0.99) && (app.lens !== 2 || stage.opacity > 0.99)
    }
    readonly property bool stageSliding: stage.slideP < 1

    // --- Focus cursor ---------------------------------------------------------------------------
    // What the controller / keyboard is on right now, with the action its confirm key takes.
    // Hidden in pure mouse mode, behind overlays, mid-zoom, and over the write-up.
    readonly property var cursorSpec: {
        const deps = [zoom, width, height, uiScale, lens, selectedIsland, artifactIndex, constellation.width,
                      constellation.height, constellation.scrollY, constellation.islandH, constellation.restCenters,
                      constellation.count, constellation.slots, constellation.cardPositions, answers.inputFocused, answers.hasSelection, answers.inputText,
                      answers.height, answers.pickerOpen, stage.width, stage.height, stage.barH, stage.kind,
                      stage.terms.active, stage.terms.selected, stage.terms.count, stage.terms.expanded];
        if (deps.length === 0 || (navMode === "mouse" && !padConnected))
            return null;
        if (settings.open || onboarding.open || dialog.open || controllerMap.open || briefing.open || review.open || music.open)
            return null;
        if (lens === 0 && constellation.dragging)
            return null;  // the card under the pointer is the focus while it moves
        if (!(zoom < 0.001 || zoom > 0.999) || (proseOpen && !answers.inputFocused))
            return null;
        if (answers.inputFocused && lens > 0) {
            if (answers.pickerOpen)
                return null;
            const sel = answers.hasSelection;
            return { item: answers, r: answers.cursorRect(), radius: sel ? 8 : 22, round: false, side: "bottom", hint: "" };
        }
        if (lens === 0) {
            if (constellation.count === 0)
                return null;
            const i = constellation.slotOf(selectedIsland) >= 0 ? selectedIsland : constellation.slots[0];
            if (!constellation.islandInBand(i))
                return null;  // scrolled out of view: no ring or hint over the top bar / footer
            return { item: constellation, r: constellation.islandRect(i), radius: 26, round: false, side: "bottomTuck",
                     hint: confirmKey.length > 0 ? confirmKey + " Expand" : "" };
        }
        if (lens === 2) {
            // A term card is down (clicked or Cmd+P): wrap its pill, never the artifact behind the
            // card, so no ring line crosses the card's text.
            if (stage.terms.active || stage.terms.expanded >= 0)
                return { item: stage, r: stage.cursorRect(), radius: 16, round: true, side: "bottom", hint: "" };
            if (!artifact)
                return null;
            // Pulled up 12 px at the bottom: the Stage ends 18 px above the composer, whose own
            // predictor ring would otherwise touch this one.
            const v = stage.cursorRect();
            return { item: stage, r: Qt.rect(v.x, v.y, v.width, Math.max(0, v.height - 12)), radius: 18, round: false, side: "right",
                     hint: confirmKey.length > 0 ? confirmKey + " Prompt" : "" };
        }
        return null;
    }
    readonly property rect cursorRect: {
        const s = cursorSpec;
        if (!s || s.r.width <= 0)
            return Qt.rect(0, 0, 0, 0);
        return s.item.mapToItem(app, s.r.x, s.r.y, s.r.width, s.r.height);
    }
    // What the controller / keyboard cursor is on, as a semantic target. It becomes the active
    // prompt target (the same registry mouse hover fills), so ✕ / L2 / Cmd+K / "/" open the ask
    // card beside it with its chip attached. Mouse mode leaves the target to the pointer.
    readonly property var cursorTarget: {
        const s = cursorSpec;
        const r = cursorRect;
        if (!s || r.width <= 0 || answers.inputFocused)
            return null;
        if (lens === 0) {
            if (constellation.count === 0)
                return null;
            const i = constellation.slotOf(selectedIsland) >= 0 ? selectedIsland : constellation.slots[0];
            const p = projects[i];
            if (!p)
                return null;
            const line = p.stateLine || p.lastLine || p.nowLine || "";
            return { kind: "island", session: p.key, artifact: "", path: "", label: p.name,
                     context: p.name + " (" + p.status + ")" + (line.length > 0 ? ": " + line : ""), r: r };
        }
        if (lens === 2) {
            if (stage.terms.active || stage.terms.expanded >= 0) {
                const term = stage.terms.current();
                if (!term)
                    return null;
                return { kind: "term", session: projectKey, artifact: String(artifactIndex), path: "",
                         label: term.name + (term.value ? " = " + term.value : ""), context: scopeLabel(term), r: r };
            }
            if (!artifact || !backend)
                return null;
            return { kind: "artifact", session: projectKey, artifact: String(artifactIndex), path: artifact.path || "",
                     label: artifact.title || artifact.name || artifact.kind || "artifact",
                     context: backend.contextSlice(projectKey, artifactIndex), r: r };
        }
        return null;
    }
    onCursorTargetChanged: syncCursorTarget()
    function syncCursorTarget() {
        const t = cursorTarget;
        if (!t || !targets || (navMode === "mouse" && !padConnected))
            return;
        targets.setHoverTarget(t.kind, t.session, t.artifact, t.path, t.label, t.context,
                               t.r.x, t.r.y, t.r.width, t.r.height);
    }

    FocusCursor {
        id: focusCursor
        anchors.fill: parent
        target: app.cursorRect
        shown: app.cursorSpec !== null
        Connections {
            target: stage
            function onCardOpenChanged() {
                if (stage.cardOpen && app.lens === 2)
                    focusCursor.snap();
            }
        }
        cornerRadius: app.cursorSpec ? app.cursorSpec.radius : 26
        round: app.cursorSpec ? app.cursorSpec.round : false
        hint: app.cursorSpec ? app.cursorSpec.hint : ""
        hintSide: app.cursorSpec ? app.cursorSpec.side : "bottom"
    }

    // Keymap pill, bottom right: "nav vim" at rest; hover shows all three for a one-click switch.
    // Clicking the current mode (or Cmd+Shift+K) cycles the modes; clicking another picks it.
    Item {
        id: navPill
        objectName: "navPill"
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.rightMargin: 18
        anchors.bottomMargin: 8
        width: navRow.implicitWidth + 24
        height: navRow.implicitHeight + 8
        // Constellation only: zoomed in, the composer (and its predictor ring) owns the bottom edge,
        // and the mode is one Cmd+, away.
        visible: !dialog.open && app.lens === 0 && app.zoom < 0.5
        opacity: navHover.hovered ? 1 : 0.8
        HoverHandler { id: navHover }
        Rectangle {
            anchors.fill: parent
            radius: height / 2
            color: navHover.hovered ? Theme.card : "transparent"
            border.width: navHover.hovered ? 1 : 0
            border.color: Theme.line
        }
        Row {
            id: navRow
            anchors.centerIn: parent
            spacing: 10
            Text {
                text: "nav"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 13
            }
            Repeater {
                model: ["controller", "vim", "wasd", "mouse"]
                Text {
                    required property string modelData
                    objectName: "navOption"
                    readonly property bool current: modelData === app.navMode
                    visible: current || navHover.hovered
                    text: modelData
                    color: current ? (navHover.hovered ? Theme.gold : Theme.dim) : Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 13
                    TapHandler {
                        onTapped: {
                            if (!app.backend)
                                return;
                            if (parent.current)
                                app.backend.cycleNavMode();
                            else
                                app.backend.navMode = parent.modelData;
                        }
                    }
                }
            }
        }
    }

    // Gamepad pill, bottom left: dim, only while a controller is connected. Constellation only,
    // like the keymap pill: zoomed in, the composer owns the bottom edge.
    Rectangle {
        id: padPill
        objectName: "padPill"
        anchors.left: parent.left
        anchors.bottom: parent.bottom
        anchors.leftMargin: 18
        anchors.bottomMargin: 8
        width: padRow.implicitWidth + 24
        height: padRow.implicitHeight + 8
        radius: height / 2
        color: "transparent"
        border.width: 1
        border.color: Theme.line
        visible: app.padConnected && !dialog.open && !onboarding.open && app.lens === 0 && app.zoom < 0.5
        opacity: 0.8
        Row {
            id: padRow
            anchors.centerIn: parent
            spacing: 8
            Rectangle {
                width: 6; height: 6; radius: 3
                anchors.verticalCenter: parent.verticalCenter
                color: Theme.gold
            }
            Text {
                objectName: "padPillText"
                text: app.gamepad && app.gamepad.controllerName.length > 0 ? app.gamepad.controllerName : "Gamepad"
                color: Theme.faint
                font.family: Theme.mono
                font.pixelSize: 13
            }
        }
    }

    // Middle-click anywhere, or Shift+Click on a hovered target, opens the floating ask card.
    // Every other press is declined, so it falls through to whatever is underneath.
    MouseArea {
        objectName: "omnibarTrigger"
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.MiddleButton
        enabled: !omnibar.suppressed
        onPressed: (m) => {
            const middle = m.button === Qt.MiddleButton;
            const shiftTarget = m.button === Qt.LeftButton && (m.modifiers & Qt.ShiftModifier) !== 0
                                && app.targets && app.targets.hasActiveTarget;
            if (!middle && !shiftTarget) {
                m.accepted = false;
                return;
            }
            omnibar.openAt(m.x, m.y);
        }
    }

    FloatingOmnibar {
        id: omnibar
        anchors.fill: parent
        backend: app.backend
        projectKey: app.lens > 0 ? app.projectKey
                                 : (app.projects.length > 0 ? app.projects[Math.min(app.selectedIsland, app.projects.length - 1)].key : "")
        artifactIndex: app.lens === 2 ? app.artifactIndex : -1
        mouseMode: app.navMode === "mouse"
        padHints: app.padHints
        padGlyphs: app.padGlyphs
        voiceListening: app.gamepad ? app.gamepad.voiceActive : false
        suppressed: settings.open || dialog.open || onboarding.open || briefing.open || controllerMap.open || review.open || music.open
        onClosed: app.takeFocus()
    }

    NewSessionDialog {
        id: dialog
        anchors.fill: parent
        backend: app.backend
        padHints: app.padHints
        padGlyphs: app.padGlyphs
        onClosed: app.takeFocus()
    }

    SettingsOverlay {
        id: settings
        anchors.fill: parent
        backend: app.backend
        onClosed: app.takeFocus()
        onOpenControllerMap: {
            settings.hide();
            controllerMap.show();
        }
    }

    ControllerMapOverlay {
        id: controllerMap
        anchors.fill: parent
        backend: app.backend
        onClosed: settings.show()
    }

    // --- Morning Briefing ------------------------------------------------------------------------
    // B / R2 on the Constellation, or by itself when the window comes back after an overnight or
    // long-idle stretch (live window only: offscreen tests and benches never trigger it).
    property real lastInputMs: Date.now()
    readonly property real briefingIdleMs: 4 * 3600 * 1000
    function maybeAutoBriefing() {
        if (!backend || !backend.live || briefing.open || settings.open || dialog.open || onboarding.open || controllerMap.open)
            return;
        const away = Date.now() - lastInputMs;
        const overnight = new Date(lastInputMs).toDateString() !== new Date().toDateString() && away > 90 * 60 * 1000;
        if (away > briefingIdleMs || overnight) {
            takeFocus();
            goLens(0);
            briefing.show();
        }
    }
    Connections {
        target: Qt.application
        function onStateChanged() {
            if (Qt.application.state === Qt.ApplicationActive)
                app.maybeAutoBriefing();
            else
                app.lastInputMs = Math.max(app.lastInputMs, Date.now());  // away from here on
        }
    }
    MorningBriefingOverlay {
        id: briefing
        anchors.fill: parent
        backend: app.backend
        padHints: app.padHints
        padGlyphs: app.padGlyphs
        onClosed: {
            app.lastInputMs = Date.now();
            app.takeFocus();
        }
        onOpenProject: (i) => app.enterProject(i)
    }
    // --- Session review --------------------------------------------------------------------------
    // Cmd+Shift+R or the Review pill: intent vs outcome, the human's steers, taste rules to keep.
    SessionReviewOverlay {
        id: review
        anchors.fill: parent
        backend: app.backend
        padHints: app.padHints
        padGlyphs: app.padGlyphs
        onClosed: app.takeFocus()
    }
    // --- Music -----------------------------------------------------------------------------------
    // Cmd+M or the ♪ pill: the curated slowmix rotation and its five controls.
    // Ambient voice reply: what was heard, faint, and the short answer or action confirmation.
    // Anchored under the top bar's right side, quiet, gone after 9 s. [[IGNORE]] never shows.
    Rectangle {
        id: ambientCard
        objectName: "ambientAnswerCard"
        readonly property string answer: app.backend ? app.backend.ambientAnswer : ""
        readonly property string heard: app.backend ? app.backend.ambientHeard : ""
        anchors.top: parent.top
        anchors.topMargin: 70
        anchors.right: parent.right
        anchors.rightMargin: 24
        width: Math.min(460, parent.width * 0.4)
        height: ambientCol.implicitHeight + 24
        radius: 14
        color: "#0b0a08"
        border.width: 1
        border.color: Theme.line
        opacity: answer.length > 0 && !music.open && !settings.open ? 1 : 0
        visible: opacity > 0.01
        Behavior on opacity { NumberAnimation { duration: 180; easing.type: Easing.OutCubic } }
        Column {
            id: ambientCol
            x: 16
            y: 12
            width: parent.width - 32
            spacing: 6
            Text {
                objectName: "ambientHeard"
                width: parent.width
                text: ambientCard.heard.length > 0 ? "“" + ambientCard.heard + "”" : ""
                visible: text.length > 0
                color: Theme.faint
                elide: Text.ElideRight
                font.family: Theme.sans
                font.pixelSize: 14
            }
            Text {
                objectName: "ambientAnswerText"
                width: parent.width
                text: ambientCard.answer
                color: Theme.text
                wrapMode: Text.Wrap
                maximumLineCount: 4
                elide: Text.ElideRight
                font.family: Theme.sans
                font.pixelSize: 16
            }
        }
        TapHandler { onTapped: if (app.backend) app.backend.clearAmbientAnswer() }
        onAnswerChanged: if (answer.length > 0) ambientTimer.restart()
        Timer {
            id: ambientTimer
            interval: 9000
            onTriggered: if (app.backend) app.backend.clearAmbientAnswer()
        }
    }

    SlowmixOverlay {
        id: music
        anchors.fill: parent
        backend: app.backend
        padHints: app.padHints
        onClosed: app.takeFocus()
    }
    // Quiet hint, Constellation only: where the briefing lives.
    Text {
        objectName: "briefingHint"
        anchors.right: navPill.left
        anchors.rightMargin: 14
        anchors.verticalCenter: navPill.verticalCenter
        visible: navPill.visible && !briefing.open && app.projects.length > 0
        text: (app.padHints ? (app.padKey("send").length > 0 ? app.padKey("send") : "B") : "B") + " briefing"
        color: Theme.faint
        font.family: Theme.mono
        font.pixelSize: 13
    }

    // Phonon voice-typing live badge while L2 (or remapped voice_hold) is held.
    Rectangle {
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.top: parent.top
        anchors.topMargin: 16
        width: voiceRow.implicitWidth + 24
        height: 30
        radius: 15
        color: "#1b1408"
        border.width: 1
        border.color: Theme.gold
        visible: app.gamepad && app.gamepad.voiceActive
        Row {
            id: voiceRow
            anchors.centerIn: parent
            spacing: 8
            Rectangle { width: 8; height: 8; radius: 4; color: Theme.gold; anchors.verticalCenter: parent.verticalCenter }
            Text {
                text: "Phonon listening…"
                color: Theme.gold
                font.family: Theme.mono
                font.pixelSize: 12
            }
        }
    }

    // First launch only: shown while backend.firstLaunch is true (never in offscreen modes,
    // which have no prefs path, unless a test calls showOnboarding()).
    OnboardingModal {
        id: onboarding
        anchors.fill: parent
        backend: app.backend
        onDone: app.takeFocus()
    }
}
