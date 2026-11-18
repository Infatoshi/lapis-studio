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

    // UI zoom (0.75..1.60) and input mode ("vim" | "wasd" | "mouse"), persisted by the backend.
    // There is no arrows mode; arrow keys are a silent fallback in every mode.
    readonly property real uiScale: backend ? backend.uiScale : 1.0
    readonly property string navMode: backend ? backend.navMode : "controller"

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
        idle = false;
        idleTimer.restart();
    }
    function markSeen() {
        if (lens === 2 && seenArtifacts.indexOf(artifactIndex) < 0)
            seenArtifacts = seenArtifacts.concat([artifactIndex]);
    }
    onArtifactIndexChanged: markSeen()
    onLensChanged: {
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
        if (settings.open || onboarding.open || dialog.open)
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
                      constellation.width, selectedIsland, stage.slideP, stage.opacity, ladder.opacity];
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
    readonly property var artifact: artifacts.length > 0 ? artifacts[Math.min(artifactIndex, artifacts.length - 1)] : null

    // Shared-element zoom. The island of the current project (ix, iy, iw, ih) grows into the
    // full-bleed Ladder/Stage while its siblings recede, and Esc shrinks it back into that island.
    readonly property int projectIdx: {
        for (let i = 0; i < projects.length; ++i)
            if (projects[i].key === projectKey)
                return i;
        return -1;
    }
    readonly property rect islandRect: projectIdx >= 0 && projectIdx < constellation.count
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

    Connections {
        target: app.backend
        function onSessionStarted(key) {
            app.projectKey = key;
            app.artifactIndex = 0;
            app.backend.clearAnswers();
            const i = app.backend.projectIndex(key);
            app.selectedIsland = Math.max(0, i);
            app.lens = 1;
        }
    }

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
            e.accepted = true;
            return;
        }
        if (ctrl && e.key === Qt.Key_K) {
            if (backend)
                backend.cycleNavMode();
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
        if (settings.open)
            return;
        if (ctrl && e.key === Qt.Key_M) {
            if (lens > 0) {
                answers.openPicker();
            } else {
                settings.show();
                settings.row = 2;  // Default Model
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
        if (e.key === Qt.Key_Tab && !ctrl) {
            acceptPrediction();
            e.accepted = true;
            return;
        }
        if (ctrl && (e.key === Qt.Key_N || e.key === Qt.Key_T)) {
            dialog.show();
            e.accepted = true;
            return;
        }
        if (ctrl && e.key >= Qt.Key_1 && e.key <= Qt.Key_3) {
            app.takeFocus();
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
            if (proseOpen)
                proseOpen = false;
            else if (lens > 0)
                goLens(0);
            e.accepted = true;
            return;
        }
        if (ctrl)
            return;
        if (navMode === "vim" && e.key === Qt.Key_G && !(e.modifiers & (Qt.AltModifier | Qt.MetaModifier))) {
            const last = (e.modifiers & Qt.ShiftModifier) !== 0;
            if (lens === 0 && projects.length > 0) {
                selectedIsland = last ? Math.min(projects.length, constellation.count) - 1 : 0;
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
            const n = projects.length;
            if (k >= Qt.Key_1 && k <= Qt.Key_9) {
                enterProject(k - Qt.Key_1);
                e.accepted = true;
            } else if ((k === Qt.Key_Return || k === Qt.Key_Enter) && n > 0) {
                enterProject(selectedIsland);
                e.accepted = true;
            } else if (k === Qt.Key_Right) {
                selectedIsland = (selectedIsland + 1) % Math.max(1, n);
                e.accepted = true;
            } else if (k === Qt.Key_Left || k === Qt.Key_Backtab) {
                selectedIsland = (selectedIsland - 1 + n) % Math.max(1, n);
                e.accepted = true;
            } else if (k === Qt.Key_Down) {
                selectedIsland = Math.min(n - 1, selectedIsland + constellation.columns);
                e.accepted = true;
            } else if (k === Qt.Key_Up) {
                selectedIsland = Math.max(0, selectedIsland - constellation.columns);
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
            if (k === Qt.Key_Down) { if (proseOpen) prose.scrollBy(1); else if (lens === 2) stage.scrollBy(1); }
            else if (k === Qt.Key_Up) { if (proseOpen) prose.scrollBy(-1); else if (lens === 2) stage.scrollBy(-1); }
            e.accepted = true;
        } else if (k === Qt.Key_Down || k === Qt.Key_PageDown) {
            if (proseOpen) prose.scrollBy(1); else if (lens === 2) stage.scrollBy(1);
            e.accepted = true;
        } else if (k === Qt.Key_Up || k === Qt.Key_PageUp) {
            if (proseOpen) prose.scrollBy(-1); else if (lens === 2) stage.scrollBy(-1);
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
    // then the lens. ✕ confirm · ○ back · △ Tab · □ prose / picker · L1 R1 slide · L2 R2 zoom.
    readonly property QtObject gamepad: backend ? backend.gamepad : null
    readonly property bool padConnected: gamepad ? gamepad.connected : false
    function padNav(dx, dy) {
        poke();
        if (onboarding.open) {
            onboarding.choice = Math.max(0, Math.min(onboarding.cards.length - 1, onboarding.choice + dx + dy));
        } else if (dialog.open) {
            if (dx !== 0) dialog.cycleHarness(dx); else dialog.cycleFolder(dy);
        } else if (settings.open) {
            if (dy !== 0) settings.row = (settings.row + dy + settings.rowCount) % settings.rowCount;
            else settings.change(dx);
        } else if (answers.pickerOpen && backend) {
            if (dy !== 0) backend.cycleModel(dy); else backend.cycleEffort(dx);
        } else if (answers.inputFocused) {
            if (dy < 0) answers.openPicker();          // Up from the composer: the model strip
            else if (dy > 0) takeFocus();               // Down: back to the lens, text kept
        } else if (lens === 2 && stage.terms.active) {
            stage.terms.step(dx + dy);
        } else if (lens === 0) {
            const n = projects.length;
            if (n === 0) return;
            if (dx !== 0) selectedIsland = (selectedIsland + dx + n) % n;
            else if (dy > 0) selectedIsland = Math.min(n - 1, selectedIsland + constellation.columns);
            else selectedIsland = Math.max(0, selectedIsland - constellation.columns);
        } else if (dx !== 0) {
            if (lens === 2) stepArtifact(dx);
        } else if (proseOpen) {
            prose.scrollBy(dy);
        } else if (lens === 2) {
            stage.scrollBy(dy);
        }
    }
    function padConfirm() {
        poke();
        if (onboarding.open) onboarding.confirm();
        else if (dialog.open) dialog.start();
        else if (settings.open) settings.hide();
        else if (answers.pickerOpen) answers.pickerOpen = false;  // the composer keeps focus
        else if (answers.inputFocused) answers.send(0);
        else if (lens === 2 && stage.terms.active) scopeToParam(stage.terms.current());
        else if (lens === 0) { if (projects.length > 0) enterProject(selectedIsland); }
        else if (lens === 1 && artifacts.length > 0) goLens(2);
        else answers.focusInput("");
    }
    function padBack() {
        poke();
        if (onboarding.open) return;  // a mode has to be picked once
        if (dialog.open) dialog.hide();
        else if (settings.open) settings.hide();
        else if (answers.pickerOpen) answers.pickerOpen = false;
        else if (answers.inputFocused) answers.escapeInput();
        else if (lens === 2 && stage.terms.active) stage.terms.close();
        else if (proseOpen) proseOpen = false;
        else if (lens > 0) { takeFocus(); goLens(0); }
    }
    function padTriangle() {
        poke();
        if (onboarding.open || dialog.open || settings.open)
            return;
        acceptPrediction();
    }
    function padSecondary() {
        poke();
        if (onboarding.open || dialog.open)
            return;
        if (settings.open) {
            settings.row = 2;  // Default Model
        } else if (answers.inputFocused) {
            answers.pickerOpen ? (answers.pickerOpen = false) : answers.openPicker();
        } else if (lens === 0) {
            settings.show();
            settings.row = 2;
        } else {
            proseOpen = !proseOpen;
        }
    }
    function padStep(d) {
        poke();
        if (onboarding.open) {
            onboarding.choice = Math.max(0, Math.min(onboarding.cards.length - 1, onboarding.choice + d));
        } else if (dialog.open) {
            dialog.cycleModel(d);
        } else if (settings.open) {
            settings.row = (settings.row + d + settings.rowCount) % settings.rowCount;
        } else if (answers.pickerOpen && backend) {
            backend.cycleModel(d);
        } else if (lens === 0) {
            const n = projects.length;
            if (n > 0) selectedIsland = (selectedIsland + d + n) % n;
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
        if (proseOpen && lens > 0) prose.scrollPixels(dy);
        else if (lens === 2) stage.scrollPixels(dy);
    }
    Connections {
        target: app.gamepad
        function onNav(dx, dy) { app.padNav(dx, dy); }
        function onConfirm() { app.padConfirm(); }
        function onBack() { app.padBack(); }
        function onAcceptPrediction() { app.padTriangle(); }
        function onSecondary() { app.padSecondary(); }
        function onStep(d) { app.padStep(d); }
        function onZoomStep(d) { app.padZoom(d); }
        function onToggleSettings() { app.padSettings(); }
        function onToggleNewSession() { app.padNewSession(); }
        function onScrollDelta(dy) { app.padScroll(dy); }
    }

    Rectangle {
        anchors.fill: parent
        color: Theme.bg
    }

    // Lens 1: constellation of project islands.
    Constellation {
        id: constellation
        anchors.fill: parent
        projects: app.projects
        selectedIndex: app.selectedIsland
        recede: app.zoom
        focusIndex: app.projectIdx
        visible: app.zoom < 0.999
        visited: app.visited
        onEnter: (i) => app.enterProject(i)
        onZoomBy: (f) => app.zoomBy(f)
        onNewSession: dialog.show()
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
            number: app.projectIdx + 1
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
                    anchors.right: lensDots.left
                    anchors.rightMargin: 24
                    anchors.verticalCenter: parent.verticalCenter
                    visible: text.length > 0
                    text: app.project && app.project.jobName ? "·  " + app.project.jobName : ""
                    color: Theme.text
                    elide: Text.ElideRight
                    font.family: Theme.sans
                    font.pixelSize: Theme.px(20)
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
                anchors.bottom: answers.top
                anchors.bottomMargin: 16
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
                anchors.bottom: answers.top
                anchors.bottomMargin: 18
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

            Composer {
                id: answers
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.bottom: parent.bottom
                anchors.leftMargin: 56
                anchors.rightMargin: 56
                anchors.bottomMargin: 32
                backend: app.backend
                hasSession: app.project !== null && app.project.sessionCount > 0
                navMode: app.navMode
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

    // Tab predictor ring + "⇥ Tab" badge, over whichever lens is settled on screen.
    PredictorRing {
        id: predictor
        anchors.fill: parent
        readonly property var p: app.predictedAction
        readonly property string tid: p ? p.targetId : ""
        target: app.predictedRect
        label: p ? p.label : ""
        round: tid.startsWith("pager:") || tid.startsWith("rung:") || tid === "newSession"
        cardRadius: tid === "composer" ? 22 : 26
        badgeSide: tid.startsWith("pager:") ? "right" : tid.startsWith("rung:") ? "left"
                   : tid === "newSession" ? "bottom" : "top"
        idle: app.idle
        shown: p !== null && (app.zoom < 0.001 || app.zoom > 0.999) && !(app.lens === 2 && app.stageSliding)
               && !(app.lens === 2 && (stage.cardOpen || stage.terms.active))
               && !app.proseOpen && (app.lens !== 1 || ladder.opacity > 0.99) && (app.lens !== 2 || stage.opacity > 0.99)
    }
    readonly property bool stageSliding: stage.slideP < 1

    // Keymap pill, bottom right: "nav vim" at rest; hover shows all three for a one-click switch.
    // Clicking the current mode (or Cmd+K) cycles arrows -> vim -> wasd; clicking another picks it.
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

    NewSessionDialog {
        id: dialog
        anchors.fill: parent
        backend: app.backend
        onClosed: app.takeFocus()
    }

    SettingsOverlay {
        id: settings
        anchors.fill: parent
        backend: app.backend
        onClosed: app.takeFocus()
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
