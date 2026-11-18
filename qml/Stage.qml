import QtQuick

// Lens 3: one full-bleed artifact, a <=5-word title and one metric.
// Two panes take turns: stepping Left/Right slides the new artifact in from that side while the
// old one slides out the other way (Easing.OutQuint, Theme.slideMs), so modalities feel physical.
Item {
    id: root
    objectName: "stage"
    property var artifact
    property int artifactIndex: 0
    property int artifactCount: 0
    property bool live: false
    property QtObject backend
    property int direction: 1     // +1: the next artifact enters from the right
    property bool animate: false  // only slide while the Stage is on screen and settled
    property var params: []       // hyperparameter / term pills for this project
    property bool padHints: false // controller glyphs in the pill card hints (StudioApp decides)
    property var padGlyphs: ({})
    signal pick(int index)
    signal askAbout(var param)

    // The pill row sits right under the header; the artifact panes start below it.
    readonly property alias terms: termBar
    readonly property real barH: termBar.visible ? termBar.height + 16 : 0
    // An open param card floats over the artifact, which dims behind it (nothing moves).
    readonly property bool cardOpen: termBar.expanded >= 0
    property real dim: cardOpen ? 0.22 : 1
    Behavior on dim { NumberAnimation { duration: Theme.hopMs; easing.type: Easing.OutCubic } }

    property int activePane: 0
    readonly property Item pane: activePane === 0 ? paneA : paneB
    readonly property string kind: pane.kind
    property real slideP: 1       // 0 -> 1 across one slide (OutQuint: position and scale)
    property real fadeP: 1        // 0 -> 1 linear: the opacity handoff, so both panes are seen moving
    property int slideDir: 1
    // A full-width push: the panes travel side by side with a gap, so they never overlap.
    readonly property real slideDist: width + 96

    function scrollBy(d) {
        pane.scrollBy(d);
    }
    function scrollPixels(px) {  // gamepad right stick: smooth, unanimated
        pane.scrollPixels(px);
    }
    function scrollToEnd(d) {
        for (let i = 0; i < 40; ++i)
            pane.scrollBy(d);
    }

    // Tab predictor. Results lead to their cause: a chart predicts the diff that made it, a diff
    // predicts the numbers, numbers predict the write-up (PDF) or the visual (video, image).
    // Once every artifact has been seen, it suggests the question this artifact invites instead.
    property var seen: []  // artifact indices already shown for this project (StudioApp keeps it)
    readonly property var kindLabels: ({ chart: "Chart", line: "Chart", diff: "Diff", table: "Table", pdf: "PDF",
                                         video: "Video", image: "Image", svg: "Image", audio: "Audio" })
    readonly property var nextKind: ({ chart: ["diff", "table"], line: ["diff", "table"], diff: ["table", "chart", "line"],
                                       table: ["pdf", "video", "image", "chart"], pdf: ["video", "image", "chart"],
                                       video: ["chart", "diff"], image: ["chart", "diff"], svg: ["chart", "diff"],
                                       audio: ["chart", "diff"] })
    function promptFor(a) {
        if (!a)
            return "What is this session doing?";
        switch (a.kind) {
        case "chart":
        case "line": return a.winnerLabel ? "Why did " + a.winnerLabel + " win?" : "What stands out in this chart?";
        case "diff": return "What does this diff change?";
        case "table": return "Which row is best, and why?";
        case "pdf": return "Summarize this page in one line.";
        case "video":
        case "image":
        case "svg": return "What does this visual show?";
        case "audio": return "What is in this recording?";
        }
        return "What should we try next?";
    }
    readonly property var predictedAction: {
        const n = artifacts.length;
        if (n === 0)
            return null;
        const cur = artifacts[Math.min(artifactIndex, n - 1)];
        const unseen = (i) => seen.indexOf(i) < 0 && i !== artifactIndex;
        let pick = -1;
        const wants = cur && nextKind[cur.kind] ? nextKind[cur.kind] : [];
        for (let w = 0; w < wants.length && pick < 0; ++w)
            for (let d = 1; d < n && pick < 0; ++d) {
                const i = (artifactIndex + d) % n;
                if (unseen(i) && artifacts[i].kind === wants[w]) pick = i;
            }
        for (let d = 1; d < n && pick < 0; ++d) {
            const i = (artifactIndex + d) % n;
            if (unseen(i)) pick = i;
        }
        if (pick >= 0)
            return { targetId: "pager:" + pick, label: "View " + (kindLabels[artifacts[pick].kind] || "Next"),
                     promptSuggestion: "", actionKind: "artifact", index: pick };
        return { targetId: "composer", label: "", promptSuggestion: promptFor(cur), actionKind: "prompt", index: -1 };
    }
    function targetRect(id) {
        // artifactCount, not pager.visible: visibility is inherited, and the Stage is invisible
        // for the first frames of a lens crossfade, when this is first asked.
        if (!id.startsWith("pager:") || artifactCount <= 1)
            return Qt.rect(0, 0, 0, 0);
        const i = parseInt(id.slice(6));
        const x = pager.x + i * (7 + pager.spacing) + (i > artifactIndex ? 13 : 0);
        return Qt.rect(x, pager.y, i === artifactIndex ? 20 : 7, 7);
    }

    // What the focus cursor wraps, in this item's coordinates: the open term pill while its card is
    // down (clicked or Cmd+P), the selected pill while the keyboard is in the row, else the
    // artifact viewer of the settled pane. Never the artifact while a card is open: the big
    // artifact ring would cross the card's text.
    function cursorRect() {
        if (termBar.expanded >= 0)
            return termBar.pillRect(termBar.expanded);
        if (termBar.active)
            return termBar.pillRect(termBar.selected);
        return pane.viewerRect();
    }

    function sameArtifact(a, b) {
        return a && b && a.kind === b.kind && a.title === b.title;
    }

    onArtifactChanged: {
        const cur = pane.artifact;
        if (sameArtifact(cur, artifact)) {
            pane.artifact = artifact;  // same artifact, refreshed data
            return;
        }
        if (slide.running) {
            slide.complete();
            (activePane === 0 ? paneB : paneA).artifact = null;
        }
        if (!animate || !cur || !artifact) {
            pane.artifact = artifact;
            return;
        }
        const next = activePane === 0 ? paneB : paneA;
        next.artifact = artifact;
        slideDir = direction >= 0 ? 1 : -1;
        slideP = 0;
        fadeP = 0;
        activePane = 1 - activePane;
        slide.restart();
    }

    ParallelAnimation {
        id: slide
        NumberAnimation { target: root; property: "slideP"; from: 0; to: 1; duration: Theme.slideMs; easing.type: Easing.OutQuint }
        NumberAnimation { target: root; property: "fadeP"; from: 0; to: 1; duration: Theme.slideMs }
        onFinished: {
            const old = root.activePane === 0 ? paneB : paneA;
            old.artifact = null;  // drop the outgoing viewer once it is off stage
        }
    }

    component StagePane: Item {
        id: p
        property Item st  // the Stage (inline components cannot see the file's ids)
        property var artifact
        property bool current: false
        readonly property string kind: artifact ? artifact.kind : ""
        readonly property real titleHeight: title.height
        function viewerRect() {  // where it lands (x = 0), not where the slide has it right now
            return Qt.rect(viewer.x, p.y + viewer.y, viewer.width, viewer.height);
        }
        function scrollBy(d) {
            if (viewer.item && viewer.item.scrollBy)
                viewer.item.scrollBy(d);
        }
        function scrollPixels(px) {
            if (viewer.item && viewer.item.scrollPixels)
                viewer.item.scrollPixels(px);
        }

        readonly property real sp: st ? st.slideP : 1
        readonly property real fp: st ? st.fadeP : 1
        readonly property real sd: st ? st.slideDir * st.slideDist : 0
        readonly property QtObject backend: st ? st.backend : null
        readonly property bool live: st ? st.live : false
        width: st ? st.width : 0
        y: st ? st.barH : 0
        height: st ? st.height - st.barH : 0
        // Incoming: from +dir*dist to 0, scale 0.97 -> 1. Outgoing: 0 to -dir*dist, scale 1 -> 0.97,
        // dimming on a linear clock so it stays visible for the first frames of the push.
        x: current ? sd * (1 - sp) : -sd * sp
        opacity: (current ? Math.min(1, 0.55 + fp) : Math.max(0, 1 - fp * 1.25)) * (st ? st.dim : 1)
        scale: current ? 0.97 + 0.03 * sp : 1 - 0.03 * sp
        visible: opacity > 0.005 && (current || artifact !== null)

        Text {
            id: title
            objectName: p.current ? "stageTitle" : ""
            anchors.left: parent.left
            anchors.right: metric.left
            anchors.rightMargin: 40
            y: 0
            text: p.artifact ? p.artifact.title : "Nothing to show yet"
            color: Theme.ink
            elide: Text.ElideRight
            font.family: Theme.sans
            font.pixelSize: Theme.px(42)
            font.weight: Font.DemiBold
        }
        Text {
            id: metric
            objectName: p.current ? "stageMetric" : ""
            anchors.right: parent.right
            anchors.baseline: title.baseline
            readonly property bool monoKind: p.kind === "diff" || p.kind === "table" || p.kind === "pdf"
            text: !p.artifact ? "" : (p.kind === "pdf" ? pdfPageLabel() : p.artifact.metric)
            color: Theme.gold
            font.family: monoKind ? Theme.mono : Theme.sans
            font.pixelSize: Theme.px(monoKind ? 32 : 46)
            font.weight: Font.DemiBold
            function pdfPageLabel() {
                return viewer.item && viewer.item.pageLabel !== undefined ? viewer.item.pageLabel : "";
            }
        }

        Loader {
            id: viewer
            anchors.top: title.bottom
            anchors.topMargin: 47  // room for the pager dots, which stay put while panes slide
            anchors.horizontalCenter: parent.horizontalCenter
            width: Math.min(parent.width, Math.max(760, parent.width * 0.72))
            anchors.bottom: parent.bottom
            sourceComponent: {
                switch (p.kind) {
                case "chart": return chartC;
                case "line": return chartC;
                case "diff": return diffC;
                case "table": return tableC;
                case "pdf": return pdfC;
                case "image": return imageC;
                case "svg": return imageC;
                case "video": return mediaC;
                case "audio": return mediaC;
                default: return p.current ? emptyC : null;
                }
            }
        }

        Component { id: chartC; ChartView { artifact: p.artifact } }
        Component { id: diffC; DiffView { artifact: p.artifact } }
        Component { id: tableC; CsvTable { artifact: p.artifact } }
        Component { id: pdfC; PdfView { artifact: p.artifact; backend: p.backend; live: p.live } }
        Component { id: imageC; ImageView { artifact: p.artifact; live: p.live } }
        Component { id: mediaC; MediaView { artifact: p.artifact; backend: p.backend; live: p.live } }
        Component {
            id: emptyC
            Item {
                Rectangle {
                    anchors.centerIn: parent
                    width: 12; height: 12; radius: 6
                    color: Theme.faint
                }
            }
        }
    }

    // Decode the project's PDF pages off the GUI thread ahead of time, at the exact URL, width and
    // fill mode PdfView asks for, so the pixmap cache already holds the page when its slide
    // starts (no blank card while PDFKit renders, no 30 ms GUI-thread stall).
    property var artifacts: []
    Repeater {
        model: root.artifacts.length
        Image {
            required property int index
            readonly property var art: root.artifacts[index]
            visible: false
            asynchronous: true
            cache: true
            fillMode: Image.PreserveAspectFit
            sourceSize.width: Math.round(Math.min(root.width, 1180))
            source: art && art.kind === "pdf" && art.path && root.width > 0
                    ? "image://pdf/0/" + encodeURIComponent(art.path) : ""
        }
    }

    StagePane {
        id: paneA
        objectName: "stagePane"
        st: root
        current: root.activePane === 0
    }
    StagePane {
        id: paneB
        objectName: "stagePane"
        st: root
        current: root.activePane === 1
    }
    Component.onCompleted: paneA.artifact = artifact

    // Pager: one dot per artifact. It stays fixed; the active dot glides with the slide.
    Row {
        id: pager
        x: 0
        y: root.barH + root.pane.titleHeight + 10
        spacing: 9
        visible: root.artifactCount > 1
        opacity: root.dim
        Repeater {
            model: Math.min(root.artifactCount, 12)
            Rectangle {
                required property int index
                width: index === root.artifactIndex ? 20 : 7
                height: 7
                radius: 3.5
                color: index === root.artifactIndex ? Theme.gold : Theme.faint
                Behavior on width { NumberAnimation { duration: Theme.slideMs; easing.type: Easing.OutQuint } }
                MouseArea {
                    anchors.fill: parent
                    anchors.margins: -8
                    onClicked: root.pick(parent.index)
                }
            }
        }
    }

    // Declared last: its dropdown card floats over the artifact and the pager.
    TermZoomBar {
        id: termBar
        x: 0
        y: 0
        width: root.width
        params: root.params
        padHints: root.padHints
        padGlyphs: root.padGlyphs
        // The card fades in as the artifact dims: squared, so the two are never both above half
        // opacity on the same frame (dim 0.5 -> card 0.41).
        cardReveal: Math.pow((1 - root.dim) / 0.78, 2)
        onAskAbout: (p) => root.askAbout(p)
    }
}
