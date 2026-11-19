import QtQuick
import QtQuick.Shapes

// Full-width bar chart (winner in gold) or a line chart for numeric sweeps.
Item {
    id: root
    objectName: "chartView"
    property var artifact
    readonly property bool line: artifact && artifact.kind === "line"
    readonly property var values: artifact ? artifact.values : []
    readonly property var labels: artifact ? artifact.labels : []
    readonly property var valueText: artifact && artifact.valueText ? artifact.valueText : []
    readonly property int best: artifact ? artifact.best : -1
    readonly property real maxV: {
        let m = 0;
        for (let i = 0; i < values.length; ++i) m = Math.max(m, values[i]);
        return m > 0 ? m : 1;
    }
    readonly property real minV: {
        let m = Infinity;
        for (let i = 0; i < values.length; ++i) m = Math.min(m, values[i]);
        return isFinite(m) ? m : 0;
    }

    // Hover-to-ask: one HoverHandler; the bar under the pointer, or the line point nearest in x.
    property QtObject targets: null
    readonly property string series: artifact && artifact.title ? artifact.title : "series"
    function valueAt(i) {
        return valueText[i] !== undefined && String(valueText[i]).length > 0 ? String(valueText[i]) : String(values[i]);
    }
    function reportHover(pos) {
        const n = values.length;
        if (!targets || n === 0)
            return;
        let i = -1;
        let r;
        if (line) {
            const lp = linePlot.mapFromItem(root, pos.x, pos.y);
            i = Math.max(0, Math.min(n - 1, Math.round(lp.x / Math.max(1, linePlot.width) * Math.max(1, n - 1))));
            const pt = linePlot.pt(i);
            r = linePlot.mapToItem(null, pt.x - 14, pt.y - 14, 28, 28);
        } else {
            const lp = plot.mapFromItem(root, pos.x, pos.y);
            i = Math.floor(lp.x / plot.slot);
            if (i < 0 || i >= n) {
                clearHover();
                return;
            }
            r = plot.mapToItem(null, plot.slot * i + (plot.slot - plot.barW) / 2, 0, plot.barW, plot.height);
        }
        let sum = 0;
        for (let j = 0; j < n; ++j) sum += values[j];
        const name = labels[i] !== undefined ? String(labels[i]) : "#" + (i + 1);
        const all = [];
        for (let j = 0; j < Math.min(n, 24); ++j)
            all.push((labels[j] !== undefined ? labels[j] : "#" + (j + 1)) + "=" + valueAt(j));
        const ctx = ["Chart: " + series + " (" + (line ? "line" : "bar") + ", " + n + " points)",
                     "min " + minV + " · max " + maxV + " · mean " + (sum / n).toFixed(2)
                         + (best >= 0 && labels[best] !== undefined ? " · best " + labels[best] : ""),
                     "Point " + (i + 1) + ": " + name + " = " + valueAt(i) + (i === best ? " (best)" : ""),
                     "All: " + all.join(", ") + (n > 24 ? ", ..." : "")];
        targets.setHoverTarget("chart_point", "", "", "series=" + series + "/idx=" + i, name + ": " + valueAt(i),
                               ctx.join("\n"), r.x, r.y, r.width, r.height);
    }
    function clearHover() {
        if (targets && targets.activeTarget.kind === "chart_point")
            targets.clearHoverTarget("chart_point");
    }
    HoverHandler {
        onPointChanged: if (hovered) root.reportHover(point.position)
        onHoveredChanged: if (!hovered) root.clearHover()
    }

    property real grow: 0
    Component.onCompleted: grow = 1
    Behavior on grow { NumberAnimation { duration: 520; easing.type: Easing.OutCubic } }

    // Bars
    Item {
        id: plot
        visible: !root.line
        anchors.fill: parent
        anchors.bottomMargin: 48
        anchors.topMargin: 52
        readonly property int n: Math.max(1, root.values.length)
        readonly property real slot: width / n
        readonly property real barW: Math.min(260, slot * 0.62)

        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 2
            color: Theme.line
        }
        Repeater {
            model: root.line ? 0 : root.values.length
            Item {
                required property int index
                readonly property bool win: index === root.best
                x: plot.slot * index + (plot.slot - plot.barW) / 2
                width: plot.barW
                height: plot.height
                Rectangle {
                    id: bar
                    anchors.bottom: parent.bottom
                    anchors.bottomMargin: 2
                    width: parent.width
                    height: Math.max(6, (parent.height - 8) * root.values[index] / root.maxV * root.grow)
                    radius: 10
                    color: parent.win ? Theme.gold : Theme.copper
                    opacity: parent.win ? 1 : 0.62
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.bottom: bar.top
                    anchors.bottomMargin: 12
                    text: root.valueText[index] !== undefined ? root.valueText[index] : ""
                    color: parent.win ? Theme.gold : Theme.dim
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(parent.win ? 30 : 24)
                    font.weight: parent.win ? Font.Bold : Font.Normal
                    opacity: root.grow
                }
                Text {
                    anchors.horizontalCenter: parent.horizontalCenter
                    anchors.top: parent.bottom
                    anchors.topMargin: 16
                    width: plot.slot - 10
                    horizontalAlignment: Text.AlignHCenter
                    elide: Text.ElideRight
                    text: root.labels[index] !== undefined ? root.labels[index] : ""
                    color: parent.win ? Theme.ink : Theme.dim
                    font.family: Theme.mono
                    font.pixelSize: Theme.px(21)
                }
            }
        }
    }

    // Line
    Item {
        id: linePlot
        visible: root.line
        anchors.fill: parent
        anchors.margins: 20
        anchors.bottomMargin: 48
        readonly property real span: Math.max(1e-9, root.maxV - root.minV)
        function pt(i) {
            const n = Math.max(1, root.values.length - 1);
            return Qt.point(width * i / n, height - 10 - (height - 20) * (root.values[i] - root.minV) / span);
        }
        Rectangle {
            anchors.bottom: parent.bottom
            width: parent.width
            height: 2
            color: Theme.line
        }
        Shape {
            anchors.fill: parent
            preferredRendererType: Shape.CurveRenderer
            ShapePath {
                strokeColor: Theme.gold
                strokeWidth: 3.5
                fillColor: "transparent"
                capStyle: ShapePath.RoundCap
                joinStyle: ShapePath.RoundJoin
                PathPolyline {
                    path: {
                        const pts = [];
                        const n = Math.max(1, Math.ceil(root.values.length * root.grow));
                        for (let i = 0; i < Math.min(n, root.values.length); ++i)
                            pts.push(linePlot.pt(i));
                        return pts;
                    }
                }
            }
        }
        Rectangle {
            readonly property point p: root.values.length > 0 ? linePlot.pt(root.values.length - 1) : Qt.point(0, 0)
            x: p.x - 8; y: p.y - 8
            width: 16; height: 16; radius: 8
            color: Theme.gold
            opacity: root.grow > 0.98 ? 1 : 0
        }
        Text {
            anchors.left: parent.left
            anchors.top: parent.bottom
            anchors.topMargin: 14
            text: root.labels.length > 0 ? root.labels[0] : ""
            color: Theme.dim
            font.family: Theme.mono
            font.pixelSize: Theme.px(19)
        }
        Text {
            anchors.right: parent.right
            anchors.top: parent.bottom
            anchors.topMargin: 14
            text: root.labels.length > 0 ? root.labels[root.labels.length - 1] : ""
            color: Theme.dim
            font.family: Theme.mono
            font.pixelSize: Theme.px(19)
        }
    }
}
