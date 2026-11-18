import QtQuick

// PDFKit-rendered page (image://pdf), inked for the OLED stage and fit to width.
// Up/Down scroll within the page; at the page edge they turn to the next or previous page.
Item {
    id: root
    objectName: "pdfView"
    property var artifact
    property QtObject backend
    property bool live: false
    property int page: 0
    readonly property string path: artifact ? artifact.path : ""
    readonly property int pages: backend && path.length > 0 ? backend.pdfPages(path) : 0
    readonly property string pageLabel: pages > 0 ? (page + 1) + " / " + pages : ""

    function scrollBy(d) {
        const maxY = Math.max(0, flick.contentHeight - flick.height);
        if (d > 0 && flick.contentY >= maxY - 1 && page < pages - 1) {
            page += 1;
            flick.contentY = 0;
        } else if (d < 0 && flick.contentY <= 1 && page > 0) {
            page -= 1;
            flick.contentY = 0;
        } else {
            flick.contentY = Math.max(0, Math.min(maxY, flick.contentY + d * flick.height * 0.4));
        }
    }

    // Gamepad right stick: pixel deltas, set directly. Holding the stick past an end of the page
    // for ~240 px of travel turns it, like a reader.
    property bool padScrolling: false
    property real overscroll: 0
    Timer { id: padIdle; interval: 120; onTriggered: { root.padScrolling = false; root.overscroll = 0; } }
    function scrollPixels(px) {
        padScrolling = true;
        padIdle.restart();
        const maxY = Math.max(0, flick.contentHeight - flick.height);
        const y = flick.contentY + px;
        if ((y > maxY && page < pages - 1) || (y < 0 && page > 0)) {
            overscroll += px;
            if (Math.abs(overscroll) >= 240) {
                page += overscroll > 0 ? 1 : -1;
                overscroll = 0;
                flick.contentY = px > 0 ? 0 : Math.max(0, flick.contentHeight - flick.height);
                return;
            }
        } else {
            overscroll = 0;
        }
        flick.contentY = Math.max(0, Math.min(maxY, y));
    }

    Rectangle {
        id: frame
        anchors.horizontalCenter: parent.horizontalCenter
        height: parent.height
        width: Math.min(parent.width, 1180)
        color: Theme.card
        radius: 18
        clip: true

        Flickable {
            id: flick
            objectName: "pdfFlick"
            anchors.fill: parent
            contentWidth: width
            contentHeight: img.height
            boundsBehavior: Flickable.StopAtBounds
            Behavior on contentY {
                enabled: !root.padScrolling
                NumberAnimation { duration: 180; easing.type: Easing.OutCubic }
            }
            Image {
                id: img
                objectName: "pdfImage"
                width: flick.width
                height: implicitWidth > 0 ? width * implicitHeight / implicitWidth : flick.height
                fillMode: Image.PreserveAspectFit
                asynchronous: true  // PDFKit renders off the GUI thread (was a 30 ms stall per page)
                cache: true
                smooth: true
                sourceSize.width: Math.round(flick.width)
                source: root.path.length > 0 ? "image://pdf/" + root.page + "/" + encodeURIComponent(root.path) : ""
            }
        }
    }
}
