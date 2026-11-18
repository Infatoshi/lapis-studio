import QtQuick

// Image or SVG, fit to the stage at native sharpness.
Item {
    id: root
    objectName: "imageView"
    property var artifact
    property bool live: false
    Image {
        anchors.fill: parent
        fillMode: Image.PreserveAspectFit
        asynchronous: root.live
        smooth: true
        mipmap: true
        sourceSize.width: root.artifact && root.artifact.kind === "svg" ? Math.round(width) : undefined
        sourceSize.height: root.artifact && root.artifact.kind === "svg" ? Math.round(height) : undefined
        source: root.artifact && root.artifact.path ? "file://" + root.artifact.path : ""
    }
}
