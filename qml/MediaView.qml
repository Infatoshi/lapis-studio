import QtQuick
import QtMultimedia

// Video (QtMultimedia) or an audio waveform with playback. Click to play or pause.
Item {
    id: root
    objectName: "mediaView"
    property var artifact
    property QtObject backend
    property bool live: false
    readonly property bool audio: artifact && artifact.kind === "audio"
    readonly property string path: artifact && artifact.path ? artifact.path : ""
    readonly property var peaks: audio && backend && path.length > 0 ? backend.audioPeaks(path) : []
    readonly property real progress: player.duration > 0 ? player.position / player.duration : 0

    // Videos (lapis-viz renders, screen captures) play inline, muted and looping, as soon as the
    // Stage shows them; audio waits for a click. Offscreen modes never load a file (live is false).
    MediaPlayer {
        id: player
        source: root.live && root.path.length > 0 ? ("file://" + encodeURI(root.path)) : ""
        audioOutput: AudioOutput { muted: !root.audio }
        videoOutput: root.audio ? null : video
        loops: root.audio ? 1 : MediaPlayer.Infinite
        onSourceChanged: if (!root.audio && source.toString().length > 0) play()
    }

    VideoOutput {
        id: video
        anchors.fill: parent
        visible: !root.audio
        fillMode: VideoOutput.PreserveAspectFit
    }

    Text {
        anchors.centerIn: parent
        visible: !root.audio && (!root.live || player.error !== MediaPlayer.NoError || !player.hasVideo)
        text: player.error !== MediaPlayer.NoError
              ? ("Video unavailable: " + player.errorString)
              : (root.path.length > 0 ? "Video ready (click to play)" : "No video frame")
        color: Theme.dim
        font.family: Theme.mono
        font.pixelSize: Theme.px(14)
    }

    // Waveform
    Row {
        id: wave
        visible: root.audio
        anchors.centerIn: parent
        height: parent.height * 0.6
        spacing: 3
        readonly property real barW: Math.max(2, (root.width - 3 * Math.max(1, root.peaks.length)) / Math.max(1, root.peaks.length))
        Repeater {
            model: root.peaks
            Rectangle {
                required property var modelData
                required property int index
                anchors.verticalCenter: parent.verticalCenter
                width: wave.barW
                height: Math.max(3, wave.height * modelData)
                radius: width / 2
                color: index / Math.max(1, root.peaks.length) <= root.progress && root.progress > 0 ? Theme.gold : Theme.copper
                opacity: 0.85
            }
        }
    }

    Rectangle {
        anchors.bottom: parent.bottom
        width: parent.width * root.progress
        height: 3
        color: Theme.gold
        visible: !root.audio
    }

    MouseArea {
        anchors.fill: parent
        cursorShape: Qt.PointingHandCursor
        onClicked: player.playbackState === MediaPlayer.PlayingState ? player.pause() : player.play()
    }
}
