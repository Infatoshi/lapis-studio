import QtQuick

// A cursor that blinks on the QML animation clock, so offscreen recordings stay deterministic.
Rectangle {
    id: caret
    width: 2
    color: Theme.gold
    visible: parent && parent.cursorVisible
    SequentialAnimation on opacity {
        running: caret.visible
        loops: Animation.Infinite
        PropertyAction { value: 1 }
        PauseAnimation { duration: 530 }
        PropertyAction { value: 0 }
        PauseAnimation { duration: 470 }
    }
}
