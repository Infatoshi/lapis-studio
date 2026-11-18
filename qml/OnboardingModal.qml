import QtQuick

// First launch only (no ~/.lapis-next/studio_prefs.json yet): pick how to drive the app.
// Three cards, Neovim pre-selected. j/k, w/s or Up/Down switch; Enter confirms; 1/2/3 or a
// click picks directly. Confirming writes the prefs file, so it never shows again.
FocusScope {
    id: root
    objectName: "onboardingModal"
    property QtObject backend
    readonly property bool open: backend ? backend.firstLaunch : false
    property int choice: 0
    signal done()

    readonly property var cards: [
        { id: "controller", name: "Controller Mode", tag: "Recommended",
          line: "L-stick / d-pad move · ✕ confirm · ○ back · □ write-up · △ tab",
          keys: "DualShock 4 / gamepad. Keys and ⇥ Tab still work." },
        { id: "vim", name: "Neovim Mode", tag: "",
          line: "h j k l move · g G first and last · ⏎ zoom · esc back · space write-up",
          keys: "Zero mouse. ⇥ Tab accepts the predicted next step." },
        { id: "wasd", name: "Gaming Mode", tag: "",
          line: "w a s d move · e interact · q back · space write-up",
          keys: "Left hand on the keys. ⇥ Tab accepts the predicted next step." },
        { id: "mouse", name: "Mouse Mode", tag: "",
          line: "Click islands and artifacts · wheel to zoom · hover for options",
          keys: "Best if you are new here. ⇥ Tab still works." }
    ]

    function confirm() {
        if (!backend)
            return;
        backend.completeOnboarding(cards[choice].id);
        done();
    }

    visible: opacity > 0.01
    opacity: open ? 1 : 0
    Behavior on opacity { NumberAnimation { duration: 200; easing.type: Easing.OutCubic } }
    enabled: open
    onOpenChanged: if (open) { choice = 0; forceActiveFocus(); }
    Component.onCompleted: if (open) forceActiveFocus()

    Keys.onPressed: (e) => {
        const k = e.key;
        if (e.modifiers & Qt.ControlModifier) {
            e.accepted = true;
            return;
        }
        if (k === Qt.Key_Down || k === Qt.Key_J || k === Qt.Key_S || k === Qt.Key_Tab)
            choice = Math.min(cards.length - 1, choice + 1);
        else if (k === Qt.Key_Up || k === Qt.Key_K || k === Qt.Key_W || k === Qt.Key_Backtab)
            choice = Math.max(0, choice - 1);
        else if (k >= Qt.Key_1 && k <= Qt.Key_4)
            choice = k - Qt.Key_1;
        else if (k === Qt.Key_Return || k === Qt.Key_Enter || k === Qt.Key_E || k === Qt.Key_Space)
            confirm();
        e.accepted = true;
    }

    Rectangle {
        anchors.fill: parent
        color: "#000000"
        MouseArea { anchors.fill: parent; onWheel: (w) => w.accepted = true }
    }

    Column {
        anchors.centerIn: parent
        width: 820
        spacing: 18

        Text {
            width: parent.width
            text: "How do you want to drive lapis studio?"
            color: Theme.ink
            font.family: Theme.sans
            font.pixelSize: 34
            font.weight: Font.DemiBold
        }
        Text {
            width: parent.width
            text: root.backend && root.backend.gamepad && root.backend.gamepad.connected
                  ? "Change it any time with ⌘,  ·  controller: ↕ pick, ✕ confirm"
                  : "Change it any time with ⌘,"
            color: Theme.dim
            font.family: Theme.sans
            font.pixelSize: 18
        }
        Item { width: 1; height: 14 }

        Repeater {
            model: root.cards
            Rectangle {
                required property var modelData
                required property int index
                objectName: "onboardingCard"
                readonly property bool current: root.choice === index
                width: parent.width
                height: 124
                radius: 24
                color: current ? "#0d0b08" : "#000000"
                border.width: current ? 2 : 1
                border.color: current ? Theme.gold : Theme.line
                Behavior on border.color { ColorAnimation { duration: 140 } }

                Text {
                    id: cardNum
                    x: 30
                    y: 26
                    text: String(index + 1)
                    color: parent.current ? Theme.gold : Theme.faint
                    font.family: Theme.mono
                    font.pixelSize: 18
                }
                Text {
                    id: cardName
                    x: 66
                    y: 20
                    text: modelData.name
                    color: parent.current ? Theme.gold : Theme.ink
                    font.family: Theme.sans
                    font.pixelSize: 26
                    font.weight: Font.DemiBold
                }
                Rectangle {
                    visible: modelData.tag.length > 0
                    anchors.left: cardName.right
                    anchors.leftMargin: 16
                    anchors.verticalCenter: cardName.verticalCenter
                    width: tagText.implicitWidth + 20
                    height: 26
                    radius: 13
                    color: Qt.rgba(0.878, 0.624, 0.243, 0.16)
                    Text {
                        id: tagText
                        anchors.centerIn: parent
                        text: modelData.tag
                        color: Theme.gold
                        font.family: Theme.mono
                        font.pixelSize: 13
                    }
                }
                Text {
                    x: 66
                    y: 60
                    width: parent.width - 96
                    text: modelData.line
                    color: Theme.text
                    elide: Text.ElideRight
                    font.family: Theme.mono
                    font.pixelSize: 15
                }
                Text {
                    x: 66
                    y: 88
                    width: parent.width - 96
                    text: modelData.keys
                    color: Theme.dim
                    elide: Text.ElideRight
                    font.family: Theme.sans
                    font.pixelSize: 15
                }
                MouseArea {
                    anchors.fill: parent
                    cursorShape: Qt.PointingHandCursor
                    hoverEnabled: true
                    onEntered: root.choice = index
                    onClicked: { root.choice = index; root.confirm(); }
                }
            }
        }

        Item { width: 1; height: 8 }
        Text {
            width: parent.width
            horizontalAlignment: Text.AlignHCenter
            text: "j/k or w/s to choose   ⏎ confirm"
            color: Theme.faint
            font.family: Theme.mono
            font.pixelSize: 15
        }
    }
}
