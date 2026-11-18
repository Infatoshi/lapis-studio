pragma Singleton
import QtQuick

QtObject {
    // "gold" (default) | "lapis". StudioApp binds it to backend.theme. Only the accent pair
    // changes: gold #e09f3e / copper #b5653b, or Lapis Blue #4f8cff / #2d5aa8, both on OLED black.
    property string theme: "gold"
    readonly property bool lapis: theme === "lapis"
    readonly property color bg: "#000000"
    readonly property color card: "#0d0c0b"
    readonly property color cardHi: "#14120f"
    readonly property color line: "#1d1915"
    readonly property color track: "#1a1714"
    readonly property color gold: lapis ? "#4f8cff" : "#e09f3e"    // primary accent
    readonly property color copper: lapis ? "#2d5aa8" : "#b5653b"  // secondary accent
    readonly property color ink: "#e8d5b5"
    readonly property color text: "#d9cfc3"
    readonly property color dim: "#6e665e"
    readonly property color faint: "#3a332c"
    readonly property color addBg: "#1a1309"
    readonly property color delBg: "#170c07"
    readonly property string sans: "Avenir Next"
    readonly property string mono: "Menlo"
    readonly property int zoomMs: 220   // shared-element zoom, Easing.OutQuint
    readonly property int slideMs: 180  // Stage artifact slide, Easing.OutQuint
    // Focus / hover lift spring: stiff and near-critically damped, so it settles in ~120 ms with
    // no visible wobble (the old 4.5 / 0.36 rang for ~700 ms).
    readonly property real springK: 22.0
    readonly property real springDamping: 0.92
    // Island-to-island hop of the focus ring: a 120 ms OutExpo glide on x/y only (a GPU
    // transform; the ring's size never changes, so its border is never re-tessellated).
    readonly property int hopMs: 120
    // UI zoom (0.75..1.60): StudioApp binds it to backend.uiScale; text sizes multiply by it.
    property real fontScale: 1.0
    function px(n) { return Math.round(n * fontScale); }
    // The primary accent at alpha A (tints, rings, borders), so they follow the theme.
    function goldA(a) { return Qt.rgba(gold.r, gold.g, gold.b, a); }
}
