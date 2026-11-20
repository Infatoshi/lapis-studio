#pragma once
#include <QColor>
#include <QImage>
#include <QPointF>
#include <QRectF>
#include <QSizeF>
#include <QString>
#include <cstdint>
#include <memory>

namespace studio::cua {

// Pure geometry behind the visible agent cursor (unit-tested, no AppKit).
// A target box the overlay will outline: finite, positive, and no larger than any real display.
bool reticleRectValid(const QRectF &globalRect);
// The quadratic Bezier control point for a glide from `from` to `to`: the midpoint pushed
// sideways by 18% of the distance (at most 64 pt), always bowing upward on screen.
QPointF glideControlPoint(QPointF from, QPointF to);
QPointF quadBezier(QPointF a, QPointF c, QPointF b, double t);
// Where the status pill goes: 10 pt below `anchor` if it fits on `display`, else 10 pt above,
// else pinned inside; x starts at the anchor's left and is clamped 8 pt inside the display.
QRectF reticlePillRect(const QRectF &anchor, QSizeF pill, const QRectF &display);

// The floating HUD bar's dot: gold pulse while listening, sapphire pulse while gliding or
// clicking, a green check when done, coral on error.
enum class HudTone { Listening, Working, Done, Error };
constexpr double kHudWidth = 340, kHudHeight = 38, kHudBottomGap = 44;
constexpr double kHudLingerSec = 2.5;  // Done / Error hold this long, then fade out
// Where the HUD capsule sits on `display` (global points): centred, 44 pt above the bottom edge.
QRectF cuaHudRect(const QRectF &display);

// What the overlay is showing right now (tests, snapshots).
struct ReticleState {
    bool visible = false;
    QPointF tip;              // current, possibly mid-glide, global point
    QPointF target;           // where the glide ends
    bool gliding = false;
    QRectF focusRect;         // empty when no focus ring is shown
    bool rippleActive = false;
    QString label;
    QRectF pillRect;          // empty when no pill is shown
    QRectF overlayFrame;      // the display the overlay covers, global points
    // The all-Spaces HUD bar (independent of the cursor: it stays up while the cursor hides).
    bool hudVisible = false;  // any alpha, including mid fade-out
    QString hudStatus;
    HudTone hudTone = HudTone::Working;
    QRectF hudRect;           // global points; empty when hidden
    double hudAlpha = 0;
};

// The agent's visible cursor: a matte obsidian arrow with a sapphire rim that glides along a
// slight Bezier arc with a fading motion tail, a sapphire focus ring around the target, a click
// ripple, and a frosted status pill. One borderless, click-through, non-activating panel covers
// the active display; sharingType = NSWindowSharingNone, so it never shows up in captures.
//
// All methods, construction, and destruction must run on the AppKit main thread.
// Pass windowNumber() to captureDisplayJpeg for explicit ScreenCaptureKit exclusion.
class CuaReticle {
public:
    // Offscreen: no window ever opens; geometry runs against `virtualDisplay` and time only moves
    // through advanceClock(). Used by tests and --snap-cursor.
    enum class Mode { Visible, Offscreen };
    CuaReticle();
    explicit CuaReticle(Mode mode, const QRectF &virtualDisplay = QRectF(0, 0, 1440, 900));
    ~CuaReticle();
    CuaReticle(const CuaReticle &) = delete;
    CuaReticle &operator=(const CuaReticle &) = delete;

    void moveTo(QPointF globalPt, const QString &label = {});
    // Returns false (and changes nothing) for an invalid rect or one on no display.
    bool highlightRect(QRectF globalRect, const QString &label = {});
    // Ripple at the pointer tip; glides there first when the pointer is elsewhere. An empty label
    // keeps the current one. Returns false for a non-finite point or one on no display.
    bool pulseClick(QPointF globalPt, const QString &label = {});
    void hide();
    uint32_t windowNumber() const;  // 0 in Offscreen mode

    // The floating HUD bar: a frosted obsidian capsule pinned bottom-centre of the display it
    // first appears on, on every Space (so it stays put while fullscreen Lapis slides away), click-
    // through, never activating, excluded from captures. A new status while it is up updates it in
    // place. Done and Error fade out on their own after 2.5 s.
    void showHud(const QString &status, HudTone tone);
    void showHud(const QString &status, bool listening = false) {
        showHud(status, listening ? HudTone::Listening : HudTone::Working);
    }
    // Fades the HUD out, starting `afterSeconds` from now (0: at once).
    void hideHud(double afterSeconds = 0);
    uint32_t hudWindowNumber() const;  // 0 in Offscreen mode

    ReticleState state() const;
    void advanceClock(double seconds);  // Offscreen mode only
    // Renders the real overlay view (the overlay display's top-left at the image origin) over a
    // transparent or `background` fill, at `scale` pixels per point.
    QImage render(double scale = 2, const QColor &background = Qt::transparent) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

// --snap-cursor OUT.png: two frames side by side on a dark desktop (mid-glide with the motion
// tail toward a focused button; then the click ripple with the status pill), each with the HUD
// bar at bottom-centre, over a strip of the remaining HUD tones. 0 on success.
int snapCursor(const QString &outPath);

} // namespace studio::cua
