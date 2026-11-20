#pragma once

#include "mac/cua.hpp"
#include "mac/cua_reticle.hpp"
#include <QByteArray>
#include <QPointF>
#include <QRectF>
#include <QSizeF>
#include <QString>
#include <QVector>
#include <cstdint>
#include <functional>

namespace studio::cua {

// What the L3 pixel grounder answered. point_1000 is a 0..1000 fraction of the captured image on
// each axis; point_px is a pixel in the captured JPEG. imageSize is the JPEG size when known.
struct GroundResult {
    bool ok = false;
    bool hasPoint1000 = false;
    QPointF point1000;
    bool hasPointPx = false;
    QPointF pointPx;
    QSizeF imageSize;
    QString detail;  // error text when !ok
};

// Parses a grounder JSON reply: point_1000 / point_px as [x, y] or {x, y}, optional
// image_size ([w, h] or {width, height}) and error. ok=false when neither point is usable.
GroundResult parseGroundReply(const QByteArray &json);
// The grounder request body: {image_b64, instruction, refine_crop}.
QByteArray buildGroundRequest(const QByteArray &jpeg, const QString &instruction, bool refineCrop);
// LAPIS_CUA_GROUNDER_URL, else http://127.0.0.1:18120/ground.
QString grounderUrl();
// libcurl POST to grounderUrl() with a 4 s total timeout; ok=false (no throw) when offline.
GroundResult httpGroundPixel(const QByteArray &jpeg, const QString &instruction, bool refineCrop);

// The pixel size captureDisplayJpeg(display, maxWidth) produces (same rounding as cua.mm).
QSizeF expectedCaptureSize(const DisplayInfo &display, int maxWidth);

// What activating (or launching) an app gave: its pid, the display of its largest window (0 when
// none is on screen yet), and whether it had to be launched.
struct AppActivation {
    bool ok = false;
    uint32_t displayId = 0;
    int pid = 0;
    bool launched = false;
};

// Every side effect the executor has, injectable. defaultCuaHooks() wires the real
// studio::cua::* calls, a shared CuaReticle (always touched on the main thread) and libcurl.
struct CuaHooks {
    std::function<QVector<DisplayInfo>()> listDisplays;
    std::function<QVector<AxNode>(uint32_t displayId)> inspectDisplay;
    std::function<QByteArray(uint32_t displayId, int maxWidth, const QVector<uint32_t> &excludeWindowIds, QString *err)>
        captureDisplayJpeg;
    std::function<GroundResult(const QByteArray &jpeg, const QString &instruction, bool refineCrop)> groundPixel;
    std::function<ActionResult(QPointF globalPt, bool doubleClick)> clickAt;
    std::function<ActionResult(QPointF globalPt, int deltaY)> scrollAt;
    std::function<ActionResult(const QString &text)> typeText;
    std::function<ActionResult(int pid, const QString &titleOrRole, const QRectF &nearRect)> axPress;
    std::function<void(QPointF globalPt, const QString &label)> reticleMove;
    std::function<void(QRectF globalRect, const QString &label)> reticleHighlight;
    std::function<void(QPointF globalPt, const QString &label)> reticlePulse;
    std::function<void()> reticleHide;
    std::function<uint32_t()> reticleWindowId;  // excluded from captures; 0 when none
    std::function<uint32_t()> hudWindowId;      // the HUD bar, also excluded; 0 when none
    // The floating HUD bar's status line ("Chrome · Clicking address bar"); Done/Error fade out.
    std::function<void(const QString &status, HudTone tone)> reticleHud;
    std::function<void(int ms)> pause;          // lets a glide land before the click; no-op in tests
    // Multi-monitor routing; every one is optional. Unset listAppWindows means "no window info",
    // unset activateApp means no activation, unset selfPid means no self filtering, unset
    // lapisDisplayId is derived from listAppWindows (largest selfPid window).
    std::function<QVector<AppWindowInfo>()> listAppWindows;
    std::function<AppActivation(const QString &appHint)> activateApp;  // activates, else launches
    std::function<int()> selfPid;
    std::function<uint32_t()> lapisDisplayId;  // 0 when Lapis has no on-screen window
    // Space switching. Unset frontmostPid means a fixed 220 ms settle after activation instead of
    // waiting for the app to come to the front; unset hideSelf means Lapis is never hidden.
    std::function<int()> frontmostPid;  // 0 when unknown
    std::function<bool()> hideSelf;     // hides Lapis so its fullscreen Space slides away
};
CuaHooks defaultCuaHooks();

// Runs fn on the Qt main thread: directly when already there, else via
// QMetaObject::invokeMethod(qApp, fn, Qt::BlockingQueuedConnection). No-op without a qApp.
void runOnMainThread(const std::function<void()> &fn);

// The app-wide HUD bar on the shared overlay (the one defaultCuaHooks drives), from any thread.
// Live app only: these open a real window.
void showCuaHud(const QString &status, HudTone tone);
void hideCuaHud(double afterSeconds = 0);

// Space-switch timing: after activation, poll the frontmost app every 40 ms for up to 900 ms
// (2.5 s for a cold launch), then give the Space slide 180 ms to finish.
constexpr int kFrontPollMs = 40, kFrontWaitMs = 900, kLaunchWaitMs = 2500, kSpaceSlideMs = 180;
constexpr int kActivateSettleMs = 220;  // the fixed fallback when there is no frontmost signal

struct CuaExecResult {
    bool ok = false;
    QString tier;  // "L1_AX", "L3_VLM", "ERR"; scrollTarget's display-centre fallback is "CENTER"
    QPointF globalPt;
    QString detail;
};

// Scores one AX node against a target description: 0 = no match. Exact title > exact value >
// title substring > value substring, plus a bonus for clickable roles. Requires a valid rect.
int axMatchScore(const AxNode &node, const QString &targetDesc);
// The index of the single best AX match for targetDesc, -1 when none or when the top score is
// tied between nodes with different rects (ambiguous: let the VLM decide).
int bestAxMatch(const QVector<AxNode> &nodes, const QString &targetDesc);
// The app a target names, as a short hint ("Chrome address bar" -> "Chrome", "Send button in
// Discord" -> "Discord", "VS Code terminal" -> "VS Code"); empty when it names none.
QString extractAppHint(const QString &targetDesc);
// targetDesc without the app phrase and its connector ("Send button in Discord" -> "Send button").
QString stripAppHint(const QString &targetDesc, const QString &appHint);
// Model tags carry "\n" as two characters; turn each literal backslash-n into a real newline.
QString normalizeTypedText(const QString &text);

// Visible computer use: L1 Accessibility first, L3 local VLM pixel grounding as fallback, with
// the reticle showing every step. Blocking; run it on a worker thread (capture requires one).
class CuaExecutor {
public:
    explicit CuaExecutor(CuaHooks hooks = defaultCuaHooks());

    CuaExecResult clickTarget(const QString &targetDesc, bool doubleClick = false, bool refineCrop = false);
    CuaExecResult typeIntoTarget(const QString &targetDesc, const QString &text);
    // Glides to the target (AX, then VLM, else the display centre), then scrolls there.
    CuaExecResult scrollTarget(const QString &targetDesc, int deltaY);

    // L1 hits press through AXPress instead of a visible synthetic click (off by default, so
    // Chrome and native apps behave the same and the click shows on screen).
    void setPreferAxPress(bool on) { m_preferAxPress = on; }
    const CuaHooks &hooks() const { return m_hooks; }

private:
    struct Located {
        bool ok = false;
        QString tier;
        QPointF pt;
        QRectF rect;  // AX rect (L1 only)
        AxNode node;
        QString detail;
    };
    // Where to look for one target: displays in search order (the hinted app's display, then
    // displays without Lapis, primary first, then Lapis's own), plus the self filter.
    struct Route {
        QVector<DisplayInfo> displays;
        QString appHint;
        uint32_t targetDisplayId = 0;
        uint32_t lapisDisplayId = 0;
        int selfPid = 0;
        bool allowSelf = false;
    };
    // Activates the hinted app (waiting for it to come to the front) when activate is true; on a
    // single display, hides Lapis when it is still frontmost and the target does not name it.
    bool route(const QString &targetDesc, bool activate, Route *out, QString *err);
    // `verb` ("Clicking") builds the HUD status once the route is known.
    Located locate(const QString &targetDesc, bool refineCrop, const QString &label, const QString &verb,
                   Route *routeOut = nullptr);
    // Polls frontmostPid until pid (else anything but Lapis) is in front, then lets the Space
    // slide finish. False on timeout.
    bool waitFrontmost(int pid, bool launched, int selfPid);
    CuaExecResult click(const QString &targetDesc, bool doubleClick, bool refineCrop, bool finish, Route *routeOut);
    void hud(const QString &status, HudTone tone) const;
    void hudFail(const QString &targetDesc, const QString &why) const;

    CuaHooks m_hooks;
    bool m_preferAxPress = false;
};

}  // namespace studio::cua
