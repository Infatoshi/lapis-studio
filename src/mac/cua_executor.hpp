#pragma once

#include "mac/cua.hpp"
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

// Every side effect the executor has, injectable. defaultCuaHooks() wires the real
// studio::cua::* calls, a shared CuaReticle (always touched on the main thread) and libcurl.
struct CuaHooks {
    std::function<QVector<DisplayInfo>()> listDisplays;
    std::function<QVector<AxNode>(uint32_t displayId)> inspectDisplay;
    std::function<QByteArray(uint32_t displayId, int maxWidth, uint32_t excludeWindowId, QString *err)> captureDisplayJpeg;
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
    std::function<void(int ms)> pause;          // lets a glide land before the click; no-op in tests
    // Multi-monitor routing; every one is optional. Unset listAppWindows means "no window info",
    // unset activateApp means no activation, unset selfPid means no self filtering, unset
    // lapisDisplayId is derived from listAppWindows (largest selfPid window).
    std::function<QVector<AppWindowInfo>()> listAppWindows;
    std::function<bool(const QString &appHint, uint32_t *outDisplayId)> activateApp;
    std::function<int()> selfPid;
    std::function<uint32_t()> lapisDisplayId;  // 0 when Lapis has no on-screen window
};
CuaHooks defaultCuaHooks();

// Runs fn on the Qt main thread: directly when already there, else via
// QMetaObject::invokeMethod(qApp, fn, Qt::BlockingQueuedConnection). No-op without a qApp.
void runOnMainThread(const std::function<void()> &fn);

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
    // Activates the hinted app (with a settle pause) when activate is true.
    bool route(const QString &targetDesc, bool activate, Route *out, QString *err);
    Located locate(const QString &targetDesc, bool refineCrop, const QString &label, Route *routeOut = nullptr);

    CuaHooks m_hooks;
    bool m_preferAxPress = false;
};

}  // namespace studio::cua
