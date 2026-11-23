#include "mac/cua.hpp"
#include "mac/cua_reticle.hpp"
#include "mac/cua_executor.hpp"
#include "mac/cua_shadow.hpp"
#include "mac/cua_spotlight.hpp"
#include "mac/cua_pad_icon.hpp"
#include <CoreGraphics/CGEventTypes.h>
#include <QPainter>
#include <functional>
#include <iterator>
#include <utility>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QFile>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>
#include <QThread>
#include <cmath>
#include <cstdio>
#include <limits>
#include <thread>

using namespace studio::cua;
namespace {
int passed = 0, failed = 0;
void check(bool condition, const char *name) {
    if (condition) ++passed;
    else { ++failed; std::printf("FAIL: %s\n", name); }
}
bool close(QPointF a, QPointF b) { return std::hypot(a.x() - b.x(), a.y() - b.y()) < 1e-8; }
void geometryTests() {
    const DisplayInfo primary{1, "primary", {0, 0, 1440, 900}, 2, true};
    const DisplayInfo left{2, "left", {-1920, -300, 1920, 1080}, 1, false};
    const DisplayInfo above{3, "above", {200, -1200, 1600, 1200}, 2, false};
    for (const auto &d : {primary, left, above}) {
        for (QPointF p : {QPointF(0, 0), QPointF(10.5, 20.25), QPointF(1280, 720)}) {
            check(close(globalPointToCapturePixel(capturePixelToGlobalPoint(p, d), d), p), "native pixel round trip");
            check(close(globalPointToCapturePixel(capturePixelToGlobalPoint(p, d, {1280, 720}), d, {1280, 720}), p), "resized pixel round trip");
        }
        check(close(capturePixelToGlobalPoint({0, 0}, d), d.globalPoints.topLeft()), "capture origin is display origin");
    }
    check(close(capturePixelToGlobalPoint({200, 100}, primary), {100, 50}), "Retina halves native pixels");
    check(close(capturePixelToGlobalPoint({200, 100}, left), {-1720, -200}), "negative secondary origin");
    check(close(globalPointToAppKit({-1500, -200}, primary.globalPoints), {-1500, 1100}), "AppKit flip uses primary height");
    check(close(globalPointToAppKit({0, 900}, primary.globalPoints), {0, 0}), "primary bottom maps to AppKit zero");
    check(!std::isfinite(capturePixelToGlobalPoint({1, 1}, {}).x()), "invalid display fails without division by zero");
}
void leaseTests() {
    const QPointF saved(-1000, -100), target(200, 300);
    check(shouldRestoreCursor(saved, target, target, 12), "restore when pointer unchanged");
    check(shouldRestoreCursor(saved, {212, 300}, target, 12), "threshold inclusive");
    check(!shouldRestoreCursor(saved, {212.01, 300}, target, 12), "abort beyond threshold");
    check(!shouldRestoreCursor(saved, {209, 309}, target, 12), "Euclidean diagonal distance");
    check(shouldRestoreCursor(saved, target, target, 0), "zero threshold exact match");
    check(!shouldRestoreCursor(saved, target, target, -1), "negative threshold rejected");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(!shouldRestoreCursor({nan, 0}, target, target, 12), "invalid saved point rejected");
    check(!shouldRestoreCursor(saved, {nan, 0}, target, 12), "invalid current point rejected");
    check(!shouldRestoreCursor(saved, target, {nan, 0}, 12), "invalid target rejected");
    check(!shouldRestoreCursor(saved, target, target, nan), "invalid threshold rejected");
}
void formattingTests() {
    AxNode n{1, "AXButton", "Save\n\"document\"", "a\\b", "Example", 42, {-100, -50, 80, 24}, {"AXPress"}};
    const QString text = formatNodesForModel({n});
    const auto parsed = QJsonDocument::fromJson(text.toUtf8());
    check(parsed.isObject() && parsed.object().value("title").toString() == n.title, "JSON escaping round trip");
    check(parsed.object().value("pid").toInt() == 42 && parsed.object().contains("rect") && parsed.object().contains("actions"), "model target fields present");
    check(text.count('\n') == 1, "one structured line per node");
    check(formatNodesForModel({}).isEmpty(), "empty nodes");
    check(formatNodesForModel({n}, 0).isEmpty() && formatNodesForModel({n}, -1).isEmpty(), "nonpositive budgets");
    for (int cap : {1, 2, 10, 100, 200, 6000}) check(formatNodesForModel({n, n}, cap).size() <= cap, "character budget hard bound");
    check(formatNodesForModel({n}, 1) == QString(QChar(0x2026)), "tiny budget truncation marker");
    check(formatNodesForModel({n}, int(text.size())) == text, "exact fit");
    const auto truncated = formatNodesForModel({n, n}, int(text.size()) + 1);
    check(truncated == text + QChar(0x2026), "whole lines retained on truncation");
    n.value = QString(100000, QChar('x'));
    check(formatNodesForModel({n}, 500).size() <= 500, "large value bounded");
}
void reticleGeometryTests() {
    const double nan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
    check(reticleRectValid({10, 10, 120, 36}), "reticle: normal rect valid");
    check(!reticleRectValid({}) && !reticleRectValid({10, 10, 0, 20}) && !reticleRectValid({10, 10, -5, 20}), "reticle: empty / negative rect rejected");
    check(!reticleRectValid({nan, 0, 10, 10}) && !reticleRectValid({0, 0, inf, 10}), "reticle: non-finite rect rejected");
    check(!reticleRectValid({0, 0, 50000, 10}) && !reticleRectValid({2e6, 0, 10, 10}), "reticle: absurd rect rejected");
    const QPointF a(100, 400), b(500, 400);
    const QPointF c = glideControlPoint(a, b);
    check(c.y() < 400 && std::abs(c.x() - 300) < 1e-9 && std::abs(400 - c.y() - 64) < 1e-9, "reticle: glide bows upward, offset capped at 64 pt");
    check(glideControlPoint(b, a).y() < 400, "reticle: reverse glide also bows upward");
    check(close(glideControlPoint(a, a), a), "reticle: zero-length glide control is the point");
    const QPointF s = glideControlPoint({0, 0}, {0, 100});
    check(std::isfinite(s.x()) && s.x() < 0 && std::abs(s.y() - 50) < 1e-9, "reticle: vertical glide bows left, finite");
    check(close(quadBezier(a, c, b, 0), a) && close(quadBezier(a, c, b, 1), b), "reticle: Bezier endpoints exact");
    const QRectF disp(0, 0, 1440, 900);
    const QSizeF pill(160, 24);
    const QRectF below = reticlePillRect({100, 100, 120, 36}, pill, disp);
    check(std::abs(below.top() - 146) < 1e-9 && std::abs(below.left() - 100) < 1e-9, "reticle: pill 10 pt below the target");
    const QRectF above = reticlePillRect({100, 860, 120, 30}, pill, disp);
    check(std::abs(above.bottom() - 850) < 1e-9, "reticle: pill flips above at the bottom edge");
    const QRectF right = reticlePillRect({1400, 300, 30, 30}, pill, disp);
    check(right.right() <= 1432 + 1e-9 && right.left() >= 8, "reticle: pill clamped 8 pt inside the right edge");
    const QRectF tall = reticlePillRect({100, 10, 120, 880}, pill, disp);
    check(disp.adjusted(8, 8, -8, -8).contains(tall), "reticle: pill pinned inside when neither side fits");
    const QRectF neg(-1920, -300, 1920, 1080);
    check(neg.contains(reticlePillRect({-60, 700, 50, 50}, pill, neg)), "reticle: pill stays on a negative-origin display");
}
void reticleOffscreenTests() {
    // Offscreen mode: no window, no screen; the real view still draws into a bitmap.
    const QRectF disp(0, 0, 800, 500);
    CuaReticle r(CuaReticle::Mode::Offscreen, disp);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(r.windowNumber() == 0 && !r.state().visible, "reticle: offscreen has no window, starts hidden");
    check(!r.highlightRect({nan, 0, 10, 10}) && !r.highlightRect({}) && !r.highlightRect({10, 10, -4, 4}), "reticle: invalid rects rejected");
    check(!r.highlightRect({5000, 5000, 40, 20}), "reticle: rect on no display rejected");
    check(!r.pulseClick({nan, 1}) && !r.pulseClick({-900, -900}), "reticle: invalid / offscreen click rejected");
    r.moveTo({nan, 3});
    r.moveTo({-50, -50});
    check(!r.state().visible, "reticle: invalid / offscreen moveTo ignored");
    r.moveTo({100, 100}, "Looking");
    auto st = r.state();
    check(st.visible && close(st.tip, {100, 100}) && !st.gliding && st.label == "Looking", "reticle: first moveTo appears in place");
    check(!st.pillRect.isEmpty() && disp.contains(st.pillRect), "reticle: pill beside the pointer, on screen");
    // Cartoon palette pixel classes (unpremultiplied). Gold is the Lapis accent #e09f3e, kept apart
    // from the warm cel shadows (#dfc79a, #cfa96e) and parchment text by g < 190 and r > b + 110.
    const auto cream = [](const QColor &c) { return c.alpha() > 200 && c.red() > 210 && c.green() > 190; };
    const auto gold = [](const QColor &c) {
        return c.alpha() > 100 && c.red() > 180 && c.green() > 110 && c.green() < 190 && c.red() > c.blue() + 110;
    };
    const auto ink = [](const QColor &c) { return c.alpha() > 180 && c.red() < 35 && c.green() < 35 && c.blue() < 35; };
    const auto lapisBlue = [](const QColor &c) { return c.alpha() > 100 && c.blue() > 180 && c.blue() > c.red() + 60; };
    // Counts pixels of a class inside `pts` (global points) of a 2x render.
    const auto count = [](const QImage &im, const QRectF &pts, auto pred) {
        int n = 0;
        const QRect px = QRectF(pts.topLeft() * 2, pts.size() * 2).toAlignedRect() & im.rect();
        for (int y = px.top(); y <= px.bottom(); ++y)
            for (int x = px.left(); x <= px.right(); ++x)
                n += pred(im.pixelColor(x, y));
        return n;
    };
    check(std::abs(glideDuration({0, 0}, {50, 0}) - kGlideMinSec) < 1e-9
              && std::abs(glideDuration({0, 0}, {2000, 0}) - kGlideMaxSec) < 1e-9
              && glideDuration({0, 0}, {300, 0}) > kGlideMinSec && glideDuration({0, 0}, {300, 0}) < kGlideMaxSec,
        "reticle: glide 0.24 s short, eases up to a 0.36 s cap with distance");
    check(std::abs(r.state().handAngleDeg + 28) < 1e-9 && std::abs(r.state().stretch - 1) < 1e-9, "reticle: hand rests at -28°, unstretched");
    r.advanceClock(0.5);
    r.moveTo({600, 300}, "Going");
    const double glide = glideDuration({100, 100}, {600, 300});
    r.advanceClock(0.02);
    check(r.state().tip.x() < 100, "reticle: glide winds up (anticipation) before the swoosh");
    r.advanceClock(glide / 2 - 0.02);
    st = r.state();
    check(st.gliding && st.tip.x() > 100 && st.tip.x() < 600 && close(st.target, {600, 300}), "reticle: glides mid-arc");
    // Half time is eased progress 0.5: the straight line would be at (350, 200).
    check(st.tip.y() < 200 - 10, "reticle: glide arcs upward, not a straight line");
    check(st.stretch > 1.2 && st.stretch <= 1.26 + 1e-9, "reticle: hand stretches along the motion at peak speed");
    check(st.handAngleDeg > 0, "reticle: hand banks toward the motion mid-glide");
    {
        // Mid-glide pixels: the swoosh ribbon and speed lines behind the hand, the glove itself.
        const QImage mid = r.render(2);
        const QPointF tip = st.tip;
        const int ribbon = count(mid, QRectF(110, 60, tip.x() - 60 - 110, 120), gold);
        const int glove = count(mid, QRectF(tip.x() - 50, tip.y() - 50, 100, 100), cream);
        const int outline = count(mid, QRectF(tip.x() - 50, tip.y() - 50, 100, 100), ink);
        check(ribbon > 150 && glove > 150 && outline > 150, "reticle: mid-glide renders the gold swoosh behind a cream, ink-outlined glove");
        std::printf("reticle mid-glide: ribbon=%d glove=%d outline=%d\n", ribbon, glove, outline);
    }
    r.advanceClock(glide / 2 + 0.05);
    st = r.state();
    check(!st.gliding && close(st.tip, {600, 300}), "reticle: glide lands exactly");
    check(std::abs(st.stretch - 1) < 1e-9 && std::abs(st.handAngleDeg + 28) < 1e-9, "reticle: landed hand back at rest");
    check(r.highlightRect({560, 280, 100, 40}, "Clicking Send"), "reticle: valid highlight accepted");
    st = r.state();
    check(st.focusRect == QRectF(560, 280, 100, 40) && st.label == "Clicking Send", "reticle: focus ring on the target");
    check(st.pillRect.top() >= 320 + 4 && disp.contains(st.pillRect), "reticle: pill below the focus ring");
    check(r.highlightRect({300, 470, 100, 25}, "Low"), "reticle: bottom-edge highlight accepted");
    check(r.state().pillRect.bottom() <= 470 - 4 && disp.contains(r.state().pillRect), "reticle: pill flips above near the bottom");
    check(r.pulseClick({610, 300}), "reticle: pulse accepted");
    st = r.state();
    check(st.label == "Low" && st.gliding, "reticle: pulse keeps label, glides to the click point first");
    r.advanceClock(glideDuration({600, 300}, {610, 300}) + 0.07);
    check(r.state().rippleActive && close(r.state().tip, {610, 300}), "reticle: click burst runs once landed");
    const QImage img = r.render(2);
    check(img.width() == 1600 && img.height() == 1000, "reticle: render at 2x");
    // The resting hand extends down-right of the fingertip, so up-left of the click point is all burst.
    const int burst = count(img, QRectF(580, 270, 28, 28), gold) + count(img, QRectF(580, 270, 28, 28), cream);
    const int glove = count(img, QRectF(600, 290, 50, 50), cream);
    const int cuff = count(img, QRectF(600, 290, 50, 50), gold);
    const int outline = count(img, QRectF(570, 260, 80, 80), ink);
    int blue = 0;
    for (int y = 0; y < img.height(); y += 2)
        for (int x = 0; x < img.width(); x += 2)
            blue += lapisBlue(img.pixelColor(x, y));
    check(burst > 200 && glove > 150 && cuff > 40 && outline > 200 && blue == 0,
        "reticle: click renders the comic starburst, cream glove, gold cuff, ink outline, no blue");
    std::printf("reticle click: burst=%d glove=%d cuff=%d outline=%d blue=%d\n", burst, glove, cuff, outline, blue);
    r.advanceClock(0.4);
    check(!r.state().rippleActive, "reticle: click burst ends");
    {
        // Lapis-blue theme: the accents shift to #4f8cff / #2d5aa8; glove and ink stay.
        CuaReticle b(CuaReticle::Mode::Offscreen, disp);
        b.setGoldTheme(false);
        b.moveTo({300, 200});
        b.highlightRect({280, 190, 120, 40}, "Clicking Send");
        b.pulseClick({300, 200});
        b.advanceClock(0.07);
        const QImage bi = b.render(2);
        const QRectF area(200, 120, 300, 200);
        const int bBlue = count(bi, area, lapisBlue), bGold = count(bi, area, gold), bCream = count(bi, area, cream);
        const int bInk = count(bi, area, ink);
        check(bBlue > 300 && bGold == 0 && bCream > 150 && bInk > 150, "reticle: setGoldTheme(false) swaps the accent to Lapis blue only");
        std::printf("reticle blue theme: blue=%d gold=%d cream=%d ink=%d\n", bBlue, bGold, bCream, bInk);
    }
    r.advanceClock(3.0);
    st = r.state();
    check(st.visible && st.focusRect.isEmpty() && st.pillRect.isEmpty(), "reticle: ring and pill fade after idle, pointer stays");
    r.hide();
    const QImage hidden = r.render(1);
    bool clear = !hidden.isNull();
    for (int y = 0; clear && y < hidden.height(); ++y)
        for (int x = 0; clear && x < hidden.width(); ++x)
            clear = hidden.pixelColor(x, y).alpha() == 0;
    check(!r.state().visible && clear, "reticle: hide clears the overlay");
    check(r.render(0).isNull() && r.render(std::numeric_limits<double>::quiet_NaN()).isNull(), "reticle: invalid render scale");
}
// A CuaHooks whose every side effect appends to `calls` and touches nothing real.
struct MockRig {
    QStringList calls;
    QVector<DisplayInfo> displays{{7, "primary", {0, 0, 1440, 900}, 2, true}, {9, "side", {1440, 0, 1920, 1080}, 1, false}};
    QVector<AxNode> nodes;
    QByteArray jpeg = "not-a-real-jpeg";
    GroundResult ground;
    uint32_t windowId = 4242, hudId = 5151;
    uint32_t capturedDisplay = 0;
    QVector<uint32_t> capturedExclude;
    int capturedWidth = 0;
    QString groundInstruction;
    bool groundRefine = false;
    QPointF clickPt, movePt, pulsePt, scrollPt;
    QRectF highlightRect;
    bool clickDouble = false;
    int scrollDy = 0;
    QString typed;
    bool clearOk = true;
    QString hotkeyCombo;
    QPointF dragFrom, dragTo;
    bool dragOk = true;
    // Multi-monitor: per-display AX nodes (falls back to `nodes`), on-screen windows, activation.
    QMap<uint32_t, QVector<AxNode>> nodesByDisplay;
    QVector<AppWindowInfo> windows;
    int selfPid = 0;
    bool activateOk = false;
    uint32_t activateDisplay = 0;
    int activatePid = 0;
    bool activateLaunched = false;
    QVector<AppWindowInfo> windowsAfterActivate;  // replaces `windows` once activation succeeds
    QStringList pauses;
    // Space switching: frontmostPid answers frontSeq in order, then repeats the last entry.
    bool frontmostHook = false;
    QVector<int> frontSeq;
    int frontPolls = 0;
    bool hideSelfOk = true;
    // HUD statuses, "<tone>|<status>" (kept out of `calls` so effect orderings stay focused).
    QStringList huds;
    CuaHooks hooks() {
        CuaHooks h;
        h.listDisplays = [this] { calls << "listDisplays"; return displays; };
        h.inspectDisplay = [this](uint32_t id) {
            calls << QString("inspectDisplay:%1").arg(id);
            return nodesByDisplay.isEmpty() ? nodes : nodesByDisplay.value(id);
        };
        h.listAppWindows = [this] { calls << "listAppWindows"; return windows; };
        h.activateApp = [this](const QString &hint) {
            calls << QString("activate:%1").arg(hint);
            AppActivation a;
            a.ok = activateOk;
            if (activateOk) {
                a.displayId = activateDisplay; a.pid = activatePid; a.launched = activateLaunched;
                if (!windowsAfterActivate.isEmpty()) windows = windowsAfterActivate;
            }
            return a;
        };
        if (frontmostHook) {
            h.frontmostPid = [this] {
                const int i = std::min<int>(frontPolls++, int(frontSeq.size()) - 1);
                return i >= 0 ? frontSeq[i] : 0;
            };
            h.hideSelf = [this] { calls << "hideSelf"; return hideSelfOk; };
        }
        h.reticleHud = [this](const QString &status, HudTone tone) {
            static const char *names[] = {"listening", "working", "done", "error"};
            huds << QString("%1|%2").arg(names[int(tone)], status);
        };
        h.hudWindowId = [this] { return hudId; };
        h.selfPid = [this] { return selfPid; };
        h.captureDisplayJpeg = [this](uint32_t id, int w, const QVector<uint32_t> &ex, QString *err) {
            calls << "capture";
            capturedDisplay = id; capturedWidth = w; capturedExclude = ex;
            if (jpeg.isEmpty() && err) *err = "denied";
            return jpeg;
        };
        h.groundPixel = [this](const QByteArray &, const QString &instr, bool refine) {
            calls << "ground"; groundInstruction = instr; groundRefine = refine; return ground;
        };
        h.clickAt = [this](QPointF p, bool dbl) { calls << "click"; clickPt = p; clickDouble = dbl; return ActionResult{true, "ok"}; };
        h.scrollAt = [this](QPointF p, int dy) { calls << "scroll"; scrollPt = p; scrollDy = dy; return ActionResult{true, "ok"}; };
        h.typeText = [this](const QString &t) { calls << "type"; typed = t; return ActionResult{true, "ok"}; };
        h.selectAllAndDelete = [this] { calls << "clear"; return ActionResult{clearOk, clearOk ? "Cleared" : "denied"}; };
        h.axPress = [this](int, const QString &, const QRectF &) { calls << "axPress"; return ActionResult{true, "ok"}; };
        h.pressHotkey = [this](const QString &c) { calls << "hotkey"; hotkeyCombo = c; return ActionResult{true, "ok"}; };
        h.dragBetween = [this](QPointF a, QPointF b) {
            calls << "drag"; dragFrom = a; dragTo = b; return ActionResult{dragOk, dragOk ? "ok" : "denied"};
        };
        h.reticleMove = [this](QPointF p, const QString &) { calls << "reticleMove"; movePt = p; };
        h.reticleHighlight = [this](QRectF r, const QString &) { calls << "reticleHighlight"; highlightRect = r; };
        h.reticlePulse = [this](QPointF p, const QString &) { calls << "reticlePulse"; pulsePt = p; };
        h.reticleHide = [this] { calls << "reticleHide"; };
        h.reticleWindowId = [this] { return windowId; };
        h.pause = [this](int ms) { pauses << QString::number(ms); };
        return h;
    }
    // The calls that matter for ordering (drops the read-only listing / inspection steps).
    QStringList effects() const {
        QStringList out;
        for (const QString &c : calls)
            if (c != "listDisplays" && c != "listAppWindows" && !c.startsWith("inspectDisplay")) out << c;
        return out;
    }
    QStringList inspected() const {
        QStringList out;
        for (const QString &c : calls) if (c.startsWith("inspectDisplay:")) out << c.mid(15);
        return out;
    }
};
AxNode axNode(int id, const QString &role, const QString &title, QRectF rect, const QString &value = {}) {
    return AxNode{id, role, title, value, "Example", 100 + id, rect, {"AXPress"}};
}
MockRig dualMonitorRig();
void executorTests() {
    // Matching.
    const QRectF r(100, 200, 80, 30);
    check(axMatchScore(axNode(1, "AXButton", "Send", r), "send") > axMatchScore(axNode(2, "AXStaticText", "Send", r), "send"),
        "exec: clickable role outranks static text on the same title");
    check(axMatchScore(axNode(1, "AXButton", "Send", {}), "send") == 0, "exec: empty rect never matches");
    check(axMatchScore(axNode(1, "AXButton", "Send", r), "") == 0, "exec: empty target never matches");
    check(axMatchScore(axNode(1, "AXTextField", "", r, "Search docs"), "search") > 0, "exec: value substring matches");
    check(bestAxMatch({axNode(1, "AXButton", "OK", r), axNode(2, "AXButton", "OK", r.translated(300, 0))}, "ok") == -1,
        "exec: tie between two distinct rects is ambiguous");
    check(bestAxMatch({axNode(1, "AXButton", "OK", r), axNode(2, "AXButton", "OK", r)}, "OK") == 0,
        "exec: same-rect duplicates are one target");

    // L1 hit: best role wins, highlight -> pulse -> click, no capture or grounding.
    {
        MockRig m;
        const QRectF text(10, 10, 60, 20), button(400, 300, 120, 40), hidden(0, 0, 0, 0);
        m.nodes = {axNode(1, "AXStaticText", "Send", text), axNode(2, "AXButton", "Send", button),
                   axNode(3, "AXButton", "Send", hidden), axNode(4, "AXGroup", "Unrelated", button)};
        CuaExecutor ex(m.hooks());
        const auto res = ex.clickTarget("send");
        check(res.ok && res.tier == "L1_AX", "exec L1: ok via AX");
        check(close(res.globalPt, button.center()) && close(m.clickPt, button.center()), "exec L1: clicks the AXButton centre");
        check(m.highlightRect == button && close(m.pulsePt, button.center()), "exec L1: ring on the button, pulse at its centre");
        check(m.effects() == QStringList({"reticleHighlight", "reticlePulse", "click"}), "exec L1: highlight -> pulse -> click only");
        check(!m.calls.contains("capture") && !m.calls.contains("ground"), "exec L1: no capture, no grounding");
        check(m.calls.contains("inspectDisplay:7"), "exec L1: inspects the primary display");
        check(!m.clickDouble, "exec L1: single click by default");
        MockRig d;
        d.nodes = {axNode(2, "AXButton", "Open", button)};
        CuaExecutor exd(d.hooks());
        exd.setPreferAxPress(true);
        check(exd.clickTarget("Open").ok && d.calls.contains("axPress") && !d.calls.contains("click"), "exec L1: AXPress when preferred");
        d.calls.clear();
        exd.clickTarget("Open", true);
        check(d.calls.contains("click") && d.clickDouble, "exec L1: double click always posts a visible click");
    }

    // L3 fallback: capture excludes the reticle, point_1000 maps onto a 2x Retina display.
    {
        MockRig m;
        m.nodes = {axNode(1, "AXGroup", "Toolbar", {0, 0, 1440, 40})};
        m.ground.ok = true; m.ground.hasPoint1000 = true; m.ground.point1000 = {250, 500};
        CuaExecutor ex(m.hooks());
        const auto res = ex.clickTarget("the gear icon", false, true);
        check(res.ok && res.tier == "L3_VLM", "exec L3: ok via VLM");
        check(m.capturedDisplay == 7 && m.capturedWidth == 1536 && m.capturedExclude == QVector<uint32_t>({4242, 5151}),
            "exec L3: captures the primary display at 1536 px, excluding the reticle and HUD windows");
        check(m.groundInstruction == "the gear icon" && m.groundRefine, "exec L3: instruction and refine_crop forwarded");
        // 1440x900 pt at 2x = 2880x1800 px, captured at 1536x960: 250/1000 -> 360 pt, 500/1000 -> 450 pt.
        check(close(res.globalPt, {360, 450}) && close(m.clickPt, {360, 450}), "exec L3: point_1000 -> Quartz global points");
        check(m.effects() == QStringList({"capture", "ground", "reticleMove", "reticlePulse", "click"}),
            "exec L3: capture -> ground -> move -> pulse -> click");
        check(close(m.movePt, {360, 450}) && close(m.pulsePt, {360, 450}), "exec L3: reticle lands on the click point");
        check(expectedCaptureSize(m.displays[0], 1536) == QSizeF(1536, 960), "exec L3: expected capture size 1536x960");
    }
    {
        // point_px on the capture with a grounder-reported image size.
        MockRig m;
        m.displays = {{3, "primary", {0, 0, 1512, 982}, 2, true}};
        m.ground.ok = true; m.ground.hasPointPx = true; m.ground.pointPx = {640, 100}; m.ground.imageSize = {1280, 831};
        CuaExecutor ex(m.hooks());
        const auto res = ex.clickTarget("Canvas brush");
        check(res.ok && std::abs(res.globalPt.x() - 756) < 1e-6 && std::abs(res.globalPt.y() - 100 * 982.0 / 831) < 1e-6,
            "exec L3: point_px uses the reported image size");
    }
    {
        // An off-display VLM point never clicks.
        MockRig m;
        m.ground.ok = true; m.ground.hasPointPx = true; m.ground.pointPx = {5000, 5000}; m.ground.imageSize = {1280, 800};
        CuaExecutor ex(m.hooks());
        const auto res = ex.clickTarget("ghost");
        check(!res.ok && res.tier == "ERR" && !m.calls.contains("click"), "exec L3: off-display point is rejected");
    }
    {
        // Grounder offline: clean failure, no click, no pulse, reticle hidden.
        MockRig m;
        m.ground.ok = false; m.ground.detail = "Grounder offline (Couldn't connect to server)";
        CuaExecutor ex(m.hooks());
        const auto res = ex.clickTarget("unlabeled icon");
        check(!res.ok && res.tier == "ERR" && res.detail.contains("offline"), "exec L3 offline: ok=false with the reason");
        check(!m.calls.contains("click") && !m.calls.contains("reticlePulse") && !m.calls.contains("axPress"),
            "exec L3 offline: no click posted");
        check(m.calls.contains("reticleHide"), "exec L3 offline: reticle hidden");
        MockRig c;
        c.jpeg.clear();
        CuaExecutor exc(c.hooks());
        const auto rc = exc.clickTarget("anything");
        check(!rc.ok && rc.detail.contains("denied") && !c.calls.contains("ground") && !c.calls.contains("click"),
            "exec L3: capture failure skips grounding and clicking");
        check(!exc.clickTarget("   ").ok, "exec: blank target rejected");
        // The real HTTP hook against a closed port fails fast and cleanly.
        qputenv("LAPIS_CUA_GROUNDER_URL", "http://127.0.0.1:9/ground");
        QElapsedTimer t; t.start();
        const GroundResult g = httpGroundPixel("x", "button", false);
        check(!g.ok && !g.detail.isEmpty() && t.elapsed() < 4500, "exec: real grounder hook offline -> ok=false within 4 s");
        check(grounderUrl() == "http://127.0.0.1:9/ground", "exec: LAPIS_CUA_GROUNDER_URL override");
        qunsetenv("LAPIS_CUA_GROUNDER_URL");
        check(grounderUrl() == "http://127.0.0.1:18120/ground", "exec: default grounder URL");
    }

    // Grounder JSON.
    {
        const auto a = parseGroundReply(R"({"point_1000":[120,880],"image_size":[1280,800]})");
        check(a.ok && a.hasPoint1000 && close(a.point1000, {120, 880}) && a.imageSize == QSizeF(1280, 800), "ground json: array point + size");
        const auto b = parseGroundReply(R"({"point_px":{"x":10,"y":20},"image_size":{"width":640,"height":400}})");
        check(b.ok && !b.hasPoint1000 && close(b.pointPx, {10, 20}) && b.imageSize == QSizeF(640, 400), "ground json: object point + size");
        check(!parseGroundReply(R"({"ok":false,"error":"no match"})").ok && parseGroundReply(R"({"ok":false,"error":"no match"})").detail == "no match",
            "ground json: explicit failure");
        check(!parseGroundReply("garbage").ok && !parseGroundReply(R"({"point_1000":[1200,5]})").ok, "ground json: junk and out-of-range rejected");
        const auto req = QJsonDocument::fromJson(buildGroundRequest("abc", "Send", true)).object();
        check(req.value("image_b64").toString() == "YWJj" && req.value("instruction").toString() == "Send"
                  && req.value("refine_crop").toBool(),
            "ground json: request carries image_b64, instruction, refine_crop");
    }

    // typeIntoTarget: click visibly, then type; nothing typed when the click fails.
    {
        MockRig m;
        m.nodes = {axNode(1, "AXTextField", "Search", {50, 60, 300, 28})};
        CuaExecutor ex(m.hooks());
        const auto res = ex.typeIntoTarget("search", "hello world");
        check(res.ok && res.tier == "L1_AX" && m.typed == "hello world", "exec type: ok, text typed");
        check(m.effects() == QStringList({"reticleHighlight", "reticlePulse", "click", "clear", "type"}),
            "exec type: highlight -> pulse -> click -> clear -> type");
        MockRig f;
        f.ground.ok = false;
        CuaExecutor exf(f.hooks());
        check(!exf.typeIntoTarget("missing", "secret").ok && !f.calls.contains("type") && !f.calls.contains("clear"),
            "exec type: no clearing or typing after a failed click");
    }

    // Clear-before-type: a retry replaces "John Gout" instead of inserting "John" into it.
    {
        MockRig m;
        m.nodes = {axNode(1, "AXTextField", "Search", {50, 60, 300, 28})};
        CuaExecutor ex(m.hooks());
        check(ex.typeIntoTarget("search", "John").ok, "clear: type ok");
        const qsizetype clearAt = m.calls.indexOf("clear"), typeAt = m.calls.indexOf("type");
        check(clearAt >= 0 && typeAt > clearAt && m.calls.count("clear") == 1, "clear: Cmd+A/Backspace runs once, before typing");
        check(clearAt > m.calls.indexOf("click") && m.pauses.indexOf("60") >= 0, "clear: after the click and the focus beat");
        MockRig n;
        n.nodes = m.nodes;
        CuaExecutor exn(n.hooks());
        check(exn.typeIntoTarget("search", "John", false).ok && !n.calls.contains("clear") && n.calls.contains("type"),
            "clear: clearFirst=false skips the clear");
        MockRig d;
        d.nodes = m.nodes;
        d.clearOk = false;
        CuaExecutor exd(d.hooks());
        const auto bad = exd.typeIntoTarget("search", "John");
        check(!bad.ok && !d.calls.contains("type") && bad.detail.contains("Clearing failed"),
            "clear: a failed clear never types into a dirty field");
        MockRig u;
        u.nodes = m.nodes;
        CuaHooks noClear = u.hooks();
        noClear.selectAllAndDelete = nullptr;
        CuaExecutor exu(noClear);
        check(exu.typeIntoTarget("search", "John").ok && u.typed == "John", "clear: an unset hook is a no-op");
    }

    // Post-action observation: after a click, the planner sees the display's AX tree again.
    {
        MockRig m;
        m.selfPid = 500;
        m.nodes = {axNode(1, "AXButton", "Send", {400, 300, 120, 40}),
                   AxNode{9, "AXButton", "Lapis secret", {}, "lapis_studio", 500, {10, 10, 80, 20}, {"AXPress"}}};
        CuaExecutor ex(m.hooks());
        const auto res = ex.clickTarget("Send");
        check(res.ok && res.observation.contains("\"title\":\"Send\""), "observe: clickTarget fills observation");
        check(!res.observation.contains("Lapis secret"), "observe: Lapis's own nodes are left out");
        check(m.pauses.contains(QString::number(kObserveSettleMs)) && m.inspected().size() == 2
                  && m.calls.lastIndexOf("inspectDisplay:7") > m.calls.indexOf("click"),
            "observe: settles, then re-reads the display after the click");
        check(res.observation.size() <= kPostActionChars, "observe: post-action budget");
        const auto typed = ex.typeIntoTarget("Send", "x");
        check(typed.ok && !typed.observation.isEmpty(), "observe: typeIntoTarget fills observation");
        const auto scrolled = ex.scrollTarget("Send", -3);
        check(scrolled.ok && !scrolled.observation.isEmpty(), "observe: scrollTarget fills observation");
        MockRig f;
        f.ground.ok = false;
        f.nodes = {axNode(2, "AXButton", "Compose", {400, 300, 120, 40})};
        CuaExecutor exf(f.hooks());
        const auto miss = exf.clickTarget("Send");
        check(!miss.ok && miss.observation.contains("Compose"), "observe: a miss still shows the model what is there");
    }

    // observeApp: activates the app, reads its display, Lapis filtered, the app's nodes first.
    {
        MockRig m = dualMonitorRig();
        m.activateOk = true; m.activateDisplay = 2; m.activatePid = 700;
        m.windowsAfterActivate = {{500, "lapis_studio", 1, {0, 0, 1920, 1080}, true},
                                  {700, "Discord", 2, {1920, 25, 1920, 1055}, true}};
        m.nodesByDisplay[1] = {AxNode{1, "AXTextField", "Ask Lapis", {}, "lapis_studio", 500, {10, 10, 300, 30}, {}}};
        m.nodesByDisplay[2] = {AxNode{2, "AXButton", "Dock item", {}, "Dock", 300, {1930, 1040, 40, 40}, {"AXPress"}},
                               AxNode{3, "AXTextField", "Find or start a conversation", {}, "Discord", 700, {1940, 40, 240, 28}, {}},
                               AxNode{4, "AXStaticText", "John Gout", {}, "Discord", 700, {1940, 90, 200, 20}, {}},
                               AxNode{5, "AXTextField", "Lapis overlay", {}, "lapis_studio", 500, {2000, 500, 50, 20}, {}}};
        CuaExecutor ex(m.hooks());
        const auto r = ex.observeApp("Discord");
        check(r.ok && r.tier == "L1_OBS", "observeApp: ok, tier L1_OBS");
        check(m.calls.contains("activate:Discord") && m.inspected().value(0) == "2", "observeApp: activates Discord, reads its display");
        check(r.observation.contains("Find or start a conversation") && r.observation.contains("John Gout"),
            "observeApp: Discord's search bar and DM list reach the model");
        check(!r.observation.contains("lapis_studio") && !r.observation.contains("Ask Lapis"), "observeApp: selfPid filtered out");
        check(r.observation.indexOf("Discord") < r.observation.indexOf("Dock item"), "observeApp: the app's nodes come first");
        check(r.observation.size() <= kObserveChars && r.detail.contains("Discord"), "observeApp: budget and detail");
        check(!m.calls.contains("click") && !m.calls.contains("type") && !m.calls.contains("capture"),
            "observeApp: reads only, no input, no capture");
        std::printf("observeApp detail: %s\n", qPrintable(r.detail));
    }
    {
        // Naming Lapis keeps Lapis's own nodes.
        MockRig m;
        m.selfPid = 500;
        m.nodes = {AxNode{1, "AXTextField", "Ask Lapis", {}, "lapis_studio", 500, {10, 10, 300, 30}, {}}};
        CuaExecutor ex(m.hooks());
        check(ex.observeApp("Lapis").observation.contains("Ask Lapis"), "observeApp: naming Lapis keeps its nodes");
        MockRig e;
        e.selfPid = 500;
        e.nodes = m.nodes;
        CuaExecutor exe(e.hooks());
        const auto bare = exe.observeApp("");
        check(bare.ok && bare.observation.isEmpty() && !e.calls.join(",").contains("activate"),
            "observeApp: no app reads the front display without activating, Lapis filtered");
    }

    // scrollTarget: glide to the target, then scroll there; centre fallback when it is not found.
    {
        MockRig m;
        const QRectF list(200, 100, 400, 600);
        m.nodes = {axNode(1, "AXButton", "Results", list)};
        CuaExecutor ex(m.hooks());
        const auto res = ex.scrollTarget("results", -5);
        check(res.ok && res.tier == "L1_AX" && close(m.scrollPt, list.center()) && m.scrollDy == -5, "exec scroll: scrolls the AX target");
        check(m.effects() == QStringList({"reticleHighlight", "reticleMove", "scroll"}), "exec scroll: highlight -> move -> scroll");
        MockRig c;
        c.ground.ok = false;
        CuaExecutor exc(c.hooks());
        const auto rc = exc.scrollTarget("nowhere", 3);
        check(rc.ok && rc.tier == "CENTER" && close(c.scrollPt, {720, 450}), "exec scroll: unknown target scrolls the display centre");
        check(c.effects().mid(c.effects().indexOf("reticleMove")) == QStringList({"reticleMove", "scroll"}), "exec scroll: centre glide before scroll");
        MockRig e;
        CuaExecutor exe(e.hooks());
        const auto re = exe.scrollTarget("", 2);
        check(re.ok && !e.calls.contains("capture") && e.effects() == QStringList({"reticleMove", "scroll"}), "exec scroll: empty target skips grounding");
        check(!exe.scrollTarget("x", 0).ok, "exec scroll: zero delta rejected");
    }

    // runOnMainThread: direct on the main thread, marshalled from a worker.
    {
        QThread *mainThread = QThread::currentThread();
        QThread *ranOn = nullptr;
        runOnMainThread([&] { ranOn = QThread::currentThread(); });
        check(ranOn == mainThread, "exec: main-thread call runs inline");
        ranOn = nullptr;
        QEventLoop loop;
        auto *worker = QThread::create([&] { runOnMainThread([&] { ranOn = QThread::currentThread(); }); });
        QObject::connect(worker, &QThread::finished, &loop, &QEventLoop::quit);
        worker->start();
        loop.exec();
        worker->wait();
        delete worker;
        check(ranOn == mainThread, "exec: worker call is marshalled onto the main thread");
    }
}
AxNode appNode(const QString &role, const QString &title, QRectF rect, int pid, const QString &app) {
    return AxNode{1, role, title, {}, app, pid, rect, {"AXPress"}};
}
// Lapis fullscreen on display 1 (primary), Chrome on display 2 to its right.
MockRig dualMonitorRig() {
    MockRig m;
    m.displays = {{1, "left", {0, 0, 1920, 1080}, 1, true}, {2, "right", {1920, 0, 1920, 1080}, 1, false}};
    m.selfPid = 500;
    m.windows = {{500, "lapis_studio", 1, {0, 0, 1920, 1080}, true},
                 {600, "Google Chrome", 2, {1920, 25, 1920, 1055}, true}};
    return m;
}
void multiMonitorTests() {
    // App hints and stripping.
    check(extractAppHint("Chrome address bar") == "Chrome", "hint: Chrome address bar -> Chrome");
    check(extractAppHint("Send button in Discord") == "Discord", "hint: Send button in Discord -> Discord");
    check(extractAppHint("address bar in Google Chrome") == "Chrome", "hint: Google Chrome -> Chrome");
    check(extractAppHint("VS Code terminal tab") == "VS Code", "hint: leftmost app wins (VS Code before Terminal)");
    check(extractAppHint("the code block") == "" && extractAppHint("send") == "" && extractAppHint("searchbar") == "",
        "hint: generic words and substrings name no app");
    check(extractAppHint("Spotify play button") == "Spotify" && extractAppHint("Slack message box") == "Slack"
              && extractAppHint("Finder sidebar") == "Finder" && extractAppHint("Safari tab") == "Safari",
        "hint: Spotify, Slack, Finder, Safari");
    check(stripAppHint("Send button in Discord", "Discord") == "Send button", "strip: trailing 'in Discord'");
    check(stripAppHint("Chrome address bar", "Chrome") == "address bar", "strip: leading app name");
    check(stripAppHint("address bar inside of the Chrome window", "Chrome") == "address bar", "strip: 'inside of the ... window'");
    check(stripAppHint("Discord", "Discord") == "Discord", "strip: never empties the target");
    check(canonicalAppName("chrome") == "Google Chrome" && canonicalAppName("vscode") == "Visual Studio Code"
              && canonicalAppName("code") == "Visual Studio Code" && canonicalAppName(" Discord ") == "Discord",
        "alias: chrome, vscode, code");
    check(appNameMatches("Google Chrome", "Chrome") && appNameMatches("Visual Studio Code", "VS Code")
              && appNameMatches("discord", "Discord") && !appNameMatches("Safari", "Chrome") && !appNameMatches("", "Chrome"),
        "alias: running-app name matching");
    check(bestAxMatch({axNode(1, "AXButton", "Send", {10, 10, 50, 20})}, "Send button") == 0, "match: role noun dropped");
    check(axMatchScore(axNode(1, "AXTextField", "Address and search bar", {10, 10, 500, 30}), "address bar") > 0
              && axMatchScore(axNode(1, "AXButton", "Address and search bar", {10, 10, 500, 30}), "address bar") == 0,
        "match: address bar synonym needs an editable field");

    // "\n" normalization.
    check(normalizeTypedText("x.com\\n") == "x.com\n" && normalizeTypedText("a\\r\\nb") == "a\nb"
              && normalizeTypedText("plain") == "plain",
        "type: literal \\n becomes a real newline");

    // L1 across displays: Lapis's own field on display 1 is skipped, Chrome's on display 2 wins.
    {
        MockRig m = dualMonitorRig();
        m.activateOk = true; m.activateDisplay = 2;
        const QRectF omni(2100, 60, 900, 32);
        m.nodesByDisplay[1] = {appNode("AXTextField", "Address bar", {600, 20, 700, 40}, 500, "lapis_studio")};
        m.nodesByDisplay[2] = {appNode("AXTextField", "Address and search bar", omni, 600, "Google Chrome")};
        CuaExecutor ex(m.hooks());
        const auto res = ex.typeIntoTarget("Chrome address bar", "x.com\\n");
        check(res.ok && res.tier == "L1_AX", "multi L1: ok via AX");
        check(close(m.clickPt, omni.center()) && m.clickPt.x() >= 1920, "multi L1: clicks Chrome's omnibox on display 2");
        check(m.typed == "x.com\n" && res.detail.contains("Return"), "multi L1: typed x.com then Return");
        check(m.calls.contains("activate:Chrome") && m.pauses.contains("220"), "multi L1: Chrome activated, 220 ms settle");
        check(m.inspected().value(0) == "2", "multi L1: the Chrome display is inspected first");
        check(m.effects() == QStringList({"activate:Chrome", "reticleHighlight", "reticlePulse", "click", "clear", "type"}),
            "multi L1: activate -> highlight -> pulse -> click -> clear -> type");
    }
    {
        // No hint: Lapis's matching node is never a candidate; the other display is searched first.
        MockRig m = dualMonitorRig();
        const QRectF field(2300, 400, 400, 30);
        m.nodesByDisplay[1] = {appNode("AXTextField", "Search", {600, 20, 700, 40}, 500, "lapis_studio")};
        m.nodesByDisplay[2] = {appNode("AXTextField", "Search", field, 700, "Finder")};
        CuaExecutor ex(m.hooks());
        const auto res = ex.clickTarget("search");
        check(res.ok && close(m.clickPt, field.center()), "multi L1: non-Lapis display first, Lapis node skipped");
        check(m.inspected().value(0) == "2" && !m.calls.join(",").contains("activate"), "multi L1: no hint, no activation");
        MockRig only = dualMonitorRig();
        only.nodesByDisplay[1] = {appNode("AXTextField", "Search", {600, 20, 700, 40}, 500, "lapis_studio")};
        only.ground.ok = false;
        CuaExecutor exo(only.hooks());
        check(!exo.clickTarget("search").ok && !only.calls.contains("click"), "multi L1: a Lapis-only match is never clicked");
        MockRig self = dualMonitorRig();
        self.nodesByDisplay[1] = {appNode("AXButton", "Lapis settings", {10, 10, 80, 20}, 500, "lapis_studio")};
        CuaExecutor exs(self.hooks());
        check(exs.clickTarget("Lapis settings").ok && close(self.clickPt, {50, 20}), "multi L1: a target naming Lapis may hit Lapis");
    }

    // L3: capture the Chrome display, not fullscreen Lapis, and map onto display 2's rect.
    {
        MockRig m = dualMonitorRig();
        m.activateOk = true; m.activateDisplay = 0;  // activated, display found via the window list
        m.ground.ok = true; m.ground.hasPoint1000 = true; m.ground.point1000 = {500, 50};
        CuaExecutor ex(m.hooks());
        const auto res = ex.typeIntoTarget("Chrome address bar", "x.com\\n");
        check(res.ok && res.tier == "L3_VLM", "multi L3: ok via VLM");
        check(m.capturedDisplay == 2, "multi L3: captures display 2 (Chrome), not display 1 (Lapis)");
        // 1920x1080 at 1x captured at 1536x864: 500/1000 -> 960 pt, 50/1000 -> 54 pt, offset by x=1920.
        check(close(res.globalPt, {2880, 54}) && m.clickPt.x() >= 1920 && close(m.movePt, {2880, 54}),
            "multi L3: point maps onto display 2's global rect");
        check(m.groundInstruction == "Chrome address bar" && m.typed == "x.com\n", "multi L3: full target grounded, Return typed");
    }
    {
        // No hint and no app info beyond Lapis: any non-Lapis display beats Lapis's.
        MockRig m = dualMonitorRig();
        m.windows = {{500, "lapis_studio", 1, {0, 0, 1920, 1080}, true}};
        m.ground.ok = true; m.ground.hasPoint1000 = true; m.ground.point1000 = {100, 100};
        CuaExecutor ex(m.hooks());
        check(ex.clickTarget("the gear icon").ok && m.capturedDisplay == 2 && m.clickPt.x() >= 1920,
            "multi L3: avoids the Lapis display without a hint");
        // Single monitor with Lapis fullscreen: the primary after activation is all there is.
        MockRig s;
        s.displays = {{1, "only", {0, 0, 1440, 900}, 2, true}};
        s.selfPid = 500;
        s.windows = {{500, "lapis_studio", 1, {0, 0, 1440, 900}, true}};
        s.activateOk = true;
        s.ground.ok = true; s.ground.hasPoint1000 = true; s.ground.point1000 = {500, 500};
        CuaExecutor exs(s.hooks());
        check(exs.clickTarget("Send button in Discord").ok && s.capturedDisplay == 1 && s.calls.contains("activate:Discord"),
            "multi L3: single monitor activates the app and captures the primary");
        // Scroll's centre fallback also lands on the non-Lapis display.
        MockRig c = dualMonitorRig();
        CuaExecutor exc(c.hooks());
        check(exc.scrollTarget("", 3).ok && close(c.scrollPt, {2880, 540}), "multi scroll: centre of the non-Lapis display");
    }
}
void hudTests() {
    // Pure placement: centred, 44 pt above the bottom edge, on any display origin.
    check(cuaHudRect({0, 0, 1440, 900}) == QRectF(550, 818, 340, 38), "hud: bottom-centre of a 1440x900 display");
    check(cuaHudRect({1440, -200, 1920, 1080}) == QRectF(2230, 798, 340, 38), "hud: offset secondary display");
    check(cuaHudRect({}).isEmpty(), "hud: no display, no rect");

    // Offscreen lifecycle: never a window, time moves only through advanceClock.
    const QRectF disp(0, 0, 1440, 900);
    CuaReticle r(CuaReticle::Mode::Offscreen, disp);
    check(r.hudWindowNumber() == 0 && !r.state().hudVisible, "hud: offscreen has no window, starts hidden");
    r.showHud(QStringLiteral("Listening…"), true);
    auto st = r.state();
    check(st.hudVisible && st.hudTone == HudTone::Listening && st.hudStatus == QStringLiteral("Listening…")
              && st.hudRect == QRectF(550, 818, 340, 38) && st.hudAlpha < 0.01,
        "hud: showHud(listening) appears bottom-centre, fading in");
    check(!st.visible, "hud: the cursor stays hidden while only the HUD shows");
    r.advanceClock(0.3);
    check(std::abs(r.state().hudAlpha - 1) < 1e-9, "hud: fully in after 0.3 s");
    r.advanceClock(60);
    check(r.state().hudVisible, "hud: listening never times out on its own");
    // Status updates keep the same pinned rect, even when the cursor moves to the far corner.
    r.moveTo({1400, 40}, "Gliding");
    r.showHud(QStringLiteral("Chrome · Clicking address bar"), HudTone::Working);
    st = r.state();
    check(st.hudTone == HudTone::Working && st.hudRect == QRectF(550, 818, 340, 38) && std::abs(st.hudAlpha - 1) < 1e-9,
        "hud: a new status updates in place, no re-fade");
    // Rendered: Lapis espresso-obsidian #0d0c0b body, #e09f3e gold star mark and gold rim, no cold
    // blue anywhere in it, and nothing above it.
    const QImage img = r.render(2);
    const auto gold = [](const QColor &c) {
        return c.alpha() > 150 && c.red() > 180 && c.green() > 110 && c.green() < 190 && c.red() > c.blue() + 110;
    };
    int body = 0, dot = 0, rim = 0, blue = 0, strayAbove = 0;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); x += 2) {
            const QColor c = img.pixelColor(x, y);
            const QPointF pt(x / 2.0, y / 2.0);
            if (QRectF(560, 822, 320, 30).contains(pt)) {
                body += c.alpha() > 230 && c.red() < 20 && c.green() < 20 && c.blue() < 20;
                dot += gold(c) && pt.x() < 580;
                blue += c.blue() > c.red() + 30;
            } else if (QRectF(600, 817, 240, 3).contains(pt)) {
                rim += gold(c);
            } else if (QRectF(400, 600, 640, 180).contains(pt)) {
                strayAbove += c.alpha() > 0;
            }
        }
    check(body > 1000 && dot > 4 && rim > 100 && blue == 0 && strayAbove == 0,
        "hud: render draws the obsidian capsule, gold mark and gold rim, only there");
    std::printf("hud render: body=%d dot=%d rim=%d blue=%d stray=%d\n", body, dot, rim, blue, strayAbove);
    {
        // Lapis-blue theme: the HUD rim and mark follow the accent.
        CuaReticle b(CuaReticle::Mode::Offscreen, disp);
        b.setGoldTheme(false);
        b.showHud(QStringLiteral("Working"), HudTone::Working);
        b.advanceClock(0.3);
        const QImage bi = b.render(2);
        int bBlue = 0, bGold = 0;
        for (int y = int(818 * 2); y < int(856 * 2); ++y)
            for (int x = int(550 * 2); x < int(890 * 2); ++x) {
                const QColor c = bi.pixelColor(x, y);
                bBlue += c.alpha() > 100 && c.blue() > 180 && c.blue() > c.red() + 60;
                bGold += gold(c);
            }
        check(bBlue > 100 && bGold == 0, "hud: setGoldTheme(false) turns the rim and mark Lapis blue");
        std::printf("hud blue theme: blue=%d gold=%d\n", bBlue, bGold);
    }
    // Hiding the cursor leaves the HUD alone.
    r.hide();
    check(!r.state().visible && r.state().hudVisible, "hud: cursor hide() keeps the HUD up");
    // Done: holds 2.5 s, then fades out over 0.4 s.
    r.showHud(QStringLiteral("Done · Chrome address bar"), HudTone::Done);
    r.advanceClock(2.4);
    st = r.state();
    check(st.hudVisible && st.hudTone == HudTone::Done && std::abs(st.hudAlpha - 1) < 1e-9, "hud: done holds for 2.5 s");
    r.advanceClock(0.3);
    st = r.state();
    check(st.hudVisible && st.hudAlpha > 0 && st.hudAlpha < 1, "hud: done fades out after 2.5 s");
    // A new action mid fade-out takes the HUD back without a pop.
    r.showHud(QStringLiteral("Switching to Discord…"), HudTone::Working);
    const double resumed = r.state().hudAlpha;
    r.advanceClock(5);
    check(resumed > 0.05 && r.state().hudVisible && std::abs(r.state().hudAlpha - 1) < 1e-9,
        "hud: a new status mid fade-out resumes and stays");
    r.hideHud();
    r.advanceClock(0.2);
    check(r.state().hudVisible && r.state().hudAlpha < 1, "hud: hideHud fades out");
    r.advanceClock(0.25);
    st = r.state();
    check(!st.hudVisible && st.hudRect.isEmpty() && st.hudStatus.isEmpty(), "hud: gone after the fade");
    r.hideHud();  // hiding a hidden HUD is a no-op
    r.showHud(QStringLiteral("Error"), HudTone::Error);
    r.hideHud(1.0);
    r.advanceClock(0.9);
    check(r.state().hudVisible, "hud: hideHud(1 s) waits");
    r.advanceClock(0.6);
    check(!r.state().hudVisible, "hud: delayed hide completes");
    // Error lingers 2.5 s like Done.
    r.showHud(QStringLiteral("Couldn't find Send"), HudTone::Error);
    r.advanceClock(2.45);
    check(r.state().hudVisible && r.state().hudTone == HudTone::Error, "hud: error holds");
    r.advanceClock(0.5);
    check(!r.state().hudVisible, "hud: error fades out on its own");
}
// Lapis fullscreen on the only display; Chrome comes forward after a few polls.
MockRig singleScreenRig() {
    MockRig m;
    m.displays = {{1, "only", {0, 0, 1512, 982}, 2, true}};
    m.selfPid = 500;
    m.windows = {{500, "lapis_studio", 1, {0, 0, 1512, 982}, true}};
    m.frontmostHook = true;
    return m;
}
int pauseTotal(const QStringList &pauses, const QString &ms) { return int(pauses.count(ms)) * ms.toInt(); }
void spaceSwitchTests() {
    {
        // Activation waits for Chrome to be frontmost (3 polls), then the 180 ms slide, then looks.
        MockRig m = singleScreenRig();
        m.activateOk = true; m.activatePid = 600; m.activateDisplay = 0;
        m.windowsAfterActivate = {{600, "Google Chrome", 1, {0, 25, 1512, 957}, true}};
        m.frontSeq = {500, 500, 500, 600};
        m.ground.ok = true; m.ground.hasPoint1000 = true; m.ground.point1000 = {300, 60};
        CuaExecutor ex(m.hooks());
        const auto res = ex.typeIntoTarget("Chrome address bar", "x.com\\n");
        check(res.ok && res.tier == "L3_VLM" && m.capturedDisplay == 1, "space: single screen types into Chrome via L3");
        check(m.pauses.count("40") == 3 && m.pauses.contains("180") && !m.pauses.contains("220"),
            "space: polls frontmost every 40 ms, then 180 ms for the slide (no fixed 220 ms)");
        const int settled = int(m.pauses.indexOf("180"));
        check(settled >= 3 && m.pauses.mid(0, settled) == QStringList({"40", "40", "40"}),
            "space: the slide settle follows the frontmost polls");
        check(!m.calls.contains("hideSelf"), "space: Chrome in front, Lapis is not hidden");
        check(m.capturedExclude == QVector<uint32_t>({4242, 5151}), "space: capture excludes the cursor and the HUD bar");
        check(m.huds == QStringList({"working|Switching to Chrome…", "working|Chrome · Clicking address bar",
                                     "working|Chrome · Typing x.com ↩", "done|Done · Typed x.com ↩"}),
            "space: HUD lifecycle switch -> click -> type -> done");
        std::printf("hud statuses: %s\n", qPrintable(m.huds.join(" / ")));
    }
    {
        // Chrome never comes forward: bounded at 900 ms, no slide settle, still proceeds.
        MockRig m = singleScreenRig();
        m.activateOk = true; m.activatePid = 600;
        m.frontSeq = {500};
        m.hideSelfOk = false;
        m.ground.ok = true; m.ground.hasPoint1000 = true; m.ground.point1000 = {500, 500};
        CuaExecutor ex(m.hooks());
        ex.clickTarget("Chrome address bar");
        const int polled = pauseTotal(m.pauses, "40");
        check(polled >= 860 && polled <= 900, "space: frontmost wait is bounded at 900 ms");
        std::printf("space: frontmost timeout after %d ms of polling\n", polled);
        // Lapis still frontmost after a failed bring-forward: it hides itself (here refused).
        check(m.calls.contains("hideSelf"), "space: Lapis still in front after activation -> hide Lapis");
        MockRig l = singleScreenRig();
        l.activateOk = true; l.activatePid = 600; l.activateLaunched = true;
        l.frontSeq = {0};
        l.hideSelfOk = false;
        CuaExecutor exl(l.hooks());
        exl.clickTarget("Chrome address bar");
        const int launchPolled = pauseTotal(l.pauses, "40");
        check(launchPolled >= 2400 && launchPolled <= 2600, "space: a cold launch waits up to 2.5 s");
    }
    {
        // No hint, Lapis frontmost on the only screen: hide Lapis, wait for another app, then look.
        MockRig m = singleScreenRig();
        m.frontSeq = {500, 500, 700};  // check, then two polls after hideSelf
        m.ground.ok = true; m.ground.hasPoint1000 = true; m.ground.point1000 = {500, 500};
        CuaExecutor ex(m.hooks());
        const auto res = ex.clickTarget("the gear icon");
        check(res.ok && m.calls.contains("hideSelf") && m.calls.indexOf("hideSelf") < m.calls.indexOf("capture"),
            "space: single screen hides fullscreen Lapis before capturing");
        check(m.pauses.contains("180") && m.huds.value(0) == "working|Switching to the desktop…",
            "space: waits for the slide and says so on the HUD");
        check(m.huds.last() == "done|Done · the gear icon", "space: HUD ends on done");
        // Another app already in front: nothing hidden.
        MockRig o = singleScreenRig();
        o.frontSeq = {700};
        o.ground = m.ground;
        CuaExecutor exo(o.hooks());
        check(exo.clickTarget("the gear icon").ok && !o.calls.contains("hideSelf"), "space: Lapis not frontmost, not hidden");
        // A target naming Lapis keeps it.
        MockRig s = singleScreenRig();
        s.frontSeq = {500};
        s.nodes = {AxNode{1, "AXButton", "Lapis settings", {}, "lapis_studio", 500, {10, 10, 80, 20}, {"AXPress"}}};
        CuaExecutor exs(s.hooks());
        check(exs.clickTarget("Lapis settings").ok && !s.calls.contains("hideSelf"), "space: a target naming Lapis never hides it");
        // Two displays: never hides (routing already avoids Lapis's display).
        MockRig d = dualMonitorRig();
        d.frontmostHook = true;
        d.frontSeq = {500};
        d.ground = m.ground;
        CuaExecutor exd(d.hooks());
        check(exd.clickTarget("the gear icon").ok && !d.calls.contains("hideSelf"), "space: multi-monitor never hides Lapis");
    }
    {
        // Failure reaches the HUD as an error.
        MockRig m;
        m.ground.ok = false; m.ground.detail = "Grounder offline";
        CuaExecutor ex(m.hooks());
        ex.clickTarget("unlabeled icon");
        check(m.huds.value(0) == "working|Clicking unlabeled icon" && m.huds.last() == "error|Couldn't find unlabeled icon",
            "hud: a miss ends on an error status");
        MockRig sc;
        CuaExecutor exs(sc.hooks());
        exs.scrollTarget("", -3);
        check(sc.huds.last() == "done|Done · Scrolled", "hud: scroll ends on done");
    }
    // Launch support: known bundle ids, and canonical names for the aliases.
    check(knownBundleId("Google Chrome") == "com.google.Chrome" && knownBundleId("discord") == "com.hnc.Discord"
              && knownBundleId("Nonexistent App").isEmpty(),
        "launch: known bundle ids");
    check(canonicalAppName("safari") == "Safari" && canonicalAppName("slack") == "Slack" && canonicalAppName("terminal") == "Terminal"
              && canonicalAppName("spotify") == "Spotify" && canonicalAppName("finder") == "Finder",
        "launch: aliases map to the installed app names");
}
void liveTests() {
    const auto displays = listDisplays();
    check(!displays.isEmpty(), "live display enumeration");
    int primary = 0;
    for (const auto &d : displays) {
        check(d.displayId != 0 && !d.uuid.isEmpty() && d.globalPoints.isValid() && d.scale > 0 && std::isfinite(d.scale), "live display geometry");
        primary += d.primary;
    }
    check(primary == 1, "exactly one primary display");
    const auto permissions = checkPermissions();
    std::printf("Permission preflight: accessibility=%d screenRecording=%d inputMonitoring=%d (no prompts)\n",
        permissions.accessibility, permissions.screenRecording, permissions.inputMonitoring);
    check(inspectApp(-1).isEmpty(), "invalid PID rejected");
    check(inspectApp(1, 0).isEmpty(), "zero node budget");
    check(inspectApp(1, 20, -1).isEmpty(), "negative depth rejected");
    check(inspectApp(1, 20, 8, 0).isEmpty(), "zero time budget");
    check(inspectApp(1, 20, 8, std::numeric_limits<double>::quiet_NaN()).isEmpty(), "invalid time budget");
    check(inspectDisplay(0).isEmpty(), "invalid display rejected");
    check(!clickAt({std::numeric_limits<double>::quiet_NaN(), 0}).ok, "invalid click rejected before actuation");
    check(!scrollAt({std::numeric_limits<double>::infinity(), 0}, 1).ok, "invalid scroll rejected before actuation");
    check(!axPress(1, "", {}).ok, "underspecified AX target rejected");
    QString error;
    check(captureDisplayJpeg(0, 0, 0, &error).isEmpty() && !error.isEmpty(), "invalid capture width returns error");
    check(captureDisplayJpeg(0, 1280, 0, &error).isEmpty() && error.contains("worker"), "main-thread capture rejected without prompt");
}
void cancellationTests() {
    cancelFlag.store(true);
    QElapsedTimer timer; timer.start();
    check(!clickAt({0, 0}).ok, "cancel click");
    check(!clickAt({0, 0}, true).ok, "cancel double click");
    check(!scrollAt({0, 0}, 100).ok, "cancel scroll");
    check(!typeText("test").ok, "cancel typing");
    check(!axPress(1, "AXButton", {0, 0, 10, 10}).ok, "cancel AX press");
    check(!axSetValue(1, "AXTextField", {0, 0, 10, 10}, "test").ok, "cancel AX set");
    check(inspectApp(1).isEmpty() && inspectDisplay(0).isEmpty(), "cancel inspections");
    QString error;
    check(captureDisplayJpeg(0, 1280, 0, &error).isEmpty() && error == "Cancelled", "cancel capture");
    std::thread worker([] { for (int i = 0; i < 100; ++i) if (typeText("test").ok) std::abort(); });
    worker.join();
    check(timer.elapsed() < 100, "cancelled actions return immediately");
    check(cancelFlag.load(), "cancellation remains latched");
    cancelFlag.store(false);
}
void captureSmokeTest() {
    // Opt-in only: inspect JPEG metadata in memory, never persist screen contents.
    if (!checkPermissions().screenRecording) {
        check(false, "capture smoke requires existing screen recording permission");
        return;
    }
    const auto displays = listDisplays();
    if (displays.isEmpty()) { check(false, "capture smoke requires a display"); return; }
    QByteArray jpeg;
    QString error, invalidError;
    QEventLoop loop;
    auto *worker = QThread::create([&] {
        captureDisplayJpeg(0, 1280, 0, &invalidError);
        jpeg = captureDisplayJpeg(displays.first().displayId, 1280, 0, &error);
    });
    QObject::connect(worker, &QThread::finished, &loop, &QEventLoop::quit);
    worker->start();
    loop.exec();
    worker->wait();
    delete worker;
    check(invalidError == "Display not found", "worker capture rejects unknown display");
    if (!error.isEmpty()) std::printf("Capture error: %s\n", qPrintable(error));
    const auto image = QImage::fromData(jpeg, "JPEG");
    check(error.isEmpty() && !image.isNull(), "live ScreenCaptureKit JPEG decodes");
    check(image.width() > 0 && image.width() <= 1280 && image.height() > 0, "live capture width bounded");
    if (!image.isNull()) std::printf("Capture smoke: %d x %d JPEG (memory only)\n", image.width(), image.height());
}
}
// Every non-empty JSONL line of the trace log, parsed.
QVector<QJsonObject> traceLines(const QString &dir) {
    QFile f(dir + "/traces.jsonl");
    QVector<QJsonObject> out;
    if (!f.open(QIODevice::ReadOnly)) return out;
    for (const QByteArray &l : f.readAll().split('\n'))
        if (!l.trimmed().isEmpty()) out << QJsonDocument::fromJson(l).object();
    return out;
}
int traceFiles(const QString &dir) {
    return int(QDir(dir + "/frames").entryList(QDir::Files).size()) + (QFile::exists(dir + "/traces.jsonl") ? 1 : 0);
}
bool intArray(const QJsonValue &v, const QVector<int> &want) {
    const QJsonArray a = v.toArray();
    if (a.size() != want.size()) return false;
    for (int i = 0; i < want.size(); ++i)
        if (a.at(i).toInt(-1) != want[i]) return false;
    return true;
}
void traceTests() {
    check(traceBlocked(true, {}) && traceBlocked(false, {"x", "Slack · INFINITY ops"}) && !traceBlocked(false, {"Chrome", ""}),
        "trace: redacted or any 'infinity' field blocks");
    check(close(toDisplay1000(QPointF(1440 + 960, 540), {9, "side", {1440, 0, 1920, 1080}, 1, false}), {500, 500}),
        "trace: point_1000 is relative to the step's own display");
    const QRectF button(400, 300, 120, 40);  // centre (460, 320) on the 1440x900 primary
    // L1 click: before + after frames, one JSONL line with point_1000 and the AX bbox_1000.
    {
        QTemporaryDir tmp;
        MockRig m;
        m.nodes = {axNode(2, "AXButton", "Send", button)};
        CuaHooks h = m.hooks();
        h.traceDir = tmp.path();
        CuaExecutor ex(h);
        ex.setTraceContext({"send the message", "lapis", false});
        const auto res = ex.clickTarget("Send");
        const auto lines = traceLines(tmp.path());
        check(res.ok && lines.size() == 1, "trace L1 click: one JSONL line");
        check(m.calls.count("capture") == 2 && m.capturedWidth == 1536 && m.capturedExclude == QVector<uint32_t>({4242, 5151}),
            "trace L1 click: two 1536-wide captures excluding reticle + HUD");
        check(m.effects().indexOf("capture") < m.effects().indexOf("click") && m.effects().lastIndexOf("capture") > m.effects().indexOf("click"),
            "trace L1 click: before frame precedes the click, after frame follows it");
        if (lines.size() == 1) {
            const QJsonObject o = lines[0];
            check(o["action"] == "click" && o["tier"] == "L1_AX" && o["ok"].toBool() && o["app"] == "Example"
                      && o["goal"] == "send the message" && o["target"] == "Send" && o["text"] == "",
                "trace L1 click: action, tier, ok, app, goal");
            check(o["display_w"].toDouble() == 1440 && o["display_h"].toDouble() == 900, "trace L1 click: display size in points");
            check(intArray(o["point_1000"], {319, 356}), "trace L1 click: point_1000 normalized to the display");
            check(intArray(o["bbox_1000"], {278, 333, 361, 378}), "trace L1 click: bbox_1000 from the AX rect");
            const QString before = o["before_image"].toString(), after = o["after_image"].toString();
            check(before.startsWith("frames/") && before.endsWith("_before.jpg") && after.endsWith("_after.jpg"),
                "trace L1 click: frame names <ts_ms>_before/after.jpg");
            QFile bf(tmp.path() + "/" + before), af(tmp.path() + "/" + after);
            check(bf.open(QIODevice::ReadOnly) && bf.readAll() == m.jpeg && af.exists(), "trace L1 click: both frames on disk");
            check(QDateTime::fromString(o["ts"].toString(), Qt::ISODateWithMs).isValid(), "trace L1 click: ISO 8601 ts");
        }
    }
    // Type: the typed text is recorded; one before (pre-click) and one after frame.
    {
        QTemporaryDir tmp;
        MockRig m;
        m.nodes = {axNode(5, "AXTextField", "Search", button)};
        CuaHooks h = m.hooks();
        h.traceDir = tmp.path();
        CuaExecutor ex(h);
        const auto res = ex.typeIntoTarget("Search", "hello world");
        const auto lines = traceLines(tmp.path());
        check(res.ok && lines.size() == 1 && lines[0]["action"] == "type" && lines[0]["text"] == "hello world"
                  && intArray(lines[0]["bbox_1000"], {278, 333, 361, 378}),
            "trace type: text and bbox recorded");
        check(m.calls.count("capture") == 2 && traceFiles(tmp.path()) == 3, "trace type: before + after + log");
    }
    // L3: the grounding capture is the before frame (no second capture), no bbox.
    {
        QTemporaryDir tmp;
        MockRig m;
        m.ground.ok = true; m.ground.hasPoint1000 = true; m.ground.point1000 = {250, 500};
        CuaHooks h = m.hooks();
        h.traceDir = tmp.path();
        CuaExecutor ex(h);
        const auto res = ex.clickTarget("the gear icon");
        const auto lines = traceLines(tmp.path());
        check(res.ok && m.calls.count("capture") == 2, "trace L3: reuses the grounding capture as the before frame");
        check(lines.size() == 1 && lines[0]["tier"] == "L3_VLM" && intArray(lines[0]["point_1000"], {250, 500})
                  && lines[0]["bbox_1000"].isNull(),
            "trace L3: point_1000 kept, bbox_1000 null");
    }
    // Scroll records its delta.
    {
        QTemporaryDir tmp;
        MockRig m;
        m.nodes = {axNode(2, "AXScrollArea", "Feed", button)};
        CuaHooks h = m.hooks();
        h.traceDir = tmp.path();
        CuaExecutor ex(h);
        ex.scrollTarget("Feed", -5);
        const auto lines = traceLines(tmp.path());
        check(lines.size() == 1 && lines[0]["action"] == "scroll" && lines[0]["delta_y"].toInt() == -5, "trace scroll: delta_y recorded");
    }
    // Privacy: infinity in the target, the app, the goal, or a redacted project writes nothing.
    {
        struct Case { const char *name; QString target; QString app; TraceContext ctx; };
        const QVector<Case> cases{
            {"trace skip: 'infinity' in targetDesc", "Infinity", "Example", {"open it", "p", false}},
            {"trace skip: 'infinity' in appName", "Send", "Infinity Slack", {"send", "p", false}},
            {"trace skip: redacted project", "Send", "Example", {"send", "p", true}},
            {"trace skip: 'infinity' in goal", "Send", "Example", {"reply in INFINITY", "p", false}},
        };
        for (const Case &c : cases) {
            QTemporaryDir tmp;
            MockRig m;
            AxNode n = axNode(2, "AXButton", c.target, button);
            n.appName = c.app;
            m.nodes = {n};
            CuaHooks h = m.hooks();
            h.traceDir = tmp.path();
            CuaExecutor ex(h);
            ex.setTraceContext(c.ctx);
            const bool ok = ex.clickTarget(c.target).ok && ex.typeIntoTarget(c.target, "hi").ok;
            check(ok && traceFiles(tmp.path()) == 0 && !QDir(tmp.path() + "/frames").exists() && !m.calls.contains("capture"),
                c.name);
        }
    }
    // Off when traceDir is empty (every test rig and LAPIS_CUA_TRACE=0): no extra capture.
    {
        MockRig m;
        m.nodes = {axNode(2, "AXButton", "Send", button)};
        CuaExecutor ex(m.hooks());
        check(ex.clickTarget("Send").ok && !m.calls.contains("capture"), "trace off: no traceDir, no capture");
        qputenv("LAPIS_CUA_TRACE", "0");
        check(defaultTraceDir().isEmpty(), "trace off: LAPIS_CUA_TRACE=0");
        qunsetenv("LAPIS_CUA_TRACE");
        check(defaultTraceDir().endsWith("/.lapis-next/cua_traces"), "trace on: default dir ~/.lapis-next/cua_traces");
    }
    // Disk cap: the newest traceMaxSteps steps survive, older frames are deleted.
    {
        QTemporaryDir tmp;
        MockRig m;
        m.nodes = {axNode(2, "AXButton", "Send", button)};
        CuaHooks h = m.hooks();
        h.traceDir = tmp.path();
        h.traceMaxSteps = 2;
        CuaExecutor ex(h);
        for (int i = 0; i < 5; ++i) ex.clickTarget("Send");
        const auto lines = traceLines(tmp.path());
        bool framesMatch = lines.size() == 2;
        for (const QJsonObject &o : lines)
            framesMatch = framesMatch && QFile::exists(tmp.path() + "/" + o["before_image"].toString())
                          && QFile::exists(tmp.path() + "/" + o["after_image"].toString());
        check(framesMatch && QDir(tmp.path() + "/frames").entryList(QDir::Files).size() == 4,
            "trace cap: keeps the newest 2 steps (4 frames), deletes the rest");
    }
}
void hotkeyDragTests() {
    // parseHotkeyCombo: modifiers in any order and case, one key, a clean label.
    {
        const ParsedHotkey l = parseHotkeyCombo("cmd+l");
        check(l.valid && l.keyCode == 37 && l.flags == kCGEventFlagMaskCommand && l.display == "Cmd+L", "hotkey: cmd+l");
        const ParsedHotkey t = parseHotkeyCombo(" Shift + CMD + t ");
        check(t.valid && t.keyCode == 17 && t.flags == (kCGEventFlagMaskCommand | kCGEventFlagMaskShift) && t.display == "Cmd+Shift+T",
            "hotkey: shift+cmd+t is Cmd+Shift+T");
        check(parseHotkeyCombo("cmd+shift+t").display == "Cmd+Shift+T", "hotkey: cmd+shift+t label");
        const ParsedHotkey s = parseHotkeyCombo("ctrl+opt+space");
        check(s.valid && s.keyCode == 49 && s.flags == (kCGEventFlagMaskControl | kCGEventFlagMaskAlternate)
                  && s.display == "Ctrl+Opt+Space",
            "hotkey: ctrl+opt+space");
        check(parseHotkeyCombo("control+option+space").flags == s.flags && parseHotkeyCombo("alt+ctrl+space").display == s.display,
            "hotkey: control/option/alt aliases");
        const ParsedHotkey e = parseHotkeyCombo("escape");
        check(e.valid && e.keyCode == 53 && e.flags == 0 && e.display == "Esc" && parseHotkeyCombo("esc").keyCode == 53,
            "hotkey: escape / esc, no modifiers");
        const ParsedHotkey en = parseHotkeyCombo("enter");
        check(en.valid && en.keyCode == 36 && en.display == "Enter" && parseHotkeyCombo("return").keyCode == 36, "hotkey: enter / return");
        check(parseHotkeyCombo("command+meta+a").flags == kCGEventFlagMaskCommand, "hotkey: command and meta are cmd");
        check(parseHotkeyCombo("cmd+f12").keyCode == 111 && parseHotkeyCombo("cmd+=").keyCode == 24
                  && parseHotkeyCombo("cmd+-").keyCode == 27 && parseHotkeyCombo("cmd+[").keyCode == 33
                  && parseHotkeyCombo("cmd+,").display == "Cmd+," && parseHotkeyCombo("up").keyCode == 126
                  && parseHotkeyCombo("pagedown").keyCode == 121 && parseHotkeyCombo("cmd+delete").keyCode == 51
                  && parseHotkeyCombo("forwarddelete").keyCode == 117 && parseHotkeyCombo("cmd+0").keyCode == 29,
            "hotkey: f-keys, punctuation, arrows, paging, deletes, digits");
        bool allInvalid = true;
        for (const char *bad : {"", "  ", "cmd", "cmd+shift", "cmd+", "+l", "cmd++l", "cmd+l+t", "hyper+l", "cmd+f13", "cmd+ll",
                                "lapis", "cmd|l"})
            allInvalid = allInvalid && !parseHotkeyCombo(bad).valid;
        check(allInvalid, "hotkey: modifier-only, empty parts, two keys, unknown names rejected");
    }
    // hotkeyAction with an app: activates it, HUD "Chrome · Pressing Cmd+L", presses, observes.
    {
        MockRig m;
        m.activateOk = true; m.activatePid = 600; m.activateDisplay = 9;
        m.nodesByDisplay[9] = {AxNode{1, "AXTextField", "Address and search bar", "", "Google Chrome", 600, {1500, 40, 600, 30}, {}}};
        CuaExecutor ex(m.hooks());
        const auto res = ex.hotkeyAction("Chrome", "cmd+l");
        check(res.ok && res.tier == "KEY" && res.globalPt.isNull() && res.detail == "Pressed Cmd+L in Chrome",
            "hotkey exec: ok, tier KEY, no point");
        check(m.calls.contains("activate:Chrome") && m.hotkeyCombo == "cmd+l", "hotkey exec: activates Chrome, presses cmd+l");
        check(m.effects().indexOf("activate:Chrome") < m.effects().indexOf("hotkey"), "hotkey exec: activation precedes the press");
        check(m.huds.contains("working|Chrome · Pressing Cmd+L") && m.huds.last() == "done|Done · Pressed Cmd+L",
            "hotkey exec: HUD shows Chrome · Pressing Cmd+L, then Done");
        check(res.observation.contains("Address and search bar") && m.pauses.contains("200")
                  && m.inspected().last() == "9",
            "hotkey exec: settles 200 ms, then observes the app's display");
        check(!m.calls.contains("click") && !m.calls.contains("reticlePulse"), "hotkey exec: no pointer input");
    }
    // No app: nothing is activated; a fullscreen single-screen Lapis is hidden first.
    {
        MockRig m = singleScreenRig();
        m.frontSeq = {500, 700};
        CuaExecutor ex(m.hooks());
        const auto res = ex.hotkeyAction("", "cmd+space");
        check(res.ok && m.hotkeyCombo == "cmd+space" && !m.calls.join(",").contains("activate:"),
            "hotkey bare: presses without activating");
        check(m.calls.contains("hideSelf") && m.effects().indexOf("hideSelf") < m.effects().indexOf("hotkey"),
            "hotkey bare: hides fullscreen Lapis before pressing");
        check(m.huds.contains("working|Pressing Cmd+Space"), "hotkey bare: HUD 'Pressing Cmd+Space'");
    }
    // Invalid combo: no route, no press, an error.
    {
        MockRig m;
        CuaExecutor ex(m.hooks());
        const auto res = ex.hotkeyAction("Chrome", "cmd+banana");
        check(!res.ok && res.tier == "ERR" && !m.calls.contains("hotkey") && !m.calls.join(",").contains("activate:"),
            "hotkey exec: invalid combo fails before activating or pressing");
    }
    // Trace: one JSONL line with action hotkey, text = combo, no point, before + after frames.
    {
        QTemporaryDir tmp;
        MockRig m;
        CuaHooks h = m.hooks();
        h.traceDir = tmp.path();
        CuaExecutor ex(h);
        ex.setTraceContext({"open a new tab", "lapis", false});
        const auto res = ex.hotkeyAction("", "cmd+t");
        const auto lines = traceLines(tmp.path());
        check(res.ok && lines.size() == 1 && lines[0]["action"] == "hotkey" && lines[0]["text"] == "cmd+t"
                  && lines[0]["tier"] == "KEY" && lines[0]["point_1000"].isNull() && lines[0]["bbox_1000"].isNull()
                  && lines[0]["goal"] == "open a new tab",
            "trace hotkey: action, text, KEY tier, null point");
        check(m.calls.count("capture") == 2 && m.effects().indexOf("capture") < m.effects().indexOf("hotkey")
                  && m.effects().lastIndexOf("capture") > m.effects().indexOf("hotkey") && traceFiles(tmp.path()) == 3,
            "trace hotkey: before frame, press, after frame");
    }

    const QRectF handle(100, 400, 20, 20), maxMark(600, 400, 40, 20);  // centres (110, 410), (620, 410)
    // dragTarget: both ends via AX, reticle glides from -> pulse -> to, drag with exact points.
    {
        MockRig m;
        m.nodes = {axNode(1, "AXValueIndicator", "slider handle", handle), axNode(2, "AXButton", "volume max", maxMark)};
        CuaExecutor ex(m.hooks());
        const auto res = ex.dragTarget("slider handle", "volume max");
        check(res.ok && res.tier == "L1_AX" && close(res.globalPt, {110, 410}), "drag exec: ok via AX, globalPt is from");
        check(close(m.dragFrom, {110, 410}) && close(m.dragTo, {620, 410}), "drag exec: dragBetween gets the exact centres");
        const QStringList fx = m.effects();
        check(fx == QStringList({"reticleHighlight", "reticleHighlight", "reticleMove", "reticlePulse", "reticleMove", "drag"}),
            "drag exec: locate both, glide to from, pulse, glide to to, drag");
        check(close(m.pulsePt, {110, 410}) && close(m.movePt, {620, 410}), "drag exec: pulse at from, reticle ends at to");
        check(m.huds.contains("working|Dragging slider handle to volume max")
                  && m.huds.last() == "done|Done · Dragged slider handle to volume max",
            "drag exec: HUD 'Dragging <from> to <to>', then Done");
        check(m.pauses.contains("200") && res.observation.contains("volume max"), "drag exec: settles, then observes");
    }
    // To via the VLM: from is AX, to is not in the tree, so the grounder places it.
    {
        MockRig m;
        m.nodes = {axNode(1, "AXValueIndicator", "slider handle", handle)};
        m.ground.ok = true; m.ground.hasPoint1000 = true; m.ground.point1000 = {500, 500};  // (720, 450) on the primary
        CuaExecutor ex(m.hooks());
        const auto res = ex.dragTarget("slider handle", "the far right end");
        check(res.ok && res.tier == "L3_VLM" && m.groundInstruction == "the far right end" && close(m.dragFrom, {110, 410})
                  && close(m.dragTo, {720, 450}),
            "drag exec: AX from + VLM to, exact points");
    }
    // A missing end: nothing is pressed, the reticle hides, the screen comes back.
    {
        MockRig m;
        m.nodes = {axNode(2, "AXButton", "volume max", maxMark)};
        CuaExecutor ex(m.hooks());
        const auto res = ex.dragTarget("slider handle", "volume max");
        check(!res.ok && res.tier == "ERR" && !m.calls.contains("drag") && m.calls.contains("reticleHide")
                  && res.detail.startsWith("From target \"slider handle\"") && res.observation.contains("volume max"),
            "drag exec: missing from never drags");
        MockRig n;
        n.nodes = {axNode(1, "AXValueIndicator", "slider handle", handle)};
        CuaExecutor exn(n.hooks());
        const auto r2 = exn.dragTarget("slider handle", "volume max");
        check(!r2.ok && !n.calls.contains("drag") && !n.calls.contains("reticlePulse") && r2.detail.startsWith("To target \"volume max\""),
            "drag exec: missing to never drags or pulses");
        check(!exn.dragTarget("slider handle", "  ").ok && !n.calls.contains("drag"), "drag exec: empty to rejected");
        MockRig f;
        f.dragOk = false;
        f.nodes = {axNode(1, "AXValueIndicator", "slider handle", handle), axNode(2, "AXButton", "volume max", maxMark)};
        CuaExecutor exf(f.hooks());
        const auto r3 = exf.dragTarget("slider handle", "volume max");
        check(!r3.ok && r3.detail.contains("denied") && f.huds.last().startsWith("error|"), "drag exec: a failed drag reports the error");
    }
    // A bare to inherits from's app: the second locate is routed to the same app.
    {
        MockRig m;
        m.activateOk = true; m.activatePid = 600; m.activateDisplay = 9;
        AxNode h = axNode(1, "AXValueIndicator", "slider handle", {1500, 400, 20, 20});
        AxNode x = axNode(2, "AXButton", "volume max", {1900, 400, 40, 20});
        h.appName = x.appName = "Google Chrome";
        m.nodesByDisplay[9] = {h, x};
        CuaExecutor ex(m.hooks());
        const auto res = ex.dragTarget("Chrome slider handle", "volume max");
        check(res.ok && m.calls.count("activate:Chrome") == 2 && close(m.dragFrom, {1510, 410}) && close(m.dragTo, {1920, 410}),
            "drag exec: bare to inherits Chrome, both found on its display");
    }
    // Trace: one JSONL line, point_1000 = from, to_point_1000 = to, bbox from the from rect.
    {
        QTemporaryDir tmp;
        MockRig m;
        m.nodes = {axNode(1, "AXValueIndicator", "slider handle", handle), axNode(2, "AXButton", "volume max", maxMark)};
        CuaHooks h = m.hooks();
        h.traceDir = tmp.path();
        CuaExecutor ex(h);
        const auto res = ex.dragTarget("slider handle", "volume max");
        const auto lines = traceLines(tmp.path());
        check(res.ok && lines.size() == 1 && lines[0]["action"] == "drag" && lines[0]["target"] == "slider handle"
                  && lines[0]["text"] == "volume max" && lines[0]["tier"] == "L1_AX",
            "trace drag: action, from target, to text");
        if (lines.size() == 1) {
            // 1440x900: (110, 410) -> (76, 456); (620, 410) -> (431, 456); handle rect -> 69..83 x 444..467.
            check(intArray(lines[0]["point_1000"], {76, 456}) && intArray(lines[0]["to_point_1000"], {431, 456})
                      && intArray(lines[0]["bbox_1000"], {69, 444, 83, 467}),
                "trace drag: from point, to point and from bbox normalized");
        }
        check(m.calls.count("capture") == 2 && m.effects().indexOf("capture") < m.effects().indexOf("drag")
                  && m.effects().lastIndexOf("capture") > m.effects().indexOf("drag"),
            "trace drag: before frame, drag, after frame");
        QTemporaryDir tmp2;
        MockRig n;
        n.nodes = {axNode(1, "AXValueIndicator", "slider handle", handle)};
        CuaHooks h2 = n.hooks();
        h2.traceDir = tmp2.path();
        CuaExecutor ex2(h2);
        ex2.dragTarget("slider handle", "volume max");
        const auto miss = traceLines(tmp2.path());
        check(miss.size() == 1 && !miss[0]["ok"].toBool() && miss[0]["tier"] == "ERR" && !miss[0].contains("to_point_1000"),
            "trace drag: a miss logs ERR without a to point");
    }
    // Real actuators under cancellation: nothing posts, both return at once.
    cancelFlag.store(true);
    check(!pressHotkey("cmd+l").ok && !dragBetween({0, 0}, {10, 10}).ok, "cancel hotkey and drag");
    cancelFlag.store(false);
    check(!pressHotkey("cmd+banana").ok && pressHotkey("cmd+banana").detail.startsWith("Invalid hotkey"),
        "pressHotkey: invalid combo rejected before posting");
    const double nan = std::numeric_limits<double>::quiet_NaN();
    check(!dragBetween({nan, 0}, {10, 10}).ok && !dragBetween({0, 0}, {1e9, 1e9}).ok, "dragBetween: off-display endpoints rejected");
}
// Human-click shadow recorder: recordClickSample against fake displays / AX / capture.
struct ShadowRig {
    QVector<DisplayInfo> displays{{7, "primary", {0, 0, 1440, 900}, 2, true}, {9, "side", {1440, 0, 1920, 1080}, 1, false}};
    // Chrome / Electron shape: empty AXTitle, the name in AXDescription.
    ShadowAxHit hit{true, "AXButton", "", "", "Send", "Example", 321, {400, 300, 120, 40}};
    QByteArray jpeg = "fake-jpeg";
    int captures = 0;
    uint32_t capturedDisplay = 0;
    int capturedWidth = 0;
    ShadowHooks hooks() {
        ShadowHooks h;
        h.listDisplays = [this] { return displays; };
        h.elementAt = [this](QPointF) { return hit; };
        h.captureJpeg = [this](uint32_t id, int w) { ++captures; capturedDisplay = id; capturedWidth = w; return jpeg; };
        h.selfPid = 999;
        return h;
    }
};
QVector<QJsonObject> humanLines(const QString &dir) {
    QFile f(dir + "/human_traces.jsonl");
    QVector<QJsonObject> out;
    if (!f.open(QIODevice::ReadOnly)) return out;
    for (const QByteArray &l : f.readAll().split('\n'))
        if (!l.trimmed().isEmpty()) out << QJsonDocument::fromJson(l).object();
    return out;
}
void shadowTests() {
    const qint64 t0 = 1763600000000;  // fixed ms, so frame names are deterministic
    // Left click on the primary Retina display: one line, point_1000 + bbox_1000, one frame.
    {
        QTemporaryDir tmp;
        ShadowRig r;
        const auto res = recordClickSample(tmp.path(), {{460, 320}, false, t0}, r.hooks());
        const auto lines = humanLines(tmp.path());
        check(res.outcome == ShadowOutcome::Recorded && lines.size() == 1, "shadow left: one JSONL line");
        check(r.captures == 1 && r.capturedDisplay == 7 && r.capturedWidth == 1536, "shadow left: one 1536-wide capture of the click's display");
        if (lines.size() == 1) {
            const QJsonObject o = lines[0];
            check(o["source"] == "human_click" && o["button"] == "left" && o["app"] == "Example" && o["role"] == "AXButton"
                      && o["title"] == "" && o["description"] == "Send",
                "shadow left: source, button, app, role, title, description");
            check(o["display_w"].toDouble() == 1440 && o["display_h"].toDouble() == 900, "shadow left: display size in points");
            check(intArray(o["point_1000"], {319, 356}), "shadow left: point_1000 normalized (460,320 on 1440x900)");
            check(intArray(o["bbox_1000"], {278, 333, 361, 378}), "shadow left: bbox_1000 from the AX rect");
            check(o["image"] == QString("human_frames/%1.jpg").arg(t0), "shadow left: image is human_frames/<ts_ms>.jpg");
            QFile jf(tmp.path() + "/" + o["image"].toString());
            check(jf.open(QIODevice::ReadOnly) && jf.readAll() == r.jpeg, "shadow left: frame bytes on disk");
            check(QDateTime::fromString(o["ts"].toString(), Qt::ISODateWithMs).toMSecsSinceEpoch() == t0
                      && o["ts"].toString().endsWith("Z"),
                "shadow left: ts is ISO 8601 UTC of the click");
            check(o.keys().size() == 12, "shadow left: exactly the 12 documented fields");
        }
        check(res.line == QJsonDocument(lines.value(0)).toJson(QJsonDocument::Compact), "shadow left: result carries the compact line");
    }
    // Right click on the 1x side display (negative of nothing: origin x=1440), element half off it.
    {
        QTemporaryDir tmp;
        ShadowRig r;
        r.hit.globalRect = {1340, 540, 200, 108};  // straddles the left edge of the side display
        const auto res = recordClickSample(tmp.path(), {{1440 + 960, 540}, true, t0}, r.hooks());
        const auto lines = humanLines(tmp.path());
        check(res.outcome == ShadowOutcome::Recorded && r.capturedDisplay == 9 && lines.size() == 1, "shadow right: side display captured");
        if (lines.size() == 1) {
            check(lines[0]["button"] == "right" && lines[0]["display_w"].toDouble() == 1920, "shadow right: button right, side display size");
            check(intArray(lines[0]["point_1000"], {500, 500}), "shadow right: point_1000 relative to its own display");
            check(intArray(lines[0]["bbox_1000"], {0, 500, 52, 600}), "shadow right: bbox_1000 clipped to the display");
        }
    }
    // No AX rect: bbox_1000 is null, the point is still kept.
    {
        QTemporaryDir tmp;
        ShadowRig r;
        r.hit.globalRect = {};
        recordClickSample(tmp.path(), {{0, 0}, false, t0}, r.hooks());
        const auto lines = humanLines(tmp.path());
        check(lines.size() == 1 && lines[0]["bbox_1000"].isNull() && intArray(lines[0]["point_1000"], {0, 0}), "shadow: no rect -> bbox_1000 null");
    }
    // Privacy: skipped BEFORE any capture, nothing on disk.
    {
        struct Case { const char *name; ShadowAxHit hit; };
        const QVector<Case> cases{
            {"shadow skip: role AXSecureTextField", {true, "AXSecureTextField", "", "Password", "", "Safari", 5, {10, 10, 100, 20}}},
            {"shadow skip: subrole AXSecureTextField (real macOS shape)", {true, "AXTextField", "AXSecureTextField", "", "", "Safari", 5, {10, 10, 100, 20}}},
            {"shadow skip: 'infinity' in app", {true, "AXButton", "", "Send", "", "Infinity Slack", 5, {}}},
            {"shadow skip: 'INFINITY' in title", {true, "AXStaticText", "", "#infinity-ops", "", "Slack", 5, {}}},
            {"shadow skip: '1Password' app", {true, "AXButton", "", "Unlock", "", "1Password 8", 5, {}}},
            {"shadow skip: 'keychain' in description", {true, "AXButton", "", "", "Keychain Access item", "Safari", 5, {}}},
            {"shadow skip: 'bitwarden' in title", {true, "AXButton", "", "Bitwarden vault", "", "Chrome", 5, {}}},
            {"shadow skip: unknown app", {false, "", "", "", "", "", 0, {}}},
            {"shadow skip: Lapis itself", {true, "AXButton", "", "Send", "", "lapis_studio", 999, {}}},
        };
        for (const Case &c : cases) {
            QTemporaryDir tmp;
            ShadowRig r;
            r.hit = c.hit;
            const auto res = recordClickSample(tmp.path(), {{100, 100}, false, t0}, r.hooks());
            check(res.outcome == ShadowOutcome::SkippedSensitive && r.captures == 0 && QDir(tmp.path()).entryList(QDir::NoDotAndDotDot | QDir::AllEntries).isEmpty(),
                c.name);
        }
        check(!shadowSensitive({true, "AXButton", "", "Send", "Compose", "Mail", 5, {}}), "shadow: ordinary target is not sensitive");
    }
    // Off-display point and a failed capture write nothing.
    {
        QTemporaryDir tmp;
        ShadowRig r;
        check(recordClickSample(tmp.path(), {{-5000, -5000}, false, t0}, r.hooks()).outcome == ShadowOutcome::SkippedNoDisplay && r.captures == 0,
            "shadow: off-display click skipped without capture");
        r.jpeg.clear();
        check(recordClickSample(tmp.path(), {{10, 10}, false, t0}, r.hooks()).outcome == ShadowOutcome::SkippedNoFrame
                  && !QFile::exists(tmp.path() + "/human_traces.jsonl"),
            "shadow: capture denied -> no line");
    }
    // Rate limit: 350 ms between captures.
    check(shadowRateAllows(0, t0) && !shadowRateAllows(t0, t0 + 349) && shadowRateAllows(t0, t0 + 350) && shadowRateAllows(t0, t0 - 10),
        "shadow: rate limit 1 per 350 ms (first always, clock step back allowed)");
    // Disk cap: 10 steps cap -> trims to 10 newest once past 11, deleting unreferenced frames.
    {
        QTemporaryDir tmp;
        ShadowRig r;
        const int cap = 10;
        for (int i = 0; i < 11; ++i) recordClickSample(tmp.path(), {{100, 100}, false, t0 + i * 1000}, r.hooks(), cap);
        check(humanLines(tmp.path()).size() == 11, "shadow cap: within 10% slack nothing is trimmed");
        QFile orphan(tmp.path() + "/human_frames/orphan.jpg");
        check(orphan.open(QIODevice::WriteOnly) && orphan.write("x") == 1, "shadow cap: orphan frame planted");
        orphan.close();
        recordClickSample(tmp.path(), {{100, 100}, false, t0 + 11 * 1000}, r.hooks(), cap);
        const auto lines = humanLines(tmp.path());
        bool ok = lines.size() == cap && lines.first()["image"] == QString("human_frames/%1.jpg").arg(t0 + 2 * 1000)
                  && lines.last()["image"] == QString("human_frames/%1.jpg").arg(t0 + 11 * 1000);
        for (const QJsonObject &o : lines) ok = ok && QFile::exists(tmp.path() + "/" + o["image"].toString());
        check(ok && QDir(tmp.path() + "/human_frames").entryList(QDir::Files).size() == cap,
            "shadow cap: keeps the newest 10 lines + frames, deletes the oldest and orphans");
    }
    // The recorder object: never running before start, start refuses an empty dir, stop is idempotent.
    {
        CuaShadowRecorder rec;
        check(!rec.isRunning() && rec.recordedCount() == 0 && !rec.start(QString()), "shadow recorder: idle, empty dir refused");
        rec.stop();
        check(!rec.isRunning(), "shadow recorder: stop without start is a no-op");
    }
}
// The MacBook-wide Spotlight bar: pure geometry, then the Offscreen state machine (no window).
void spotlightTests() {
    {
        const QRectF screen(0, 0, 1512, 982);  // AppKit: origin bottom-left
        const QRectF f = spotlightFrame(screen, kSpotlightBaseHeight);
        check(f.width() == kSpotlightWidth && std::abs(f.center().x() - screen.center().x()) <= 1,
            "spotlight: 620 wide, centred horizontally");
        check(std::abs((f.y() + f.height()) - (982 - std::round(982 * 0.20))) < 1e-9,
            "spotlight: top edge 20% below the screen top (AppKit coords)");
        const QRectF g = spotlightFrame(QRectF(-1920, 200, 1920, 1080), 200);
        check(g.x() == -1920 + 650 && std::abs(g.y() + g.height() - (200 + 1080 - 216)) < 1e-9,
            "spotlight: a secondary screen with a negative origin");
        check(spotlightFrame(screen, 10).height() == kSpotlightBaseHeight
                  && spotlightFrame(screen, 1000).height() == kSpotlightMaxHeight,
            "spotlight: height clamped to 96..340");
        check(spotlightHeight(false, 500) == 96 && spotlightHeight(true, 0) > 96 && spotlightHeight(true, 5000) <= 340
                  && spotlightHeight(true, 40) < spotlightHeight(true, 120),
            "spotlight: the drawer grows with the answer, capped at 340");
        check(spotlightHint(true) == QStringLiteral("● Listening…")
                  && spotlightHint(false) == QStringLiteral("L2 voice · R2 send · ○ close"),
            "spotlight: hint text listening / idle");
        check(spotlightPlaceholder("Google Chrome") == QStringLiteral("Command or ask about Google Chrome…"),
            "spotlight: placeholder names the target app");
    }
    {
        CuaSpotlight sp(CuaSpotlight::Mode::Offscreen);
        int frontPid = 4321;
        QString frontName = "Google Chrome";
        sp.setFrontmostProbe([&](int *pid, QString *name) { *pid = frontPid; *name = frontName; });
        QString submitted;
        int dismissed = 0;
        sp.onSubmit = [&](const QString &t) { submitted = t; };
        sp.onDismiss = [&] { ++dismissed; };
        check(!sp.isOpen() && sp.windowNumber() == 0 && sp.state().opens == 0, "spotlight off: closed, no window");
        const bool wasKey = sp.openOrFocus();
        SpotlightState st = sp.state();
        check(!wasKey && st.open && st.key && st.alpha == 1.0 && st.targetPid == 4321 && st.targetApp == "Google Chrome",
            "spotlight off: openOrFocus captures the frontmost app as the target");
        check(st.frame.height() == kSpotlightBaseHeight && !st.drawerVisible && sp.windowNumber() == 0,
            "spotlight off: base height without a drawer, still no window");
        check(sp.openOrFocus(), "spotlight off: a second openOrFocus is already key (no voice-key delay)");
        sp.setListening(true);
        check(sp.state().listening && sp.state().hint.contains("Listening"), "spotlight off: listening hint");
        sp.setInputText("click sign in");
        sp.setListening(false);
        check(sp.inputText() == "click sign in" && sp.state().hint.startsWith("L2 voice"),
            "spotlight off: dictated text kept after release");
        const QString q = sp.takeInputText();
        sp.beginTurn(q);
        st = sp.state();
        check(q == "click sign in" && st.inputText.isEmpty() && st.lastQuestion == "click sign in" && st.busy
                  && st.statusLine == QStringLiteral("Thinking…") && st.drawerVisible && st.frame.height() > kSpotlightBaseHeight,
            "spotlight off: a turn shows the question + Thinking… and grows the drawer");
        const double top = st.frame.y() + st.frame.height();
        sp.setAnswer(QString("A long answer. ").repeated(60));
        st = sp.state();
        check(std::abs(st.frame.y() + st.frame.height() - top) < 1e-9 && st.frame.height() <= kSpotlightMaxHeight,
            "spotlight off: growing keeps the top edge, capped at 340");
        sp.setActionRunning(true, "Google Chrome · Clicking sign in…");
        st = sp.state();
        check(st.actionRunning && st.alpha == 0.0 && st.open && st.actionRuns == 1,
            "spotlight off: a cua_* action ghosts the bar (alpha 0)");
        sp.finishAfterAction("Done · Clicked sign in");
        st = sp.state();
        check(!st.open && !st.actionRunning && st.lastQuestion == "click sign in" && st.statusLine == "Done · Clicked sign in"
                  && !st.answer.isEmpty(),
            "spotlight off: after the action it closes but keeps the exchange");
        frontPid = 999; frontName = "Discord";
        check(!sp.openOrFocus(), "spotlight off: reopening after an action is not key yet (voice key waits)");
        st = sp.state();
        check(st.open && st.lastQuestion == "click sign in" && st.drawerVisible && st.targetApp == "Discord" && st.opens == 2,
            "spotlight off: the second L2 pops it back with the previous exchange and a fresh target");
        sp.setBusy(false);
        sp.dismiss(true);
        check(!sp.isOpen() && sp.state().dismissals == 2 && !sp.reactivateTarget(),
            "spotlight off: dismiss closes; Offscreen never activates an app");
        check(submitted.isEmpty() && dismissed == 0, "spotlight off: state calls never fire the user callbacks");
    }
    {
        // A step naming no app goes to the Spotlight's target app, which is activated first.
        MockRig m = singleScreenRig();
        m.activateOk = true; m.activatePid = 600;
        m.windowsAfterActivate = {{600, "Google Chrome", 1, {0, 25, 1512, 957}, true}};
        m.frontSeq = {600};
        m.hotkeyCombo.clear();
        CuaExecutor ex(m.hooks());
        ex.setDefaultAppHint("Google Chrome");
        const auto res = ex.hotkeyAction(QString(), "cmd+t");
        check(res.ok && m.calls.contains("activate:Google Chrome") && !m.calls.contains("hideSelf"),
            "spotlight exec: an app-less hotkey activates the bar's target app, Lapis is not hidden");
        MockRig n = singleScreenRig();
        n.activateOk = true; n.activatePid = 700;
        n.frontSeq = {700};
        CuaExecutor ex2(n.hooks());
        ex2.setDefaultAppHint("Google Chrome");
        ex2.hotkeyAction("Slack", "cmd+k");
        check(n.calls.contains("activate:Slack") && !n.calls.contains("activate:Google Chrome"),
            "spotlight exec: an explicit app still wins over the default");
    }
}
// Mirrors forbiddenPixels in src/demo.cpp: blue- or green-tinted pixels (after flattening on black).
int forbiddenPixels(const QImage &src) {
    const QImage img = src.convertToFormat(QImage::Format_RGB32);
    int bad = 0;
    for (int y = 0; y < img.height(); ++y) {
        const auto *row = reinterpret_cast<const QRgb *>(img.constScanLine(y));
        for (int x = 0; x < img.width(); ++x) {
            const int r = qRed(row[x]), g = qGreen(row[x]), b = qBlue(row[x]);
            bad += (b > r + 24 && b > g + 8) || (g > r + 30 && g > b + 30);
        }
    }
    return bad;
}
// Amber Gold #e09f3e-ish (and its glow): warm, saturated, bright.
bool goldish(const QColor &c) {
    return c.alpha() > 100 && c.red() > 140 && c.red() - c.blue() > 70 && c.green() > c.red() * 0.5 && c.green() < c.red();
}
int countIn(const QImage &img, const QRectF &pts, double scale, const std::function<bool(const QColor &)> &pred) {
    int n = 0;
    const QRect px = QRectF(pts.x() * scale, pts.y() * scale, pts.width() * scale, pts.height() * scale).toAlignedRect() & img.rect();
    for (int y = px.top(); y <= px.bottom(); ++y)
        for (int x = px.left(); x <= px.right(); ++x)
            n += pred(img.pixelColor(x, y));
    return n;
}
constexpr PadButton kAllPadButtons[] = {PadButton::Cross, PadButton::Circle, PadButton::Square, PadButton::Triangle,
                                        PadButton::L1, PadButton::R1, PadButton::L2, PadButton::R2,
                                        PadButton::L3, PadButton::R3, PadButton::Options, PadButton::Touchpad};
void padIconTests() {
    check(std::size(kAllPadButtons) == size_t(kPadButtonCount), "pad icon: all 12 buttons covered");
    const CGSize face = padIconSize(PadButton::Cross, 20), bump = padIconSize(PadButton::L1, 20),
                 trig = padIconSize(PadButton::R2, 20), opt = padIconSize(PadButton::Options, 20),
                 pad = padIconSize(PadButton::Touchpad, 20), stick = padIconSize(PadButton::L3, 20);
    check(face.width == 20 && face.height == 20 && stick.width == 20, "pad icon: face buttons and sticks are 1:1");
    check(std::abs(bump.width / 20 - 1.45) < 0.03 && std::abs(trig.width / 20 - 1.45) < 0.03 && std::abs(pad.width / 20 - 1.45) < 0.03,
        "pad icon: bumpers, triggers, touchpad ~1.45:1");
    check(opt.width < opt.height && opt.width > 10, "pad icon: Options is a vertical stadium");
    check(renderPadIcon(PadButton::Cross, 0, 2).isNull() && renderPadIcon(PadButton::Cross, 20, -1).isNull(),
        "pad icon: invalid size / scale render null");
    int okInk = 0, okForbidden = 0, okActiveGold = 0, okBlue = 0;
    for (PadButton b : kAllPadButtons) {
        for (bool active : {false, true}) {
            const QImage img = renderPadIcon(b, 24, 3, active, false);
            const CGSize sz = padIconSize(b, 24);
            int ink = 0, gold = 0, edge = 0;
            for (int y = 0; y < img.height(); ++y)
                for (int x = 0; x < img.width(); ++x) {
                    const QColor c = img.pixelColor(x, y);
                    // Idle ink #e8d5b5: bright and warm. Active: gold.
                    ink += c.alpha() > 180 && c.red() > 170 && c.red() >= c.blue();
                    gold += goldish(c);
                    edge += (x == 0 || y == 0 || x == img.width() - 1 || y == img.height() - 1) && c.alpha() > 200;
                }
            const bool sized = img.width() == int(std::ceil(sz.width * 3)) && img.height() == 72;
            okInk += sized && ink > 12 && edge < img.width();  // visible glyph, not clipped into a solid edge
            okForbidden += forbiddenPixels(img) == 0;
            if (active)
                okActiveGold += gold > 20;
            if (!(ink > 12 && forbiddenPixels(img) == 0))
                std::printf("pad icon %s active=%d: ink=%d forbidden=%d edge=%d\n", qPrintable(padButtonName(b)), int(active),
                    ink, forbiddenPixels(img), edge);
        }
        const QImage blue = renderPadIcon(b, 24, 2, true, true);
        int bluePx = 0;
        for (int y = 0; y < blue.height(); ++y)
            for (int x = 0; x < blue.width(); ++x) {
                const QColor c = blue.pixelColor(x, y);
                bluePx += c.alpha() > 150 && c.blue() > 180 && c.blue() > c.red() + 60;
            }
        okBlue += bluePx > 10;
    }
    check(okInk == 24, "pad icon: all 12 icons, idle + active, have visible ink and fit their box");
    check(okForbidden == 24, "pad icon: zero blue/green pixels in gold mode (idle + active)");
    check(okActiveGold == 12, "pad icon: active icons are Amber Gold");
    check(okBlue == 12, "pad icon: lapis-blue active icons take #4f8cff");
    std::printf("pad icons: ink=%d/24 forbidden-free=%d/24 gold=%d/12 blue=%d/12\n", okInk, okForbidden, okActiveGold, okBlue);
}
void goldThemeTests() {
    const QRectF disp(0, 0, 800, 500);
    {
        CuaReticle plain(CuaReticle::Mode::Offscreen, disp);
        check(plain.goldTheme(), "gold: a plain CuaReticle starts in Lapis Gold");
        plain.setGoldTheme(false);
        check(!plain.goldTheme() && plain.windowNumber() == 0, "gold: setGoldTheme(false) flips to blue, still no window");
        plain.setGoldTheme(true);
    }
    const QRectF hud = cuaHudRect(disp);
    const std::pair<HudTone, QString> tones[] = {{HudTone::Listening, QStringLiteral("Listening…")},
                                                 {HudTone::Working, QStringLiteral("Mail · Clicking Send")},
                                                 {HudTone::Done, QStringLiteral("Done · Send in Mail")},
                                                 {HudTone::Error, QStringLiteral("Couldn't find Send")}};
    int okForbidden = 0, okDot = 0, okBody = 0;
    for (const auto &[tone, status] : tones) {
        CuaReticle r(CuaReticle::Mode::Offscreen, disp);
        r.setGoldTheme(true);
        r.moveTo({100, 100});
        r.advanceClock(0.5);
        r.highlightRect({560, 280, 100, 40}, QStringLiteral("Clicking Send"));
        r.showHud(status, tone);
        r.advanceClock(0.1);
        r.pulseClick({610, 300});
        r.advanceClock(0.46);
        const QImage img = r.render(2);
        const int bad = forbiddenPixels(img);
        okForbidden += bad == 0;
        // The dot / check sits in the left 36 pt of the capsule.
        const QRectF dotArea(hud.x() + 8, hud.y() + 6, 26, hud.height() - 12);
        const int dot = tone == HudTone::Error
            ? countIn(img, dotArea, 2, [](const QColor &c) { return c.alpha() > 150 && c.red() > 180 && c.green() > 80 && c.green() < 130 && c.blue() < 90; })
            : countIn(img, dotArea, 2, goldish);
        okDot += dot > 4;
        // Warm obsidian body: dark, red >= blue (no cold slate), fully covering the middle.
        const int body = countIn(img, QRectF(hud.center().x() + 40, hud.y() + 8, 60, hud.height() - 16), 2,
            [](const QColor &c) { return c.alpha() > 230 && c.red() < 30 && c.red() >= c.blue() && c.green() >= c.blue(); });
        okBody += body > 1500;
        if (bad || dot <= 4 || body <= 1500)
            std::printf("gold hud tone %d: forbidden=%d dot=%d body=%d\n", int(tone), bad, dot, body);
        if (tone == HudTone::Working) {
            const int rimRing = countIn(img, QRectF(550, 270, 120, 60), 2, goldish);
            const int pointer = countIn(img, QRectF(600, 290, 30, 40), 2, goldish);
            check(rimRing > 50 && pointer > 10, "gold: focus ring, ripple and pointer rim are Amber Gold");
            const QRectF pill = r.state().pillRect;
            check(!pill.isEmpty() && countIn(img, pill, 2, goldish) > 10
                      && countIn(img, pill.adjusted(30, 6, -8, -6), 2, [](const QColor &c) { return c.alpha() > 200 && c.red() > 170 && c.red() > c.blue() + 25; }) > 20,
                "gold: the pill has a gold rim/dot and #e8d5b5 text");
        }
    }
    check(okForbidden == 4, "gold: reticle + ring + pill + HUD in every tone have zero blue/green pixels");
    check(okDot == 4, "gold: HUD dot/check is gold (Listening, Working, Done), copper-coral on Error");
    check(okBody == 4, "gold: HUD capsule body is warm obsidian");
    {
        // A trailing "L3+R3 Run" turns into controller badges on the right of the capsule.
        const auto rightInk = [&](const QString &status) {
            CuaReticle r(CuaReticle::Mode::Offscreen, disp);
            r.setGoldTheme(true);
            r.showHud(status, HudTone::Working);
            r.advanceClock(0.5);
            const QImage img = r.render(2);
            return std::make_pair(countIn(img, QRectF(hud.right() - 90, hud.y() + 4, 80, hud.height() - 8), 2,
                                      [](const QColor &c) { return c.alpha() > 200 && c.red() > 90; }),
                                  forbiddenPixels(img));
        };
        const auto plain = rightInk(QStringLiteral("Draft ready"));
        const auto badge = rightInk(QStringLiteral("Draft ready · L3+R3 Run"));
        check(badge.first > plain.first + 150 && badge.second == 0, "gold: \"L3+R3 Run\" draws [L3]+[R3] badges at the right");
        std::printf("hud run badge: right-ink plain=%d badge=%d\n", plain.first, badge.first);
    }
    {
        // Lapis blue: the accent is #4f8cff (forbidden pixels are expected there).
        CuaReticle r(CuaReticle::Mode::Offscreen, disp);
        r.setGoldTheme(false);
        r.showHud(QStringLiteral("Working"), HudTone::Working);
        r.advanceClock(0.5);
        const QImage img = r.render(2);
        check(countIn(img, QRectF(hud.x() + 8, hud.y() + 6, 26, hud.height() - 12), 2,
                  [](const QColor &c) { return c.alpha() > 150 && c.blue() > 200 && c.blue() > c.red() + 100; }) > 4
                  && countIn(img, hud, 2, goldish) == 0,
            "gold: lapis-blue mode swaps the accent to #4f8cff, no gold");
    }
}
void themeSyncTests() {
    check(!overlayLapisBlue() && !spotlightLapisBlue(), "theme: overlays default to Gold");
    setOverlayTheme(true);
    check(overlayLapisBlue() && spotlightLapisBlue(), "theme: setOverlayTheme(true) -> lapis blue everywhere");
    setOverlayTheme(false);
    check(!overlayLapisBlue() && !spotlightLapisBlue(), "theme: back to Gold");
}
void spotlightRenderTests() {
    CuaSpotlight sp(CuaSpotlight::Mode::Offscreen);
    sp.setFrontmostProbe([](int *pid, QString *name) { *pid = 77; *name = QStringLiteral("Google Chrome"); });
    sp.openOrFocus();
    QImage idle = sp.render(2);
    check(!idle.isNull() && idle.width() == int(kSpotlightWidth * 2) && idle.height() == int(kSpotlightBaseHeight * 2),
        "spotlight render: 2x image of the base bar");
    check(forbiddenPixels(idle) == 0, "spotlight render: idle bar has zero blue/green pixels");
    check(countIn(idle, QRectF(0, 0, kSpotlightWidth, kSpotlightBaseHeight), 2, goldish) > 100,
        "spotlight render: idle bar has its gold border");
    // The hint draws vector controller icons: warm ink in the top-right, gold only while listening.
    const QRectF hintArea(kSpotlightWidth - 230, 12, 214, 22);
    const int hintInk = countIn(idle, hintArea, 2, [](const QColor &c) { return c.alpha() > 200 && c.red() > 150 && c.red() > c.blue(); });
    check(hintInk > 40 && countIn(idle, hintArea, 2, goldish) < 10, "spotlight render: idle hint is [L2] [R2] [○] icons in Lapis ink");
    // The context pill: #14120f body.
    check(countIn(idle, QRectF(90, 15, 20, 16), 2, [](const QColor &c) {
              return c.alpha() > 240 && std::abs(c.red() - 0x14) <= 3 && std::abs(c.green() - 0x12) <= 3 && std::abs(c.blue() - 0x0F) <= 3; }) > 8,
        "spotlight render: the context pill is #14120f");
    sp.setListening(true);
    const QImage listening = sp.render(2);
    check(forbiddenPixels(listening) == 0 && countIn(listening, hintArea, 2, goldish) > 40,
        "spotlight render: listening shows the gold dot + Listening… and no blue/green");
    sp.setListening(false);
    sp.beginTurn(QStringLiteral("why does the sign in page keep reloading?"));
    sp.setAnswer(QStringLiteral("The page wants a cookie the browser blocks, so the form never gets its session token.\n\n"
                                "Allow third-party cookies for the accounts site, then reload the tab."));
    sp.setBusy(false);
    const SpotlightState st = sp.state();
    const QImage drawer = sp.render(2);
    check(!drawer.isNull() && std::abs(drawer.height() / 2.0 - st.frame.height()) <= 1 && st.frame.height() > kSpotlightBaseHeight + 60,
        "spotlight render: the drawer image matches the state frame height (two paragraphs)");
    const int bad = forbiddenPixels(drawer);
    check(bad == 0, "spotlight render: drawer with question + answer has zero blue/green pixels");
    const int q = countIn(drawer, QRectF(16, kSpotlightBaseHeight + 10, 300, 20), 2, goldish);
    const int body = countIn(drawer, QRectF(16, kSpotlightBaseHeight + 35, kSpotlightWidth - 32, st.frame.height() - kSpotlightBaseHeight - 40), 2,
        [](const QColor &c) { return c.alpha() > 200 && c.red() > 160 && std::abs(c.red() - c.blue()) < 40; });
    check(q > 30 && body > 300, "spotlight render: gold question line and #d9cfc3 answer body");
    std::printf("spotlight render: forbidden=%d question-gold=%d body=%d\n", bad, q, body);
    check(sp.windowNumber() == 0 && spotlightWindowNumber() == 0, "spotlight render: Offscreen still never creates a window");
    check(sp.render(0).isNull() && sp.render(std::numeric_limits<double>::quiet_NaN()).isNull(), "spotlight render: invalid scale");
    setSpotlightLapisBlue(true);
    const QImage blue = sp.render(2);
    setSpotlightLapisBlue(false);
    check(countIn(blue, QRectF(16, kSpotlightBaseHeight + 10, 300, 20), 2,
              [](const QColor &c) { return c.alpha() > 150 && c.blue() > 200 && c.blue() > c.red() + 100; }) > 30
              && countIn(blue, QRectF(0, 0, kSpotlightWidth, st.frame.height()), 2, goldish) == 0,
        "spotlight render: lapis-blue mode swaps every accent to #4f8cff");
}
// --proofs DIR: the snap PNGs plus a pad-icon sheet, for eyeballing.
void writeProofs(const QString &dir) {
    QDir().mkpath(dir);
    check(snapCursor(dir + "/snap-cursor.png") == 0, "proofs: snap-cursor");
    check(snapSpotlight(dir + "/snap-spotlight.png") == 0, "proofs: snap-spotlight");
    const double scale = 3, rowH = 64, cell = 64, h = 28;
    QImage sheet(int((cell * 12 + 32) * scale), int((rowH * 4 + 32) * scale), QImage::Format_ARGB32_Premultiplied);
    sheet.fill(QColor(0, 0, 0));
    QPainter p(&sheet);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    p.scale(scale, scale);
    const QColor rowBg[] = {QColor(0, 0, 0), QColor(0x0D, 0x0C, 0x0B), QColor(0x0D, 0x0C, 0x0B), QColor(0, 0, 0)};
    for (int row = 0; row < 4; ++row) {
        p.fillRect(QRectF(16, 16 + row * rowH, cell * 12, rowH), rowBg[row]);
        for (int i = 0; i < 12; ++i) {
            const double height = row == 3 ? 16 : h;  // the last row at hint-bar size
            QImage img = renderPadIcon(kAllPadButtons[i], height, scale, row == 1 || row == 2, row == 2);
            const CGSize sz = padIconSize(kAllPadButtons[i], height);
            p.drawImage(QRectF(16 + i * cell + (cell - sz.width) / 2, 16 + row * rowH + (rowH - sz.height) / 2, sz.width, sz.height), img);
        }
    }
    p.end();
    check(sheet.save(dir + "/pad-icons.png"), "proofs: pad icon sheet");
    std::printf("proofs: %s\n", qPrintable(dir));
}
int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    if (const int i = int(app.arguments().indexOf("--proofs")); i > 0 && i + 1 < app.arguments().size()) {
        writeProofs(app.arguments().at(i + 1));
        std::printf("test_cua: %d passed, %d failed\n", passed, failed);
        return failed ? 1 : 0;
    }
    padIconTests();
    goldThemeTests();
    themeSyncTests();
    spotlightRenderTests();
    traceTests();
    shadowTests();
    geometryTests(); leaseTests(); formattingTests(); cancellationTests(); liveTests();
    reticleGeometryTests(); reticleOffscreenTests(); hudTests(); executorTests(); multiMonitorTests();
    spaceSwitchTests();
    hotkeyDragTests();
    spotlightTests();
    if (app.arguments().contains("--capture-smoke")) captureSmokeTest();
    std::printf("test_cua: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
