#include "mac/cua.hpp"
#include "mac/cua_reticle.hpp"
#include "mac/cua_executor.hpp"
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QEventLoop>
#include <QImage>
#include <QJsonDocument>
#include <QJsonObject>
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
    r.advanceClock(0.5);
    r.moveTo({600, 300}, "Going");
    r.advanceClock(0.07);
    st = r.state();
    check(st.gliding && st.tip.x() > 100 && st.tip.x() < 600 && close(st.target, {600, 300}), "reticle: glides mid-arc");
    // Half time is eased progress 0.5: the straight line would be at (350, 200).
    check(st.tip.y() < 200 - 10, "reticle: glide arcs upward, not a straight line");
    r.advanceClock(0.2);
    st = r.state();
    check(!st.gliding && close(st.tip, {600, 300}), "reticle: glide lands exactly");
    check(r.highlightRect({560, 280, 100, 40}, "Clicking Send"), "reticle: valid highlight accepted");
    st = r.state();
    check(st.focusRect == QRectF(560, 280, 100, 40) && st.label == "Clicking Send", "reticle: focus ring on the target");
    check(st.pillRect.top() >= 320 + 4 && disp.contains(st.pillRect), "reticle: pill below the focus ring");
    check(r.highlightRect({300, 470, 100, 25}, "Low"), "reticle: bottom-edge highlight accepted");
    check(r.state().pillRect.bottom() <= 470 - 4 && disp.contains(r.state().pillRect), "reticle: pill flips above near the bottom");
    check(r.pulseClick({610, 300}), "reticle: pulse accepted");
    st = r.state();
    check(st.label == "Low" && st.gliding, "reticle: pulse keeps label, glides to the click point first");
    r.advanceClock(0.15);
    check(r.state().rippleActive && close(r.state().tip, {610, 300}), "reticle: ripple runs once landed");
    const QImage img = r.render(2);
    int sapphire = 0, dark = 0;
    for (int y = 0; y < img.height(); y += 2)
        for (int x = 0; x < img.width(); x += 2) {
            const QColor c = img.pixelColor(x, y);
            sapphire += c.alpha() > 100 && c.blue() > 180 && c.blue() > c.red() + 60;
            dark += c.alpha() > 200 && c.red() < 40 && c.blue() < 50;
        }
    check(img.width() == 1600 && img.height() == 1000, "reticle: render at 2x");
    check(sapphire > 50 && dark > 20, "reticle: render has sapphire rim/ring and obsidian body");
    std::printf("reticle render: sapphire=%d dark=%d samples\n", sapphire, dark);
    r.advanceClock(0.4);
    check(!r.state().rippleActive, "reticle: ripple ends");
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
        check(m.capturedDisplay == 7 && m.capturedWidth == 1280 && m.capturedExclude == QVector<uint32_t>({4242, 5151}),
            "exec L3: captures the primary display at 1280 px, excluding the reticle and HUD windows");
        check(m.groundInstruction == "the gear icon" && m.groundRefine, "exec L3: instruction and refine_crop forwarded");
        // 1440x900 pt at 2x = 2880x1800 px, captured at 1280x800: 250/1000 -> 360 pt, 500/1000 -> 450 pt.
        check(close(res.globalPt, {360, 450}) && close(m.clickPt, {360, 450}), "exec L3: point_1000 -> Quartz global points");
        check(m.effects() == QStringList({"capture", "ground", "reticleMove", "reticlePulse", "click"}),
            "exec L3: capture -> ground -> move -> pulse -> click");
        check(close(m.movePt, {360, 450}) && close(m.pulsePt, {360, 450}), "exec L3: reticle lands on the click point");
        check(expectedCaptureSize(m.displays[0], 1280) == QSizeF(1280, 800), "exec L3: expected capture size 1280x800");
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
        // 1920x1080 at 1x captured at 1280x720: 500/1000 -> 960 pt, 50/1000 -> 54 pt, offset by x=1920.
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
    // Rendered: frosted obsidian body with a sapphire dot in the capsule, nothing above it.
    const QImage img = r.render(2);
    int body = 0, dot = 0, strayAbove = 0;
    for (int y = 0; y < img.height(); ++y)
        for (int x = 0; x < img.width(); x += 2) {
            const QColor c = img.pixelColor(x, y);
            const QPointF pt(x / 2.0, y / 2.0);
            if (QRectF(560, 822, 320, 30).contains(pt)) {
                body += c.alpha() > 200 && c.red() < 40 && c.blue() < 50;
                dot += c.alpha() > 150 && c.blue() > 180 && c.blue() > c.red() + 60 && pt.x() < 580;
            } else if (QRectF(400, 600, 640, 180).contains(pt)) {
                strayAbove += c.alpha() > 0;
            }
        }
    check(body > 1000 && dot > 4 && strayAbove == 0, "hud: render draws the capsule and sapphire dot, only there");
    std::printf("hud render: body=%d dot=%d stray=%d\n", body, dot, strayAbove);
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
int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    geometryTests(); leaseTests(); formattingTests(); cancellationTests(); liveTests();
    reticleGeometryTests(); reticleOffscreenTests(); hudTests(); executorTests(); multiMonitorTests();
    spaceSwitchTests();
    if (app.arguments().contains("--capture-smoke")) captureSmokeTest();
    std::printf("test_cua: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
