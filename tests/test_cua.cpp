#include "mac/cua.hpp"
#include "mac/cua_reticle.hpp"
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
    reticleGeometryTests(); reticleOffscreenTests();
    if (app.arguments().contains("--capture-smoke")) captureSmokeTest();
    std::printf("test_cua: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
