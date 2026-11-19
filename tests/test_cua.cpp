#include "mac/cua.hpp"
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
    if (app.arguments().contains("--capture-smoke")) captureSmokeTest();
    std::printf("test_cua: %d passed, %d failed\n", passed, failed);
    return failed ? 1 : 0;
}
