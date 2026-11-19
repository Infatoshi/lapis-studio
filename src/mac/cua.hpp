#pragma once

#include <QByteArray>
#include <QPointF>
#include <QRectF>
#include <QSizeF>
#include <QStringList>
#include <QVector>
#include <atomic>
#include <cstdint>

namespace studio::cua {

struct PermissionStatus { bool accessibility, screenRecording, inputMonitoring; };
PermissionStatus checkPermissions(); // Preflight only: never prompts.
void requestAccessibility();
void requestScreenRecording();

struct DisplayInfo {
    uint32_t displayId = 0;
    QString uuid;
    QRectF globalPoints; // Quartz global points, origin at primary display's top left.
    double scale = 1;
    bool primary = false;
};
QVector<DisplayInfo> listDisplays();
// Empty captureSize means native backing pixels; pass actual JPEG dimensions for resized captures.
QPointF capturePixelToGlobalPoint(QPointF pixel, const DisplayInfo &display, QSizeF captureSize = {});
QPointF globalPointToCapturePixel(QPointF point, const DisplayInfo &display, QSizeF captureSize = {});
// AppKit flips around the primary display, NOT the union of all displays.
QPointF globalPointToAppKit(QPointF point, const QRectF &primaryGlobalPoints);

struct AxNode {
    int id = 0;
    QString role, title, value, appName;
    int pid = 0;
    QRectF globalRect;
    QStringList actions;
};
// Snapshots are cached for 150 ms, at most 8 apps and 2000 nodes/app.
// timeoutSec is a total traversal budget; each IPC also has a remaining-budget timeout.
QVector<AxNode> inspectApp(int pid, int maxNodes = 200, int maxDepth = 8, double timeoutSec = 0.25);
QVector<AxNode> inspectDisplay(uint32_t displayId, int maxNodes = 200, double timeoutSec = 0.25);
QString formatNodesForModel(const QVector<AxNode> &nodes, int maxChars = 6000);
// macOS 14+. Worker-thread-only, bounded to 5 s, no implicit permission prompt.
// Always pass the reticle windowNumber: modern ScreenCaptureKit may ignore sharingType.
QByteArray captureDisplayJpeg(uint32_t displayId, int maxWidth = 1280,
                            uint32_t excludeWindowId = 0, QString *errOut = nullptr);

struct ActionResult { bool ok; QString detail; };
ActionResult axPress(int pid, const QString &titleOrRole, const QRectF &nearRect);
ActionResult axSetValue(int pid, const QString &titleOrRole, const QRectF &nearRect, const QString &value);
struct PointerLeaseConfig { double userMoveAbortPx = 12.0; bool restoreCursor = true; };
// Threshold is measured in Quartz global points (historical Px name), inclusive.
bool shouldRestoreCursor(QPointF saved, QPointF beforeRestore, QPointF expectedTarget, double abortPx);
ActionResult clickAt(QPointF globalPt, bool doubleClick = false, const PointerLeaseConfig &cfg = {});
ActionResult scrollAt(QPointF globalPt, int deltaY);
ActionResult typeText(const QString &text);
// Caller latches true on abort and explicitly resets before a new operation.
// A posted mouse/key down is always balanced by an up, even on cancellation.
extern std::atomic<bool> cancelFlag;

} // namespace studio::cua
