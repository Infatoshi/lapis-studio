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

// One normal-layer (layer 0) on-screen window, front to back. displayId is the display holding
// the window's centre (else the one it overlaps most); 0 when it is on no active display.
struct AppWindowInfo {
    int pid = 0;
    QString appName;
    uint32_t displayId = 0;
    QRectF globalRect;  // Quartz global points
    bool onScreen = false;
};
QVector<AppWindowInfo> listAppWindows();
// Pure: maps a voice/model app hint to the macOS app name ("chrome" -> "Google Chrome",
// "vscode" / "code" -> "Visual Studio Code"); unknown hints come back trimmed, unchanged.
QString canonicalAppName(const QString &appHint);
// Pure: true when a running app's localized name matches a hint (canonical name equality, else
// case-insensitive substring either way of the canonical name).
bool appNameMatches(const QString &appName, const QString &appHint);
// The installed-app bundle id for a canonical app name ("Google Chrome" -> "com.google.Chrome");
// empty when unknown. Pure.
QString knownBundleId(const QString &canonicalName);
// Activates the regular running app matching appHint (never this process); when none runs, it
// launches the installed app (known bundle id, else /Applications/<Name>.app and the system app
// folders). Returns at once: it does not wait for the Space slide (see frontmostAppPid). On
// success, *outDisplayId is the display of its largest on-screen window (0 when none is on screen
// yet), *outPid the app's pid, *outLaunched true when it had to be launched.
// Call on the main thread (AppKit activation).
bool activateAppByName(const QString &appHint, uint32_t *outDisplayId = nullptr, int *outPid = nullptr,
                       bool *outLaunched = nullptr);
// NSWorkspace's frontmost app pid, 0 when none. Main thread (the value updates on its run loop).
int frontmostAppPid();
// Hides this app (NSApp hide:), so a single-screen fullscreen Lapis slides away. Main thread.
bool hideSelfApp();

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
// Same, excluding every listed window (the reticle and the HUD bar); zero ids are ignored.
QByteArray captureDisplayJpeg(uint32_t displayId, int maxWidth, const QVector<uint32_t> &excludeWindowIds,
                            QString *errOut = nullptr);

struct ActionResult { bool ok; QString detail; };
ActionResult axPress(int pid, const QString &titleOrRole, const QRectF &nearRect);
ActionResult axSetValue(int pid, const QString &titleOrRole, const QRectF &nearRect, const QString &value);
struct PointerLeaseConfig { double userMoveAbortPx = 12.0; bool restoreCursor = true; };
// Threshold is measured in Quartz global points (historical Px name), inclusive.
bool shouldRestoreCursor(QPointF saved, QPointF beforeRestore, QPointF expectedTarget, double abortPx);
ActionResult clickAt(QPointF globalPt, bool doubleClick = false, const PointerLeaseConfig &cfg = {});
ActionResult scrollAt(QPointF globalPt, int deltaY);
// '\n' (and "\r\n" / '\r') posts a real Return key (keycode 36) instead of a newline character,
// so address bars and search fields submit.
ActionResult typeText(const QString &text);
// Cmd+A, ~25 ms, then Backspace: clears the focused field before typing. Honours cancelFlag.
ActionResult selectAllAndDelete();
// A desktop keyboard shortcut: "+"-separated modifiers (cmd/command/meta, shift, opt/option/alt,
// ctrl/control) and exactly one key (a..z, 0..9, space, enter/return, tab, esc/escape,
// backspace/delete, forwarddelete, arrows, home/end, pageup/pagedown, f1..f12, - = [ ] / . ,).
// Case-insensitive. display is the clean label ("Cmd+Shift+T"); flags are CGEventFlags. Pure.
struct ParsedHotkey { bool valid = false; uint16_t keyCode = 0; uint64_t flags = 0; QString display; };
ParsedHotkey parseHotkeyCombo(const QString &combo);
// Posts one real keyDown + keyUp carrying the modifier flags (the up always follows a posted down).
ActionResult pressHotkey(const QString &combo);
// A visible left-button drag: move to fromPt, press, `steps` interpolated drags over durationMs,
// release at toPt. The release is always posted once the press went out, even on cancellation.
ActionResult dragBetween(QPointF fromPt, QPointF toPt, int steps = 14, int durationMs = 220,
                         const PointerLeaseConfig &cfg = {});
// Caller latches true on abort and explicitly resets before a new operation.
// A posted mouse/key down is always balanced by an up, even on cancellation.
extern std::atomic<bool> cancelFlag;

} // namespace studio::cua
