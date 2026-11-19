#include "mac/cua.hpp"

#import <AppKit/AppKit.h>
#import <ApplicationServices/ApplicationServices.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <memory>
#include <limits>
#include <mutex>
#include <thread>

namespace studio::cua {
std::atomic<bool> cancelFlag{false};
namespace {
using Clock = std::chrono::steady_clock;
template<class T> struct CFDrop { void operator()(T p) const { if (p) CFRelease(p); } };
template<class T> using CF = std::unique_ptr<std::remove_pointer_t<T>, CFDrop<T>>;
using Element = std::shared_ptr<std::remove_pointer_t<AXUIElementRef>>;
Element own(AXUIElementRef p) { return Element(p, [](AXUIElementRef v) { if (v) CFRelease(v); }); }
bool finite(QPointF p) { return std::isfinite(p.x()) && std::isfinite(p.y()); }
QPointF invalidPoint() { const double n = std::numeric_limits<double>::quiet_NaN(); return {n, n}; }
QSizeF pixels(const DisplayInfo &d, QSizeF size) {
    return size.isEmpty() ? QSizeF(d.globalPoints.width() * d.scale, d.globalPoints.height() * d.scale) : size;
}
struct Budget {
    Clock::time_point end;
    explicit Budget(double seconds) : end(Clock::now() + std::chrono::duration_cast<Clock::duration>(
        std::chrono::duration<double>(std::isfinite(seconds) ? std::clamp(seconds, 0.0, 5.0) : 0.0))) {}
    double left() const { return std::chrono::duration<double>(end - Clock::now()).count(); }
    bool prepare(AXUIElementRef e) const {
        const double remaining = left();
        return !cancelFlag.load() && remaining > 0 && e &&
            AXUIElementSetMessagingTimeout(e, float(std::min(remaining, 0.025))) == kAXErrorSuccess;
    }
};
CF<CFTypeRef> attribute(AXUIElementRef e, CFStringRef name, const Budget &b) {
    CFTypeRef value = nullptr;
    if (b.prepare(e)) AXUIElementCopyAttributeValue(e, name, &value);
    return CF<CFTypeRef>(value);
}
QString stringValue(CFTypeRef value) {
    if (!value) return {};
    if (CFGetTypeID(value) == CFStringGetTypeID()) {
        const auto s = (CFStringRef)value;
        const CFIndex count = std::min<CFIndex>(CFStringGetLength(s), 2048);
        QString out(count, Qt::Uninitialized);
        CFStringGetCharacters(s, CFRangeMake(0, count), reinterpret_cast<UniChar *>(out.data()));
        if (!out.isEmpty() && out.back().isHighSurrogate()) out.chop(1);
        return out;
    }
    if (CFGetTypeID(value) == CFBooleanGetTypeID()) return CFBooleanGetValue((CFBooleanRef)value) ? "true" : "false";
    if (CFGetTypeID(value) == CFNumberGetTypeID()) {
        double n = 0; CFNumberGetValue((CFNumberRef)value, kCFNumberDoubleType, &n); return QString::number(n);
    }
    return {};
}
struct Entry { AxNode node; Element element; };
struct Snapshot { int pid, limit, depth; Clock::time_point time; QVector<Entry> entries; };
std::mutex cacheMutex;
std::deque<Snapshot> cache;
void invalidate(int pid) {
    std::lock_guard lock(cacheMutex);
    std::erase_if(cache, [pid](const Snapshot &s) { return s.pid == pid; });
}
QVector<Entry> scan(int pid, int maxNodes, int maxDepth, const Budget &budget, bool cached) {
    if (pid <= 0 || maxNodes <= 0 || maxDepth < 0 || budget.left() <= 0 || cancelFlag.load() || !AXIsProcessTrusted()) return {};
    maxNodes = std::min(maxNodes, 2000); maxDepth = std::min(maxDepth, 64);
    if (cached) {
        std::lock_guard lock(cacheMutex);
        for (const auto &s : cache)
            if (s.pid == pid && s.limit == maxNodes && s.depth == maxDepth &&
                Clock::now() - s.time < std::chrono::milliseconds(150)) return s.entries;
    }
    QVector<Entry> result;
    std::deque<std::pair<Element, int>> pending;
    pending.push_back({own(AXUIElementCreateApplication(pid)), 0});
    CF<CFMutableSetRef> visited(CFSetCreateMutable(nullptr, 0, &kCFTypeSetCallBacks));
    QString appName;
    while (!pending.empty() && result.size() < maxNodes && budget.left() > 0 && !cancelFlag.load()) {
        auto [e, depth] = pending.front(); pending.pop_front();
        if (!e || CFSetContainsValue(visited.get(), e.get())) continue;
        CFSetAddValue(visited.get(), e.get());
        AxNode n;
        n.id = int(result.size()) + 1; n.pid = pid;
        n.role = stringValue(attribute(e.get(), kAXRoleAttribute, budget).get());
        n.title = stringValue(attribute(e.get(), kAXTitleAttribute, budget).get());
        if (depth == 0) appName = n.title;
        // Chrome's toolbar and omnibox carry their names in AXDescription, not AXTitle.
        else if (n.title.isEmpty()) n.title = stringValue(attribute(e.get(), kAXDescriptionAttribute, budget).get());
        n.appName = appName;
        // Never copy secure text-field contents into model context.
        const auto subrole = stringValue(attribute(e.get(), kAXSubroleAttribute, budget).get());
        if (subrole != "AXSecureTextField") n.value = stringValue(attribute(e.get(), kAXValueAttribute, budget).get());
        auto position = attribute(e.get(), kAXPositionAttribute, budget);
        auto size = attribute(e.get(), kAXSizeAttribute, budget);
        CGPoint p{}; CGSize s{};
        if (position && size && CFGetTypeID(position.get()) == AXValueGetTypeID() && CFGetTypeID(size.get()) == AXValueGetTypeID() &&
            AXValueGetValue((AXValueRef)position.get(), kAXValueTypeCGPoint, &p) &&
            AXValueGetValue((AXValueRef)size.get(), kAXValueTypeCGSize, &s)) n.globalRect = {p.x, p.y, s.width, s.height};
        CFArrayRef actions = nullptr;
        if (budget.prepare(e.get())) AXUIElementCopyActionNames(e.get(), &actions);
        CF<CFArrayRef> actionOwner(actions);
        if (actions) for (CFIndex i = 0; i < std::min<CFIndex>(CFArrayGetCount(actions), 32); ++i)
            n.actions.push_back(stringValue(CFArrayGetValueAtIndex(actions, i)));
        result.push_back({n, e});
        if (depth >= maxDepth || !budget.prepare(e.get())) continue;
        // Bounded array fetch: do not materialize an unbounded AXChildren attribute.
        CFArrayRef children = nullptr;
        const auto capacity = maxNodes - result.size() - qsizetype(pending.size());
        if (capacity > 0) AXUIElementCopyAttributeValues(e.get(), kAXChildrenAttribute, 0, capacity, &children);
        CF<CFArrayRef> childOwner(children);
        if (children) for (CFIndex i = 0; i < CFArrayGetCount(children); ++i) {
            auto child = CFArrayGetValueAtIndex(children, i);
            if (CFGetTypeID(child) == AXUIElementGetTypeID()) pending.push_back({own((AXUIElementRef)CFRetain(child)), depth + 1});
        }
    }
    if (!cancelFlag.load()) {
        std::lock_guard lock(cacheMutex);
        std::erase_if(cache, [pid](const Snapshot &s) { return s.pid == pid; });
        if (cache.size() >= 8) cache.pop_front();
        cache.push_back({pid, maxNodes, maxDepth, Clock::now(), result});
    }
    return result;
}
std::timed_mutex actuator;
ActionResult cancelled() { return {false, "Cancelled"}; }
bool acquire(std::unique_lock<std::timed_mutex> &lock) {
    while (!cancelFlag.load()) if (lock.try_lock_for(std::chrono::milliseconds(2))) return !cancelFlag.load();
    return false;
}
ActionResult axAction(int pid, const QString &key, const QRectF &near, const QString *value) {
    if (cancelFlag.load()) return cancelled();
    if (key.isEmpty() || !near.isValid() || !finite(near.topLeft()) || !finite(near.bottomRight())) return {false, "A title/role and valid target rectangle are required"};
    std::unique_lock lock(actuator, std::defer_lock);
    if (!acquire(lock)) return cancelled();
    Budget budget(0.25);
    // Fresh resolution avoids acting through stale cached UI elements.
    auto entries = scan(pid, 200, 8, budget, false);
    Element best; double distance = std::numeric_limits<double>::infinity(); bool ambiguous = false;
    for (const auto &entry : entries) {
        const auto &n = entry.node;
        if ((n.title != key && n.role != key) || !n.globalRect.intersects(near)) continue;
        if (!value && !n.actions.contains("AXPress")) continue;
        const QPointF d = n.globalRect.center() - near.center();
        const double score = d.x()*d.x() + d.y()*d.y();
        if (score < distance) { best = entry.element; distance = score; ambiguous = false; }
        else if (score == distance) ambiguous = true;
    }
    if (cancelFlag.load()) return cancelled();
    if (!best || ambiguous) return {false, "No unique matching accessible target"};
    AXError error;
    if (value) {
        Boolean settable = false;
        if (!budget.prepare(best.get()) || AXUIElementIsAttributeSettable(best.get(), kAXValueAttribute, &settable) != kAXErrorSuccess || !settable)
            return {false, "Target value is not settable or inspection timed out"};
        CF<CFStringRef> text(value->toCFString());
        if (!budget.prepare(best.get())) return {false, "Cancelled or timed out"};
        error = AXUIElementSetAttributeValue(best.get(), kAXValueAttribute, text.get());
    } else {
        if (!budget.prepare(best.get())) return {false, "Cancelled or timed out"};
        error = AXUIElementPerformAction(best.get(), kAXPressAction);
    }
    invalidate(pid);
    return {error == kAXErrorSuccess, error == kAXErrorSuccess ? "OK" : QString("AX error %1").arg(error)};
}
QPointF cursor() {
    CF<CGEventRef> e(CGEventCreate(nullptr));
    if (!e) return invalidPoint();
    CGPoint p = CGEventGetLocation(e.get()); return {p.x, p.y};
}
bool post(CGEventRef e) { if (!e || cancelFlag.load()) return false; CGEventPost(kCGHIDEventTap, e); return true; }
bool move(QPointF point) {
    // Warping is synchronous. An asynchronously posted move can otherwise leave
    // the old location visible to the immediate user-movement check.
    if (cancelFlag.load()) return false;
    return CGWarpMouseCursorPosition(CGPointMake(point.x(), point.y())) == kCGErrorSuccess;
}
bool activePoint(QPointF p) {
    if (!finite(p)) return false;
    for (const auto &d : listDisplays()) if (d.globalPoints.contains(p)) return true;
    return false;
}
struct CursorLease {
    QPointF saved, target;
    PointerLeaseConfig config;
    bool moved = false;
    ~CursorLease() {
        if (moved && config.restoreCursor && !cancelFlag.load() &&
            shouldRestoreCursor(saved, cursor(), target, config.userMoveAbortPx)) move(saved);
    }
};
} // namespace

PermissionStatus checkPermissions() {
    return {bool(AXIsProcessTrusted()), bool(CGPreflightScreenCaptureAccess()), bool(CGPreflightListenEventAccess())};
}
void requestAccessibility() {
    @autoreleasepool {
        NSDictionary *options = @{(id)kAXTrustedCheckOptionPrompt: @YES};
        AXIsProcessTrustedWithOptions((CFDictionaryRef)options);
    }
}
void requestScreenRecording() { CGRequestScreenCaptureAccess(); }
QVector<DisplayInfo> listDisplays() {
    uint32_t count = 0;
    if (CGGetActiveDisplayList(0, nullptr, &count) != kCGErrorSuccess || !count) return {};
    QVector<CGDirectDisplayID> ids(count);
    if (CGGetActiveDisplayList(count, ids.data(), &count) != kCGErrorSuccess) return {};
    QVector<DisplayInfo> result;
    for (uint32_t i = 0; i < count; ++i) {
        const auto id = ids[i]; const CGRect b = CGDisplayBounds(id);
        CF<CFUUIDRef> uuid(CGDisplayCreateUUIDFromDisplayID(id));
        CF<CFStringRef> text(uuid ? CFUUIDCreateString(nullptr, uuid.get()) : nullptr);
        CF<CGDisplayModeRef> mode(CGDisplayCopyDisplayMode(id));
        // PixelWidth / Width is backing scale, unlike physical-pixel / desktop bounds
        // which is incorrect for scaled Retina display modes.
        double scale = 1;
        if (mode && CGDisplayModeGetWidth(mode.get())) scale = double(CGDisplayModeGetPixelWidth(mode.get())) / CGDisplayModeGetWidth(mode.get());
        result.push_back({id, text ? QString::fromCFString(text.get()) : QString(), {b.origin.x, b.origin.y, b.size.width, b.size.height}, scale, id == CGMainDisplayID()});
    }
    return result;
}
QVector<AppWindowInfo> listAppWindows() {
    QVector<AppWindowInfo> result;
    @autoreleasepool {
        const auto displays = listDisplays();
        CF<CFArrayRef> windows(CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID));
        if (!windows) return result;
        for (CFIndex i = 0; i < CFArrayGetCount(windows.get()); ++i) {
            auto info = (CFDictionaryRef)CFArrayGetValueAtIndex(windows.get(), i);
            auto rawBounds = (CFDictionaryRef)CFDictionaryGetValue(info, kCGWindowBounds);
            auto rawPid = (CFNumberRef)CFDictionaryGetValue(info, kCGWindowOwnerPID);
            auto rawLayer = (CFNumberRef)CFDictionaryGetValue(info, kCGWindowLayer);
            CGRect rect{}; int pid = 0, layer = 0;
            if (!rawBounds || !rawPid || !CGRectMakeWithDictionaryRepresentation(rawBounds, &rect)) continue;
            if (rawLayer) CFNumberGetValue(rawLayer, kCFNumberIntType, &layer);
            if (layer != 0) continue;  // menu bar, Dock, overlays (our own reticle) are not app windows
            CFNumberGetValue(rawPid, kCFNumberIntType, &pid);
            AppWindowInfo w;
            w.pid = pid;
            w.globalRect = {rect.origin.x, rect.origin.y, rect.size.width, rect.size.height};
            if (w.globalRect.isEmpty()) continue;
            w.appName = stringValue(CFDictionaryGetValue(info, kCGWindowOwnerName));
            auto rawOnScreen = (CFBooleanRef)CFDictionaryGetValue(info, kCGWindowIsOnscreen);
            w.onScreen = !rawOnScreen || CFBooleanGetValue(rawOnScreen);
            double bestArea = 0;
            for (const auto &d : displays) {
                if (d.globalPoints.contains(w.globalRect.center())) { w.displayId = d.displayId; break; }
                const QRectF overlap = d.globalPoints.intersected(w.globalRect);
                if (overlap.width() * overlap.height() > bestArea) { bestArea = overlap.width() * overlap.height(); w.displayId = d.displayId; }
            }
            result.push_back(w);
        }
    }
    return result;
}
QString canonicalAppName(const QString &hint) {
    const QString h = hint.simplified();
    const QString k = h.toLower();
    static const QList<std::pair<QStringList, QString>> aliases{
        {{"chrome", "google chrome"}, "Google Chrome"},
        {{"vscode", "vs code", "code", "visual studio code"}, "Visual Studio Code"},
        {{"iterm", "iterm2"}, "iTerm2"},
        {{"edge", "microsoft edge"}, "Microsoft Edge"},
        {{"brave", "brave browser"}, "Brave Browser"},
        {{"settings", "system settings", "system preferences"}, "System Settings"},
        {{"whatsapp"}, "WhatsApp"},
    };
    for (const auto &[keys, name] : aliases) if (keys.contains(k)) return name;
    return h;
}
bool appNameMatches(const QString &appName, const QString &hint) {
    const QString canon = canonicalAppName(hint), name = appName.simplified();
    if (canon.isEmpty() || name.isEmpty()) return false;
    if (name.compare(canon, Qt::CaseInsensitive) == 0) return true;
    return (canon.size() >= 3 && name.contains(canon, Qt::CaseInsensitive)) ||
        (name.size() >= 3 && canon.contains(name, Qt::CaseInsensitive));
}
bool activateAppByName(const QString &hint, uint32_t *outDisplayId) {
    if (outDisplayId) *outDisplayId = 0;
    const QString canon = canonicalAppName(hint);
    if (canon.isEmpty()) return false;
    @autoreleasepool {
        const pid_t self = getpid();
        NSRunningApplication *exact = nil, *loose = nil;
        for (NSRunningApplication *app in NSWorkspace.sharedWorkspace.runningApplications) {
            if (app.processIdentifier == self || app.terminated || app.activationPolicy != NSApplicationActivationPolicyRegular) continue;
            const QString name = QString::fromNSString(app.localizedName ?: @"");
            if (name.compare(canon, Qt::CaseInsensitive) == 0) { exact = app; break; }
            if (!loose && appNameMatches(name, canon)) loose = app;
        }
        NSRunningApplication *app = exact ?: loose;
        if (!app) return false;
        bool activated = false;
        if (@available(macOS 14.0, *)) {
            // Cooperative activation: we are usually frontmost, so hand activation over first.
            if (NSApp) [NSApp yieldActivationToApplication:app];
            activated = [app activateFromApplication:NSRunningApplication.currentApplication options:NSApplicationActivateAllWindows]
                || [app activateWithOptions:NSApplicationActivateAllWindows];
        } else {
            activated = [app activateWithOptions:NSApplicationActivateAllWindows];
        }
        if (!activated) return false;
        if (outDisplayId) {
            double bestArea = 0;
            for (const auto &w : listAppWindows())
                if (w.pid == app.processIdentifier && w.displayId && w.globalRect.width() * w.globalRect.height() > bestArea) {
                    bestArea = w.globalRect.width() * w.globalRect.height();
                    *outDisplayId = w.displayId;
                }
        }
        return true;
    }
}
QPointF capturePixelToGlobalPoint(QPointF p, const DisplayInfo &d, QSizeF size) {
    size = pixels(d, size);
    if (!d.globalPoints.isValid() || !size.isValid() || size.isEmpty() || !std::isfinite(size.width()) || !std::isfinite(size.height())) return invalidPoint();
    return d.globalPoints.topLeft() + QPointF(p.x()*d.globalPoints.width()/size.width(), p.y()*d.globalPoints.height()/size.height());
}
QPointF globalPointToCapturePixel(QPointF p, const DisplayInfo &d, QSizeF size) {
    size = pixels(d, size);
    if (!d.globalPoints.isValid() || !size.isValid() || size.isEmpty() || !std::isfinite(size.width()) || !std::isfinite(size.height())) return invalidPoint();
    p -= d.globalPoints.topLeft(); return {p.x()*size.width()/d.globalPoints.width(), p.y()*size.height()/d.globalPoints.height()};
}
QPointF globalPointToAppKit(QPointF p, const QRectF &primary) { return {p.x(), primary.y() + primary.height() - p.y()}; }
QVector<AxNode> inspectApp(int pid, int maxNodes, int maxDepth, double timeoutSec) {
    @autoreleasepool {
        QVector<AxNode> result;
        for (const auto &e : scan(pid, maxNodes, maxDepth, Budget(timeoutSec), true)) result.push_back(e.node);
        return result;
    }
}
QVector<AxNode> inspectDisplay(uint32_t displayId, int maxNodes, double timeoutSec) {
    Budget budget(timeoutSec); QVector<AxNode> result;
    if (maxNodes <= 0 || cancelFlag.load() || budget.left() <= 0) return result;
    QRectF bounds;
    for (const auto &d : listDisplays()) if (d.displayId == displayId) bounds = d.globalPoints;
    if (!bounds.isValid()) return result;
    maxNodes = std::min(maxNodes, 2000);
    CF<CFArrayRef> windows(CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, kCGNullWindowID));
    if (!windows) return result;
    QVector<int> seen;
    for (CFIndex i = 0; i < CFArrayGetCount(windows.get()) && result.size() < maxNodes && budget.left() > 0 && !cancelFlag.load(); ++i) {
        auto info = (CFDictionaryRef)CFArrayGetValueAtIndex(windows.get(), i);
        CGRect rect{}; int pid = 0;
        auto rawBounds = (CFDictionaryRef)CFDictionaryGetValue(info, kCGWindowBounds);
        auto rawPid = (CFNumberRef)CFDictionaryGetValue(info, kCGWindowOwnerPID);
        if (!rawBounds || !rawPid || !CGRectMakeWithDictionaryRepresentation(rawBounds, &rect)) continue;
        if (!bounds.intersects({rect.origin.x, rect.origin.y, rect.size.width, rect.size.height})) continue;
        CFNumberGetValue(rawPid, kCFNumberIntType, &pid);
        if (seen.contains(pid)) continue;
        seen.push_back(pid);
        for (const auto &e : scan(pid, maxNodes - int(result.size()), 8, budget, true)) {
            if (!e.node.globalRect.intersects(bounds)) continue;
            auto n = e.node; n.id = int(result.size()) + 1; result.push_back(n);
        }
    }
    return result;
}
QString formatNodesForModel(const QVector<AxNode> &nodes, int maxChars) {
    if (maxChars <= 0) return {};
    QString out;
    for (const auto &n : nodes) {
        QJsonObject obj{{"id", n.id}, {"pid", n.pid}, {"app", n.appName.left(maxChars)}, {"role", n.role.left(maxChars)},
            {"title", n.title.left(maxChars)}, {"value", n.value.left(maxChars)},
            {"rect", QJsonArray{n.globalRect.x(), n.globalRect.y(), n.globalRect.width(), n.globalRect.height()}}};
        QJsonArray actions; for (const auto &a : n.actions.mid(0, 32)) actions.append(a.left(maxChars));
        obj.insert("actions", actions);
        QString line = QString::fromUtf8(QJsonDocument(obj).toJson(QJsonDocument::Compact)) + '\n';
        if (out.size() + line.size() > maxChars) {
            // Keep complete JSON lines; a standalone marker signals omitted nodes.
            if (out.size() < maxChars) out += QChar(0x2026);
            break;
        }
        out += line;
    }
    return out;
}

QByteArray captureDisplayJpeg(uint32_t displayId, int maxWidth, uint32_t excludeWindowId, QString *errOut) {
    if (errOut) errOut->clear();
    auto fail = [errOut](const QString &s) { if (errOut) *errOut = s; return QByteArray(); };
    if (cancelFlag.load()) return fail("Cancelled");
    if (maxWidth <= 0 || maxWidth > 16384) return fail("Invalid maximum width");
    if ([NSThread isMainThread]) return fail("Capture requires a worker thread");
    if (!CGPreflightScreenCaptureAccess()) return fail("Screen recording permission is required");
    if (@available(macOS 14.0, *)) {
        struct State {
            std::mutex mutex; std::condition_variable ready; bool done = false;
            std::atomic<bool> abandoned{false};
            QByteArray bytes; QString error;
            void finish(QByteArray b, QString e) { std::lock_guard lock(mutex); bytes = std::move(b); error = std::move(e); done = true; ready.notify_all(); }
        };
        auto state = std::make_shared<State>();
        @autoreleasepool {
            [SCShareableContent getShareableContentWithCompletionHandler:^(SCShareableContent *content, NSError *error) {
                @autoreleasepool {
                    if (error || !content) { state->finish({}, error ? QString::fromNSString(error.localizedDescription) : "No shareable content"); return; }
                    if (cancelFlag.load() || state->abandoned.load()) { state->finish({}, "Cancelled"); return; }
                    SCDisplay *display = nil;
                    for (SCDisplay *d in content.displays) if (d.displayID == displayId) { display = d; break; }
                    if (!display) { state->finish({}, "Display not found"); return; }
                    NSMutableArray<SCWindow *> *excluded = [NSMutableArray array];
                    for (SCWindow *w in content.windows) if (excludeWindowId && w.windowID == excludeWindowId) [excluded addObject:w];
                    SCContentFilter *filter = [[SCContentFilter alloc] initWithDisplay:display excludingWindows:excluded];
                    SCStreamConfiguration *config = [[SCStreamConfiguration alloc] init];
                    const double scale = filter.pointPixelScale;
                    config.width = size_t(std::max(1.0, std::min(double(maxWidth), double(display.width)*scale)));
                    config.height = size_t(std::max(1.0, std::round(double(config.width)*display.height/std::max<NSInteger>(1, display.width))));
                    config.showsCursor = NO;
                    [SCScreenshotManager captureImageWithFilter:filter configuration:config completionHandler:^(CGImageRef img, NSError *captureError) {
                        @autoreleasepool {
                            if (!img || captureError) { state->finish({}, captureError ? QString::fromNSString(captureError.localizedDescription) : "Capture returned no image"); return; }
                            NSBitmapImageRep *rep = [[NSBitmapImageRep alloc] initWithCGImage:img];
                            NSData *jpeg = [rep representationUsingType:NSBitmapImageFileTypeJPEG properties:@{NSImageCompressionFactor: @0.8}];
                            QByteArray bytes((const char *)jpeg.bytes, qsizetype(jpeg.length));
                            [rep release];
                            state->finish(bytes, bytes.isEmpty() ? QString("JPEG encoding failed") : QString());
                        }
                    }];
                    [config release]; [filter release];
                }
            }];
        }
        const auto deadline = Clock::now() + std::chrono::seconds(5);
        std::unique_lock lock(state->mutex);
        while (!state->done && !cancelFlag.load() && Clock::now() < deadline) state->ready.wait_for(lock, std::chrono::milliseconds(10));
        if (cancelFlag.load()) { state->abandoned.store(true); return fail("Cancelled"); }
        if (!state->done) { state->abandoned.store(true); return fail("Capture timed out"); }
        if (errOut) *errOut = state->error;
        return state->bytes;
    }
    return fail("Screen capture requires macOS 14 or later");
}
ActionResult axPress(int pid, const QString &key, const QRectF &rect) { return axAction(pid, key, rect, nullptr); }
ActionResult axSetValue(int pid, const QString &key, const QRectF &rect, const QString &value) { return axAction(pid, key, rect, &value); }
bool shouldRestoreCursor(QPointF saved, QPointF before, QPointF target, double threshold) {
    return finite(saved) && finite(before) && finite(target) && std::isfinite(threshold) && threshold >= 0 &&
        std::hypot(before.x() - target.x(), before.y() - target.y()) <= threshold;
}
ActionResult clickAt(QPointF point, bool doubleClick, const PointerLeaseConfig &cfg) {
    if (cancelFlag.load()) return cancelled();
    if (!activePoint(point) || !std::isfinite(cfg.userMoveAbortPx) || cfg.userMoveAbortPx < 0) return {false, "Invalid pointer target or threshold"};
    std::unique_lock lock(actuator, std::defer_lock);
    if (!acquire(lock)) return cancelled();
    if (!AXIsProcessTrusted()) return {false, "Accessibility permission is required"};
    const auto saved = cursor();
    if (!finite(saved)) return {false, "Cannot read cursor"};
    CursorLease lease{saved, point, cfg};
    if (!move(point)) return cancelFlag.load() ? cancelled() : ActionResult{false, "Pointer move failed"};
    lease.moved = true;
    for (int i = 1; i <= (doubleClick ? 2 : 1); ++i) {
        if (cancelFlag.load()) return cancelled();
        if (!shouldRestoreCursor(saved, cursor(), point, cfg.userMoveAbortPx)) return {false, "User moved pointer; lease aborted"};
        CF<CGEventRef> down(CGEventCreateMouseEvent(nullptr, kCGEventLeftMouseDown, CGPointMake(point.x(), point.y()), kCGMouseButtonLeft));
        CF<CGEventRef> up(CGEventCreateMouseEvent(nullptr, kCGEventLeftMouseUp, CGPointMake(point.x(), point.y()), kCGMouseButtonLeft));
        if (!down || !up) return {false, "Cannot create mouse events"};
        CGEventSetIntegerValueField(down.get(), kCGMouseEventClickState, i);
        CGEventSetIntegerValueField(up.get(), kCGMouseEventClickState, i);
        if (!post(down.get())) return cancelled();
        // Safety cleanup is unconditional: never strand a pressed mouse button.
        CGEventPost(kCGHIDEventTap, up.get());
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    if (cancelFlag.load()) return cancelled();
    return cancelFlag.load() ? cancelled() : ActionResult{true, "Click posted"};
}
ActionResult scrollAt(QPointF point, int deltaY) {
    if (cancelFlag.load()) return cancelled();
    if (!activePoint(point)) return {false, "Invalid scroll target"};
    std::unique_lock lock(actuator, std::defer_lock);
    if (!acquire(lock)) return cancelled();
    if (!AXIsProcessTrusted()) return {false, "Accessibility permission is required"};
    const auto saved = cursor();
    if (!finite(saved)) return {false, "Cannot read cursor"};
    CursorLease lease{saved, point, {}};
    if (!move(point)) return cancelFlag.load() ? cancelled() : ActionResult{false, "Pointer move failed"};
    lease.moved = true;
    if (!shouldRestoreCursor(saved, cursor(), point, 12)) return {false, "User moved pointer; lease aborted"};
    CF<CGEventRef> event(CGEventCreateScrollWheelEvent(nullptr, kCGScrollEventUnitPixel, 1, deltaY));
    if (!event) return {false, "Cannot create scroll event"};
    CGEventSetLocation(event.get(), CGPointMake(point.x(), point.y()));
    if (!post(event.get())) return cancelled();
    return cancelFlag.load() ? cancelled() : ActionResult{true, "Scroll posted"};
}
ActionResult typeText(const QString &text) {
    if (cancelFlag.load()) return cancelled();
    std::unique_lock lock(actuator, std::defer_lock);
    if (!acquire(lock)) return cancelled();
    if (!AXIsProcessTrusted()) return {false, "Accessibility permission is required"};
    for (qsizetype offset = 0; offset < text.size();) {
        if (cancelFlag.load()) return cancelled();
        const QChar c = text.at(offset);
        if (c == '\n' || c == '\r') {
            // A real Return key: a typed newline character is ignored by address bars.
            offset += (c == '\r' && offset + 1 < text.size() && text.at(offset + 1) == '\n') ? 2 : 1;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let autocomplete settle
            CF<CGEventRef> down(CGEventCreateKeyboardEvent(nullptr, CGKeyCode(36), true));
            CF<CGEventRef> up(CGEventCreateKeyboardEvent(nullptr, CGKeyCode(36), false));
            if (!down || !up) return {false, "Cannot create keyboard events"};
            CGEventSetFlags(down.get(), 0);
            CGEventSetFlags(up.get(), 0);
            if (!post(down.get())) return cancelled();
            CGEventPost(kCGHIDEventTap, up.get());
            continue;
        }
        qsizetype count = std::min<qsizetype>(20, text.size() - offset);
        for (qsizetype i = 0; i < count; ++i)
            if (text.at(offset + i) == '\n' || text.at(offset + i) == '\r') { count = i; break; }
        if (offset + count < text.size() && text.at(offset + count - 1).isHighSurrogate()) --count;
        CF<CGEventRef> down(CGEventCreateKeyboardEvent(nullptr, 0, true));
        CF<CGEventRef> up(CGEventCreateKeyboardEvent(nullptr, 0, false));
        if (!down || !up) return {false, "Cannot create keyboard events"};
        CGEventSetFlags(down.get(), 0);
        CGEventSetFlags(up.get(), 0);
        CGEventKeyboardSetUnicodeString(down.get(), count, reinterpret_cast<const UniChar *>(text.utf16() + offset));
        CGEventKeyboardSetUnicodeString(up.get(), count, reinterpret_cast<const UniChar *>(text.utf16() + offset));
        if (!post(down.get())) return cancelled();
        CGEventPost(kCGHIDEventTap, up.get());
        offset += count;
    }
    return cancelFlag.load() ? cancelled() : ActionResult{true, "Text posted"};
}
} // namespace studio::cua
