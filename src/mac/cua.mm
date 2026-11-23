#include "mac/cua.hpp"

#import <AppKit/AppKit.h>
#import <ApplicationServices/ApplicationServices.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
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
    Element app = own(AXUIElementCreateApplication(pid));
    if (!app) return result;
    // Unlock full DOM accessibility trees in Electron/Chromium apps (Discord, Slack, VS Code).
    AXUIElementSetAttributeValue(app.get(), CFSTR("AXManualAccessibility"), kCFBooleanTrue);
    // Google Chrome only exposes its live web DOM tree (buttons, inputs, sliders, links) with this flag.
    AXUIElementSetAttributeValue(app.get(), CFSTR("AXEnhancedUserInterface"), kCFBooleanTrue);
    CF<CFMutableSetRef> visitedSet(CFSetCreateMutable(nullptr, 0, &kCFTypeSetCallBacks));
    QString appName;
    const auto visit = [&](const Element &e) {
        if (!e || CFSetContainsValue(visitedSet.get(), e.get())) return false;
        CFSetAddValue(visitedSet.get(), e.get());
        return true;
    };
    const auto read = [&](const Element &e, int depth) {
        AxNode n;
        n.pid = pid;
        n.role = stringValue(attribute(e.get(), kAXRoleAttribute, budget).get());
        n.title = stringValue(attribute(e.get(), kAXTitleAttribute, budget).get());
        if (depth == 0) appName = n.title;
        // Chrome's toolbar and omnibox carry their names in AXDescription, not AXTitle.
        else if (n.title.isEmpty()) n.title = stringValue(attribute(e.get(), kAXDescriptionAttribute, budget).get());
        n.appName = appName;
        // Never copy secure text-field contents into model context.
        const auto subrole = stringValue(attribute(e.get(), kAXSubroleAttribute, budget).get());
        if (subrole != "AXSecureTextField") n.value = stringValue(attribute(e.get(), kAXValueAttribute, budget).get());
        // An unnamed wrapper is walked through, never shown: skip its rect and action IPCs.
        if (axStructuralNoise(n, depth)) return n;
        auto position = attribute(e.get(), kAXPositionAttribute, budget);
        auto size = attribute(e.get(), kAXSizeAttribute, budget);
        CGPoint p{}; CGSize s{};
        if (position && size && CFGetTypeID(position.get()) == AXValueGetTypeID() && CFGetTypeID(size.get()) == AXValueGetTypeID() &&
            AXValueGetValue((AXValueRef)position.get(), kAXValueTypeCGPoint, &p) &&
            AXValueGetValue((AXValueRef)size.get(), kAXValueTypeCGSize, &s)) n.globalRect = {p.x, p.y, s.width, s.height};
        if (axTinyRect(n, depth)) return n;
        CFArrayRef actions = nullptr;
        if (budget.prepare(e.get())) AXUIElementCopyActionNames(e.get(), &actions);
        CF<CFArrayRef> actionOwner(actions);
        if (actions) for (CFIndex i = 0; i < std::min<CFIndex>(CFArrayGetCount(actions), 32); ++i)
            n.actions.push_back(stringValue(CFArrayGetValueAtIndex(actions, i)));
        return n;
    };
    const auto children = [&](const Element &e, int capacity) {
        QVector<Element> out;
        if (!budget.prepare(e.get())) return out;
        // Bounded array fetch: do not materialize an unbounded AXChildren attribute.
        CFArrayRef kids = nullptr;
        AXUIElementCopyAttributeValues(e.get(), kAXChildrenAttribute, 0, capacity, &kids);
        CF<CFArrayRef> owner(kids);
        if (kids) for (CFIndex i = 0; i < CFArrayGetCount(kids); ++i) {
            auto child = CFArrayGetValueAtIndex(kids, i);
            if (CFGetTypeID(child) == AXUIElementGetTypeID()) out.push_back(own((AXUIElementRef)CFRetain(child)));
        }
        return out;
    };
    const auto more = [&] { return budget.left() > 0 && !cancelFlag.load(); };
    const AxWalkLimits limits{maxNodes, maxDepth, std::max(maxNodes * 8, 3200)};
    for (auto &[n, e] : walkAxTree(std::move(app), limits, visit, read, children, more)) result.push_back({n, e});
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
    Budget budget(0.6);
    // Fresh resolution avoids acting through stale cached UI elements (as deep as inspectDisplay).
    auto entries = scan(pid, 500, 32, budget, false);
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
        {{"safari"}, "Safari"},
        {{"finder"}, "Finder"},
        {{"discord"}, "Discord"},
        {{"slack"}, "Slack"},
        {{"spotify"}, "Spotify"},
        {{"terminal"}, "Terminal"},
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
QString knownBundleId(const QString &name) {
    static const QList<std::pair<QString, QString>> ids{
        {"Google Chrome", "com.google.Chrome"}, {"Visual Studio Code", "com.microsoft.VSCode"},
        {"Safari", "com.apple.Safari"}, {"Finder", "com.apple.finder"}, {"Discord", "com.hnc.Discord"},
        {"Slack", "com.tinyspeck.slackmacgap"}, {"Spotify", "com.spotify.client"}, {"Terminal", "com.apple.Terminal"},
        {"System Settings", "com.apple.systempreferences"}, {"iTerm2", "com.googlecode.iterm2"},
        {"Microsoft Edge", "com.microsoft.edgemac"}, {"Brave Browser", "com.brave.Browser"},
        {"Firefox", "org.mozilla.firefox"}, {"Xcode", "com.apple.dt.Xcode"},
    };
    for (const auto &[n, id] : ids) if (n.compare(name, Qt::CaseInsensitive) == 0) return id;
    return {};
}
namespace {
// The installed .app for a canonical name: its bundle id first, then the usual app folders.
NSURL *installedAppUrl(const QString &canon) {
    const QString bundle = knownBundleId(canon);
    if (!bundle.isEmpty())
        if (NSURL *u = [NSWorkspace.sharedWorkspace URLForApplicationWithBundleIdentifier:bundle.toNSString()]) return u;
    if (canon.contains('/') || canon.startsWith('.')) return nil;  // a name, never a path
    const QString file = canon + ".app";
    for (const QString &dir : {QStringLiteral("/Applications"), QStringLiteral("/System/Applications"),
                               QStringLiteral("/System/Applications/Utilities"), QDir::homePath() + "/Applications"}) {
        const QString path = dir + '/' + file;
        if (QFileInfo(path).isDir()) return [NSURL fileURLWithPath:path.toNSString() isDirectory:YES];
    }
    return nil;
}
// Launches and activates an installed app; waits (bounded, 4 s) for the process to exist.
NSRunningApplication *launchApp(NSURL *url) {
    __block NSRunningApplication *launched = nil;
    dispatch_semaphore_t done = dispatch_semaphore_create(0);
    NSWorkspaceOpenConfiguration *config = [NSWorkspaceOpenConfiguration configuration];
    config.activates = YES;
    // The completion handler runs on a concurrent queue, so waiting here cannot deadlock.
    [NSWorkspace.sharedWorkspace openApplicationAtURL:url configuration:config
                                    completionHandler:^(NSRunningApplication *app, NSError *error) {
        if (app && !error) launched = [app retain];
        dispatch_semaphore_signal(done);
    }];
    const bool finished = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 4 * NSEC_PER_SEC)) == 0;
    dispatch_release(done);
    return finished ? [launched autorelease] : nil;  // on timeout the late handler's retain leaks one object, never crashes
}
// The display of pid's largest on-screen window; 0 when it has none.
uint32_t largestWindowDisplay(int pid) {
    uint32_t id = 0;
    double bestArea = 0;
    for (const auto &w : listAppWindows())
        if (w.pid == pid && w.displayId && w.globalRect.width() * w.globalRect.height() > bestArea) {
            bestArea = w.globalRect.width() * w.globalRect.height();
            id = w.displayId;
        }
    return id;
}
// Polls (60 ms apart, up to 750 ms) for pid's first on-screen window; its display, else 0.
uint32_t waitForWindow(int pid) {
    for (int waited = 0;; waited += 60) {
        if (const uint32_t id = largestWindowDisplay(pid)) return id;
        if (waited + 60 > 750 || cancelFlag.load()) return 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
    }
}
// The regular running app matching a canonical name (never this process); nil when none runs.
NSRunningApplication *runningApp(const QString &canon) {
    const pid_t self = getpid();
    NSRunningApplication *loose = nil;
    for (NSRunningApplication *app in NSWorkspace.sharedWorkspace.runningApplications) {
        if (app.processIdentifier == self || app.terminated || app.activationPolicy != NSApplicationActivationPolicyRegular) continue;
        const QString name = QString::fromNSString(app.localizedName ?: @"");
        if (name.compare(canon, Qt::CaseInsensitive) == 0) return app;
        if (!loose && appNameMatches(name, canon)) loose = app;
    }
    return loose;
}
}
bool activateAppByName(const QString &hint, uint32_t *outDisplayId, int *outPid, bool *outLaunched) {
    if (outDisplayId) *outDisplayId = 0;
    if (outPid) *outPid = 0;
    if (outLaunched) *outLaunched = false;
    const QString canon = canonicalAppName(hint);
    if (canon.isEmpty()) return false;
    @autoreleasepool {
        NSRunningApplication *app = runningApp(canon);
        bool activated = false;
        if (app) {
            if (@available(macOS 14.0, *)) {
                // Cooperative activation: we are usually frontmost, so hand activation over first.
                if (NSApp) [NSApp yieldActivationToApplication:app];
                activated = [app activateFromApplication:NSRunningApplication.currentApplication options:NSApplicationActivateAllWindows]
                    || [app activateWithOptions:NSApplicationActivateAllWindows];
            } else {
                activated = [app activateWithOptions:NSApplicationActivateAllWindows];
            }
            // Running with no window (Chrome in the Dock, every window closed): reopen it the way
            // a Dock click does (openApplicationAtURL sends kAEReopenApplication), so Chrome,
            // Finder, Safari and Terminal open a fresh window instead of activating nothing.
            if (activated && app.bundleURL && !largestWindowDisplay(app.processIdentifier)) {
                if (NSRunningApplication *reopened = launchApp(app.bundleURL)) app = reopened;
                const uint32_t id = waitForWindow(app.processIdentifier);
                if (outPid) *outPid = app.processIdentifier;
                if (outDisplayId) *outDisplayId = id;
                return true;
            }
        } else if (NSURL *url = installedAppUrl(canon)) {
            // Not running: launch it ("open x.com in Chrome" works with Chrome closed).
            if (@available(macOS 14.0, *)) {
                NSString *bundle = [NSBundle bundleWithURL:url].bundleIdentifier;
                if (NSApp && bundle) [NSApp yieldActivationToApplicationWithBundleIdentifier:bundle];
            }
            app = launchApp(url);
            activated = app != nil;
            if (activated && outLaunched) *outLaunched = true;
        }
        if (!activated) return false;
        if (outPid) *outPid = app.processIdentifier;
        if (outDisplayId) *outDisplayId = largestWindowDisplay(app.processIdentifier);
        return true;
    }
}
QString normalizeOpenUrl(const QString &raw) {
    const QString u = raw.trimmed();
    if (u.isEmpty() || u.contains(QRegularExpression(QStringLiteral("\\s")))) return {};
    static const QRegularExpression scheme(QStringLiteral("^[a-zA-Z][a-zA-Z0-9+.-]*:"));
    if (scheme.match(u).hasMatch() && !u.contains(QRegularExpression(QStringLiteral("^[^/:]+:\\d"))))
        return u;  // https:, file:, mailto: ... ("localhost:8080" is a port, not a scheme)
    // domain.tld[/...] or localhost[:port][/...]
    static const QRegularExpression host(QStringLiteral(
        "^(localhost|[a-zA-Z0-9-]+(\\.[a-zA-Z0-9-]+)*\\.[a-zA-Z]{2,}|\\d{1,3}(\\.\\d{1,3}){3})(:\\d+)?([/?#].*)?$"));
    if (!host.match(u).hasMatch()) return {};
    return QStringLiteral("https://") + u;
}
bool openUrlInApp(const QString &hint, const QString &rawUrl, uint32_t *outDisplayId, int *outPid, bool *outLaunched,
                  QString *outAppName) {
    if (outDisplayId) *outDisplayId = 0;
    if (outPid) *outPid = 0;
    if (outLaunched) *outLaunched = false;
    if (outAppName) outAppName->clear();
    const QString url = normalizeOpenUrl(rawUrl);
    if (url.isEmpty()) return false;
    @autoreleasepool {
        NSURL *nsUrl = [NSURL URLWithString:url.toNSString()];
        if (!nsUrl) return false;
        const QString canon = canonicalAppName(hint);
        NSURL *appUrl = nil;
        NSRunningApplication *before = nil;
        if (!canon.isEmpty()) {
            before = runningApp(canon);
            appUrl = before.bundleURL ?: installedAppUrl(canon);
            if (!appUrl) return false;  // a named app that is neither running nor installed
        } else {
            appUrl = [NSWorkspace.sharedWorkspace URLForApplicationToOpenURL:nsUrl];
            if (appUrl) {
                NSString *bundle = [NSBundle bundleWithURL:appUrl].bundleIdentifier;
                for (NSRunningApplication *a in NSWorkspace.sharedWorkspace.runningApplications)
                    if (bundle && [a.bundleIdentifier isEqualToString:bundle]) { before = a; break; }
            }
        }
        if (@available(macOS 14.0, *)) {
            NSString *bundle = appUrl ? [NSBundle bundleWithURL:appUrl].bundleIdentifier : nil;
            if (NSApp && bundle) [NSApp yieldActivationToApplicationWithBundleIdentifier:bundle];
        }
        __block NSRunningApplication *opened = nil;
        dispatch_semaphore_t done = dispatch_semaphore_create(0);
        NSWorkspaceOpenConfiguration *config = [NSWorkspaceOpenConfiguration configuration];
        config.activates = YES;
        // A browser opens a URL handed over by the system in a new tab: the user's current tab
        // and the omnibox (inline autocomplete) are never involved.
        void (^handler)(NSRunningApplication *, NSError *) = ^(NSRunningApplication *app, NSError *error) {
            if (app && !error) opened = [app retain];
            dispatch_semaphore_signal(done);
        };
        if (appUrl)
            [NSWorkspace.sharedWorkspace openURLs:@[nsUrl] withApplicationAtURL:appUrl configuration:config completionHandler:handler];
        else
            [NSWorkspace.sharedWorkspace openURL:nsUrl configuration:config completionHandler:handler];
        const bool finished = dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 4 * NSEC_PER_SEC)) == 0;
        dispatch_release(done);
        NSRunningApplication *app = finished ? [opened autorelease] : nil;
        if (!app) return false;
        if (outPid) *outPid = app.processIdentifier;
        if (outLaunched) *outLaunched = before == nil;
        if (outAppName) *outAppName = QString::fromNSString(app.localizedName ?: @"");
        if (outDisplayId) *outDisplayId = waitForWindow(app.processIdentifier);
        return true;
    }
}
int frontmostAppPid() {
    @autoreleasepool {
        NSRunningApplication *front = NSWorkspace.sharedWorkspace.frontmostApplication;
        return front ? int(front.processIdentifier) : 0;
    }
}
bool hideSelfApp() {
    if (!NSApp) return false;
    [NSApp hide:nil];
    return true;
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
    // Candidate PIDs with a normal-layer window on this display, in front-to-back z-order.
    QVector<int> pids;
    for (CFIndex i = 0; i < CFArrayGetCount(windows.get()); ++i) {
        auto info = (CFDictionaryRef)CFArrayGetValueAtIndex(windows.get(), i);
        CGRect rect{}; int pid = 0, layer = 0;
        auto rawBounds = (CFDictionaryRef)CFDictionaryGetValue(info, kCGWindowBounds);
        auto rawPid = (CFNumberRef)CFDictionaryGetValue(info, kCGWindowOwnerPID);
        auto rawLayer = (CFNumberRef)CFDictionaryGetValue(info, kCGWindowLayer);
        if (!rawBounds || !rawPid || !CGRectMakeWithDictionaryRepresentation(rawBounds, &rect)) continue;
        if (rawLayer) CFNumberGetValue(rawLayer, kCFNumberIntType, &layer);
        if (layer != 0) continue;  // menu bar, Dock, WindowServer and overlays must not starve app windows
        if (!bounds.intersects({rect.origin.x, rect.origin.y, rect.size.width, rect.size.height})) continue;
        CFNumberGetValue(rawPid, kCFNumberIntType, &pid);
        if (pid > 0 && !pids.contains(pid)) pids.push_back(pid);
    }
    // The frontmost app is scanned first with the full budget; Lapis itself only when frontmost or alone.
    const int front = frontmostAppPid();
    if (front > 0 && pids.removeAll(front) > 0) pids.prepend(front);
    const int self = int(getpid());
    if (self != front && pids.size() > 1) pids.removeAll(self);
    for (int pid : pids) {
        if (result.size() >= maxNodes || budget.left() <= 0 || cancelFlag.load()) break;
        // Background apps get a capped slice so they can never burn the timeout.
        const Budget appBudget = pid == front ? budget : Budget(std::min(budget.left(), 0.35));
        for (const auto &e : scan(pid, maxNodes - int(result.size()), 32, appBudget, true)) {
            if (!e.node.globalRect.intersects(bounds)) continue;
            auto n = e.node; n.id = int(result.size()) + 1; result.push_back(n);
        }
    }
    return result;
}
bool axStructuralNoise(const AxNode &n, int depth) {
    static const QStringList wrappers{"AXGroup", "AXGenericElement", "AXUnknown", "AXSection", "AXList", "AXScrollArea",
        "AXSplitGroup", "AXLayoutArea", "AXLandmarkMain", "AXLandmarkNavigation", "AXLandmarkBanner", "AXLandmarkRegion",
        "AXWebArea"};
    return depth > 0 && n.title.trimmed().isEmpty() && n.value.trimmed().isEmpty() && wrappers.contains(n.role);
}
bool axTinyRect(const AxNode &n, int depth) {
    return depth > 0 && !(n.globalRect.width() >= 2 && n.globalRect.height() >= 2);
}
QString formatNodesForModel(const QVector<AxNode> &nodes, int maxChars) {
    if (maxChars <= 0) return {};
    // One giant paragraph must never eat the whole budget.
    constexpr int kTextCap = 280;
    const auto cap = [&](const QString &s) {
        const int n = std::min(maxChars, kTextCap);
        return s.size() > n ? s.left(n - 1) + QChar(0x2026) : s;
    };
    // A JSON-style string literal: quotes, backslashes and control characters escaped, so every
    // node stays on one physical line.
    const auto quoted = [](const QString &s) {
        QString out; out.reserve(s.size() + 2); out += QLatin1Char('"');
        for (const QChar c : s) {
            if (c == QLatin1Char('"') || c == QLatin1Char('\\')) { out += QLatin1Char('\\'); out += c; }
            else if (c == QLatin1Char('\n')) out += QStringLiteral("\\n");
            else if (c == QLatin1Char('\r')) out += QStringLiteral("\\r");
            else if (c == QLatin1Char('\t')) out += QStringLiteral("\\t");
            else if (c.unicode() < 0x20) out += QStringLiteral("\\u%1").arg(uint(c.unicode()), 4, 16, QLatin1Char('0'));
            else out += c;
        }
        out += QLatin1Char('"');
        return out;
    };
    const auto bare = [](const QString &ax) { return ax.startsWith(QLatin1String("AX")) ? ax.mid(2) : ax; };
    // Compact lines, about a third of the tokens of a JSON object per node: an app header when the
    // app changes, then `Role "title" = "value" @x,y wxh {Press}` (actions every node has dropped).
    QString out;
    QString lastApp;
    int lastPid = -1;
    for (const auto &n : nodes) {
        QString block;
        if (n.appName != lastApp || n.pid != lastPid)
            block += QStringLiteral("[%1 pid %2]\n").arg(n.appName.left(maxChars)).arg(n.pid);
        QString line = bare(n.role.left(maxChars));
        if (!n.title.isEmpty()) line += QLatin1Char(' ') + quoted(cap(n.title));
        if (!n.value.isEmpty()) line += QStringLiteral(" = ") + quoted(cap(n.value));
        const QRectF &r = n.globalRect;
        line += QStringLiteral(" @%1,%2 %3x%4").arg(qRound(r.x())).arg(qRound(r.y())).arg(qRound(r.width())).arg(qRound(r.height()));
        QStringList acts;
        for (const auto &a : n.actions.mid(0, 32))
            if (a != QLatin1String("AXShowMenu") && a != QLatin1String("AXScrollToVisible")) acts << bare(a.left(maxChars));
        if (!acts.isEmpty()) line += QStringLiteral(" {") + acts.join(QLatin1Char(',')) + QLatin1Char('}');
        block += line + QLatin1Char('\n');
        if (out.size() + block.size() > maxChars) {
            // Keep complete lines; a standalone marker signals omitted nodes.
            if (out.size() < maxChars) out += QChar(0x2026);
            break;
        }
        out += block;
        lastApp = n.appName;
        lastPid = n.pid;
    }
    return out;
}

QByteArray captureDisplayJpeg(uint32_t displayId, int maxWidth, uint32_t excludeWindowId, QString *errOut) {
    return captureDisplayJpeg(displayId, maxWidth, excludeWindowId ? QVector<uint32_t>{excludeWindowId} : QVector<uint32_t>{}, errOut);
}
QByteArray captureDisplayJpeg(uint32_t displayId, int maxWidth, const QVector<uint32_t> &excludeIds, QString *errOut, double quality) {
    if (errOut) errOut->clear();
    const double jpegQuality = std::isfinite(quality) ? std::clamp(quality, 0.05, 1.0) : 0.8;
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
        const QVector<uint32_t> exclude = excludeIds;  // the block may outlive this call (timeout)
        @autoreleasepool {
            [SCShareableContent getShareableContentWithCompletionHandler:^(SCShareableContent *content, NSError *error) {
                @autoreleasepool {
                    if (error || !content) { state->finish({}, error ? QString::fromNSString(error.localizedDescription) : "No shareable content"); return; }
                    if (cancelFlag.load() || state->abandoned.load()) { state->finish({}, "Cancelled"); return; }
                    SCDisplay *display = nil;
                    for (SCDisplay *d in content.displays) if (d.displayID == displayId) { display = d; break; }
                    if (!display) { state->finish({}, "Display not found"); return; }
                    NSMutableArray<SCWindow *> *excluded = [NSMutableArray array];
                    for (SCWindow *w in content.windows) if (w.windowID && exclude.contains(uint32_t(w.windowID))) [excluded addObject:w];
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
                            NSData *jpeg = [rep representationUsingType:NSBitmapImageFileTypeJPEG properties:@{NSImageCompressionFactor: @(jpegQuality)}];
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
    // deltaY is in scroll notches (positive = up, "down" = -5): each is kScrollPxPerNotch points,
    // posted as a few smooth pixel ticks so web pages and native lists both move visibly.
    const int totalPx = std::clamp(deltaY * kScrollPxPerNotch, -kScrollMaxPx, kScrollMaxPx);
    int sent = 0;
    for (int i = 1; i <= kScrollTicks; ++i) {
        const int target = int(std::lround(double(totalPx) * i / kScrollTicks));
        const int step = target - sent;
        sent = target;
        if (!step) continue;
        CF<CGEventRef> event(CGEventCreateScrollWheelEvent(nullptr, kCGScrollEventUnitPixel, 1, step));
        if (!event) return {false, "Cannot create scroll event"};
        CGEventSetLocation(event.get(), CGPointMake(point.x(), point.y()));
        if (!post(event.get())) return cancelled();
        if (i < kScrollTicks) std::this_thread::sleep_for(std::chrono::milliseconds(kScrollTickMs));
    }
    return cancelFlag.load() ? cancelled() : ActionResult{true, QString("Scrolled %1 px").arg(totalPx)};
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
            const bool afterText = offset > 0 && text.at(offset - 1) != '\n' && text.at(offset - 1) != '\r';
            offset += (c == '\r' && offset + 1 < text.size() && text.at(offset + 1) == '\n') ? 2 : 1;
            std::this_thread::sleep_for(std::chrono::milliseconds(30));  // let autocomplete settle
            if (afterText) {
                // Omnibox inline autocomplete selects a suffix ("x.com/pmarca[/followers]"):
                // ForwardDelete drops exactly that selection. At the end of a plain field (the
                // caret after what we just typed) it deletes nothing.
                CF<CGEventRef> fdDown(CGEventCreateKeyboardEvent(nullptr, CGKeyCode(117), true));
                CF<CGEventRef> fdUp(CGEventCreateKeyboardEvent(nullptr, CGKeyCode(117), false));
                if (!fdDown || !fdUp) return {false, "Cannot create keyboard events"};
                CGEventSetFlags(fdDown.get(), 0);
                CGEventSetFlags(fdUp.get(), 0);
                if (!post(fdDown.get())) return cancelled();
                CGEventPost(kCGHIDEventTap, fdUp.get());
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
                if (cancelFlag.load()) return cancelled();
            }
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
ActionResult selectAllAndDelete() {
    if (cancelFlag.load()) return cancelled();
    std::unique_lock lock(actuator, std::defer_lock);
    if (!acquire(lock)) return cancelled();
    if (!AXIsProcessTrusted()) return {false, "Accessibility permission is required"};
    // One key press with explicit flags; the up is always posted once the down went out.
    const auto press = [](CGKeyCode key, CGEventFlags flags) -> ActionResult {
        CF<CGEventRef> down(CGEventCreateKeyboardEvent(nullptr, key, true));
        CF<CGEventRef> up(CGEventCreateKeyboardEvent(nullptr, key, false));
        if (!down || !up) return {false, "Cannot create keyboard events"};
        CGEventSetFlags(down.get(), flags);
        CGEventSetFlags(up.get(), flags);
        if (!post(down.get())) return cancelled();
        CGEventPost(kCGHIDEventTap, up.get());
        return {true, {}};
    };
    if (const ActionResult a = press(CGKeyCode(0), kCGEventFlagMaskCommand); !a.ok) return a;  // Cmd+A
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    if (cancelFlag.load()) return cancelled();
    if (const ActionResult d = press(CGKeyCode(51), CGEventFlags(0)); !d.ok) return d;  // Backspace
    return cancelFlag.load() ? cancelled() : ActionResult{true, "Cleared"};
}
ParsedHotkey parseHotkeyCombo(const QString &combo) {
    struct Key { const char *name; uint16_t code; const char *label; };
    // ANSI virtual key codes (HIToolbox Events.h).
    static const Key keys[] = {
        {"a", 0, "A"}, {"b", 11, "B"}, {"c", 8, "C"}, {"d", 2, "D"}, {"e", 14, "E"}, {"f", 3, "F"}, {"g", 5, "G"},
        {"h", 4, "H"}, {"i", 34, "I"}, {"j", 38, "J"}, {"k", 40, "K"}, {"l", 37, "L"}, {"m", 46, "M"}, {"n", 45, "N"},
        {"o", 31, "O"}, {"p", 35, "P"}, {"q", 12, "Q"}, {"r", 15, "R"}, {"s", 1, "S"}, {"t", 17, "T"}, {"u", 32, "U"},
        {"v", 9, "V"}, {"w", 13, "W"}, {"x", 7, "X"}, {"y", 16, "Y"}, {"z", 6, "Z"},
        {"0", 29, "0"}, {"1", 18, "1"}, {"2", 19, "2"}, {"3", 20, "3"}, {"4", 21, "4"}, {"5", 23, "5"}, {"6", 22, "6"},
        {"7", 26, "7"}, {"8", 28, "8"}, {"9", 25, "9"},
        {"space", 49, "Space"}, {"enter", 36, "Enter"}, {"return", 36, "Enter"}, {"tab", 48, "Tab"},
        {"esc", 53, "Esc"}, {"escape", 53, "Esc"}, {"backspace", 51, "Backspace"}, {"delete", 51, "Backspace"},
        {"forwarddelete", 117, "ForwardDelete"}, {"up", 126, "Up"}, {"down", 125, "Down"}, {"left", 123, "Left"},
        {"right", 124, "Right"}, {"home", 115, "Home"}, {"end", 119, "End"}, {"pageup", 116, "PageUp"},
        {"pagedown", 121, "PageDown"}, {"f1", 122, "F1"}, {"f2", 120, "F2"}, {"f3", 99, "F3"}, {"f4", 118, "F4"},
        {"f5", 96, "F5"}, {"f6", 97, "F6"}, {"f7", 98, "F7"}, {"f8", 100, "F8"}, {"f9", 101, "F9"}, {"f10", 109, "F10"},
        {"f11", 103, "F11"}, {"f12", 111, "F12"},
        {"-", 27, "-"}, {"=", 24, "="}, {"[", 33, "["}, {"]", 30, "]"}, {"/", 44, "/"}, {".", 47, "."}, {",", 43, ","}};
    ParsedHotkey out;
    const QStringList parts = combo.trimmed().toLower().split(QLatin1Char('+'));
    bool cmd = false, shift = false, opt = false, ctrl = false, haveKey = false;
    const Key *key = nullptr;
    for (const QString &raw : parts) {
        const QString p = raw.trimmed();
        if (p.isEmpty()) return {};
        if (p == "cmd" || p == "command" || p == "meta") cmd = true;
        else if (p == "shift") shift = true;
        else if (p == "opt" || p == "option" || p == "alt") opt = true;
        else if (p == "ctrl" || p == "control") ctrl = true;
        else {
            if (haveKey) return {};  // one key per shortcut
            for (const Key &k : keys)
                if (p == QLatin1String(k.name)) { key = &k; break; }
            if (!key) return {};
            haveKey = true;
        }
    }
    if (!key) return {};
    QStringList label;
    if (cmd) { label << "Cmd"; out.flags |= kCGEventFlagMaskCommand; }
    if (ctrl) { label << "Ctrl"; out.flags |= kCGEventFlagMaskControl; }
    if (opt) { label << "Opt"; out.flags |= kCGEventFlagMaskAlternate; }
    if (shift) { label << "Shift"; out.flags |= kCGEventFlagMaskShift; }
    label << QLatin1String(key->label);
    out.valid = true;
    out.keyCode = key->code;
    out.display = label.join(QLatin1Char('+'));
    return out;
}
ActionResult pressHotkey(const QString &combo) {
    if (cancelFlag.load()) return cancelled();
    const ParsedHotkey hk = parseHotkeyCombo(combo);
    if (!hk.valid) return {false, QString("Invalid hotkey: %1").arg(combo.simplified())};
    std::unique_lock lock(actuator, std::defer_lock);
    if (!acquire(lock)) return cancelled();
    if (!AXIsProcessTrusted()) return {false, "Accessibility permission is required"};
    CF<CGEventRef> down(CGEventCreateKeyboardEvent(nullptr, CGKeyCode(hk.keyCode), true));
    CF<CGEventRef> up(CGEventCreateKeyboardEvent(nullptr, CGKeyCode(hk.keyCode), false));
    if (!down || !up) return {false, "Cannot create keyboard events"};
    CGEventSetFlags(down.get(), CGEventFlags(hk.flags));
    CGEventSetFlags(up.get(), CGEventFlags(hk.flags));
    if (!post(down.get())) return cancelled();
    // Safety cleanup is unconditional: never strand a pressed key.
    CGEventPost(kCGHIDEventTap, up.get());
    return cancelFlag.load() ? cancelled() : ActionResult{true, QString("Pressed %1").arg(hk.display)};
}
ActionResult dragBetween(QPointF from, QPointF to, int steps, int durationMs, const PointerLeaseConfig &cfg) {
    if (cancelFlag.load()) return cancelled();
    if (!activePoint(from) || !activePoint(to) || !std::isfinite(cfg.userMoveAbortPx) || cfg.userMoveAbortPx < 0)
        return {false, "Invalid drag endpoints or threshold"};
    steps = std::clamp(steps, 1, 200);
    durationMs = std::clamp(durationMs, 0, 5000);
    std::unique_lock lock(actuator, std::defer_lock);
    if (!acquire(lock)) return cancelled();
    if (!AXIsProcessTrusted()) return {false, "Accessibility permission is required"};
    const auto saved = cursor();
    if (!finite(saved)) return {false, "Cannot read cursor"};
    CursorLease lease{saved, to, cfg};
    if (!move(from)) return cancelFlag.load() ? cancelled() : ActionResult{false, "Pointer move failed"};
    lease.moved = true;
    if (!shouldRestoreCursor(saved, cursor(), from, cfg.userMoveAbortPx)) return {false, "User moved pointer; lease aborted"};
    const auto mouse = [](CGEventType type, QPointF p) {
        CGEventRef e = CGEventCreateMouseEvent(nullptr, type, CGPointMake(p.x(), p.y()), kCGMouseButtonLeft);
        if (e) CGEventSetIntegerValueField(e, kCGMouseEventClickState, 1);
        return CF<CGEventRef>(e);
    };
    if (CF<CGEventRef> hover = mouse(kCGEventMouseMoved, from); !post(hover.get()))
        return cancelFlag.load() ? cancelled() : ActionResult{false, "Cannot create mouse events"};
    CF<CGEventRef> down = mouse(kCGEventLeftMouseDown, from);
    if (!down) return {false, "Cannot create mouse events"};
    if (!post(down.get())) return cancelled();
    // The button is down from here: every exit path below releases it.
    QPointF last = from;
    const auto pause = std::chrono::microseconds(int64_t(durationMs) * 1000 / steps);
    for (int i = 1; i <= steps && !cancelFlag.load(); ++i) {
        std::this_thread::sleep_for(pause);
        const double t = double(i) / steps;
        const QPointF p = from + (to - from) * t;
        CF<CGEventRef> drag = mouse(kCGEventLeftMouseDragged, p);
        if (!drag || !post(drag.get())) break;
        last = p;
    }
    CF<CGEventRef> up = mouse(kCGEventLeftMouseUp, last);
    if (!up) return {false, "Cannot create mouse-up event"};
    CGEventPost(kCGHIDEventTap, up.get());  // unconditional, unlike post(): never strand the button
    if (cancelFlag.load()) return cancelled();
    if (last != to) return {false, "Drag interrupted"};
    return {true, "Drag posted"};
}
} // namespace studio::cua
