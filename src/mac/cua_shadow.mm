#include "mac/cua_shadow.hpp"

#import <AppKit/AppKit.h>
#import <ApplicationServices/ApplicationServices.h>
#include "mac/cua_executor.hpp"
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSaveFile>
#include <QSet>
#include <QTimeZone>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>
#include <unistd.h>

namespace studio::cua {

namespace {

QString cfString(CFTypeRef v)
{
    if (!v || CFGetTypeID(v) != CFStringGetTypeID())
        return {};
    const auto s = (CFStringRef)v;
    const CFIndex n = std::min<CFIndex>(CFStringGetLength(s), 512);
    QString out(n, Qt::Uninitialized);
    CFStringGetCharacters(s, CFRangeMake(0, n), reinterpret_cast<UniChar *>(out.data()));
    if (!out.isEmpty() && out.back().isHighSurrogate())
        out.chop(1);
    return out;
}

QString axString(AXUIElementRef e, CFStringRef attr)
{
    CFTypeRef v = nullptr;
    if (AXUIElementCopyAttributeValue(e, attr, &v) != kAXErrorSuccess || !v)
        return {};
    const QString s = cfString(v);
    CFRelease(v);
    return s;
}

QRectF axRect(AXUIElementRef e)
{
    CFTypeRef pos = nullptr, size = nullptr;
    AXUIElementCopyAttributeValue(e, kAXPositionAttribute, &pos);
    AXUIElementCopyAttributeValue(e, kAXSizeAttribute, &size);
    QRectF r;
    CGPoint p{};
    CGSize s{};
    if (pos && size && CFGetTypeID(pos) == AXValueGetTypeID() && CFGetTypeID(size) == AXValueGetTypeID()
        && AXValueGetValue((AXValueRef)pos, kAXValueTypeCGPoint, &p) && AXValueGetValue((AXValueRef)size, kAXValueTypeCGSize, &s))
        r = QRectF(p.x, p.y, s.width, s.height);
    if (pos)
        CFRelease(pos);
    if (size)
        CFRelease(size);
    return r;
}

// The owner of the frontmost on-screen window containing pt (thread-safe Quartz window list):
// fills the app name when AX is unavailable or reports none.
QString windowOwnerAt(QPointF pt, int *pidOut)
{
    CFArrayRef list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
                                                 kCGNullWindowID);
    if (!list)
        return {};
    QString name;
    for (CFIndex i = 0; i < CFArrayGetCount(list) && name.isEmpty(); ++i) {
        const auto w = (CFDictionaryRef)CFArrayGetValueAtIndex(list, i);
        CGRect b{};
        const auto bounds = (CFDictionaryRef)CFDictionaryGetValue(w, kCGWindowBounds);
        if (!bounds || !CGRectMakeWithDictionaryRepresentation(bounds, &b))
            continue;
        if (!QRectF(b.origin.x, b.origin.y, b.size.width, b.size.height).contains(pt))
            continue;
        name = cfString(CFDictionaryGetValue(w, kCGWindowOwnerName));
        int pid = 0;
        if (const auto n = (CFNumberRef)CFDictionaryGetValue(w, kCGWindowOwnerPID))
            CFNumberGetValue(n, kCFNumberIntType, &pid);
        if (pidOut)
            *pidOut = pid;
    }
    CFRelease(list);
    return name;
}

ShadowAxHit realElementAt(QPointF pt)
{
    ShadowAxHit hit;
    if (AXIsProcessTrusted()) {
        AXUIElementRef sys = AXUIElementCreateSystemWide();
        AXUIElementSetMessagingTimeout(sys, 0.25f);
        AXUIElementRef el = nullptr;
        if (AXUIElementCopyElementAtPosition(sys, float(pt.x()), float(pt.y()), &el) == kAXErrorSuccess && el) {
            AXUIElementSetMessagingTimeout(el, 0.25f);
            hit.ok = true;
            hit.role = axString(el, kAXRoleAttribute);
            hit.subrole = axString(el, kAXSubroleAttribute);
            hit.title = axString(el, kAXTitleAttribute);
            hit.description = axString(el, kAXDescriptionAttribute);
            hit.globalRect = axRect(el);
            pid_t pid = 0;
            if (AXUIElementGetPid(el, &pid) == kAXErrorSuccess && pid > 0) {
                hit.pid = int(pid);
                @autoreleasepool {
                    NSRunningApplication *app = [NSRunningApplication runningApplicationWithProcessIdentifier:pid];
                    if (app.localizedName)
                        hit.appName = QString::fromNSString(app.localizedName);
                }
            }
            CFRelease(el);
        }
        CFRelease(sys);
    }
    if (hit.appName.isEmpty()) {
        int pid = 0;
        hit.appName = windowOwnerAt(pt, &pid);
        if (!hit.pid)
            hit.pid = pid;
    }
    return hit;
}

const char *const kSensitiveWords[] = {"infinity", "1password", "keychain", "bitwarden"};

}  // namespace

ShadowHooks defaultShadowHooks(std::function<QVector<uint32_t>()> excludeIds)
{
    ShadowHooks h;
    h.listDisplays = [] { return studio::cua::listDisplays(); };
    h.elementAt = &realElementAt;
    h.captureJpeg = [excludeIds](uint32_t id, int w) {
        QString err;
        return studio::cua::captureDisplayJpeg(id, w, excludeIds ? excludeIds() : QVector<uint32_t>{}, &err);
    };
    h.selfPid = int(getpid());
    return h;
}

bool shadowSensitive(const ShadowAxHit &hit)
{
    if (hit.role == QLatin1String("AXSecureTextField") || hit.subrole == QLatin1String("AXSecureTextField"))
        return true;
    for (const QString *f : {&hit.appName, &hit.title, &hit.description})
        for (const char *w : kSensitiveWords)
            if (f->contains(QLatin1String(w), Qt::CaseInsensitive))
                return true;
    return false;
}

bool shadowRateAllows(qint64 lastMs, qint64 nowMs, int minGapMs)
{
    return lastMs <= 0 || nowMs - lastMs >= minGapMs || nowMs < lastMs;  // a clock step back never wedges it
}

void pruneHumanTraces(const QString &traceDir, int maxSteps)
{
    if (maxSteps <= 0)
        return;
    QFile f(traceDir + QLatin1Char('/') + QLatin1String(kShadowLog));
    if (!f.open(QIODevice::ReadOnly))
        return;
    QList<QByteArray> lines = f.readAll().split('\n');
    f.close();
    lines.removeIf([](const QByteArray &l) { return l.trimmed().isEmpty(); });
    static_assert(kShadowDefaultMaxSteps + kShadowDefaultMaxSteps * 3 / 22 == kShadowPruneAtSteps);
    const qint64 pruneAt = maxSteps + qint64(maxSteps) * 3 / 22;
    if (lines.size() <= pruneAt)
        return;
    lines = lines.mid(lines.size() - maxSteps);
    QSaveFile out(f.fileName());
    if (!out.open(QIODevice::WriteOnly))
        return;
    QSet<QString> keep;
    for (const QByteArray &l : lines) {
        out.write(l);
        out.write("\n");
        keep << QJsonDocument::fromJson(l).object().value(QStringLiteral("image")).toString().section(QLatin1Char('/'), -1);
    }
    if (!out.commit())
        return;
    QDir frames(traceDir + QLatin1Char('/') + QLatin1String(kShadowFrames));
    for (const QString &name : frames.entryList(QDir::Files))
        if (!keep.contains(name))
            frames.remove(name);
}

ShadowResult recordClickSample(const QString &traceDir, const ShadowClick &click, const ShadowHooks &hooks, int maxSteps)
{
    ShadowResult res;
    if (traceDir.isEmpty() || !hooks.listDisplays || !hooks.captureJpeg)
        return res;
    // Privacy first: nothing is captured until the target is known to be safe. An unknown app
    // (no AX and no window owner) cannot be checked, so it is skipped too, as is Lapis itself
    // (its cards can show redacted sessions).
    const ShadowAxHit hit = hooks.elementAt ? hooks.elementAt(click.globalPt) : ShadowAxHit{};
    if (shadowSensitive(hit) || hit.appName.trimmed().isEmpty() || (hooks.selfPid > 0 && hit.pid == hooks.selfPid)) {
        res.outcome = ShadowOutcome::SkippedSensitive;
        return res;
    }
    DisplayInfo display;
    for (const DisplayInfo &d : hooks.listDisplays())
        if (d.globalPoints.adjusted(0, 0, 0.001, 0.001).contains(click.globalPt)) {
            display = d;
            break;
        }
    if (!display.displayId || display.globalPoints.isEmpty()) {
        res.outcome = ShadowOutcome::SkippedNoDisplay;
        return res;
    }
    const QByteArray jpeg = hooks.captureJpeg(display.displayId, kShadowFrameWidth);
    if (jpeg.isEmpty()) {
        res.outcome = ShadowOutcome::SkippedNoFrame;
        return res;
    }
    const QString frames = traceDir + QLatin1Char('/') + QLatin1String(kShadowFrames);
    if (!QDir().mkpath(frames))
        return res;
    qint64 ts = click.tsMs > 0 ? click.tsMs : QDateTime::currentMSecsSinceEpoch();
    while (QFile::exists(QStringLiteral("%1/%2.jpg").arg(frames).arg(ts)))
        ++ts;  // two samples in one millisecond keep separate frames
    const QString rel = QStringLiteral("%1/%2.jpg").arg(QLatin1String(kShadowFrames)).arg(ts);
    {
        QFile f(traceDir + QLatin1Char('/') + rel);
        if (!f.open(QIODevice::WriteOnly) || f.write(jpeg) != jpeg.size())
            return res;
    }
    QJsonObject o;
    o[QStringLiteral("ts")] = QDateTime::fromMSecsSinceEpoch(ts, QTimeZone::UTC).toString(Qt::ISODateWithMs);
    o[QStringLiteral("source")] = QStringLiteral("human_click");
    o[QStringLiteral("button")] = click.right ? QStringLiteral("right") : QStringLiteral("left");
    o[QStringLiteral("app")] = hit.appName;
    o[QStringLiteral("role")] = hit.role;
    o[QStringLiteral("title")] = hit.title;
    o[QStringLiteral("description")] = hit.description;
    o[QStringLiteral("display_w")] = display.globalPoints.width();
    o[QStringLiteral("display_h")] = display.globalPoints.height();
    const QPointF p = toDisplay1000(click.globalPt, display);
    o[QStringLiteral("point_1000")] = QJsonArray{qRound(p.x()), qRound(p.y())};
    const QRectF onDisplay = hit.globalRect.intersected(display.globalPoints);
    if (hit.globalRect.isValid() && !onDisplay.isEmpty()) {
        const QRectF b = toDisplay1000(onDisplay, display);
        o[QStringLiteral("bbox_1000")] = QJsonArray{qRound(b.left()), qRound(b.top()), qRound(b.right()), qRound(b.bottom())};
    } else {
        o[QStringLiteral("bbox_1000")] = QJsonValue::Null;
    }
    o[QStringLiteral("image")] = rel;
    res.line = QJsonDocument(o).toJson(QJsonDocument::Compact);
    QFile log(traceDir + QLatin1Char('/') + QLatin1String(kShadowLog));
    if (!log.open(QIODevice::WriteOnly | QIODevice::Append) || log.write(res.line + '\n') != res.line.size() + 1) {
        QFile::remove(traceDir + QLatin1Char('/') + rel);
        res.line.clear();
        return res;
    }
    log.close();
    pruneHumanTraces(traceDir, maxSteps);
    res.outcome = ShadowOutcome::Recorded;
    res.image = rel;
    return res;
}

// --- the recorder ------------------------------------------------------------------------

struct CuaShadowRecorder::Impl {
    ShadowHooks hooks;
    bool customHooks = false;
    QString dir;
    int maxSteps = kShadowDefaultMaxSteps;

    std::atomic<bool> running{false};
    std::atomic<int> recorded{0};
    std::atomic<qint64> lastCaptureMs{0};

    // Tap thread: owns the run loop the tap's source is attached to.
    CFMachPortRef tap = nullptr;
    CFRunLoopRef tapLoop = nullptr;
    std::atomic<bool> tapStop{false};
    std::thread tapThread;

    // Worker thread: one sample at a time; the queue holds at most 2 pending clicks.
    std::mutex mu;
    std::condition_variable cv;
    std::deque<ShadowClick> queue;
    bool quit = false;
    std::thread worker;

    std::mutex excludeMu;
    QVector<uint32_t> excludeIds;

    static CGEventRef onEvent(CGEventTapProxy, CGEventType type, CGEventRef event, void *refcon)
    {
        auto *self = static_cast<Impl *>(refcon);
        if (type == kCGEventTapDisabledByTimeout || type == kCGEventTapDisabledByUserInput) {
            if (self->tap)
                CGEventTapEnable(self->tap, true);
            return event;
        }
        if (type != kCGEventLeftMouseDown && type != kCGEventRightMouseDown)
            return event;
        const qint64 now = QDateTime::currentMSecsSinceEpoch();
        qint64 last = self->lastCaptureMs.load(std::memory_order_relaxed);
        if (!shadowRateAllows(last, now) || !self->lastCaptureMs.compare_exchange_strong(last, now))
            return event;
        const CGPoint loc = CGEventGetLocation(event);
        {
            std::lock_guard lock(self->mu);
            if (self->queue.size() < 2)
                self->queue.push_back({QPointF(loc.x, loc.y), type == kCGEventRightMouseDown, now});
        }
        self->cv.notify_one();
        return event;  // listen-only: the event always passes through untouched
    }

    void workerLoop()
    {
        for (;;) {
            ShadowClick c;
            {
                std::unique_lock lock(mu);
                cv.wait(lock, [this] { return quit || !queue.empty(); });
                if (quit)
                    return;
                c = queue.front();
                queue.pop_front();
            }
            @autoreleasepool {
                if (recordClickSample(dir, c, hooks, maxSteps).outcome == ShadowOutcome::Recorded)
                    recorded.fetch_add(1);
            }
        }
    }
};

CuaShadowRecorder::CuaShadowRecorder() : d(std::make_unique<Impl>()) {}

CuaShadowRecorder::~CuaShadowRecorder()
{
    stop();
}

void CuaShadowRecorder::setHooks(ShadowHooks hooks)
{
    if (d->running.load())
        return;
    d->hooks = std::move(hooks);
    d->customHooks = true;
}

void CuaShadowRecorder::setExcludeWindowIds(const QVector<uint32_t> &ids)
{
    std::lock_guard lock(d->excludeMu);
    d->excludeIds = ids;
}

bool CuaShadowRecorder::start(const QString &traceDir, int maxSteps)
{
    if (d->running.load() || traceDir.isEmpty())
        return false;
    if (!d->customHooks) {
        Impl *impl = d.get();
        d->hooks = defaultShadowHooks([impl] {
            std::lock_guard lock(impl->excludeMu);
            return impl->excludeIds;
        });
    }
    d->dir = traceDir;
    d->maxSteps = maxSteps > 0 ? maxSteps : kShadowDefaultMaxSteps;
    const CGEventMask mask = CGEventMaskBit(kCGEventLeftMouseDown) | CGEventMaskBit(kCGEventRightMouseDown);
    d->tap = CGEventTapCreate(kCGSessionEventTap, kCGHeadInsertEventTap, kCGEventTapOptionListenOnly, mask,
                              &Impl::onEvent, d.get());
    if (!d->tap) {
        std::fprintf(stderr, "lapis_studio: shadow recorder off (event tap denied: grant Input Monitoring)\n");
        return false;
    }
    d->quit = false;
    d->tapStop.store(false);
    d->queue.clear();
    d->lastCaptureMs.store(0);
    d->worker = std::thread([impl = d.get()] { impl->workerLoop(); });

    std::mutex readyMu;
    std::condition_variable readyCv;
    bool ready = false;
    d->tapThread = std::thread([impl = d.get(), &readyMu, &readyCv, &ready] {
        CFRunLoopSourceRef src = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, impl->tap, 0);
        CFRunLoopRef loop = CFRunLoopGetCurrent();
        CFRunLoopAddSource(loop, src, kCFRunLoopCommonModes);
        CGEventTapEnable(impl->tap, true);
        {
            std::lock_guard lock(readyMu);
            impl->tapLoop = (CFRunLoopRef)CFRetain(loop);
            ready = true;
        }
        readyCv.notify_one();
        // Polled flag, not a bare CFRunLoopRun: a stop() racing this line can never be lost.
        while (!impl->tapStop.load())
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.5, false);
        CFRunLoopRemoveSource(loop, src, kCFRunLoopCommonModes);
        CFRelease(src);
    });
    {
        std::unique_lock lock(readyMu);
        readyCv.wait(lock, [&] { return ready; });
    }
    d->running.store(true);
    return true;
}

void CuaShadowRecorder::stop()
{
    if (!d->running.exchange(false))
        return;
    if (d->tap)
        CGEventTapEnable(d->tap, false);
    d->tapStop.store(true);
    if (d->tapLoop) {
        CFRunLoopStop(d->tapLoop);
        CFRunLoopWakeUp(d->tapLoop);
    }
    if (d->tapThread.joinable())
        d->tapThread.join();
    if (d->tapLoop) {
        CFRelease(d->tapLoop);
        d->tapLoop = nullptr;
    }
    if (d->tap) {
        CFMachPortInvalidate(d->tap);
        CFRelease(d->tap);
        d->tap = nullptr;
    }
    {
        std::lock_guard lock(d->mu);
        d->quit = true;
        d->queue.clear();
    }
    d->cv.notify_all();
    if (d->worker.joinable())
        d->worker.join();
}

bool CuaShadowRecorder::isRunning() const
{
    return d->running.load();
}

int CuaShadowRecorder::recordedCount() const
{
    return d->recorded.load();
}

}  // namespace studio::cua
