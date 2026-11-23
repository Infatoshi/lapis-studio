// lapis_studio --bench-cua: the real CuaExecutor click path, timed hook by hook, with no input.
//
// AX scans, screen captures, JPEG encodes, trace writes and the executor's own fixed pauses all
// run for real; the mouse, the overlay, app activation and Lapis hiding are stubbed, so nothing
// moves on screen and no window opens. The target app is treated as already frontmost (the
// common case for a second click in one task). Traces go to a temp dir, never the user's log.
#include "mac/cua_bench.hpp"

#include "mac/cua.hpp"
#include "mac/cua_executor.hpp"

#include <CoreFoundation/CoreFoundation.h>
#include <QCoreApplication>
#include <QElapsedTimer>
#include <QMap>
#include <QTemporaryDir>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <thread>

namespace studio::cua {

namespace {

struct Ledger {
    std::mutex mu;
    QMap<QString, double> ms;
    QMap<QString, int> calls;
    void add(const QString &k, double v)
    {
        std::lock_guard lock(mu);
        ms[k] += v;
        calls[k] += 1;
    }
};

template <typename F>
auto timed(Ledger &l, const QString &key, F &&f)
{
    QElapsedTimer t;
    t.start();
    if constexpr (std::is_void_v<decltype(f())>) {
        f();
        l.add(key, t.nsecsElapsed() / 1e6);
    } else {
        auto r = f();
        l.add(key, t.nsecsElapsed() / 1e6);
        return r;
    }
}

}  // namespace

int benchCua(const QString &target, int runs, const QString &vlmEndpoint)
{
    QTemporaryDir traceDir;
    const CuaHooks real = defaultCuaHooks();
    const QString hint = extractAppHint(target);
    int appPid = 0;
    uint32_t appDisplay = 0;
    for (const AppWindowInfo &w : real.listAppWindows())
        if (!hint.isEmpty() && appNameMatches(w.appName, hint) && w.displayId) {
            appPid = w.pid;
            appDisplay = w.displayId;
            break;
        }
    std::printf("bench-cua: target '%s' app hint '%s' pid %d display %u, %d run(s)\n", qPrintable(target),
                qPrintable(hint), appPid, appDisplay, runs);
    if (!appPid) {
        QStringList apps;
        for (const AppWindowInfo &w : real.listAppWindows())
            if (!apps.contains(w.appName))
                apps << w.appName;
        std::fprintf(stderr, "bench-cua: no on-screen window for '%s' (on screen: %s)\n", qPrintable(hint),
                     qPrintable(apps.join(QStringLiteral(", "))));
        return 2;
    }
    int failures = 0;
    for (int run = 0; run < runs; ++run) {
        Ledger led;
        CuaHooks h = real;
        h.listDisplays = [&] { return timed(led, "listDisplays", real.listDisplays); };
        h.listAppWindows = [&] { return timed(led, "listAppWindows", real.listAppWindows); };
        h.inspectDisplay = [&](uint32_t id) { return timed(led, "ax_scan", [&] { return real.inspectDisplay(id); }); };
        h.captureDisplayJpeg = [&](uint32_t id, int w, const QVector<uint32_t> &ex, QString *err) {
            QByteArray j = timed(led, QStringLiteral("capture_%1").arg(w), [&] { return real.captureDisplayJpeg(id, w, ex, err); });
            if (j.isEmpty() && err)
                std::fprintf(stderr, "bench-cua: capture failed: %s\n", qPrintable(*err));
            return j;
        };
        h.captureObservationJpeg = [&](uint32_t id, const QVector<uint32_t> &ex) {
            return timed(led, "capture_obs", [&] { return real.captureObservationJpeg(id, ex); });
        };
        h.inklingGround = [&](const QString &e, const QString &m, const QByteArray &j, const QString &i) {
            return timed(led, "inkling_ground", [&] { return real.inklingGround(e, m, j, i); });
        };
        h.pause = [&](int ms) {
            timed(led, QStringLiteral("pause_%1").arg(ms), [&] { real.pause(ms); });
        };
        const ActionResult ok{true, QStringLiteral("bench: no input")};
        h.clickAt = [&](QPointF, bool) { return ok; };
        h.axPress = [&](int, const QString &, const QRectF &) { return ok; };
        h.scrollAt = [&](QPointF, int) { return ok; };
        h.typeText = [&](const QString &) { return ok; };
        h.selectAllAndDelete = [&] { return ok; };
        h.reticleMove = nullptr;
        h.reticleHighlight = nullptr;
        h.reticlePulse = nullptr;
        h.reticleHide = nullptr;
        h.reticleHideAfter = nullptr;
        h.reticleHud = nullptr;
        h.reticleWindowId = nullptr;
        h.hudWindowId = nullptr;
        h.hideSelf = nullptr;
        h.activateApp = [&](const QString &) {
            AppActivation a;
            a.ok = true;
            a.pid = appPid;
            a.displayId = appDisplay;
            return a;
        };
        h.frontmostPid = [&] { return appPid; };
        h.traceDir = traceDir.path();
        CuaExecResult r;
        double totalMs = 0;
        std::atomic<bool> done{false};
        std::thread worker([&] {  // capture refuses the main thread, like the live pool
            CuaExecutor ex(h);
            ex.setVlmEndpoint(vlmEndpoint);
            ex.setAttachScreenshots(true);
            TraceContext tc;
            tc.goal = QStringLiteral("bench-cua");
            ex.setTraceContext(tc);
            QElapsedTimer t;
            t.start();
            r = ex.clickTarget(target);
            totalMs = t.nsecsElapsed() / 1e6;
            done.store(true);
        });
        // ScreenCaptureKit answers on the main queue: keep it turning, as the live app's does.
        // The offscreen platform's Qt loop never drains the main dispatch queue: run CF's too.
        while (!done.load()) {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 1);
            CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.002, true);
        }
        worker.join();
        failures += r.ok ? 0 : 1;
        QStringList parts;
        double accounted = 0;
        for (auto it = led.ms.cbegin(); it != led.ms.cend(); ++it) {
            parts << QStringLiteral("%1=%2ms(x%3)").arg(it.key()).arg(it.value(), 0, 'f', 1).arg(led.calls.value(it.key()));
            accounted += it.value();
        }
        std::printf("bench-cua: run %d %s tier %s total %.1f ms | %s | other %.1f ms | obs %lld chars, shot %lld bytes\n",
                    run + 1, r.ok ? "ok" : "FAIL", qPrintable(r.tier), totalMs, qPrintable(parts.join(QLatin1Char(' '))),
                    totalMs - accounted, static_cast<long long>(r.observation.size()),
                    static_cast<long long>(r.screenshotJpeg.size()));
    }
    return failures ? 1 : 0;
}

}  // namespace studio::cua
