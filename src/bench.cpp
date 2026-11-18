// --bench-fps: offscreen frame-time and GUI-thread stall profiler.
//
// Every frame advances the deterministic animation clock by 16.667 ms, lets the GUI thread run
// the resulting events and bindings, then renders the window with the software scene graph
// (polish -> sync -> render, the same passes a real frame takes). The QQuickWindow signals
// split each frame into phases:
//   sync    beforeSynchronizing -> afterSynchronizing (QML items -> scene graph nodes)
//   render  beforeRendering -> afterRendering (rasterising the scene graph)
//   gui     everything else: input, animation ticks, QML bindings / JS, delegate creation,
//           polish, synchronous I/O on the GUI thread
// The software renderer repaints only the dirty region of the 1920x1080 window, so render time
// tracks how much of the scene changed; the live window does the same work on the GPU.
#include "backend.hpp"
#include "demo.hpp"
#include "metalbench.hpp"
#include "providers.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QElapsedTimer>
#include <QGuiApplication>
#include <QMouseEvent>
#include <QQmlApplicationEngine>
#include <QQmlComponent>
#include <QQuickGraphicsDevice>
#include <QQuickItem>
#include <QQuickRenderControl>
#include <QQuickRenderTarget>
#include <QQuickWindow>
#include <QSGRendererInterface>
#include <QSet>

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <numeric>

namespace studio {

namespace {

struct Samples {
    QList<double> total, anim, sync, render;  // anim = the gui column
    int over = 0;  // frames above 16.67 ms
    bool verbose = true;
};

double pct(QList<double> v, double p)
{
    if (v.isEmpty())
        return 0;
    std::sort(v.begin(), v.end());
    const qsizetype i = std::clamp<qsizetype>(qsizetype(p / 100.0 * (v.size() - 1) + 0.5), 0, v.size() - 1);
    return v[i];
}

double mean(const QList<double> &v)
{
    double s = 0;
    for (double x : v)
        s += x;
    return v.isEmpty() ? 0 : s / v.size();
}

class FrameClock {
public:
    FrameClock(QQuickWindow *win, StepDriver *driver)
        : m_win(win)
        , m_driver(driver)
    {
        m_clock.start();
        // A frame can hold two passes (the render loop's own UpdateRequest during event
        // processing, then the grab), so every sync / render interval is summed.
        QObject::connect(win, &QQuickWindow::beforeSynchronizing, win, [this] { m_mark = m_clock.nsecsElapsed(); },
                         Qt::DirectConnection);
        QObject::connect(win, &QQuickWindow::afterSynchronizing, win,
                         [this] { m_syncNs += m_clock.nsecsElapsed() - m_mark; }, Qt::DirectConnection);
        QObject::connect(win, &QQuickWindow::beforeRendering, win, [this] { m_mark = m_clock.nsecsElapsed(); },
                         Qt::DirectConnection);
        QObject::connect(win, &QQuickWindow::afterRendering, win,
                         [this] { m_renderNs += m_clock.nsecsElapsed() - m_mark; }, Qt::DirectConnection);
        QObject::connect(win, &QQuickWindow::afterAnimating, win, [this] { ++m_animSignals; }, Qt::DirectConnection);
    }

    // One frame: `before` runs first (input, backend updates), then the clock steps 16.667 ms,
    // events and bindings run, and the window renders.
    void frame(Samples *s, const std::function<void()> &before = {})
    {
        m_syncNs = m_renderNs = 0;
        const qint64 t0 = m_clock.nsecsElapsed();
        if (before)
            before();
        m_ms += 1000.0 / 60.0;
        m_driver->stepTo(qRound64(m_ms));
        QCoreApplication::processEvents();
        const QImage img = m_win->grabWindow();
        const qint64 t1 = m_clock.nsecsElapsed();
        Q_UNUSED(img);
        const double total = (t1 - t0) / 1e6;
        s->total << total;
        s->sync << m_syncNs / 1e6;
        s->render << m_renderNs / 1e6;
        s->anim << total - (m_syncNs + m_renderNs) / 1e6;
        if (total > 1000.0 / 60.0)
            s->over += 1;
    }
    void idle(int n)
    {
        Samples junk;
        for (int i = 0; i < n; ++i)
            frame(&junk);
    }
    int animSignals() const { return m_animSignals; }

private:
    QQuickWindow *m_win;
    StepDriver *m_driver;
    QElapsedTimer m_clock;
    double m_ms = 0;
    qint64 m_mark = 0, m_syncNs = 0, m_renderNs = 0;
    int m_animSignals = 0;
};

void printHeader()
{
    std::printf("\n%-34s %6s %7s %7s %7s %7s %7s %5s | %7s %7s %7s\n", "scenario (ms/frame)", "frames", "p50", "p95",
                "p99", "max", "mean", ">16.7", "gui95", "sync95", "rend95");
    std::printf("%s\n", QByteArray(34 + 7 + 8 * 5 + 6 + 3 + 8 * 3, '-').constData());
}

// This build's numbers, keyed by scenario number, for the Before/After summary.
struct Stat {
    double p50 = 0, p95 = 0, p99 = 0, max = 0;
};
QHash<char, Stat> g_after;
QHash<QString, double> g_afterStall;

// Baseline: the same bench on the pre-optimisation build (instrumentation only), same Mac,
// 2026-11-17, median of 3 runs. Range across those runs in PERF_REPORT.md.
struct Base {
    char id;
    const char *name;
    Stat s;
};
const Base kBaseline[] = {
    {'1', "constellation idle+spring+hover", {0.57, 0.89, 0.93, 0.98}},
    {'2', "zoomMorph in+out", {0.67, 3.43, 3.64, 3.68}},
    {'3', "ladder fill (publish/frame)", {0.62, 1.80, 4.51, 65.04}},
    {'4', "stage slides", {2.23, 4.59, 5.00, 33.53}},
    {'5', "backend republish x10 (mock)", {0.13, 1.61, 1.71, 2.42}},
    {'7', "live constellation idle", {0.14, 0.21, 0.23, 0.23}},
    {'8', "live stage + republish x10", {0.12, 1.39, 1.47, 2.26}},
};
const QList<QPair<QString, double>> kBaselineStall = {
    {QStringLiteral("live cold start, longest GUI-thread slice (ms)"), 51.55},
    {QStringLiteral("ladder fill, longest GUI-thread slice (ms)"), 61.55},
    {QStringLiteral("mock republish, GUI ms per publish"), 1.08},
    {QStringLiteral("live republish, GUI ms per publish"), 0.91},
    {QStringLiteral("board poll, GUI ms per poll"), 3.07},
    {QStringLiteral("settled Stage, repaints per 120 idle frames"), 3},
    {QStringLiteral("transcript parse on the GUI thread (ms)"), 0},
};

void printRow(const char *name, const Samples &s)
{
    g_after.insert(name[0], Stat{pct(s.total, 50), pct(s.total, 95), pct(s.total, 99), pct(s.total, 100)});
    std::printf("%-34s %6d %7.2f %7.2f %7.2f %7.2f %7.2f %5d | %7.2f %7.2f %7.2f\n", name, int(s.total.size()),
                pct(s.total, 50), pct(s.total, 95), pct(s.total, 99), pct(s.total, 100), mean(s.total), s.over,
                pct(s.anim, 95), pct(s.sync, 95), pct(s.render, 95));
    if (s.verbose) {
        QList<int> idx(s.total.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return s.total[a] > s.total[b]; });
        std::printf("   worst frames:");
        for (int k = 0; k < std::min<int>(3, idx.size()); ++k)
            std::printf(" #%d %.2f ms (gui %.2f, render %.2f)", idx[k], s.total[idx[k]], s.anim.value(idx[k]),
                        s.render.value(idx[k]));
        std::printf("\n");
    }
    std::fflush(stdout);
}

// Counts repaint requests: while the UI is settled, each one is a frame the live window would
// render (and a full GPU frame at Retina size) for nothing.
class UpdateCounter : public QObject {
public:
    int n = 0;
    bool eventFilter(QObject *, QEvent *e) override
    {
        if (e->type() == QEvent::UpdateRequest)
            ++n;
        return false;
    }
};

void hover(QQuickWindow *win, QPointF p)
{
    QMouseEvent ev(QEvent::MouseMove, p, win->mapToGlobal(p), Qt::NoButton, Qt::NoButton, Qt::NoModifier);
    QCoreApplication::sendEvent(win, &ev);
}

QPointF islandCenter(QQuickWindow *win, int i)
{
    const QList<QQuickItem *> islands = findItems(win->contentItem(), QStringLiteral("island"));
    if (i < 0 || i >= islands.size())
        return QPointF(win->width() / 2.0, win->height() / 2.0);
    QQuickItem *it = islands[i];
    return it->mapToScene(QPointF(it->width() / 2, it->height() / 2));
}

void printStall(const char *label, const Backend::GuiStall &g)
{
    const double gui = g.refreshMs + g.applyMs + g.publishMs + g.boardMs;
    std::printf("%-34s refresh %d x %.2f ms | applyScan %d x %.2f ms | publish %d x %.2f ms (%d emitted) | board spawn+parse %d "
                "x %.2f ms\n",
                label, g.refreshes, g.refreshes ? g.refreshMs / g.refreshes : 0.0, g.applies,
                g.applies ? g.applyMs / g.applies : 0.0, g.publishes, g.publishes ? g.publishMs / g.publishes : 0.0,
                g.emits, g.boards, g.boards ? g.boardMs / g.boards : 0.0);
    std::printf("%-34s GUI-thread total %.1f ms, longest slice %.2f ms | transcript parse: %.1f ms on the GUI thread, "
                "%.1f ms on the worker (%d transcripts)\n",
                "", gui, g.maxMs, g.guiParseMs, g.workerMs, g.parsed);
    std::fflush(stdout);
}

bool waitFor(const std::function<bool()> &done, int ms)
{
    QElapsedTimer t;
    t.start();
    while (!done() && t.elapsed() < ms)
        settle(20);
    return done();
}

}  // namespace

int benchFps()
{
    StepDriver driver;
    driver.begin();
    std::printf("bench-fps: platform=%s, software scene graph, 1920x1080, 60 fps clock, dirty-region repaint per frame\n",
                qPrintable(QGuiApplication::platformName()));

    // --- mock backend: the four UI scenarios -----------------------------------------------
    double worstP95 = 0;
    {
        Backend backend(Backend::Mode::Mock);
        QQmlApplicationEngine engine;
        QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
        if (!win) {
            std::fprintf(stderr, "bench-fps: QML failed to load\n");
            return 2;
        }
        FrameClock fc(win, &driver);
        fc.idle(40);  // islands born, fonts warmed
        printHeader();

        // 1. Constellation: idle, focus-ring spring across islands, hover lift.
        Samples s1;
        for (int f = 0; f < 120; ++f) {
            fc.frame(&s1, [&] {
                if (f == 10 || f == 40)
                    sendKey(win, Qt::Key_Right);
                else if (f == 70)
                    sendKey(win, Qt::Key_Left);
                else if (f >= 90 && f % 6 == 0)
                    hover(win, islandCenter(win, (f / 6) % 4));
            });
        }
        printRow("1 constellation idle+spring+hover", s1);
        hover(win, QPointF(5, 5));
        fc.idle(20);

        // 2. Constellation -> Stage shared-element zoom and back.
        Samples s2;
        for (int f = 0; f < 60; ++f) {
            fc.frame(&s2, [&] {
                if (f == 0)
                    sendKey(win, Qt::Key_Return);
                else if (f == 30)
                    sendKey(win, Qt::Key_Escape);
            });
        }
        printRow("2 zoomMorph in+out", s2);
        fc.idle(20);

        // 3. New session (Cmd+N, goal, Enter): the island zooms into its Ladder, which the mock
        //    clock fills over 6 s. The backend republishes the projects on every frame here.
        sendKey(win, Qt::Key_N, Qt::ControlModifier);
        fc.idle(20);
        typeText(win, QStringLiteral("profile fp4 kv-cache decode on sm120"));
        fc.idle(5);
        sendKey(win, Qt::Key_Return);
        Samples s3;
        backend.resetGuiStall();
        for (int f = 0; f < 360; ++f)
            fc.frame(&s3, [&] { backend.advance(1.0 / 60); });
        printRow("3 ladder fill (publish/frame)", s3);
        const Backend::GuiStall ladderStall = backend.guiStall();

        // 4. Stage slides chart -> diff -> table -> pdf -> table.
        sendKey(win, Qt::Key_3, Qt::ControlModifier);
        fc.idle(30);
        Samples s4s;
        QStringList kinds;
        auto stageKind = [&] {
            auto *st = findItems(win->contentItem(), QStringLiteral("stage")).value(0);
            return st ? st->property("kind").toString() : QString();
        };
        kinds << stageKind();
        int pdfReadyAt = -1;  // frames from the PDF slide's key press to a decoded page on screen
        for (int f = 0; f < 60; ++f) {
            fc.frame(&s4s, [&] {
                if (f == 0 || f == 15 || f == 30)
                    sendKey(win, Qt::Key_Right);
                else if (f == 45)
                    sendKey(win, Qt::Key_Left);
                else if (f % 15 == 14)
                    kinds << stageKind();
            });
            if (f >= 30 && pdfReadyAt < 0)
                for (QQuickItem *img : findItems(win->contentItem(), QStringLiteral("pdfImage")))
                    if (img->property("status").toInt() == 1 && img->isVisible())
                        pdfReadyAt = f - 30;
        }
        printRow("4 stage slides", s4s);

        // 5. A live update landing while the Stage is up: the backend republishes the projects.
        fc.idle(20);
        Samples s4;
        backend.resetGuiStall();
        for (int f = 0; f < 60; ++f)
            fc.frame(&s4, [&] {
                if (f % 6 == 0)
                    backend.benchReapply();
            });
        printRow("5 backend republish x10 (mock)", s4);
        std::printf("   slides visited: %s; PDF page on screen %d frame(s) after its key press\n",
                    qPrintable(kinds.join(QStringLiteral(" -> "))), pdfReadyAt);
        printStall("   ladder fill", ladderStall);
        printStall("   mock republish", backend.guiStall());
        g_afterStall.insert(QStringLiteral("ladder fill, longest GUI-thread slice (ms)"), ladderStall.maxMs);
        g_afterStall.insert(QStringLiteral("mock republish, GUI ms per publish"),
                            backend.guiStall().publishMs / std::max(1, backend.guiStall().publishes));

        // Settled Stage: nothing on screen moves, so nothing should ask for a repaint.
        UpdateCounter updates;
        fc.idle(60);
        win->installEventFilter(&updates);
        fc.idle(120);
        win->removeEventFilter(&updates);
        std::printf("   settled Stage, 120 idle frames: %d repaint requests (0 = idle window sleeps)\n", updates.n);
        g_afterStall.insert(QStringLiteral("settled Stage, repaints per 120 idle frames"), updates.n);

        // 6. The Tab predictor ring breathing on an idle Stage. The bench clock outruns the app's
        // 600 ms wall-clock idle timer, so idle is set directly; this is the ring's whole cost.
        Samples s6;
        int ringRepaints = 0;
        if (QObject *app = win->findChild<QObject *>(QStringLiteral("studioApp"))) {
            app->setProperty("idle", true);
            fc.idle(20);
            UpdateCounter ringUpdates;
            win->installEventFilter(&ringUpdates);
            for (int f = 0; f < 120; ++f)
                fc.frame(&s6);
            win->removeEventFilter(&ringUpdates);
            ringRepaints = ringUpdates.n;
            app->setProperty("idle", false);
            fc.idle(30);
        }
        printRow("6 predictor ring breathing", s6);
        std::printf("   predictor ring breathing, 120 idle frames: %d repaint requests (the pulse is the only motion)\n",
                    ringRepaints);
        for (const Samples *s : {&s1, &s2, &s3, &s4s, &s4, &s6})
            worstP95 = std::max(worstP95, pct(s->total, 95));
        std::printf("   afterAnimating signals: %d (the step driver advances animations, not the window)\n",
                    fc.animSignals());
    }

    // --- live backend: GUI-thread cost of the real registry, transcripts and board ---------
    {
        Backend backend(Backend::Mode::Live, nullptr, false);  // read-only: no helpers, no new/send
        if (!waitFor([&] { return !backend.projects().isEmpty() && !backend.scanning(); }, 20000)) {
            std::printf("\nlive: no Lapis registry or no sessions, live scenarios skipped\n");
        } else {
            waitFor([&] { return backend.guiStall().boards > 0; }, 6000);
            std::printf("\nlive backend: %d projects\n", int(backend.projects().size()));
            const Backend::GuiStall cold = backend.guiStall();
            printStall("6a cold start (scan + board)", cold);
            g_afterStall.insert(QStringLiteral("live cold start, longest GUI-thread slice (ms)"), cold.maxMs);
            g_afterStall.insert(QStringLiteral("board poll, GUI ms per poll"), cold.boardMs / std::max(1, cold.boards));
            backend.resetGuiStall();
            for (int i = 0; i < 3; ++i) {
                backend.benchInvalidateCache();  // every transcript "changed"
                backend.refresh();
                waitFor([&] { return !backend.scanning(); }, 20000);
            }
            printStall("6b 3 forced full rescans", backend.guiStall());
            g_afterStall.insert(QStringLiteral("transcript parse on the GUI thread (ms)"),
                                cold.guiParseMs + backend.guiStall().guiParseMs);

            QQmlApplicationEngine engine;
            QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
            if (win) {
                FrameClock fc(win, &driver);
                fc.idle(30);
                Samples c;
                for (int f = 0; f < 60; ++f)
                    fc.frame(&c);
                printHeader();
                printRow("7 live constellation idle", c);
                sendKey(win, Qt::Key_1);
                fc.idle(30);
                Samples s;
                backend.resetGuiStall();
                for (int f = 0; f < 60; ++f)
                    fc.frame(&s, [&] {
                        if (f % 6 == 0)
                            backend.benchReapply();
                    });
                printRow("8 live stage + republish x10", s);
                printStall("   live republish", backend.guiStall());
                g_afterStall.insert(QStringLiteral("live republish, GUI ms per publish"),
                                    backend.guiStall().publishMs / std::max(1, backend.guiStall().publishes));
                UpdateCounter updates;
                fc.idle(60);
                win->installEventFilter(&updates);
                fc.idle(120);
                win->removeEventFilter(&updates);
                std::printf("   settled live Stage, 120 idle frames: %d repaint requests\n", updates.n);
                worstP95 = std::max({worstP95, pct(c.total, 95), pct(s.total, 95)});
            }
        }
    }
    std::printf("\nBEFORE vs AFTER (ms/frame; before = baseline build, median of 3 runs on this Mac)\n");
    std::printf("%-34s %15s %15s %17s\n", "scenario", "p50 bef/aft", "p95 bef/aft", "max bef/aft");
    for (const Base &b : kBaseline) {
        if (!g_after.contains(b.id))
            continue;
        const Stat a = g_after.value(b.id);
        std::printf("%c %-32s %6.2f / %6.2f %6.2f / %6.2f %7.2f / %7.2f\n", b.id, b.name, b.s.p50, a.p50, b.s.p95,
                    a.p95, b.s.max, a.max);
    }
    std::printf("%-50s %8s %8s\n", "GUI-thread stalls", "before", "after");
    for (const auto &[label, before] : kBaselineStall)
        if (g_afterStall.contains(label))
            std::printf("%-50s %8.2f %8.2f\n", qPrintable(label), before, g_afterStall.value(label));
    std::printf("\nbench-fps: worst scenario p95 %.2f ms/frame (budget 16.67 ms at 60 fps): %s\n", worstP95,
                worstP95 < 1000.0 / 60.0 ? "PASS" : "FAIL");
    return worstP95 < 1000.0 / 60.0 ? 0 : 1;
}

// --- --bench-fps --metal ---------------------------------------------------------------------
//
// The real GPU path, still with zero windows: QQuickRenderControl renders the same StudioApp
// into an offscreen 3840x2160 Metal texture (1920x1080 logical at dpr 2, a Retina window), on
// a 240 Hz animation clock (4.167 ms per frame). One frame is timed from input to GPU done:
//   gui     input, the clock step, events, bindings, polishItems()
//   sync    QML items -> scene graph nodes
//   render  scene graph -> Metal command buffer, submit, and wait until the GPU has finished
//           (an empty command buffer on the same in-order queue: no frame is counted early)
int benchFpsMetal(const QString &shotDir)
{
    constexpr double kBudget = 1000.0 / 240.0;  // 4.167 ms
    constexpr double kFrameMs = 1000.0 / 240.0;
    StepDriver driver;
    driver.begin();
    MetalTarget mt;
    if (!createMetalTarget(&mt, QSize(3840, 2160))) {
        std::fprintf(stderr, "bench-fps --metal: no Metal device\n");
        return 2;
    }
    Backend backend(Backend::Mode::Mock);
    QQuickRenderControl rc;
    QQuickWindow win(&rc);
    win.setGraphicsDevice(metalGraphicsDevice(mt));
    win.resize(1920, 1080);
    if (!rc.initialize()) {
        std::fprintf(stderr, "bench-fps --metal: QQuickRenderControl::initialize failed\n");
        destroyMetalTarget(&mt);
        return 2;
    }
    win.setRenderTarget(metalRenderTarget(mt, 2.0));
    const auto api = win.rendererInterface()->graphicsApi();
    QQmlEngine engine;
    engine.addImageProvider(QStringLiteral("pdf"), new PdfImageProvider);
    QQmlComponent comp(&engine);
    comp.loadFromModule("Studio", "StudioApp");
    QObject *obj = comp.createWithInitialProperties(
        {{QStringLiteral("backend"), QVariant::fromValue(static_cast<QObject *>(&backend))}, {QStringLiteral("demoMode"), false}});
    auto *root = qobject_cast<QQuickItem *>(obj);
    if (!root) {
        std::fprintf(stderr, "bench-fps --metal: StudioApp failed to load: %s\n", qPrintable(comp.errorString()));
        destroyMetalTarget(&mt);
        return 2;
    }
    root->setParentItem(win.contentItem());
    root->setSize(QSizeF(1920, 1080));
    win.contentItem()->setSize(QSizeF(1920, 1080));
    {
        QFocusEvent in(QEvent::FocusIn, Qt::ActiveWindowFocusReason);
        QCoreApplication::sendEvent(&win, &in);
    }
    QMetaObject::invokeMethod(root, "takeFocus");
    std::printf("bench-fps --metal: platform=%s, graphics API %s, QQuickRenderControl -> offscreen MTLTexture "
                "3840x2160 (1920x1080 @ dpr 2) on %s, 240 Hz clock, budget %.2f ms, 0 windows shown\n",
                qPrintable(QGuiApplication::platformName()), api == QSGRendererInterface::Metal ? "Metal" : "NOT METAL",
                qPrintable(metalDeviceName(mt)), kBudget);
    if (api != QSGRendererInterface::Metal) {
        destroyMetalTarget(&mt);
        return 3;
    }

    QElapsedTimer clock;
    clock.start();
    double ms = 0;
    auto frame = [&](Samples *s, const std::function<void()> &before) {
        const qint64 t0 = clock.nsecsElapsed();
        if (before)
            before();
        ms += kFrameMs;
        driver.stepTo(qRound64(ms));
        QCoreApplication::processEvents();
        rc.polishItems();
        const qint64 t1 = clock.nsecsElapsed();
        rc.beginFrame();
        rc.sync();
        const qint64 t2 = clock.nsecsElapsed();
        rc.render();
        rc.endFrame();
        metalFinish(mt);
        const qint64 t3 = clock.nsecsElapsed();
        const double total = (t3 - t0) / 1e6;
        s->total << total;
        s->anim << (t1 - t0) / 1e6;
        s->sync << (t2 - t1) / 1e6;
        s->render << (t3 - t2) / 1e6;
        if (total > kBudget)
            s->over += 1;
    };
    auto idle = [&](int n) {
        Samples junk;
        for (int i = 0; i < n; ++i)
            frame(&junk, {});
    };
    auto row = [&](const char *name, const Samples &s) {
        std::printf("%-36s %6d %7.2f %7.2f %7.2f %7.2f %7.2f %6d | %7.2f %7.2f %7.2f  %s\n", name, int(s.total.size()),
                    pct(s.total, 50), pct(s.total, 95), pct(s.total, 99), pct(s.total, 100), mean(s.total), s.over,
                    pct(s.anim, 95), pct(s.sync, 95), pct(s.render, 95), pct(s.total, 95) < kBudget ? "PASS" : "FAIL");
        std::fflush(stdout);
    };
    auto key = [&](int k, Qt::KeyboardModifiers m = Qt::NoModifier) { sendKey(&win, k, m); };

    idle(120);  // islands born, glyph atlases and pipelines warmed
    std::printf("\n%-36s %6s %7s %7s %7s %7s %7s %6s | %7s %7s %7s\n", "scenario (ms/frame, Metal, 240 Hz)", "frames",
                "p50", "p95", "p99", "max", "mean", ">4.17", "gui95", "sync95", "rend95");
    std::printf("%s\n", QByteArray(36 + 7 + 8 * 5 + 7 + 3 + 8 * 3 + 6, '-').constData());

    // A. Constellation rapid hop: a key every 6 frames (25 ms, 40 hops/s key repeat) walking
    //    right, down, left, up across the islands; the ring retargets mid-glide every time.
    Samples hop;
    const int seq[4] = {Qt::Key_Right, Qt::Key_Down, Qt::Key_Left, Qt::Key_Up};
    QSet<int> visitedIslands;
    QQuickItem *ring = findItems(win.contentItem(), QStringLiteral("focusRing")).value(0);
    double ringMinX = 1e9, ringMaxX = -1e9;
    for (int f = 0; f < 120; ++f) {
        frame(&hop, [&] {
            if (f % 6 == 0)
                key(seq[(f / 6 / 3) % 4]);
        });
        visitedIslands.insert(root->property("selectedIsland").toInt());
        if (ring) {
            ringMinX = std::min(ringMinX, ring->x());
            ringMaxX = std::max(ringMaxX, ring->x());
        }
    }
    row("A constellation rapid hop (120 frames)", hop);
    std::printf("   20 hops: %d islands visited, ring x travelled %.0f..%.0f px, ring size fixed at %.0fx%.0f\n",
                int(visitedIslands.size()), ringMinX, ringMaxX, ring ? ring->width() : 0.0, ring ? ring->height() : 0.0);
    idle(60);

    // B. Constellation -> Stage shared-element zoom (220 ms) and back.
    Samples zoom;
    for (int f = 0; f < 120; ++f)
        frame(&zoom, [&] {
            if (f == 0)
                key(Qt::Key_Return);
            else if (f == 60)
                key(Qt::Key_Escape);
        });
    row("B zoomMorph in+out", zoom);
    idle(60);

    // C. Stage slides on the four-artifact fp4 session (Cmd+N kickoff, mock clock fills it):
    //    chart -> diff -> table -> pdf and back, one slide (180 ms) every 45 frames.
    key(Qt::Key_N, Qt::ControlModifier);
    idle(20);
    typeText(&win, QStringLiteral("profile fp4 kv-cache decode on sm120"));
    key(Qt::Key_Return);
    for (int i = 0; i < 14; ++i) {
        backend.advance(0.5);
        idle(4);
    }
    key(Qt::Key_3, Qt::ControlModifier);
    idle(90);
    QStringList kinds;
    QQuickItem *stage = findItems(win.contentItem(), QStringLiteral("stage")).value(0);
    auto stageKind = [&] { return stage ? stage->property("kind").toString() : QString(); };
    kinds << stageKind();
    Samples slides;
    for (int f = 0; f < 180; ++f) {
        frame(&slides, [&] {
            if (f % 45 == 0)
                key(f < 135 ? Qt::Key_Right : Qt::Key_Left);
        });
        if (f % 45 == 44)
            kinds << stageKind();
    }
    row("C stage slides", slides);
    std::printf("   slides visited: %s\n", qPrintable(kinds.join(QStringLiteral(" -> "))));

    // D. Param card open/close on the Stage (Cmd+P, then next pill, then Esc).
    Samples cards;
    for (int f = 0; f < 120; ++f)
        frame(&cards, [&] {
            if (f == 0)
                key(Qt::Key_P, Qt::ControlModifier);
            else if (f % 20 == 0 && f < 100)
                key(Qt::Key_Right);
            else if (f == 100)
                key(Qt::Key_Escape);
        });
    row("D param pills + cards", cards);
    QImage shot;
    if (!shotDir.isEmpty()) {
        key(Qt::Key_P, Qt::ControlModifier);
        idle(30);
        shot = metalReadback(mt);
        key(Qt::Key_Escape);
    }
    key(Qt::Key_Escape);
    idle(60);

    // E. Idle Constellation (the shared border-wave clock is the only motion).
    Samples still;
    for (int f = 0; f < 240; ++f)
        frame(&still, {});
    row("E constellation idle (border waves)", still);

    double worst = 0;
    for (const Samples *s : {&hop, &zoom, &slides, &cards, &still})
        worst = std::max(worst, pct(s->total, 95));

    // F. Motion feel, before vs after, on the same 240 Hz clock: one 300 px hop of the old focus
    //    ring spring (4.5 / 0.36), the new lift spring (22 / 0.92), and the new ring glide
    //    (120 ms OutExpo). Settle = last frame more than 1 px from the target.
    {
        QQmlComponent motion(&engine);
        motion.setData(R"(import QtQuick
Item {
    property real oldSpring: 0
    property real newSpring: 0
    property real glide: 0
    Behavior on oldSpring { SpringAnimation { spring: 4.5; damping: 0.36; epsilon: 0.25 } }
    Behavior on newSpring { SpringAnimation { spring: 22.0; damping: 0.92; epsilon: 0.25 } }
    Behavior on glide { NumberAnimation { duration: 120; easing.type: Easing.OutExpo } }
})", QUrl());
        std::unique_ptr<QObject> m(motion.create());
        if (m) {
            const char *props[3] = {"oldSpring", "newSpring", "glide"};
            const char *names[3] = {"old ring spring 4.5 / 0.36", "new lift spring 22 / 0.92", "new ring glide 120 ms OutExpo"};
            for (const char *p : props)
                m->setProperty(p, 300.0);
            double settle[3] = {0, 0, 0}, peak[3] = {0, 0, 0}, half[3] = {-1, -1, -1};
            for (int f = 1; f <= 480; ++f) {
                ms += kFrameMs;
                driver.stepTo(qRound64(ms));
                QCoreApplication::processEvents();
                for (int k = 0; k < 3; ++k) {
                    const double v = m->property(props[k]).toDouble();
                    peak[k] = std::max(peak[k], v);
                    if (std::abs(v - 300.0) > 1.0)
                        settle[k] = f * kFrameMs;
                    if (half[k] < 0 && v >= 150.0)
                        half[k] = f * kFrameMs;
                }
            }
            std::printf("\nF motion feel, one 300 px hop on the 240 Hz clock:\n");
            for (int k = 0; k < 3; ++k)
                std::printf("   %-32s halfway %6.1f ms, settled (within 1 px) %6.1f ms, overshoot %5.1f px\n", names[k],
                            half[k], settle[k], std::max(0.0, peak[k] - 300.0));
        }
    }
    if (!shot.isNull()) {
        QDir().mkpath(shotDir);
        const QString path = shotDir + QStringLiteral("/metal_stage_dpr2.png");
        shot.save(path);
        std::printf("\nbench-fps --metal: Metal frame read back from the GPU texture: %s (%dx%d)\n", qPrintable(path),
                    shot.width(), shot.height());
    }
    std::printf("\nbench-fps --metal: worst scenario p95 %.2f ms/frame (budget %.2f ms at 240 Hz): %s\n", worst, kBudget,
                worst < kBudget ? "PASS" : "FAIL");
    delete obj;
    rc.invalidate();
    destroyMetalTarget(&mt);
    return worst < kBudget ? 0 : 1;
}

}  // namespace studio
