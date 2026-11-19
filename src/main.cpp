// lapis_studio: three sparse lenses (Constellation, Ladder, Stage) over live Lapis sessions.
//
//   lapis_studio                     interactive window on the live Lapis backend
//   lapis_studio --test              unit + offscreen UI tests (mock backend)
//   lapis_studio --smoke-offscreen [DIR]  load the UI offscreen once (DIR: save 1.0x/1.6x frames)
//   lapis_studio --probe-live        read-only scan of the live registry and transcripts, no window
//   lapis_studio --bench-fps [--metal [DIR]]  frame-time profiler; --metal: QQuickRenderControl on Metal, 240 Hz
//   lapis_studio --record-demo OUT   deterministic 44 s 1920x1080 60 fps demo (mock backend)
//   lapis_studio --probe-chat MODEL [EFFORT [Q]]  two real asks through one persistent daemon, no window
//   lapis_studio --snap-omnibar OUT.png  offscreen dark frame: Stage + hovered cell + floating ask card
//   lapis_studio --snap-review OUT.png   offscreen dark frames: session review card, plus
//                                        OUT-constellation.png (dragged card, Needs Me) and OUT-stage.png
//   lapis_studio --harness codex     quick asks (Enter) go to a warm codex app-server instead of claude
//
// A PS4 DualShock 4 (GameController.framework) drives every lens in the interactive window only;
// no offscreen mode ever touches GameController (src/gamepad.hpp).

#include "backend.hpp"
#include "demo.hpp"

#include <QFont>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQuickStyle>
#include <QQuickWindow>
#include <QScreen>
#include <QSGRendererInterface>

#include <cstdio>
#include <cstring>

int main(int argc, char **argv)
{
    bool test = false;
    bool smoke = false;
    bool probe = false;
    bool bench = false;
    bool metal = false;
    QString metalShots;
    QString probeShots;
    QString smokeShots;
    QString demoOut;
    QString snapOut;
    QString reviewOut;
    QString quickHarness = QStringLiteral("claude");
    QString chatModel;
    QString chatEffort = QStringLiteral("low");
    QString chatQuestion = QStringLiteral("Reply with exactly one word: the color of a clear daytime sky.");
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--test") == 0) {
            test = true;
        } else if (std::strcmp(argv[i], "--smoke-offscreen") == 0) {
            smoke = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                smokeShots = QString::fromLocal8Bit(argv[++i]);
        } else if (std::strcmp(argv[i], "--bench-fps") == 0) {
            bench = true;
        } else if (std::strcmp(argv[i], "--metal") == 0) {
            metal = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                metalShots = QString::fromLocal8Bit(argv[++i]);
        } else if (std::strcmp(argv[i], "--probe-live") == 0) {
            probe = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
                probeShots = QString::fromLocal8Bit(argv[++i]);
        } else if (std::strcmp(argv[i], "--record-demo") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--record-demo needs an output path\n");
                return 64;
            }
            demoOut = QString::fromLocal8Bit(argv[++i]);
        } else if (std::strcmp(argv[i], "--probe-chat") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--probe-chat needs a model id\n");
                return 64;
            }
            chatModel = QString::fromLocal8Bit(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-')
                chatEffort = QString::fromLocal8Bit(argv[++i]);
            if (i + 1 < argc && argv[i + 1][0] != '-')
                chatQuestion = QString::fromLocal8Bit(argv[++i]);
        } else if (std::strcmp(argv[i], "--snap-omnibar") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--snap-omnibar needs an output .png path\n");
                return 64;
            }
            snapOut = QString::fromLocal8Bit(argv[++i]);
        } else if (std::strcmp(argv[i], "--snap-review") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--snap-review needs an output .png path\n");
                return 64;
            }
            reviewOut = QString::fromLocal8Bit(argv[++i]);
        } else if (std::strcmp(argv[i], "--harness") == 0 && i + 1 < argc) {
            quickHarness = QString::fromLocal8Bit(argv[++i]);
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            std::printf("usage: lapis_studio [--harness claude|codex] [--test | --smoke-offscreen [DIR] | --probe-live | --probe-chat MODEL [EFFORT] | --bench-fps [--metal [DIR]] | --record-demo OUT.mp4 | --snap-omnibar OUT.png | --snap-review OUT.png]\n");
            return 0;
        }
    }
    const bool offscreen = test || smoke || probe || bench || !demoOut.isEmpty() || !chatModel.isEmpty()
                           || !snapOut.isEmpty() || !reviewOut.isEmpty();
    const bool metalBench = bench && metal;
    if (offscreen) {
        // Never open a visible window while building or testing on the host Mac.
        qputenv("QT_QPA_PLATFORM", "offscreen");
        qputenv("QSG_RENDER_LOOP", "basic");
        if (metalBench) {
            // Real Metal, still windowless: QQuickRenderControl renders into an offscreen texture.
            // The offscreen QPA reports no RHI support, so Qt Quick would fall back to its software
            // adaptation; ask for the default (rhi) adaptation explicitly.
            QQuickWindow::setSceneGraphBackend("rhi");
            QQuickWindow::setGraphicsApi(QSGRendererInterface::Metal);
        } else {
            QQuickWindow::setSceneGraphBackend("software");
            QQuickWindow::setGraphicsApi(QSGRendererInterface::Software);
        }
    } else {
#ifdef Q_OS_MACOS
        // The interactive window renders through Metal (QRhi), never the OpenGL or software path.
        QQuickWindow::setGraphicsApi(QSGRendererInterface::Metal);
#endif
    }

    QGuiApplication app(argc, argv);
    QGuiApplication::setApplicationName(QStringLiteral("lapis_studio"));
    QGuiApplication::setOrganizationName(QStringLiteral("lapis"));
    QFont base(QStringLiteral("Avenir Next"));
    base.setPixelSize(18);
    QGuiApplication::setFont(base);
    QQuickStyle::setStyle(QStringLiteral("Basic"));

    if (test)
        return studio::runTests();
    if (smoke)
        return studio::smokeOffscreen(smokeShots);
    if (probe)
        return studio::probeLive(probeShots);
    if (metalBench)
        return studio::benchFpsMetal(metalShots);
    if (bench)
        return studio::benchFps();
    if (!chatModel.isEmpty())
        return studio::probeChat(chatModel, chatEffort, chatQuestion);
    if (!demoOut.isEmpty())
        return studio::recordDemo(demoOut);
    if (!snapOut.isEmpty())
        return studio::snapOmnibar(snapOut);
    if (!reviewOut.isEmpty())
        return studio::snapReview(reviewOut);

    studio::Backend backend(studio::Backend::Mode::Live);
    backend.setQuickHarness(quickHarness);  // before the helpers warm up 0.4 s from now
    // Only the real window reads or writes prefs; every offscreen mode keeps them in memory.
    backend.setPrefsPath(studio::Backend::defaultPrefsPath());
    backend.setReviewsPath(studio::Backend::defaultReviewsPath());  // saved taste rules feed the filter
    // The only place GameController.framework is ever touched: offscreen modes return above.
    backend.gamepad()->setEnabled(true);
    QQmlApplicationEngine engine;
    QQuickWindow *win = studio::createStudioWindow(engine, &backend, false, QSize(1680, 1050));
    if (!win)
        return 2;
    // Log the graphics API the scene graph actually got (expected: Metal on macOS).
    QObject::connect(win, &QQuickWindow::sceneGraphInitialized, win, [win] {
        const auto api = win->rendererInterface()->graphicsApi();
        std::fprintf(stderr, "lapis_studio: scene graph on %s, dpr %.1f, screen refresh %.0f Hz\n",
                     api == QSGRendererInterface::Metal ? "Metal"
                     : api == QSGRendererInterface::OpenGL ? "OpenGL"
                     : api == QSGRendererInterface::Software ? "Software" : "another API",
                     win->devicePixelRatio(), win->screen() ? win->screen()->refreshRate() : 0.0);
    }, Qt::DirectConnection);
    return QGuiApplication::exec();
}
