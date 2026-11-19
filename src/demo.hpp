// Offscreen window helpers shared by --test, --smoke-offscreen and --record-demo.
#pragma once

#include <QAnimationDriver>
#include <QImage>
#include <QString>
#include <QStringList>

class QGuiApplication;
class QQmlApplicationEngine;
class QQuickWindow;
class QQuickItem;
class QWindow;

namespace studio {

class Backend;

// Deterministic animation clock: QML animations advance only when stepTo() is called.
//
// Like Qt's own driver, elapsed() counts from the driver's latest start(). Qt's animation timer
// stops the driver when the last animation ends and, on the next start, resets its last tick to
// 0; an absolute clock would then hand the first new animation the whole idle gap as one delta
// and finish it in a single frame (slides snapped whenever nothing else was animating).
class StepDriver : public QAnimationDriver {
public:
    void begin()
    {
        install();
        start();
    }
    void stepTo(qint64 ms)
    {
        m_ms = ms;
        if (isRunning())
            advance();
    }
    qint64 elapsed() const override { return m_ms - m_base; }

protected:
    void start() override
    {
        m_base = m_ms;
        QAnimationDriver::start();
    }

private:
    qint64 m_ms = 0;
    qint64 m_base = 0;
};

QQuickWindow *createStudioWindow(QQmlApplicationEngine &engine, Backend *backend, bool demo, QSize size);
void sendKey(QWindow *w, int key, Qt::KeyboardModifiers mods = Qt::NoModifier, const QString &text = QString());
void typeText(QWindow *w, const QString &text);
void settle(int ms);

// Pixels that read as blue or bright green (the palette forbids both).
int forbiddenPixels(const QImage &img);
// Words in Text items that are visible, opaque and on screen. Fills `texts` with them.
int visibleWords(QQuickWindow *win, QStringList *texts = nullptr);
// Pairs of visible text runs whose painted boxes intersect (layout overlap audit).
int textOverlaps(QQuickWindow *win, QStringList *pairs = nullptr);
QList<QQuickItem *> findItems(QQuickItem *root, const QString &objectName);

int recordDemo(const QString &outPath);
// shotDir non-empty: also save mock Constellation/Stage frames at 1.0x and 1.6x UI zoom there.
int smokeOffscreen(const QString &shotDir = QString());
int probeLive(const QString &shotDir = QString());
// Headless: two quick asks through the real persistent daemon for MODEL/EFFORT; prints both
// answers, their latency, and whether the second turn reused the first turn's process.
int probeChat(const QString &model, const QString &effort, const QString &question);
int runTests();
// --snap-omnibar OUT.png: the mock Stage with a hovered table cell and the floating ask card
// open over it, answered by a local fake SSE endpoint, rendered offscreen at 1920x1080.
int snapOmnibar(const QString &outPath);
// --snap-review OUT.png: the session review card over the mock Constellation; also writes
// OUT-constellation.png (a dragged card, the Needs Me pill) and OUT-stage.png (visual-first slide).
int snapReview(const QString &outPath);
// Offscreen dark frames: the Music (slowmix) overlay over the Constellation, plus OUT-bar.png
// (the top bar's ♪ pill while a track plays). Mock backend, dry run: no audio.
int snapSlowmix(const QString &outPath);
// --snap-map OUT.png: the clustered Constellation (Work · Projects · Apps · Quick Qs) with the
// free pad cursor drawing a marquee and a finished multi-selection, offscreen at 1920x1080.
int snapMap(const QString &outPath);
int benchFps();
// --bench-fps --metal: QQuickRenderControl into an offscreen Metal texture at dpr 2, 240 Hz
// budget. shotDir non-empty: also save one GPU frame (read back from the texture) there.
int benchFpsMetal(const QString &shotDir = QString());

}  // namespace studio
