#include "demo.hpp"

#include "backend.hpp"
#include "providers.hpp"
#include "transcript.hpp"
#include "llm/jarvis_legs.hpp"
#include "mac/cua.hpp"
#include "mac/cua_executor.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QJsonObject>
#include <QJsonDocument>
#include <QFile>
#include <QElapsedTimer>
#include <QMap>
#include <QEventLoop>
#include <QGuiApplication>
#include <QKeyEvent>
#include <QProcess>
#include <QQmlApplicationEngine>
#include <QQuickItem>
#include <QQuickWindow>
#include <QTimer>

#include <csignal>
#include <cstdio>

namespace studio {

QQuickWindow *createStudioWindow(QQmlApplicationEngine &engine, Backend *backend, bool demo, QSize size)
{
    engine.addImageProvider(QStringLiteral("pdf"), new PdfImageProvider);
    engine.setInitialProperties({{QStringLiteral("backend"), QVariant::fromValue(static_cast<QObject *>(backend))},
                                 {QStringLiteral("demoMode"), demo},
                                 {QStringLiteral("width"), size.width()},
                                 {QStringLiteral("height"), size.height()}});
    engine.loadFromModule("Studio", "Main");
    if (engine.rootObjects().isEmpty())
        return nullptr;
    auto *win = qobject_cast<QQuickWindow *>(engine.rootObjects().constFirst());
    if (!win)
        return nullptr;
    win->resize(size);
    win->show();
    win->requestActivate();
    QCoreApplication::processEvents();
    return win;
}

void sendKey(QWindow *w, int key, Qt::KeyboardModifiers mods, const QString &text)
{
    QKeyEvent press(QEvent::KeyPress, key, mods, text);
    QCoreApplication::sendEvent(w, &press);
    QKeyEvent release(QEvent::KeyRelease, key, mods, text);
    QCoreApplication::sendEvent(w, &release);
}

namespace {

int keyFor(QChar c)
{
    if (c.isLetter())
        return Qt::Key_A + (c.toUpper().unicode() - 'A');
    if (c.isDigit())
        return Qt::Key_0 + (c.unicode() - '0');
    switch (c.unicode()) {
    case ' ': return Qt::Key_Space;
    case '-': return Qt::Key_Minus;
    case '?': return Qt::Key_Question;
    case '.': return Qt::Key_Period;
    case '/': return Qt::Key_Slash;
    case '=': return Qt::Key_Equal;
    default: return Qt::Key_unknown;
    }
}

}  // namespace

void typeText(QWindow *w, const QString &text)
{
    for (QChar c : text)
        sendKey(w, keyFor(c), Qt::NoModifier, QString(c));
}

void settle(int ms)
{
    QEventLoop loop;
    QTimer::singleShot(ms, &loop, &QEventLoop::quit);
    loop.exec();
}

int forbiddenPixels(const QImage &src)
{
    const QImage img = src.convertToFormat(QImage::Format_RGB32);
    int bad = 0;
    for (int y = 0; y < img.height(); ++y) {
        const auto *row = reinterpret_cast<const QRgb *>(img.constScanLine(y));
        for (int x = 0; x < img.width(); ++x) {
            const int r = qRed(row[x]), g = qGreen(row[x]), b = qBlue(row[x]);
            const bool blue = b > r + 24 && b > g + 8;
            const bool green = g > r + 30 && g > b + 30;
            if (blue || green)
                ++bad;
        }
    }
    return bad;
}

namespace {

qreal effectiveOpacity(QQuickItem *item)
{
    qreal o = 1;
    for (QQuickItem *i = item; i; i = i->parentItem())
        o *= i->opacity();
    return o;
}

bool onScreen(QQuickItem *item, QQuickWindow *win)
{
    QRectF r = item->mapRectToScene(QRectF(0, 0, item->width(), item->height()));
    r = r.intersected(QRectF(0, 0, win->width(), win->height()));
    for (QQuickItem *p = item->parentItem(); p && !r.isEmpty(); p = p->parentItem())
        if (p->clip())
            r = r.intersected(p->mapRectToScene(QRectF(0, 0, p->width(), p->height())));
    return r.width() > 1 && r.height() > 1;
}

void walk(QQuickItem *item, QQuickWindow *win, int *words, QStringList *texts)
{
    if (!item->isVisible() || item->opacity() < 0.02)
        return;
    const char *cls = item->metaObject()->className();
    if ((qstrncmp(cls, "QQuickText", 10) == 0 && qstrncmp(cls, "QQuickTextInput", 15) != 0
         && qstrncmp(cls, "QQuickTextEdit", 14) != 0)
        || qstrncmp(cls, "QQuickTextInput", 15) == 0) {
        const QString t = item->property("text").toString();
        if (!t.trimmed().isEmpty() && effectiveOpacity(item) > 0.05 && onScreen(item, win)) {
            *words += countWords(t);
            if (texts)
                *texts << stripMarkup(t);
        }
    }
    for (QQuickItem *c : item->childItems())
        walk(c, win, words, texts);
}

void collect(QQuickItem *item, const QString &name, QList<QQuickItem *> *out)
{
    if (item->objectName() == name)
        *out << item;
    for (QQuickItem *c : item->childItems())
        collect(c, name, out);
}

struct TextBox {
    QRectF r;
    QString text;
};

void gatherBoxes(QQuickItem *item, QQuickWindow *win, QList<TextBox> *out)
{
    if (!item->isVisible() || item->opacity() < 0.02)
        return;
    const char *cls = item->metaObject()->className();
    if (qstrncmp(cls, "QQuickText", 10) == 0) {
        const QString t = item->property("text").toString();
        // Above half opacity only: mid-crossfade ghosts are transitions, not layout.
        if (!t.trimmed().isEmpty() && effectiveOpacity(item) > 0.5) {
            const qreal cw = std::min<qreal>(item->property("contentWidth").toReal(), item->width());
            const qreal ch = std::min<qreal>(item->property("contentHeight").toReal(), item->height() > 0 ? item->height() : 1e9);
            const int align = item->property("horizontalAlignment").toInt();
            qreal ox = 0;
            if (align & Qt::AlignRight)
                ox = item->width() - cw;
            else if (align & Qt::AlignHCenter)
                ox = (item->width() - cw) / 2;
            qreal oy = 0;
            const int valign = item->property("verticalAlignment").toInt();
            if (valign & Qt::AlignVCenter)
                oy = (item->height() - ch) / 2;
            QRectF r = item->mapRectToScene(QRectF(ox, oy, cw, ch));
            r = r.intersected(QRectF(0, 0, win->width(), win->height()));
            for (QQuickItem *p = item->parentItem(); p && !r.isEmpty(); p = p->parentItem())
                if (p->clip())
                    r = r.intersected(p->mapRectToScene(QRectF(0, 0, p->width(), p->height())));
            if (r.width() > 1 && r.height() > 1)
                *out << TextBox{r, stripMarkup(t).left(24)};
        }
    }
    for (QQuickItem *c : item->childItems())
        gatherBoxes(c, win, out);
}

}  // namespace

int textOverlaps(QQuickWindow *win, QStringList *pairs)
{
    QList<TextBox> boxes;
    // An open modal covers everything beneath it; audit only the modal then.
    QQuickItem *root = win->contentItem();
    for (const char *modal : {"newSessionDialog", "settingsOverlay", "onboardingModal", "slowmixOverlay"})
        for (QQuickItem *d : findItems(win->contentItem(), QString::fromLatin1(modal)))
            if (d->isVisible() && d->opacity() > 0.5)
                root = d;
    gatherBoxes(root, win, &boxes);
    int n = 0;
    for (int i = 0; i < boxes.size(); ++i)
        for (int j = i + 1; j < boxes.size(); ++j) {
            const QRectF x = boxes[i].r.intersected(boxes[j].r);
            if (x.width() > 2 && x.height() > 2) {
                ++n;
                if (pairs)
                    *pairs << QStringLiteral("'%1' x '%2'").arg(boxes[i].text, boxes[j].text);
            }
        }
    return n;
}

int visibleWords(QQuickWindow *win, QStringList *texts)
{
    int words = 0;
    walk(win->contentItem(), win, &words, texts);
    return words;
}

QList<QQuickItem *> findItems(QQuickItem *root, const QString &objectName)
{
    QList<QQuickItem *> out;
    collect(root, objectName, &out);
    return out;
}

// --- demo -------------------------------------------------------------------------------------

namespace {

struct DemoEvent {
    double t;
    std::function<void(QQuickWindow *)> fire;
};

QList<DemoEvent> demoScript(Backend *backend)
{
    QList<DemoEvent> ev;
    auto key = [&](double t, int k, Qt::KeyboardModifiers m = Qt::NoModifier, QString text = {}) {
        ev << DemoEvent{t, [=](QQuickWindow *w) { sendKey(w, k, m, text); }};
    };
    auto type = [&](double t0, const QString &s, double dt) {
        for (int i = 0; i < s.size(); ++i) {
            const QString c(s[i]);
            ev << DemoEvent{t0 + i * dt, [c](QQuickWindow *w) { typeText(w, c); }};
        }
    };
    auto letter = [&](double t, int k, const char *c, Qt::KeyboardModifiers m = Qt::NoModifier) {
        key(t, k, m, QString::fromLatin1(c));
    };
    // 0-3.4 s: first launch. The onboarding modal opens with Neovim Mode (Recommended) selected;
    // j/j/k/k walks Gaming and Mouse and back; Enter confirms Neovim.
    letter(0.9, Qt::Key_J, "j");
    letter(1.5, Qt::Key_J, "j");
    letter(2.1, Qt::Key_K, "k");
    letter(2.6, Qt::Key_K, "k");
    key(3.2, Qt::Key_Return);
    // 3.2-6 s: Constellation, islands named in plain English ("Serving 65B DCP=4 MTP"). Rapid
    // hops l l j h h k at key-repeat speed: the focus ring glides 120 ms per hop, retargets
    // mid-flight, no wobble. After 0.6 s idle the gold ring breathes around the predicted island
    // with "Open Stage  ⇥ Tab"; Tab opens it.
    letter(3.80, Qt::Key_L, "l");
    letter(3.95, Qt::Key_L, "l");
    letter(4.10, Qt::Key_J, "j");
    letter(4.25, Qt::Key_H, "h");
    letter(4.40, Qt::Key_H, "h");
    letter(4.55, Qt::Key_K, "k");
    letter(4.95, Qt::Key_L, "l");
    letter(5.10, Qt::Key_H, "h");
    key(6.0, Qt::Key_Tab);
    // 6-12.4 s: the cluster-serving Stage. The header says "Serving 65B DCP=4 MTP" and pills sit
    // under it (DCP: 4, TP: 4, MTP: on). Cmd+P drops DCP's card, l walks to TP and MTP (each
    // card: values across runs + one plain line), Enter scopes the composer to MTP; the question
    // streams its thinking dim, then the answer replaces it; Esc zooms out.
    key(6.8, Qt::Key_P, Qt::ControlModifier);
    letter(7.7, Qt::Key_L, "l");
    letter(8.6, Qt::Key_L, "l");
    key(9.5, Qt::Key_Return);
    type(9.9, QStringLiteral("worth it?"), 0.05);
    key(10.5, Qt::Key_Return);
    key(12.4, Qt::Key_Escape);
    // 12.4-13.9 s: back on the Constellation, the agent on kernel-search writes two turns while
    // nobody is looking: its island gets a quiet "+2" badge (nothing moves).
    if (backend) {
        auto bg = [backend](double t, const char *text) {
            return DemoEvent{t, [backend, text](QQuickWindow *) {
                                 Turn turn;
                                 turn.role = QStringLiteral("assistant");
                                 turn.text = QString::fromLatin1(text);
                                 backend->mockAppendTurn(QStringLiteral("kernel-search"), turn);
                             }};
        };
        ev << bg(12.9, "Still tuning the launch config.") << bg(13.3, "Trying warps=8 next.");
    }
    constexpr double s = 4.5;  // everything below moved 4.5 s later for the new beats above
    // 13.9-15.7 s: Cmd+N, type the goal, Enter: the new island zooms straight into its Ladder.
    key(9.4 + s, Qt::Key_N, Qt::ControlModifier);
    type(9.7 + s, QStringLiteral("profile fp4 kv-cache decode on sm120"), 0.035);
    key(11.2 + s, Qt::Key_Return);
    // 15.7-21.9 s: the ladder fills itself; the predictor rings the current rung, then the winner
    // rung with "Open Stage  ⇥ Tab"; Tab opens the Stage (its pills: tile, regs, DCP, MTP...).
    key(17.4 + s, Qt::Key_Tab);
    // 21.9-26.4 s: the chart's predictor circles the diff's pager dot ("View Diff  ⇥ Tab"); Tab
    // slides to the diff, the ring moves on ("View Table"), Tab again, then it offers the PDF.
    key(19.0 + s, Qt::Key_Tab);
    key(20.5 + s, Qt::Key_Tab);
    // 26.5-31.8 s: Cmd+, Command Settings. l walks Neovim -> Gaming -> Mouse (the keybind sheet
    // follows), h h back to Neovim, j onto UI Scale and Default Model; Esc closes.
    key(22.0 + s, Qt::Key_Comma, Qt::ControlModifier);
    letter(22.9 + s, Qt::Key_L, "l");
    letter(23.7 + s, Qt::Key_L, "l");
    letter(24.5 + s, Qt::Key_H, "h");
    letter(25.0 + s, Qt::Key_H, "h");
    letter(25.6 + s, Qt::Key_J, "j");
    letter(26.1 + s, Qt::Key_J, "j");
    key(27.3 + s, Qt::Key_Escape);
    // 32.3-36.4 s: type a question, Cmd+Shift+M opens the inline Model & Effort strip with speed tags:
    // w w (Sonnet · Fast -> Opus · Frontier -> Fable · Frontier), d d (low -> medium -> high),
    // s (back to Opus); Enter sends with Opus · high.
    type(27.8 + s, QStringLiteral("best arm?"), 0.04);
    key(28.5 + s, Qt::Key_M, Qt::ControlModifier | Qt::ShiftModifier);
    letter(29.1 + s, Qt::Key_W, "w");
    letter(29.6 + s, Qt::Key_W, "w");
    letter(30.1 + s, Qt::Key_D, "d");
    letter(30.6 + s, Qt::Key_D, "d");
    letter(31.2 + s, Qt::Key_S, "s");
    key(31.9 + s, Qt::Key_Return);
    // 36.4-44 s: thinking streams dim, the answer replaces it; the composer's ghost follow-up
    // gets the ring; Tab accepts it; Opt+Enter streams the trace, which collapses; Esc zooms out.
    key(33.6 + s, Qt::Key_Tab);
    key(34.4 + s, Qt::Key_Return, Qt::AltModifier);
    key(38.2 + s, Qt::Key_Escape);
    return ev;
}

}  // namespace

int recordDemo(const QString &outPath)
{
    StepDriver driver;
    driver.begin();
    Backend backend(Backend::Mode::Mock);
    backend.showOnboarding();  // the demo opens on the first-launch modal (no prefs file is touched)
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, true, QSize(1920, 1080));
    if (!win) {
        std::fprintf(stderr, "record-demo: QML failed to load\n");
        return 2;
    }
    const int fps = 60;
    const int frames = 44 * fps;  // 2640 frames, 44 s
    QProcess ffmpeg;
    ffmpeg.setProcessChannelMode(QProcess::ForwardedErrorChannel);
    ffmpeg.start(QStringLiteral("/opt/homebrew/bin/ffmpeg"),
                 {QStringLiteral("-y"), QStringLiteral("-loglevel"), QStringLiteral("error"), QStringLiteral("-f"),
                  QStringLiteral("rawvideo"), QStringLiteral("-pix_fmt"), QStringLiteral("rgba"), QStringLiteral("-s"),
                  QStringLiteral("1920x1080"), QStringLiteral("-r"), QString::number(fps), QStringLiteral("-i"),
                  QStringLiteral("-"), QStringLiteral("-c:v"), QStringLiteral("libx264"), QStringLiteral("-preset"),
                  QStringLiteral("medium"), QStringLiteral("-crf"), QStringLiteral("16"), QStringLiteral("-tune"),
                  QStringLiteral("animation"), QStringLiteral("-pix_fmt"), QStringLiteral("yuv420p"),
                  QStringLiteral("-threads"), QStringLiteral("2"), QStringLiteral("-movflags"), QStringLiteral("+faststart"),
                  outPath});
    if (!ffmpeg.waitForStarted(5000)) {
        std::fprintf(stderr, "record-demo: ffmpeg did not start\n");
        return 3;
    }
    const QList<DemoEvent> script = demoScript(&backend);
    qsizetype next = 0;
    QElapsedTimer wall;
    wall.start();
    qint64 grabMs = 0;
    int overlapFrames = 0;
    int colourFrames = 0;
    for (int f = 0; f < frames; ++f) {
        const double t = double(f) / fps;
        while (next < script.size() && script[next].t <= t + 1e-9)
            script[next++].fire(win);
        if (f > 0)
            backend.advance(1.0 / fps);
        driver.stepTo(qRound64(f * 1000.0 / fps));
        QCoreApplication::processEvents();
        QElapsedTimer g;
        g.start();
        QImage img = win->grabWindow();
        grabMs += g.elapsed();
        if (f % 6 == 0) {  // audit 10 frames per second, as the 30 fps recorder did
            QStringList pairs;
            if (const int n = textOverlaps(win, &pairs)) {
                overlapFrames += 1;
                std::fprintf(stderr, "record-demo: t=%.2fs %d text overlaps: %s\n", t, n, qPrintable(pairs.join("; ")));
            }
            const int bad = forbiddenPixels(img);
            if (bad > 0) {
                colourFrames += 1;
                std::fprintf(stderr, "record-demo: t=%.2fs %d blue/green pixels\n", t, bad);
            }
        }
        if (img.size() != QSize(1920, 1080))
            img = img.scaled(1920, 1080);
        img = img.convertToFormat(QImage::Format_RGBA8888);
        for (int y = 0; y < img.height(); ++y)
            ffmpeg.write(reinterpret_cast<const char *>(img.constScanLine(y)), img.width() * 4);
        while (ffmpeg.bytesToWrite() > 64 * 1024 * 1024)
            ffmpeg.waitForBytesWritten(1000);
        if (f % 180 == 0)
            std::fprintf(stderr, "record-demo: frame %d/%d (%.1fs wall)\n", f, frames, wall.elapsed() / 1000.0);
    }
    ffmpeg.closeWriteChannel();
    ffmpeg.waitForFinished(600000);
    std::fprintf(stderr, "record-demo: %d frames, %.1fs wall, grab %.1f ms/frame, ffmpeg exit %d\n", frames,
                 wall.elapsed() / 1000.0, double(grabMs) / frames, ffmpeg.exitCode());
    std::fprintf(stderr, "record-demo: audit of %d frames: %d with text overlap, %d with blue/green pixels\n",
                 (frames + 5) / 6, overlapFrames, colourFrames);
    if (overlapFrames > 0 || colourFrames > 0)
        return 6;
    if (backend.spawnCount() != 0) {
        std::fprintf(stderr, "record-demo: mock backend spawned a process\n");
        return 4;
    }
    return ffmpeg.exitStatus() == QProcess::NormalExit && ffmpeg.exitCode() == 0 ? 0 : 5;
}

int probeLive(const QString &shotDir)
{
    // Read-only: registry + transcripts + board. No helpers, no new/send; any window is offscreen.
    Backend backend(Backend::Mode::Live, nullptr, false);
    QElapsedTimer t;
    t.start();
    while (t.elapsed() < 20000 && backend.projects().isEmpty())
        settle(100);
    settle(2500);  // let the board poll land
    int arts = 0;
    int infProjects = 0, infWithArts = 0, infArts = 0, infProse = 0;
    QMap<QString, int> infKinds;
    QStringList infCtx;
    for (const QVariant &v : backend.projects()) {
        const QVariantMap p = v.toMap();
        QStringList kinds;
        for (const QVariant &a : p.value("artifacts").toList())
            kinds << a.toMap().value("kind").toString();
        arts += kinds.size();
        if (p.value("redacted").toBool()) {
            // infinity: parsed for the local window, but this output may be pasted anywhere, so
            // print counts and artifact kinds only, never names, headlines, goals or prose.
            ++infProjects;
            infWithArts += kinds.isEmpty() ? 0 : 1;
            infArts += int(kinds.size());
            infProse += p.value("prose").toString().isEmpty() ? 0 : 1;
            {
                // Shape of the Q&A context only: sizes and section presence, never the text.
                const QString ctx = backend.contextFor(p.value("key").toString(), 0);
                infCtx << QStringLiteral("%1 chars/%2 turns/%3").arg(ctx.size())
                              .arg(ctx.count(QStringLiteral("\nUser: ")) + ctx.count(QStringLiteral("\nAssistant: ")))
                              .arg(ctx.contains(QStringLiteral("\nFiles touched: ")) ? QStringLiteral("files") : QStringLiteral("no-files"));
            }
            for (const QString &k : kinds)
                infKinds[k] += 1;
            continue;
        }
        std::printf("%-28s %-9s sessions=%d redacted=%d headline='%s' artifacts=[%s] goal='%s'\n",
                    qPrintable(p.value("name").toString()), qPrintable(p.value("status").toString()),
                    p.value("sessionCount").toInt(), p.value("redacted").toBool(),
                    qPrintable(p.value("headline").toString()), qPrintable(kinds.join(',')),
                    qPrintable(p.value("ladder").toMap().value("goal").toString()));
        QStringList pills;
        for (const QVariant &pv : p.value("params").toList()) {
            const QVariantMap pm = pv.toMap();
            pills << (pm.value("value").toString().isEmpty() ? pm.value("name").toString()
                                                             : pm.value("name").toString() + ": " + pm.value("value").toString());
        }
        std::printf("%-28s   job='%s' long=%d pills=[%s]\n", "", qPrintable(p.value("jobName").toString()),
                    p.value("longRunning").toBool(), qPrintable(pills.join(", ")));
        std::printf("%-28s   key=%s category='%s' turns=%d%s last='%s'\n", "", qPrintable(p.value("key").toString()),
                    qPrintable(p.value("category").toString()), int(p.value("turns").toList().size()),
                    p.value("boardSnippet").toString().isEmpty() ? "" : " +board",
                    qPrintable(p.value("lastLine").toString()));
    }
    {
        QStringList k;
        for (auto it = infKinds.cbegin(); it != infKinds.cend(); ++it)
            k << QStringLiteral("%1=%2").arg(it.key()).arg(it.value());
        std::printf("probe-live: infinity (contents withheld): %d projects, %d with artifacts, %d with prose, "
                    "%d artifacts [%s]\n",
                    infProjects, infWithArts, infProse, infArts, qPrintable(k.join(',')));
        std::printf("probe-live: infinity Q&A context shape: [%s]\n", qPrintable(infCtx.join("; ")));
    }
    std::printf("probe-live: %d projects, %d artifacts, %d transcripts parsed, %d read-only processes, %.1fs\n",
                int(backend.projects().size()), arts, int(backend.transcriptsRead().size()), backend.spawnCount(),
                t.elapsed() / 1000.0);
    {
        const Backend::GuiStall &g = backend.guiStall();
        std::printf("probe-live: GUI-thread time %.1f ms (longest slice %.2f ms), transcript parsing %.1f ms on the GUI "
                    "thread and %.1f ms on the worker\n",
                    g.refreshMs + g.applyMs + g.publishMs + g.boardMs, g.maxMs, g.guiParseMs, g.workerMs);
    }
    if (!shotDir.isEmpty() && !backend.projects().isEmpty()) {
        QQmlApplicationEngine engine;
        QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
        if (!win)
            return 2;
        settle(500);
        win->grabWindow().save(shotDir + QStringLiteral("/live_constellation.png"));
        sendKey(win, Qt::Key_1);
        settle(600);
        for (int i = 0; i < 4; ++i) {
            QStringList pairs;
            const int overlaps = textOverlaps(win, &pairs);
            const QImage img = win->grabWindow();
            img.save(shotDir + QStringLiteral("/live_stage%1.png").arg(i));
            // Overlap pairs quote on-screen text, which may be infinity work: counts only.
            std::printf("probe-live: stage %d words=%d overlaps=%d blue/green=%d\n", i, visibleWords(win), overlaps,
                        forbiddenPixels(img));
            sendKey(win, Qt::Key_Right);
            settle(500);
        }
        sendKey(win, Qt::Key_2, Qt::ControlModifier);
        settle(500);
        win->grabWindow().save(shotDir + QStringLiteral("/live_ladder.png"));
    }
    return backend.projects().isEmpty() ? 1 : 0;
}

int probeChat(const QString &model, const QString &effort, const QString &question)
{
    // No window and no model warm-up: only the channel the first ask opens.
    Backend backend(Backend::Mode::Live, nullptr, false);
    backend.setChatModel(model);
    backend.setChatEffort(effort);
    if (backend.chatModel() != model) {
        std::fprintf(stderr, "probe-chat: unknown model %s\n", qPrintable(model));
        return 64;
    }
    std::printf("probe-chat: %s · %s via %s (channel %s)\n", qPrintable(backend.chatModel()),
                qPrintable(backend.chatEffort()), qPrintable(backend.chatProvider()),
                qPrintable(channelKeyFor(backend.chatModel(), backend.chatEffort())));
    std::printf("probe-chat: picker label '%s'\n", qPrintable(modelDisplayLabel(backend.chatModel())));
    const QString key = channelKeyFor(backend.chatModel(), backend.chatEffort());
    qint64 pids[2] = {0, 0};
    bool ok = true;
    // Every phase the answer line showed, in order ("sending > thinking > writing").
    QStringList phases;
    QObject::connect(&backend, &Backend::quickChanged, &backend, [&] {
        const QString p = backend.quickPhase();
        if (!p.isEmpty() && (phases.isEmpty() || phases.last() != p))
            phases << p;
    });
    const QStringList questions = {question, QStringLiteral("And at night? One word.")};
    for (int turn = 1; turn <= 2; ++turn) {
        phases.clear();
        QElapsedTimer t;
        t.start();
        backend.ask(QString(), 0, questions.value(turn - 1));
        while (backend.quickBusy() && t.elapsed() < 120000)
            settle(20);
        pids[turn - 1] = backend.channelPid(key);
        auto ms = [](qint64 v) { return v < 0 ? QStringLiteral("-") : QString::number(v / 1000.0, 'f', 2) + "s"; };
        std::printf("probe-chat: turn %d %.2fs busy=%d pid=%lld | first sign of life %s, first thinking %s, first token %s, "
                    "%d thinking deltas, %d thinking tokens | phases: %s | answer='%s'\n",
                    turn, t.elapsed() / 1000.0, backend.quickBusy(), static_cast<long long>(pids[turn - 1]),
                    qPrintable(ms(backend.quickFirstEventMs())), qPrintable(ms(backend.quickFirstThinkingMs())),
                    qPrintable(ms(backend.quickFirstTokenMs())), backend.quickThinkingDeltas(),
                    backend.quickThinkingTokens(), qPrintable(phases.join(QStringLiteral(" > "))),
                    qPrintable(backend.quickAnswer()));
        if (!backend.quickThinking().isEmpty())
            std::printf("probe-chat: turn %d thinking (tail): '%s'\n", turn, qPrintable(backend.quickThinking()));
        ok = ok && !backend.quickBusy() && !backend.quickAnswer().isEmpty()
            && !backend.quickAnswer().contains(QLatin1String("unavailable"));
    }
    const bool reused = pids[0] > 0 && pids[0] == pids[1];
    std::printf("probe-chat: second turn reused the daemon: %s\n", reused ? "yes" : "no");

    // Teardown: the daemon and every child it spawned must be gone after shutdown.
    auto childrenOf = [](qint64 pid) {
        QProcess pg;
        pg.start(QStringLiteral("/usr/bin/pgrep"), {QStringLiteral("-P"), QString::number(pid)});
        pg.waitForFinished(2000);
        QList<qint64> out;
        for (const QByteArray &l : pg.readAllStandardOutput().split('\n'))
            if (!l.trimmed().isEmpty())
                out << l.trimmed().toLongLong();
        return out;
    };
    QList<qint64> tree;
    if (pids[1] > 0) {
        tree << pids[1];
        for (int i = 0; i < tree.size() && i < 64; ++i)
            tree += childrenOf(tree[i]);
    }
    QElapsedTimer down;
    down.start();
    const QList<qint64> stopped = backend.shutdownChannels();
    QList<qint64> alive;
    for (int waitMs = 0; waitMs <= 3000; waitMs += 100) {
        alive.clear();
        for (qint64 p : tree)
            if (::kill(pid_t(p), 0) == 0)
                alive << p;
        if (alive.isEmpty())
            break;
        settle(100);
    }
    QStringList treeS, aliveS;
    for (qint64 p : tree)
        treeS << QString::number(p);
    for (qint64 p : alive)
        aliveS << QString::number(p);
    std::printf("probe-chat: teardown: stopped %d daemon(s) in %lld ms; process tree [%s]; still alive: [%s] -> %s\n",
                int(stopped.size()), static_cast<long long>(down.elapsed()), qPrintable(treeS.join(',')),
                qPrintable(aliveS.join(',')), alive.isEmpty() ? "clean" : "LEAKED");
    return ok && reused && alive.isEmpty() ? 0 : 1;
}

int smokeOffscreen(const QString &shotDir)
{
    Backend backend(Backend::Mode::Mock);
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    if (!win)
        return 2;
    settle(300);
    if (!shotDir.isEmpty()) {
        QDir().mkpath(shotDir);
        win->grabWindow().save(shotDir + QStringLiteral("/constellation_1.00x.png"));
        backend.setUiScale(Backend::kMaxScale);
        settle(200);  // the scale pill is up for 1.2 s after a change
        win->grabWindow().save(shotDir + QStringLiteral("/constellation_1.60x.png"));
        sendKey(win, Qt::Key_Right);
        sendKey(win, Qt::Key_Right);
        settle(700);
        sendKey(win, Qt::Key_Return);
        settle(600);
        win->grabWindow().save(shotDir + QStringLiteral("/stage_1.60x.png"));
        backend.setUiScale(Backend::kMinScale);
        settle(200);
        win->grabWindow().save(shotDir + QStringLiteral("/stage_0.75x.png"));
        backend.setUiScale(1.0);
        settle(1600);
        win->grabWindow().save(shotDir + QStringLiteral("/stage_1.00x.png"));
        sendKey(win, Qt::Key_2, Qt::ControlModifier);
        backend.setUiScale(Backend::kMaxScale);
        settle(600);
        win->grabWindow().save(shotDir + QStringLiteral("/ladder_1.60x.png"));
        backend.setUiScale(1.0);
        sendKey(win, Qt::Key_Escape);
        settle(1600);
    }
    sendKey(win, Qt::Key_1);
    settle(400);
    const QImage img = win->grabWindow();
    std::printf("smoke-offscreen ok: platform=%s grab=%dx%d words=%d spawns=%d\n",
                qPrintable(QGuiApplication::platformName()), img.width(), img.height(), visibleWords(win),
                backend.spawnCount());
    return img.isNull() ? 3 : 0;
}

int probeJarvis(const QString &outDir, const QString &app, const QString &jpegPath)
{
    // No window, no warm, no send: builds the exact Spotlight Turn 0 and visual follow-up
    // payloads from live state (shared history from studio_history.jsonl, the app's real AX tree,
    // a real JPEG) and writes them as JSON for an offline replay against the Jarvis endpoint.
    Backend backend(Backend::Mode::Live, nullptr, false);
    settle(6000);  // projects, the Phonon tail poll and the memory index land
    QVector<ChatTurn> history;
    {
        QFile f(Backend::defaultHistoryPath());
        QList<QByteArray> lines;
        if (f.open(QIODevice::ReadOnly))
            lines = f.readAll().split('\n');
        lines.removeIf([](const QByteArray &l) { return l.trimmed().isEmpty(); });
        for (const QByteArray &l : lines.mid(std::max<qsizetype>(0, lines.size() - 10))) {
            const QJsonObject o = QJsonDocument::fromJson(l).object();
            history << ChatTurn{QStringLiteral("user"), o.value("prompt").toString()}
                    << ChatTurn{QStringLiteral("assistant"), o.value("answer").toString()};
        }
    }
    backend.setSharedHistory(history);
    const QVector<JarvisLeg> legs = backend.jarvisLegs();
    if (legs.isEmpty()) {
        std::fprintf(stderr, "probe-jarvis: no Jarvis legs (set LAPIS_JARVIS_ENDPOINT)\n");
        return 2;
    }
    int pid = 0;
    uint32_t display = 0;
    for (const studio::cua::AppWindowInfo &w : studio::cua::listAppWindows())
        if (studio::cua::appNameMatches(w.appName, app) && w.displayId) {
            pid = w.pid;
            display = w.displayId;
            break;
        }
    if (!pid) {
        std::fprintf(stderr, "probe-jarvis: no on-screen window for %s\n", qPrintable(app));
        return 2;
    }
    QFile jf(jpegPath);
    const QByteArray jpeg = jf.open(QIODevice::ReadOnly) ? jf.readAll() : QByteArray();
    // Turn 0: submitSpotlight's desktop slice.
    const QString slice = QStringLiteral("Active macOS app behind Spotlight: %1\n").arg(app)
                          + studio::cua::formatNodesForModel(studio::cua::inspectApp(pid, 500, 32, 0.35),
                                                             studio::cua::kPostActionChars);
    const QString goal = QStringLiteral("Click on my profile.");
    // Follow-up: one click step whose observation is the display's AX nodes, the app's first.
    QVector<studio::cua::AxNode> nodes = studio::cua::inspectDisplay(display, 500, 0.9);
    std::stable_partition(nodes.begin(), nodes.end(),
                          [&](const studio::cua::AxNode &n) { return studio::cua::appNameMatches(n.appName, app); });
    CuaStepResult step;
    step.action = QStringLiteral("cua_click");
    step.target = app + QStringLiteral("|Me");
    step.ok = true;
    step.tier = QStringLiteral("L1_AX");
    step.detail = QStringLiteral("AXButton \"Me\" (%1)").arg(app);
    step.observation = studio::cua::formatNodesForModel(nodes, studio::cua::kPostActionChars);
    const QString spot = QStringLiteral("spotlight");
    const QString project = backend.focusedProject();
    const TieredPrompt t0 = backend.legPrompt(spot, legs, project, slice, goal, false, {});
    backend.setSharedHistory(history << ChatTurn{QStringLiteral("user"), goal}
                                     << ChatTurn{QStringLiteral("assistant"), QStringLiteral("[[ACTION:cua_click:%1|Me]]").arg(app)});
    const TieredPrompt fu = backend.legPrompt(spot, legs, project, QString(), cuaFollowupPrompt({step}), true, jpeg);
    QDir().mkpath(outDir);
    auto dump = [&](const char *name, const TieredPrompt &p) {
        QFile out(outDir + QLatin1Char('/') + QLatin1String(name) + QStringLiteral(".json"));
        if (out.open(QIODevice::WriteOnly))
            out.write(buildJarvisLegRequest(legs.first(), p));
        qsizetype hist = 0;
        for (const ChatTurn &t : p.history)
            hist += t.content.size();
        std::printf("probe-jarvis: %s effort %s | tier0 %lld trunk %lld history %lld (%lld turns) | leaf: slice %lld "
                    "live %lld recent %lld phonon %lld memory %lld question %lld | image %lld bytes\n",
                    name, qPrintable(p.effort),
                    static_cast<long long>(tier0SystemText(p.systemPrompt, p.model, p.extraSystemContext).size()),
                    static_cast<long long>(p.trunk.size()), static_cast<long long>(hist),
                    static_cast<long long>(p.history.size()), static_cast<long long>(p.targetSlice.size()),
                    static_cast<long long>(p.liveState.size()), static_cast<long long>(p.recentActions.join('\n').size()),
                    static_cast<long long>(p.phononTail.size()), static_cast<long long>(p.memory.size()),
                    static_cast<long long>(p.question.size()), static_cast<long long>(p.imageJpeg.size()));
    };
    dump("turn0", t0);
    dump("followup", fu);
    return 0;
}

}  // namespace studio
