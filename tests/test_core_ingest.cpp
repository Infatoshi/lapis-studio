// test_core_ingest: lapis_core (TargetRef, SpscRing, window geometry) and lapis_ingest (Phonon,
// Slowmix, cross-session MemoryIndex).
#include "core/ring_buffer.hpp"
#include "core/types.hpp"
#include "core/window_geometry.hpp"
#include "ingest/memory.hpp"
#include "ingest/phonon.hpp"
#include "ingest/slowmix.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLocalServer>
#include <QLocalSocket>
#include <QTemporaryDir>

#include <algorithm>
#include <chrono>

#include <cstdio>
#include <thread>
#include <utime.h>

using namespace studio;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const char *name, const QString &detail = QString())
{
    if (ok) {
        ++g_pass;
        return;
    }
    ++g_fail;
    std::printf("FAIL %s%s%s\n", name, detail.isEmpty() ? "" : ": ", qPrintable(detail));
}

bool writeFile(const QString &path, const QByteArray &data)
{
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return false;
    const bool ok = f.write(data) == data.size();
    f.close();
    return ok;
}

void setAgeSeconds(const QString &path, qint64 seconds)
{
    const time_t t = time_t(QDateTime::currentSecsSinceEpoch() - seconds);
    struct utimbuf ub {t, t};
    ::utime(QFile::encodeName(path).constData(), &ub);
}

void coreTests()
{
    TargetRef a{QStringLiteral("artifact"), QStringLiteral("s1"), QStringLiteral("a3"), QStringLiteral("/tmp/x.csv"), 7};
    TargetRef b = a;
    check(a == b && a.key() == b.key(), "target: copies compare equal");
    b.rev = 8;
    check(a != b && a.key() == b.key() && a.revKey() != b.revKey(),
          "target: a new rev keeps key() but changes revKey()", a.revKey() + " vs " + b.revKey());
    check(TargetRef().isNull() && !a.isNull(), "target: isNull");
    check(askRouteName(AskRoute::HoverAsk) == "hover" && askRouteName(AskRoute::Conversation) == "conversation"
              && askRouteName(AskRoute::AgentSession) == "agent",
          "types: AskRoute names");
    const ChatTurn t1{QStringLiteral("user"), QStringLiteral("hi")};
    check(t1 == ChatTurn{QStringLiteral("user"), QStringLiteral("hi")}, "types: ChatTurn equality");
    ContextSlice cs;
    check(cs.isEmpty() && cs.tokens == -1, "types: ContextSlice defaults");
    check(estimateTokens(QString(400, QLatin1Char('x'))) == 100, "types: estimateTokens ~4 chars/token");

    // SpscRing: capacity rounds up, push/pop order, full/empty edges.
    SpscRing<int> r(5);
    check(r.capacity() == 8, "spsc: capacity rounds up to a power of two", QString::number(r.capacity()));
    int popped = -1;
    check(!r.tryPop(popped), "spsc: empty ring pops nothing");
    int pushed = 0;
    while (r.tryPush(pushed))
        ++pushed;
    check(pushed == 8 && r.sizeApprox() == 8, "spsc: holds exactly capacity items", QString::number(pushed));
    bool order = true;
    for (int i = 0; i < 8; ++i)
        order = order && r.tryPop(popped) && popped == i;
    check(order && r.emptyApprox(), "spsc: FIFO order");

    SpscRing<QString> rs(4);
    QString s;
    check(rs.tryPush(QStringLiteral("abc")) && rs.tryPop(s) && s == "abc" && !rs.tryPop(s),
          "spsc: QString round-trip");

    // Cross-thread: 200k items through a 64-slot ring, order and sum preserved.
    SpscRing<quint64> ring(64);
    constexpr quint64 kN = 200000;
    std::thread producer([&ring] {
        for (quint64 i = 1; i <= kN;) {
            if (ring.tryPush(quint64(i)))
                ++i;
            else
                std::this_thread::yield();
        }
    });
    quint64 expect = 1, sum = 0;
    bool inOrder = true;
    while (expect <= kN) {
        quint64 v = 0;
        if (ring.tryPop(v)) {
            inOrder = inOrder && v == expect;
            sum += v;
            ++expect;
        } else {
            std::this_thread::yield();
        }
    }
    producer.join();
    check(inOrder && sum == kN * (kN + 1) / 2, "spsc: 200k items across threads, in order, none lost");
}

void phononTests()
{
    QTemporaryDir td;
    check(td.isValid(), "phonon: temp dir");

    // Missing file.
    const PhononSnapshot missing = readPhononSnapshot(td.path() + "/missing.txt");
    check(missing.tail.isEmpty() && missing.status.isEmpty() && missing.mtimeMs == 0 && !missing.live,
          "phonon: missing file -> empty snapshot");
    check(readPhononTail(td.path() + "/missing.txt").isEmpty(), "phonon: missing file -> empty tail");

    // Legacy, headerless file through a symlink (the --test case, kept working).
    const QString plain = td.path() + "/plain.txt";
    writeFile(plain, "line 1\n\nline 2\nline 3\nline 4\nline 5\nline 6\n  \nline 7\n");
    QFile::link(plain, td.path() + "/plain-link.txt");
    check(readPhononTail(td.path() + "/plain-link.txt", 5, 60000) == "line 3\nline 4\nline 5\nline 6\nline 7",
          "phonon: headerless symlinked file -> last 5 non-empty lines");

    // Phonon format: header through "\n---\n" is stripped, Status parsed.
    const QByteArray rec =
        "Meeting: weekly sync\nStarted: 2026-11-18 09:00\nStatus: recording\n---\n"
        "0:01 You: can you hear me\n0:04 Call: yes, loud and clear\n0:09 You: let's start with the KV numbers\n";
    const QString live = td.path() + "/rec.txt";
    writeFile(live, rec);
    const PhononSnapshot s1 = readPhononSnapshot(live, 15, 60000);
    check(s1.status == "recording", "phonon: Status: recording parsed", s1.status);
    check(!s1.tail.contains("Status:") && !s1.tail.contains("---") && !s1.tail.contains("Meeting:"),
          "phonon: header stripped at \\n---\\n", s1.tail);
    check(s1.tail == "0:01 You: can you hear me\n0:04 Call: yes, loud and clear\n0:09 You: let's start with the KV numbers",
          "phonon: turns kept in order", s1.tail);
    check(s1.live && s1.mtimeMs > 0, "phonon: fresh recording is live");
    check(readPhononSnapshot(live, 2, 60000).tail == "0:04 Call: yes, loud and clear\n0:09 You: let's start with the KV numbers",
          "phonon: maxLines keeps the newest turns");

    // Header only, no turns yet.
    const QString empty = td.path() + "/empty.txt";
    writeFile(empty, "Status: transcribing\n---\n");
    const PhononSnapshot se = readPhononSnapshot(empty, 15, 60000);
    check(se.status == "transcribing" && se.tail.isEmpty() && !se.live, "phonon: header with no turns -> not live");

    // Complete: tail still readable, but not live.
    const QString done = td.path() + "/done.txt";
    writeFile(done, "Status: complete\n---\n0:01 You: bye\n");
    const PhononSnapshot sd = readPhononSnapshot(done, 15, 60000);
    check(sd.status == "complete" && sd.tail == "0:01 You: bye" && !sd.live, "phonon: complete -> tail but not live");

    // Stale: status and mtime only.
    setAgeSeconds(live, 3600);
    const PhononSnapshot st = readPhononSnapshot(live, 15, 10 * 60 * 1000);
    check(st.tail.isEmpty() && !st.live && st.status == "recording" && st.mtimeMs > 0,
          "phonon: stale file -> no tail, status kept");
    check(!readPhononSnapshot(live, 15, 0).tail.isEmpty(), "phonon: maxAgeMs <= 0 ignores age");

    // Large file: tail read from the end, header still parsed from the top, partial line dropped.
    QByteArray big = "Status: recording\n---\n";
    for (int i = 0; i < 3000; ++i)
        big += QByteArray("0:") + QByteArray::number(i) + " Call: filler turn number " + QByteArray::number(i) + "\n";
    const QString bigPath = td.path() + "/big.txt";
    writeFile(bigPath, big);
    const PhononSnapshot sb = readPhononSnapshot(bigPath, 3, 60000);
    check(sb.status == "recording" && sb.tail.endsWith("filler turn number 2999") && sb.tail.split('\n').size() == 3,
          "phonon: >16 KB file -> newest 3 turns, status from the header", sb.tail);

    // Atomic replacement: latest.txt is a symlink repointed via rename(2), as Phonon does.
    const QString latest = td.path() + "/latest.txt";
    const QString m1 = td.path() + "/m1.txt", m2 = td.path() + "/m2.txt";
    writeFile(m1, "Status: recording\n---\n0:01 You: first meeting\n");
    writeFile(m2, "Status: recording\n---\n0:01 You: second meeting\n0:03 Call: hello\n");
    QFile::link(m1, latest);
    check(readPhononTail(latest, 15, 60000) == "0:01 You: first meeting", "phonon: symlink -> first meeting");
    const QString tmpLink = td.path() + "/latest.txt.tmp";
    QFile::link(m2, tmpLink);
    const bool renamed = ::rename(QFile::encodeName(tmpLink).constData(), QFile::encodeName(latest).constData()) == 0;
    check(renamed, "phonon: atomic symlink rename succeeded");
    const PhononSnapshot sr = readPhononSnapshot(latest, 15, 60000);
    check(sr.tail == "0:01 You: second meeting\n0:03 Call: hello" && sr.live,
          "phonon: after atomic symlink replacement -> new meeting", sr.tail);

    // Atomic replacement of a regular file (write temp, rename over).
    const QString reg = td.path() + "/reg.txt";
    writeFile(reg, "Status: recording\n---\n0:01 You: old\n");
    writeFile(reg + ".new", "Status: complete\n---\n0:01 You: new\n");
    ::rename(QFile::encodeName(reg + ".new").constData(), QFile::encodeName(reg).constData());
    const PhononSnapshot sg = readPhononSnapshot(reg, 15, 60000);
    check(sg.tail == "0:01 You: new" && sg.status == "complete", "phonon: atomic file rename -> new content");
}

void slowmixTests()
{
    QTemporaryDir td;
    const QString clips = td.path() + "/clips";
    QDir().mkpath(clips + "/gen/takes");
    QDir().mkpath(clips + "/sub");
    QDir().mkpath(clips + "/.hidden");
    for (const char *f : {"song_a.wav", "song_b.flac", "gen/takes/song_a_take2.wav", "song_a_runnerup.wav",
                          "song_a_nightcore_hall.wav", "song_c_0.90x_hall.wav", "never.wav", "stale.wav",
                          "sub/deep_cut.mp3", "notes.txt", ".hidden/x.wav", "my-sfx_hit.wav"})
        writeFile(clips + "/" + f, "");
    // Real "now": the controller below curates against the wall clock.
    const qint64 now = QDateTime::currentSecsSinceEpoch();
    const qint64 day = 86400;
    auto row = [&](qint64 ago, int secs, const char *speed, const char *reverb, const QString &track) {
        return QString("%1\t%2\t%3\t%4\t%5\n").arg(now - ago).arg(secs).arg(speed, reverb, track).toUtf8();
    };
    QByteArray plays = "end_unix\tseconds\tspeed\treverb\ttrack\n";
    plays += row(1 * day, 600, "slowed", "hall", "song_a.wav");
    plays += row(2 * day, 300, "slowed", "heavy", "song_a.wav");
    plays += row(3 * day, 120, "normal", "hall", "song_a.wav");
    plays += row(1 * day, 600, "1.10x", "heavy", "song_b.flac");
    plays += row(5 * day, 600, "1.10x", "heavy", "song_b.flac");
    plays += row(6 * day, 300, "nightcore", "off", "song_b.flac");
    // Old habits never outvote the last 30 days.
    plays += row(40 * day, 3000, "normal", "hall", "song_b.flac");
    plays += row(2 * day, 180, "fast", "off", "sub/deep_cut.mp3");
    plays += row(45 * day, 900, "slowed", "hall", "stale.wav");
    plays += row(1 * day, 900, "slowed", "hall", "gen/takes/song_a_take2.wav");
    plays += row(1 * day, 900, "slowed", "hall", "song_a_runnerup.wav");
    plays += row(1 * day, 900, "slowed", "hall", "song_a_nightcore_hall.wav");
    plays += row(1 * day, 900, "slowed", "hall", "song_c_0.90x_hall.wav");
    plays += "garbage line\n\n1799999999\tnotanumber\tslowed\thall\tsong_a.wav\n";
    const QString playsPath = clips + "/.plays.tsv";
    writeFile(playsPath, plays);

    check(slowmixSpeedFromTag("slowed") == 0.85 && slowmixSpeedFromTag("normal") == 1.0
              && slowmixSpeedFromTag("nightcore") == 1.25 && slowmixSpeedFromTag("fast") == 1.6
              && qFuzzyCompare(slowmixSpeedFromTag("1.10x"), 1.10) && slowmixSpeedFromTag("weird") == 0,
          "slowmix: speed tags map to multipliers");

    const auto stats = parseSlowmixPlays(playsPath, now, 30);
    const SlowmixPlayStats a = stats.value("song_a.wav");
    check(a.totalSeconds == 1020 && a.recentSeconds == 1020 && a.lastPlayed == now - day,
          "slowmix: per-track total / recent seconds and last played",
          QString("%1 %2 %3").arg(a.totalSeconds).arg(a.recentSeconds).arg(a.lastPlayed));
    check(a.favoriteSpeed == 0.85 && a.favoriteReverb == "hall", "slowmix: favorite preset weighted by seconds",
          QString("%1 %2").arg(a.favoriteSpeed).arg(a.favoriteReverb));
    const SlowmixPlayStats b = stats.value("song_b.flac");
    check(b.totalSeconds == 4500 && b.recentSeconds == 1500 && qFuzzyCompare(b.favoriteSpeed, 1.10)
              && b.favoriteReverb == "heavy",
          "slowmix: favorites come from the last 30 days, not 40-day-old habits",
          QString("%1 %2 %3 %4").arg(b.totalSeconds).arg(b.recentSeconds).arg(b.favoriteSpeed).arg(b.favoriteReverb));
    const SlowmixPlayStats st = stats.value("stale.wav");
    check(st.totalSeconds == 900 && st.recentSeconds == 0 && st.favoriteSpeed == 0.85,
          "slowmix: stale track keeps all-time favorites, zero recent seconds");
    check(parseSlowmixPlays(td.path() + "/missing.tsv", now).isEmpty(), "slowmix: missing log -> no stats");

    for (const char *junk : {"gen/takes/x.wav", "takes/x.wav", "experiments/a/x.wav", "decompiled/x.wav",
                             "graft/x.wav", "ddsp/x.wav", "x_runnerup.wav", "x_raw_mix.wav", "boom_sfx.wav",
                             "x_take3.wav", "x_slowed_hall.wav", "x_normal_off.wav", "x_0.90x_hall.wav",
                             "x_1.10x_heavy.wav", "x_nightcore.wav", "X_NIGHTCORE_HALL.WAV", ".hidden/x.wav"})
        check(isSlowmixJunk(junk), "slowmix: junk filtered", junk);
    for (const char *song : {"song.wav", "sub/song.mp3", "general_song.wav", "takeoff.wav", "nightcore_dream.wav"})
        check(!isSlowmixJunk(song), "slowmix: real song kept", song);

    const QList<SlowmixTrack> cur = curateSlowmixTracks(clips, playsPath, false, now);
    QStringList names;
    for (const SlowmixTrack &t : cur)
        names << t.relPath;
    check(names == QStringList({"song_b.flac", "song_a.wav", "sub/deep_cut.mp3"}),
          "slowmix: curated = recent songs only, by 30-day minutes (no junk, never-played or stale)",
          names.join(", "));
    if (cur.size() == 3) {
        check(cur[0].recentMinutes == 25 && qFuzzyCompare(cur[0].favoriteSpeed, 1.10) && cur[0].favoriteReverb == "heavy"
                  && cur[0].name == "song b" && cur[0].fileName == "song_b.flac" && cur[0].path == clips + "/song_b.flac",
              "slowmix: top track carries minutes, name, path and favorite preset",
              QString("%1 %2 %3 %4").arg(cur[0].recentMinutes).arg(cur[0].favoriteSpeed).arg(cur[0].favoriteReverb, cur[0].name));
        check(cur[1].recentMinutes == 17 && cur[1].favoriteSpeed == 0.85 && cur[1].favoriteReverb == "hall",
              "slowmix: second track favorite 0.85x hall");
        check(cur[2].name == "deep cut" && cur[2].favoriteSpeed == 1.6 && cur[2].favoriteReverb == "off",
              "slowmix: nested track favorite 1.6x off", cur[2].name);
    }

    const QList<SlowmixTrack> all = curateSlowmixTracks(clips, playsPath, true, now);
    QStringList allNames;
    for (const SlowmixTrack &t : all)
        allNames << t.relPath;
    check(allNames == QStringList({"song_b.flac", "song_a.wav", "sub/deep_cut.mp3", "stale.wav", "never.wav"}),
          "slowmix: archive adds stale then never-played, junk still out", allNames.join(", "));
    check(curateSlowmixTracks(td.path() + "/nope", playsPath).isEmpty(), "slowmix: missing clips dir -> empty");

    // Dry-run controller: never spawns, still tracks state for the UI.
    SlowmixController c;
    c.setLibrary(clips, playsPath);
    c.refresh();
    check(c.tracks().size() >= 3 && !c.spawnAllowed() && c.available(), "slowmix: dry-run controller lists tracks");
    c.togglePlay();  // nothing selected: plays the top of the rotation, at its favorite preset
    check(c.playing() && c.currentPath() == clips + "/song_b.flac" && c.currentTrack() == "song b"
              && qFuzzyCompare(c.speed(), 1.10) && c.reverb() == "heavy",
          "slowmix: toggle with nothing loaded plays the top track at its favorite preset",
          c.currentPath() + " " + QString::number(c.speed()) + " " + c.reverb());
    c.selectTrack(clips + "/song_a.wav");
    check(c.currentTrack() == "song a" && c.speed() == 0.85 && c.reverb() == "hall" && c.playing(),
          "slowmix: selectTrack applies that track's favorite speed / reverb");
    c.togglePlay();
    c.setSpeed(9.0);
    c.setReverb("cathedral");
    c.setReverb("OFF");
    c.setBass(4);
    c.setLooping(false);
    check(!c.playing() && c.speed() == 2.0 && c.reverb() == "off" && c.bass() == 4 && !c.looping(),
          "slowmix: pause, clamp speed, reject unknown reverb, bass, loop");
    c.setShowArchive(true);
    check(c.tracks().size() == 5, "slowmix: showArchive lists the archive", QString::number(c.tracks().size()));
    check(c.spawnCount() == 0 && !c.running(), "slowmix: dry run never spawns a process");
    SlowmixController live;
    live.setBinary(td.path() + "/no-such-slowmix");
    live.setSocketPath(td.path() + "/none.sock");  // never probe the real ~/.lapis-next daemon
    live.setSpawnAllowed(true);
    check(!live.available(), "slowmix: spawn allowed, no daemon, binary missing -> unavailable");
    check(SlowmixController::defaultSocketPath().endsWith("/slowmix.sock"), "slowmix: default socket path",
          SlowmixController::defaultSocketPath());
}

// A stand-in for `slowmix --socket`: answers `status` with its state, applies play / pause /
// load, and records every line it is sent.
struct MockSlowmixDaemon {
    QLocalServer server;
    QList<QLocalSocket *> clients;
    QStringList lines;
    int connections = 0;
    int hangups = 0;
    bool playing = true;
    QString path;

    bool listen(const QString &sockPath)
    {
        QLocalServer::removeServer(sockPath);
        QObject::connect(&server, &QLocalServer::newConnection, [this] {
            while (QLocalSocket *c = server.nextPendingConnection()) {
                ++connections;
                clients << c;
                QObject::connect(c, &QLocalSocket::disconnected, [this, c] {
                    ++hangups;
                    clients.removeAll(c);
                    c->deleteLater();
                });
                QObject::connect(c, &QLocalSocket::readyRead, [this, c] {
                    while (c->canReadLine()) {
                        const QString l = QString::fromUtf8(c->readLine()).trimmed();
                        lines << l;
                        if (l == "play")
                            playing = true;
                        else if (l == "pause")
                            playing = false;
                        else if (l.startsWith("load "))
                            path = l.mid(5), playing = true;
                        else if (l == "status")
                            c->write(QJsonDocument(QJsonObject{{"playing", playing},
                                                               {"track", QFileInfo(path).fileName()},
                                                               {"path", path},
                                                               {"speed", 0.85},
                                                               {"reverb", "heavy"},
                                                               {"bass", 3.0},
                                                               {"loop", false},
                                                               {"pos", 12.0},
                                                               {"dur", 200.0}})
                                         .toJson(QJsonDocument::Compact)
                                     + '\n');
                    }
                });
            }
        });
        return server.listen(sockPath);
    }
};

template <typename F>
bool pumpUntil(F done, int timeoutMs = 3000)
{
    QElapsedTimer t;
    t.start();
    while (!done() && t.elapsed() < timeoutMs) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return done();
}

void slowmixDaemonTests()
{
    QTemporaryDir td;
    const QString clips = td.path() + "/clips";
    QDir().mkpath(clips);
    writeFile(clips + "/night_drive.wav", "");
    const QString sock = td.path() + "/s.sock";
    MockSlowmixDaemon d;
    d.path = clips + "/night_drive.wav";
    check(d.listen(sock), "slowmix daemon: mock server listens", d.server.errorString());

    // Attach to a daemon that is already playing: the UI syncs to it, nothing is spawned.
    auto *c = new SlowmixController;
    c->setLibrary(clips, td.path() + "/plays.tsv");
    c->setBinary(td.path() + "/no-such-slowmix");
    c->setSocketPath(sock);
    c->setSpawnAllowed(true);
    check(pumpUntil([&] { return c->playing(); }), "slowmix daemon: attach syncs playing on connect");
    check(c->running() && c->available() && c->spawnCount() == 0 && d.connections == 1,
          "slowmix daemon: attached over the socket without spawning, binary missing is fine",
          QString("conns=%1 spawns=%2").arg(d.connections).arg(c->spawnCount()));
    check(c->currentPath() == d.path && c->currentTrack() == "night drive" && c->speed() == 0.85
              && c->reverb() == "heavy" && c->bass() == 3.0 && !c->looping() && c->position() == 12.0
              && c->duration() == 200.0,
          "slowmix daemon: sync fills track, path, speed, reverb, bass, loop, position",
          c->currentPath() + " | " + c->currentTrack() + " | " + QString::number(c->speed()) + " " + c->reverb());

    // Commands reach the daemon over the same connection.
    c->togglePlay();
    check(pumpUntil([&] { return d.lines.contains("pause"); }) && !c->playing() && !d.playing,
          "slowmix daemon: togglePlay sends pause", d.lines.join(", "));

    // The daemon drops us (e.g. it restarted its listener): reconnect and resync.
    QLocalSocket *first = d.clients.value(0);
    if (first)
        first->disconnectFromServer();
    check(pumpUntil([&] { return d.connections == 2 && c->running(); }),
          "slowmix daemon: reconnects after the connection drops", QString::number(d.connections));
    d.playing = true;  // someone pressed play elsewhere while we were away
    check(pumpUntil([&] { return c->playing(); }), "slowmix daemon: polling picks up remote state after reconnect");

    // Destruction hangs up but never stops the music.
    const int hangupsBefore = d.hangups;
    delete c;
    check(pumpUntil([&] { return d.hangups == hangupsBefore + 1; }), "slowmix daemon: destructor closes the socket");
    check(!d.lines.contains("quit") && d.server.isListening() && d.playing,
          "slowmix daemon: destructor never sends quit; server untouched and still playing", d.lines.join(", "));

    // A new app instance attaches to the same daemon and shows the same song.
    SlowmixController again;
    again.setSocketPath(sock);
    again.setBinary(td.path() + "/no-such-slowmix");
    again.setSpawnAllowed(true);
    check(pumpUntil([&] { return again.playing() && again.currentPath() == d.path; }) && d.connections == 3
              && again.spawnCount() == 0,
          "slowmix daemon: a restarted app reattaches and resyncs");

    // No daemon: the first play spawns `slowmix --socket <path>` detached.
    const QString fake = td.path() + "/fake-slowmix";
    const QString argsOut = td.path() + "/args.txt";
    writeFile(fake, QString("#!/bin/sh\necho \"$@\" > '%1'\n").arg(argsOut).toUtf8());
    QFile::setPermissions(fake, QFile::permissions(fake) | QFileDevice::ExeOwner);
    const QString deadSock = td.path() + "/dead/s.sock";
    SlowmixController fresh;
    fresh.setBinary(fake);
    fresh.setSocketPath(deadSock);
    fresh.setSpawnAllowed(true);
    check(fresh.spawnCount() == 0 && !fresh.running(), "slowmix daemon: probing an absent daemon spawns nothing");
    fresh.selectTrack(d.path);
    check(fresh.spawnCount() == 1 && !fresh.running(), "slowmix daemon: first play spawns once (fake never listens)",
          QString::number(fresh.spawnCount()));
    pumpUntil([&] { return QFileInfo(argsOut).size() > 0; });
    QFile af(argsOut);
    const QString args = af.open(QIODevice::ReadOnly) ? QString::fromUtf8(af.readAll()).trimmed() : QString();
    check(args == "--socket " + deadSock, "slowmix daemon: spawned with --socket <path>", args);
}

// --- Cross-session memory ---------------------------------------------------------------------

MemoryTurn turn(const QString &harness, const QString &session, qint64 mtimeMs, const QString &user,
                const QString &assistant, const QString &project = QStringLiteral("proj"))
{
    MemoryTurn t;
    t.harness = harness;
    t.sessionId = session;
    t.project = project;
    t.mtimeMs = mtimeMs;
    t.userText = user;
    t.assistantText = assistant;
    return t;
}

// 500 filler turns over a shared engineering vocabulary (deterministic), none of which mention
// the planted topics below.
QVector<MemoryTurn> syntheticTurns(qint64 now, int n = 500)
{
    static const char *vocab[] = {"build", "deploy", "branch", "rebase", "commit", "review", "config", "parser",
                                  "socket", "timeout", "retry", "cache", "index", "schema", "migration", "router",
                                  "widget", "layout", "shader", "texture", "buffer", "queue", "thread", "mutex",
                                  "logger", "metric", "dashboard", "alert", "budget", "invoice", "calendar", "email",
                                  "docker", "python", "rust", "compile", "linker", "symbol", "profile", "benchmark"};
    const int nv = int(sizeof vocab / sizeof *vocab);
    quint32 seed = 12345;
    auto rnd = [&seed] { seed = seed * 1103515245u + 12345u; return int((seed >> 16) & 0x7fff); };
    QVector<MemoryTurn> out;
    for (int i = 0; i < n; ++i) {
        QStringList u, a;
        for (int w = 0; w < 14; ++w)
            u << QString::fromLatin1(vocab[rnd() % nv]);
        for (int w = 0; w < 60; ++w)
            a << QString::fromLatin1(vocab[rnd() % nv]);
        out << turn(i % 3 == 0 ? "codex" : "claude", QStringLiteral("filler-%1").arg(i / 10), now - qint64(rnd() % 13) * 86400000,
                    u.join(' '), a.join(' '));
    }
    return out;
}

void memoryTests()
{
    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    const qint64 day = 86400000;
    QVector<MemoryTurn> corpus = syntheticTurns(now);
    const int cublas = int(corpus.size());
    corpus << turn("claude", "sess-cublas", now - 3 * day,
                   "Why is the cuBLAS GEMM slower than our sm120 FP4 kernel at 128x256 tiles?",
                   "cuBLAS picks a 64x128 tile for this shape, so it leaves the tensor cores half fed.");
    const int slowmix = int(corpus.size());
    corpus << turn("agy", "sess-slowmix", now - 2 * day, "Set the slowmix reverb preset to cathedral for night drive",
                   "Reverb is now cathedral at 0.85x speed.");
    // Identical text, two sessions: only recency can order them.
    const int oldTwin = int(corpus.size());
    corpus << turn("codex", "sess-twin-old", now - 10 * day, "Rotate the grafana dashboard api token", "Rotated it.");
    const int newTwin = int(corpus.size());
    corpus << turn("codex", "sess-twin-new", now - 1 * day, "Rotate the grafana dashboard api token", "Rotated it.");
    corpus << turn("phonon", "meeting_1700000012345678", now - 1 * day,
                   "[00:01] CALL: should we fund the kernelbench grant\n[00:05] ME: yes, fifty thousand", QString(), "meeting");

    MemoryIndex idx;
    idx.setTurns(corpus);
    check(idx.size() == corpus.size(), "memory: corpus indexed", QString::number(idx.size()));

    QVector<MemoryHit> hits = idx.search("cublas gemm fp4 tiles", QString(), 4, now);
    check(!hits.isEmpty() && hits[0].turn == cublas && hits[0].bm25Rank == 1,
          "memory: exact words recall the planted turn at rank 1",
          hits.isEmpty() ? QString() : QString::number(hits[0].turn));

    // Voice-typed misspellings share no word with the turn: only character 3-grams carry it.
    hits = idx.search("kublass", QString(), 4, now);
    check(!hits.isEmpty() && hits[0].turn == cublas && hits[0].bm25Rank == 0 && hits[0].semanticRank == 1,
          "memory: char 3-grams recall a misspelled term (kublass -> cuBLAS) with no BM25 match",
          hits.isEmpty() ? QStringLiteral("no hits") : QStringLiteral("%1 bm25 %2").arg(hits[0].turn).arg(hits[0].bm25Rank));
    hits = idx.search("slow mix reverb presets", QString(), 4, now);
    check(!hits.isEmpty() && hits[0].turn == slowmix, "memory: a paraphrase (slow mix / presets) recalls the turn");

    hits = idx.search("rotate grafana dashboard token", QString(), 4, now);
    check(hits.size() >= 2 && hits[0].turn == newTwin && hits[1].turn == oldTwin,
          "memory: recency breaks a tie between identical turns (1 day beats 10 days)",
          hits.size() >= 2 ? QStringLiteral("%1, %2").arg(hits[0].turn).arg(hits[1].turn) : QString());

    hits = idx.search("cublas gemm fp4 tiles", "sess-cublas", 4, now);
    check(std::none_of(hits.cbegin(), hits.cend(), [&](const MemoryHit &h) { return h.turn == cublas; }),
          "memory: excludeSessionId drops the active session's turns");

    const QString block = idx.query("cublas gemm fp4 tiles", QString(), 4, 3200, now);
    check(block.startsWith("[claude sess-cub] proj, 3d ago\nUser: Why is the cuBLAS") && block.contains("\nAssistant: cuBLAS picks"),
          "memory: query formats [harness session] project, age / User / Assistant", block.left(120));
    check(block.size() <= 3200, "memory: query stays within 3,200 chars", QString::number(block.size()));
    const QString meet = idx.query("kernelbench grant funding", QString(), 4, 3200, now);
    check(meet.startsWith("[phonon 12345678] meeting, 24h ago\nUser: [00:01] CALL: should we fund"),
          "memory: a Phonon meeting chunk is tagged by its meeting id", meet.left(80));

    // A tight budget with long turns: every block shares it, the total never exceeds it.
    QVector<MemoryTurn> longTurns;
    for (int i = 0; i < 8; ++i)
        longTurns << turn("claude", QStringLiteral("long-%1").arg(i), now - i * day,
                          QString("tensor parallel allreduce bandwidth ").repeated(40),
                          QString("nccl ring allreduce saturates pcie at 24 GB/s ").repeated(60));
    MemoryIndex tight;
    tight.setTurns(longTurns);
    const QString t3200 = tight.query("allreduce bandwidth", QString(), 4, 3200, now);
    const QString t600 = tight.query("allreduce bandwidth", QString(), 4, 600, now);
    check(t3200.size() <= 3200 && t3200.count("\nUser: ") == 4 && t600.size() <= 600 && !t600.isEmpty(),
          "memory: top-4 long turns fit 3,200 and 600 char budgets",
          QStringLiteral("%1 (%2 blocks) / %3").arg(t3200.size()).arg(t3200.count("\nUser: ")).arg(t600.size()));
    check(t3200.count("[claude ") == 4 && !t3200.contains("long-4"), "memory: exactly top-K turns, no neighbours");

    check(idx.query("zzqx", QString(), 4, 3200, now).isEmpty() && idx.query("the and of", QString(), 4, 3200, now).isEmpty()
              && MemoryIndex().query("cublas").isEmpty(),
          "memory: no match, stopwords only, or an empty index yield \"\"");

    // Latency on 500 synthetic turns (plus the planted ones).
    QElapsedTimer timer;
    timer.start();
    const QStringList qs{"cublas gemm fp4 tiles", "kublass", "rotate grafana dashboard token", "docker build cache",
                         "thread mutex queue timeout"};
    for (int r = 0; r < 4; ++r)
        for (const QString &q : qs)
            idx.query(q, QString(), 4, 3200, now);
    const double perQuery = timer.nsecsElapsed() / 1e6 / (4 * qs.size());
    check(perQuery < 50.0, "memory: query latency < 50 ms on 500 turns", QStringLiteral("%1 ms").arg(perQuery, 0, 'f', 2));
    std::printf("memory: %.2f ms per query over %d turns, ~%.1f MB\n", perQuery, idx.size(), idx.memoryBytes() / 1e6);

    // scan(): fixture transcripts for all four harnesses, plus a redacted path and a stale file.
    QTemporaryDir td;
    const QString root = td.path();
    QDir().mkpath(root + "/claude/-Users-me-dev-lapis");
    QDir().mkpath(root + "/claude/-Users-me-dev-infinity");
    QDir().mkpath(root + "/codex/2026/11/18");
    QDir().mkpath(root + "/agy/0f0f0f0f-1111-2222-3333-444444444444/.system_generated/logs");
    QDir().mkpath(root + "/phonon/meeting_1795000000000");
    const QByteArray claude =
        R"({"type":"user","cwd":"/Users/me/dev/lapis","message":{"role":"user","content":"why does the fn key toggle lag?"}})" "\n"
        R"({"type":"assistant","cwd":"/Users/me/dev/lapis","message":{"role":"assistant","content":[{"type":"text","text":"The HID poll runs at 125 Hz."}]}})" "\n"
        R"({"type":"user","message":{"role":"user","content":[{"type":"tool_result","content":"ok"}]}})" "\n"
        R"({"type":"user","message":{"role":"user","content":"and the fix?"}})" "\n"
        R"({"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Poll at 1 kHz."}]}})" "\n"
        R"({"type":"user","message":{"role":"user","content":"unanswered"}})" "\n";
    const QString claudeId = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    check(writeFile(root + "/claude/-Users-me-dev-lapis/" + claudeId + ".jsonl", claude)
              && writeFile(root + "/claude/-Users-me-dev-lapis/11111111-2222-3333-4444-555555555555.jsonl", claude)
              && writeFile(root + "/claude/-Users-me-dev-infinity/99999999-2222-3333-4444-555555555555.jsonl", claude),
          "memory scan: claude fixtures");
    setAgeSeconds(root + "/claude/-Users-me-dev-lapis/11111111-2222-3333-4444-555555555555.jsonl", 20 * 86400);
    const QByteArray codex =
        R"({"type":"session_meta","payload":{"id":"x","cwd":"/Users/me/dev/slowmix"}})" "\n"
        R"({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"# AGENTS.md instructions\n\n<INSTRUCTIONS>\nrules\n</INSTRUCTIONS>"},{"type":"input_text","text":"<environment_context>\n  <cwd>/Users/me</cwd>\n</environment_context>"}]}})" "\n"
        R"({"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"add a cathedral reverb"}]}})" "\n"
        R"({"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Added cathedral."}]}})" "\n";
    const QString codexId = "c0dec0de-0000-4000-8000-000000000001";
    check(writeFile(root + "/codex/2026/11/18/rollout-2026-11-18T10-00-00-" + codexId + ".jsonl", codex), "memory scan: codex fixture");
    check(writeFile(root + "/agy/0f0f0f0f-1111-2222-3333-444444444444/.system_generated/logs/transcript.jsonl",
                    R"({"step_index":0,"type":"USER_INPUT","content":"<USER_REQUEST>\nwhy did 128-reg win?\n</USER_REQUEST>"})" "\n"
                    R"({"step_index":2,"type":"PLANNER_RESPONSE","content":"Registers hold the dequant."})" "\n"),
          "memory scan: agy fixture");
    check(writeFile(root + "/phonon/meeting_1795000000000/transcript.txt", "[00:01] CALL: hello\n[00:02] ME: hi\n"),
          "memory scan: phonon fixture");

    MemorySources src;
    src.claudeProjects = root + "/claude";
    src.codexSessions = root + "/codex";
    src.agyBrain = root + "/agy";
    src.phononMeetings = root + "/phonon";
    src.excludePathParts = {"infinity"};
    MemoryCache cache;
    const QVector<MemoryTurn> scanned = MemoryIndex::scan(src, now, &cache);
    auto count = [&](const QString &h) {
        return std::count_if(scanned.cbegin(), scanned.cend(), [&](const MemoryTurn &t) { return t.harness == h; });
    };
    const auto claudeTurn = std::find_if(scanned.cbegin(), scanned.cend(), [](const MemoryTurn &t) { return t.harness == "claude"; });
    check(count("claude") == 2 && count("codex") == 1 && count("agy") == 1 && count("phonon") == 1,
          "memory scan: 2 claude + 1 codex + 1 agy + 1 phonon (stale, redacted, unanswered and AGENTS.md skipped)",
          QStringLiteral("%1/%2/%3/%4").arg(count("claude")).arg(count("codex")).arg(count("agy")).arg(count("phonon")));
    check(claudeTurn != scanned.cend() && claudeTurn->sessionId == claudeId && claudeTurn->project == "lapis"
              && claudeTurn->userText == "why does the fn key toggle lag?" && claudeTurn->assistantText == "The HID poll runs at 125 Hz.",
          "memory scan: claude turn keeps session uuid, cwd project and both sides");
    check(std::any_of(scanned.cbegin(), scanned.cend(),
                      [&](const MemoryTurn &t) { return t.sessionId == codexId && t.project == "slowmix" && t.userText == "add a cathedral reverb"; }),
          "memory scan: codex session id is the rollout uuid, project from session_meta cwd");
    check(std::none_of(scanned.cbegin(), scanned.cend(), [](const MemoryTurn &t) { return t.sessionId.startsWith("99999999"); }),
          "memory scan: an excluded (infinity) path is never read");
    src.excludeSessionIds = {claudeId};
    const QVector<MemoryTurn> again = MemoryIndex::scan(src, now, &cache);
    check(again.size() == scanned.size() - 2 && cache.size() >= 3, "memory scan: excludeSessionIds + cached rescan",
          QStringLiteral("%1 vs %2").arg(again.size()).arg(scanned.size()));
    src.maxTurns = 2;
    check(MemoryIndex::scan(src, now, nullptr).size() == 2, "memory scan: maxTurns caps the corpus");
}

// --- Window state -----------------------------------------------------------------------------

void windowStateTests()
{
    WindowState s;
    s.displayUuid = "00000000-0000-4000-8000-0000000D15A1";
    s.relX = 120;
    s.relY = -4;
    s.width = 1680;
    s.height = 1050;
    s.fullscreen = true;
    const QJsonObject o = s.toJson();
    check(WindowState::fromJson(o) == s && o.value("displayUuid").toString() == s.displayUuid && o.value("fullscreen").toBool(),
          "window: JSON round trip");
    const QJsonObject parsed = QJsonDocument::fromJson(QJsonDocument(QJsonObject{{"windowState", o}}).toJson())
                                   .object().value("windowState").toObject();
    check(WindowState::fromJson(parsed) == s, "window: round trip through serialized prefs text");
    const WindowState bad = WindowState::fromJson(QJsonObject{{"width", -5}, {"height", "x"}, {"maximized", true}});
    check(!bad.hasGeometry() && bad.maximized && !bad.fullscreen && bad.displayUuid.isEmpty(),
          "window: malformed fields read as defaults");

    const QRect avail(0, 25, 1512, 920);  // a 14" MacBook below the menu bar
    check(clampWindowRect(QRect(100, 100, 1200, 800), avail) == QRect(100, 100, 1200, 800), "window: clamp keeps a fitting rect");
    check(clampWindowRect(QRect(3000, 2000, 1200, 800), avail) == QRect(312, 145, 1200, 800),
          "window: clamp pulls an off-screen rect back inside", "");
    check(clampWindowRect(QRect(-400, -300, 1200, 800), avail) == QRect(0, 25, 1200, 800), "window: clamp from top-left");
    check(clampWindowRect(QRect(0, 0, 2560, 1600), avail) == QRect(0, 25, 1512, 920),
          "window: a rect larger than the screen shrinks to it");
    check(clampWindowRect(QRect(50, 60, 400, 300), avail) == QRect(50, 60, 1100, 700), "window: minimum size enforced");
    const QRect external(1512, -200, 2560, 1415);  // an external display right of and above the laptop
    check(clampWindowRect(QRect(1512 + 2400, 900, 1680, 1050), external) == QRect(1512 + 2560 - 1680, 165, 1680, 1050),
          "window: clamp on a secondary display with a negative origin");
}

}  // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    coreTests();
    phononTests();
    slowmixTests();
    slowmixDaemonTests();
    memoryTests();
    windowStateTests();
    std::printf("test_core_ingest: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
