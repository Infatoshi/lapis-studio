// test_core_ingest: lapis_core (TargetRef, SpscRing) and lapis_ingest (Phonon snapshot/tail).
#include "core/ring_buffer.hpp"
#include "core/types.hpp"
#include "ingest/phonon.hpp"
#include "ingest/slowmix.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QTemporaryDir>

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
    live.setSpawnAllowed(true);
    check(!live.available(), "slowmix: spawn allowed but binary missing -> unavailable");
}

}  // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    coreTests();
    phononTests();
    slowmixTests();
    std::printf("test_core_ingest: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
