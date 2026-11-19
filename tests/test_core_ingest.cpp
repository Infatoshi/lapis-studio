// test_core_ingest: lapis_core (TargetRef, SpscRing) and lapis_ingest (Phonon snapshot/tail).
#include "core/ring_buffer.hpp"
#include "core/types.hpp"
#include "ingest/phonon.hpp"

#include <QCoreApplication>
#include <QDateTime>
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

}  // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    coreTests();
    phononTests();
    std::printf("test_core_ingest: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
