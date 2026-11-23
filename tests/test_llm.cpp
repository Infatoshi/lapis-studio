// test_llm: lapis_llm (4-tier APC prefix builder, SSE parser, libcurl-multi LlmRouter).
// The router tests run against a local POSIX-socket HTTP/1.1 SSE server on its own thread that
// holds every connection open until all N have connected (a barrier), so they only pass when the
// router really keeps N streams in flight at once (QNetworkAccessManager stops at 6 per host).
#include "llm/jarvis_legs.hpp"
#include "llm/prefix.hpp"
#include "llm/router.hpp"
#include "llm/sse.hpp"

#include <QCoreApplication>
#include <QDateTime>
#include <QElapsedTimer>
#include <QFile>
#include <QHash>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryDir>

#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

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

bool waitUntil(const std::function<bool()> &pred, int timeoutMs)
{
    QElapsedTimer t;
    t.start();
    while (!pred()) {
        if (t.elapsed() > timeoutMs)
            return false;
        QCoreApplication::processEvents(QEventLoop::AllEvents, 5);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

// --- a tiny HTTP/1.1 SSE server ------------------------------------------------------------

class MiniServer {
public:
    struct Config {
        int conns = 1;             // connections to accept
        int barrier = 1;           // hold every response until this many are connected
        int rounds = 3;            // round 0: reasoning delta; rounds 1..n-1: content deltas
        int gapMs = 10;            // between rounds
        int splitDelayMs = 0;      // > 0: each SSE line is written in two halves, this far apart
        int barrierTimeoutMs = 5000;
        int status = 200;
    };

    ~MiniServer()
    {
        join();
        if (m_listen >= 0)
            ::close(m_listen);
    }

    bool listen()
    {
        m_listen = ::socket(AF_INET, SOCK_STREAM, 0);
        if (m_listen < 0)
            return false;
        int one = 1;
        ::setsockopt(m_listen, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0;
        if (::bind(m_listen, reinterpret_cast<sockaddr *>(&addr), sizeof addr) != 0 || ::listen(m_listen, 64) != 0)
            return false;
        socklen_t len = sizeof addr;
        ::getsockname(m_listen, reinterpret_cast<sockaddr *>(&addr), &len);
        m_port = ntohs(addr.sin_port);
        return true;
    }
    QString url() const { return QStringLiteral("http://127.0.0.1:%1/v1/chat/completions").arg(m_port); }

    void start(const Config &c)
    {
        m_cfg = c;
        m_thread = std::thread([this] { run(); });
    }
    void join()
    {
        if (m_thread.joinable())
            m_thread.join();
    }

    std::atomic<int> accepted{0};
    std::atomic<int> peakOpen{0};
    std::atomic<int> closedEarly{0};
    std::atomic<bool> barrierReached{false};
    std::atomic<qint64> barrierMs{-1};
    QStringList bodies() const
    {
        std::lock_guard<std::mutex> l(m_mutex);
        return m_bodies;
    }

private:
    struct Conn {
        int fd = -1;
        QByteArray tag;
        bool alive = true;
    };

    static QByteArray readRequest(int fd)
    {
        QByteArray buf;
        char tmp[4096];
        qsizetype headerEnd = -1;
        qsizetype contentLength = 0;
        while (true) {
            if (headerEnd < 0) {
                headerEnd = buf.indexOf("\r\n\r\n");
                if (headerEnd >= 0) {
                    const QByteArray head = buf.left(headerEnd).toLower();
                    const qsizetype cl = head.indexOf("content-length:");
                    if (cl >= 0) {
                        const qsizetype eol = head.indexOf("\r\n", cl);
                        contentLength = head.mid(cl + 15, eol < 0 ? -1 : eol - cl - 15).trimmed().toLongLong();
                    }
                }
            }
            if (headerEnd >= 0 && buf.size() >= headerEnd + 4 + contentLength)
                return buf.mid(headerEnd + 4, contentLength);
            const ssize_t n = ::recv(fd, tmp, sizeof tmp, 0);
            if (n <= 0)
                return {};
            buf.append(tmp, qsizetype(n));
        }
    }

    static bool sendAll(int fd, const QByteArray &data)
    {
        qsizetype off = 0;
        while (off < data.size()) {
            const ssize_t n = ::send(fd, data.constData() + off, size_t(data.size() - off), 0);
            if (n <= 0)
                return false;
            off += n;
        }
        return true;
    }

    static bool peerClosed(int fd)
    {
        char c;
        const ssize_t n = ::recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
        if (n == 0)
            return true;
        return n < 0 && errno != EAGAIN && errno != EWOULDBLOCK;
    }

    bool sendEvent(Conn &c, const QByteArray &line)
    {
        // One SSE event per HTTP chunk; optionally split mid-line to exercise reassembly.
        const QByteArray ev = line + "\n\n";
        if (m_cfg.splitDelayMs <= 0) {
            return sendAll(c.fd, QByteArray::number(ev.size(), 16) + "\r\n" + ev + "\r\n");
        }
        const qsizetype half = ev.size() / 2;
        const QByteArray a = ev.left(half), b = ev.mid(half);
        if (!sendAll(c.fd, QByteArray::number(a.size(), 16) + "\r\n" + a + "\r\n"))
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(m_cfg.splitDelayMs));
        return sendAll(c.fd, QByteArray::number(b.size(), 16) + "\r\n" + b + "\r\n");
    }

    void closeConn(Conn &c, int &open)
    {
        if (c.fd >= 0) {
            ::close(c.fd);
            c.fd = -1;
            --open;
        }
        c.alive = false;
    }

    void run()
    {
        QElapsedTimer clock;
        clock.start();
        std::vector<Conn> conns;
        int open = 0;
        while (int(conns.size()) < m_cfg.conns) {
            const int remaining = int(m_cfg.barrierTimeoutMs - clock.elapsed());
            if (remaining <= 0)
                break;
            pollfd p {m_listen, POLLIN, 0};
            if (::poll(&p, 1, remaining) <= 0)
                break;
            const int fd = ::accept(m_listen, nullptr, nullptr);
            if (fd < 0)
                continue;
#ifdef SO_NOSIGPIPE
            int one = 1;
            ::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof one);
#endif
            timeval tv {2, 0};
            ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
            const QByteArray body = readRequest(fd);
            Conn c;
            c.fd = fd;
            const qsizetype t = body.indexOf("\"tag\":\"");
            c.tag = t >= 0 ? body.mid(t + 7, body.indexOf('"', t + 7) - t - 7) : QByteArray("?");
            {
                std::lock_guard<std::mutex> l(m_mutex);
                m_bodies << QString::fromUtf8(body);
            }
            conns.push_back(c);
            ++open;
            accepted.store(int(conns.size()));
            peakOpen.store(std::max(peakOpen.load(), open));
            if (int(conns.size()) >= m_cfg.barrier && !barrierReached.load()) {
                barrierReached.store(true);
                barrierMs.store(clock.elapsed());
            }
        }

        if (m_cfg.status != 200) {
            const QByteArray body = R"({"error":{"message":"model overloaded"}})";
            for (Conn &c : conns) {
                sendAll(c.fd, "HTTP/1.1 " + QByteArray::number(m_cfg.status) + " Internal Server Error\r\n"
                              "Content-Type: application/json\r\nConnection: close\r\nContent-Length: "
                              + QByteArray::number(body.size()) + "\r\n\r\n" + body);
                closeConn(c, open);
            }
            return;
        }
        for (Conn &c : conns)
            if (!sendAll(c.fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n"
                               "Connection: close\r\n\r\n"))
                closeConn(c, open);
        for (int r = 0; r < m_cfg.rounds; ++r) {
            if (r > 0 && m_cfg.gapMs > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(m_cfg.gapMs));
            for (Conn &c : conns) {
                if (!c.alive)
                    continue;
                if (peerClosed(c.fd)) {
                    closedEarly.fetch_add(1);
                    closeConn(c, open);
                    continue;
                }
                const QByteArray line = r == 0
                    ? "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"think-" + c.tag + "\"}}]}"
                    : "data: {\"choices\":[{\"delta\":{\"content\":\"" + c.tag + "." + QByteArray::number(r) + " \"}}]}";
                if (!sendEvent(c, line)) {
                    closedEarly.fetch_add(1);
                    closeConn(c, open);
                }
            }
        }
        for (Conn &c : conns) {
            if (!c.alive)
                continue;
            sendEvent(c, "data: [DONE]");
            sendAll(c.fd, "0\r\n\r\n");
            closeConn(c, open);
        }
    }

    Config m_cfg;
    int m_listen = -1;
    quint16 m_port = 0;
    std::thread m_thread;
    mutable std::mutex m_mutex;
    QStringList m_bodies;
};

// Collects everything a router emits.
struct Sink {
    QHash<QString, QString> text, thinking, error;
    QHash<QString, int> deltas, finishes;
    QHash<QString, bool> warm;
    void attach(LlmRouter &r)
    {
        QObject::connect(&r, &LlmRouter::streamDelta, [this](const QString &id, const QString &t, const QString &th) {
            text[id] += t;
            thinking[id] += th;
            deltas[id] += 1;
        });
        QObject::connect(&r, &LlmRouter::streamFinished, [this](const QString &id, const QString &e) {
            error[id] = e;
            finishes[id] += 1;
        });
        QObject::connect(&r, &LlmRouter::warmFinished, [this](const QString &k, bool ok) { warm[k] = ok; });
    }
};

QByteArray tagPayload(const QString &tag)
{
    return QByteArray("{\"tag\":\"") + tag.toUtf8() + "\",\"stream\":true}";
}

LlmStreamRequest req(const QString &id, const QString &url, const QString &tag)
{
    LlmStreamRequest r;
    r.streamId = id;
    r.endpoint = url;
    r.payload = tagPayload(tag);
    r.timeoutMs = 20000;
    return r;
}

// --- prefix builder ------------------------------------------------------------------------

QByteArray compact(const QJsonValue &v)
{
    return v.isObject() ? QJsonDocument(v.toObject()).toJson(QJsonDocument::Compact)
                        : QJsonDocument(v.toArray()).toJson(QJsonDocument::Compact);
}

void prefixTests()
{
    TieredPrompt p;
    p.model = QStringLiteral("qwen3.8-flash-next");
    p.effort = QStringLiteral("low");
    p.systemPrompt = kCompatSystemPrompt;
    p.trunk = QStringLiteral("Project: lapis (~/dev/lapis).\nLast 2 turns:\nUser: why\nAssistant: KV");
    p.history = {ChatTurn{QStringLiteral("user"), QStringLiteral("what is the tok/s?")},
                 ChatTurn{QStringLiteral("assistant"), QStringLiteral("126 tok/s at c1.")}};
    p.targetSlice = QStringLiteral("On screen: chart \"decode tok/s\" BLOCK_M=128");
    p.recentActions = {QStringLiteral("opened pill BLOCK_M=128"), QStringLiteral("scrolled diff src/tile.cu")};
    p.phononTail = QStringLiteral("0:12 You: is MTP on?\n0:15 Call: yes");
    p.question = QStringLiteral("why did it drop?");

    const QJsonObject ask = QJsonDocument::fromJson(buildTieredChatRequest(p)).object();
    const QJsonObject warm = QJsonDocument::fromJson(buildTieredWarmRequest(p)).object();
    const QJsonArray am = ask.value("messages").toArray();
    const QJsonArray wm = warm.value("messages").toArray();
    check(am.size() == 5 && wm.size() == 4, "prefix: conversation ask = 4 prefix msgs + leaf; warm = 4 prefix msgs",
          QStringLiteral("%1/%2").arg(am.size()).arg(wm.size()));
    bool identical = wm.size() <= am.size();
    for (qsizetype i = 0; identical && i < wm.size(); ++i)
        identical = compact(wm[i]) == compact(am[i]);
    check(identical, "prefix: Tier 0+1+2 messages byte-identical between warm and ask");
    // Raw-bytes check: the warm's serialized messages (minus the closing ']') are a prefix of the ask's.
    QByteArray wBytes = compact(wm);
    wBytes.chop(1);
    check(compact(am).startsWith(wBytes), "prefix: warm messages are a raw byte prefix of the ask's messages");
    check(am[0].toObject().value("role") == "system"
              && am[0].toObject().value("content") == tier0SystemText(kCompatSystemPrompt, p.model, QString()),
          "prefix: Tier 0 is the static system prompt plus the model identity");
    check(am[1].toObject().value("content").toString() == "Pinned context:\n" + p.trunk, "prefix: Tier 1 is the trunk");
    check(am[2].toObject().value("role") == "user" && am[3].toObject().value("role") == "assistant",
          "prefix: Tier 2 history in order");

    const QString leaf = am.last().toObject().value("content").toString();
    check(am.last().toObject().value("role") == "user" && leaf.endsWith("Question: why did it drop?")
              && leaf.contains("opened pill BLOCK_M=128") && leaf.contains("0:15 Call: yes")
              && leaf.contains("decode tok/s"),
          "prefix: Tier 3 leaf is last and carries slice, actions, phonon and question", leaf);
    bool volatileLeaks = false;
    for (qsizetype i = 0; i + 1 < am.size(); ++i) {
        const QString c = am[i].toObject().value("content").toString();
        volatileLeaks = volatileLeaks || c.contains("opened pill") || c.contains("0:15 Call") || c.contains("why did it drop")
            || c.contains("decode tok/s");
    }
    check(!volatileLeaks, "prefix: no volatile content before Tier 3");

    // Volatile changes never touch the cached prefix.
    TieredPrompt p2 = p;
    p2.recentActions << QStringLiteral("zoomed chart");
    p2.phononTail += QStringLiteral("\n0:20 You: next");
    p2.targetSlice = QStringLiteral("On screen: table \"arms\"");
    p2.question = QStringLiteral("and now?");
    const QJsonArray am2 = QJsonDocument::fromJson(buildTieredChatRequest(p2)).object().value("messages").toArray();
    bool samePrefix = am2.size() == am.size();
    for (qsizetype i = 0; samePrefix && i + 1 < am.size(); ++i)
        samePrefix = compact(am[i]) == compact(am2[i]);
    check(samePrefix && compact(am.last()) != compact(am2.last()), "prefix: Tier 3 changes leave Tiers 0-2 byte-identical");

    // Hover ask: Tier 2 empty.
    TieredPrompt h = p;
    h.history.clear();
    const QJsonArray hm = QJsonDocument::fromJson(buildTieredChatRequest(h)).object().value("messages").toArray();
    check(hm.size() == 3 && compact(hm[0]) == compact(am[0]) && compact(hm[1]) == compact(am[1]),
          "prefix: hover ask shares Tiers 0+1 with the conversation");

    check(ask.value("stream").toBool() && ask.value("reasoning_effort") == "low" && !ask.contains("max_tokens"),
          "prefix: chat request streams with effort");
    check(warm.value("max_tokens").toInt() == 1 && warm.contains("stream") && !warm.value("stream").toBool(true)
              && !warm.contains("reasoning_effort") && warm.value("model") == p.model,
          "prefix: warm is max_tokens 1, stream false, no effort, same model");

    // Leaf with only a question, and the compatibility wrappers.
    TieredPrompt q;
    q.question = QStringLiteral("x?");
    check(tieredLeafText(q) == "Question: x?", "prefix: question-only leaf is exactly 'Question: <q>'");
    const QJsonObject c = QJsonDocument::fromJson(buildOpenAiCompatRequest(
        QStringLiteral("m"), QStringLiteral("SYS"), QStringLiteral("CTX"), QStringLiteral("What changed?"), QStringLiteral("low"))).object();
    const QJsonArray cm = c.value("messages").toArray();
    check(cm.size() == 3 && cm[1].toObject().value("content") == "Pinned context:\nCTX"
              && cm[2].toObject().value("content") == "Question: What changed?",
          "prefix: buildOpenAiCompatRequest wrapper unchanged");
    const QJsonArray cw = QJsonDocument::fromJson(buildOpenAiWarmRequest(QStringLiteral("m"), QStringLiteral("SYS"), QStringLiteral("CTX")))
                              .object().value("messages").toArray();
    check(cw.size() == 2 && compact(cw[0]) == compact(cm[0]) && compact(cw[1]) == compact(cm[1]),
          "prefix: buildOpenAiWarmRequest wrapper is a prefix of the compat ask");
}

// --- Tier 0 identity, local studio context, compound action parser ------------------------

void tier0Tests()
{
    TieredPrompt p;
    p.model = QStringLiteral("qwen3.8-flash-next");
    p.systemPrompt = kCompatSystemPrompt;
    p.extraSystemContext = QStringLiteral("Projects: lapis, phonon, slowmix.");
    p.trunk = QStringLiteral("Project: lapis");
    p.history = {ChatTurn{QStringLiteral("user"), QStringLiteral("hi")}};
    p.targetSlice = QStringLiteral("chart decode tok/s");
    p.question = QStringLiteral("who are you?");

    const QJsonArray am = QJsonDocument::fromJson(buildTieredChatRequest(p)).object().value("messages").toArray();
    const QJsonArray wm = QJsonDocument::fromJson(buildTieredWarmRequest(p)).object().value("messages").toArray();
    const QString sys = am[0].toObject().value("content").toString();
    check(sys.startsWith(kCompatSystemPrompt + "\n\nYour model ID is qwen3.8-flash-next (running locally via "
                                               "OpenAI-compatible vLLM)."),
          "tier0: identity follows the static system prompt", sys);
    check(sys.contains("never claim to be Claude or any other model") && sys.contains("compound [[ACTION:...]] tags"),
          "tier0: identity forbids claiming Claude and asks for compound actions");
    check(sys.endsWith("\n\nLocal studio context:\nProjects: lapis, phonon, slowmix."),
          "tier0: extraSystemContext is the Tier 0 tail", sys);
    bool same = wm.size() == 3 && am.size() == 4;
    for (qsizetype i = 0; same && i < wm.size(); ++i)
        same = compact(wm[i]) == compact(am[i]);
    QByteArray wBytes = compact(wm);
    wBytes.chop(1);
    check(same && compact(am).startsWith(wBytes), "tier0: identity + extra context byte-identical in warm and ask");

    TieredPrompt anon = p;
    anon.model.clear();
    anon.extraSystemContext.clear();
    check(QJsonDocument::fromJson(buildTieredChatRequest(anon)).object().value("messages").toArray()[0]
              .toObject().value("content") == kCompatSystemPrompt,
          "tier0: no model and no extra context -> static prompt only");
    check(tier0SystemText(QString(), QString(), QString()).isEmpty()
              && tier0SystemText(QString(), QString(), "X") == "Local studio context:\nX",
          "tier0: empty parts are skipped");

    // loadLocalStudioContext: missing, small, and capped at 4 KB without splitting UTF-8.
    QTemporaryDir td;
    check(loadLocalStudioContext(td.path() + "/none.md").isEmpty(), "context: missing file -> empty");
    {
        QFile f(td.path() + "/small.md");
        if (f.open(QIODevice::WriteOnly))
            f.write("  # Studio\nlapis is the main project.\n\n");
    }
    check(loadLocalStudioContext(td.path() + "/small.md") == "# Studio\nlapis is the main project.",
          "context: small file read and trimmed");
    {
        QByteArray big(4095, 'a');
        big += QStringLiteral("ééé").toUtf8();  // first 2-byte char straddles the 4096 cap
        big += QByteArray(10000, 'b');
        QFile f(td.path() + "/big.md");
        if (f.open(QIODevice::WriteOnly))
            f.write(big);
        QByteArray exact(4094, 'a');
        exact += QStringLiteral("é").toUtf8();  // complete char ends exactly at the cap: kept
        exact += QByteArray(100, 'b');
        QFile g(td.path() + "/exact.md");
        if (g.open(QIODevice::WriteOnly))
            g.write(exact);
    }
    const QString capped = loadLocalStudioContext(td.path() + "/big.md");
    check(capped.size() == 4095 && !capped.contains(QChar::ReplacementCharacter),
          "context: capped at 4 KB, straddling UTF-8 char dropped", QString::number(capped.size()));
    const QString exact = loadLocalStudioContext(td.path() + "/exact.md");
    check(exact.size() == 4095 && exact.endsWith(QChar(0xE9)), "context: complete UTF-8 char at the cap kept",
          QString::number(exact.size()));

    // parseStudioModelOutput.
    const ParsedModelOutput ig = parseStudioModelOutput(QStringLiteral("[[IGNORE]]"));
    check(ig.ignore && ig.cleanText.isEmpty() && ig.actions.isEmpty(), "parse: [[IGNORE]]");
    const ParsedModelOutput ig2 = parseStudioModelOutput(QStringLiteral("  just chatting [[IGNORE]] \n"));
    check(ig2.ignore && ig2.cleanText.isEmpty(), "parse: [[IGNORE]] anywhere wins");

    const ParsedModelOutput c = parseStudioModelOutput(QStringLiteral(
        "[[ACTION:focus_project:lapis]] [[ACTION:play_track:Nights (slowed)]]\n"
        "Opening lapis and starting music. [[ACTION:spawn_agent:profile the KV cache: c1 vs c8]]"));
    check(!c.ignore && c.actions.size() == 3, "parse: 3-action compound batch", QString::number(c.actions.size()));
    if (c.actions.size() == 3) {
        check(c.actions[0].name == "focus_project" && c.actions[0].arg == "lapis"
                  && c.actions[1].name == "play_track" && c.actions[1].arg == "Nights (slowed)"
                  && c.actions[2].name == "spawn_agent" && c.actions[2].arg == "profile the KV cache: c1 vs c8",
              "parse: actions in order, args kept (colons inside arg allowed)");
    }
    check(c.cleanText == "Opening lapis and starting music.", "parse: tags stripped from cleanText", c.cleanText);

    const ParsedModelOutput na = parseStudioModelOutput(QStringLiteral("Going home [[ACTION:go_home]] now."));
    check(na.actions.size() == 1 && na.actions[0].name == "go_home" && na.actions[0].arg.isEmpty()
              && na.cleanText == "Going home now.",
          "parse: no-arg action stripped mid-sentence", na.cleanText);
    const ParsedModelOutput plain = parseStudioModelOutput(QStringLiteral("  It is 126 tok/s.\n    indented  code\n"));
    check(!plain.ignore && plain.actions.isEmpty() && plain.cleanText == "It is 126 tok/s.\n    indented  code",
          "parse: plain text untouched apart from outer trim", plain.cleanText);
}

void sseTests()
{
    const OpenAiSseEvent c = parseOpenAiSseLine(R"(data: {"choices":[{"delta":{"content":"126 tok/s"}}]})");
    check(c.kind == OpenAiSseEvent::Delta && c.text == "126 tok/s" && c.thinking.isEmpty(), "sse: content");
    const OpenAiSseEvent rc = parseOpenAiSseLine(R"(data: {"choices":[{"delta":{"reasoning_content":"checking KV","content":""}}]})");
    check(rc.kind == OpenAiSseEvent::Delta && rc.thinking == "checking KV" && rc.text.isEmpty(), "sse: reasoning_content");
    const OpenAiSseEvent r = parseOpenAiSseLine(R"(data:{"choices":[{"delta":{"reasoning":"hmm"}}]})");
    check(r.kind == OpenAiSseEvent::Delta && r.thinking == "hmm", "sse: reasoning (no space after data:)");
    check(parseOpenAiSseLine("data: [DONE]").kind == OpenAiSseEvent::Done, "sse: [DONE]");
    check(parseOpenAiSseLine("data: [DONE]\r").kind == OpenAiSseEvent::Done, "sse: [DONE] with CR");
    const OpenAiSseEvent e = parseOpenAiSseLine(R"(data: {"error":{"message":"model overloaded"}})");
    check(e.kind == OpenAiSseEvent::Error && e.error == "model overloaded", "sse: error");
    check(parseOpenAiSseLine(": keep-alive").kind == OpenAiSseEvent::None
              && parseOpenAiSseLine("").kind == OpenAiSseEvent::None
              && parseOpenAiSseLine(R"(data: {"choices":[{"delta":{}}]})").kind == OpenAiSseEvent::None,
          "sse: comments, blanks and empty deltas are None");
}

// --- router --------------------------------------------------------------------------------

void concurrencyTest(int n)
{
    MiniServer srv;
    if (!srv.listen()) {
        check(false, "router: server listen");
        return;
    }
    MiniServer::Config cfg;
    cfg.conns = n;
    cfg.barrier = n;
    cfg.rounds = 3;
    cfg.gapMs = 10;
    cfg.barrierTimeoutMs = 5000;
    srv.start(cfg);

    LlmRouter router;
    Sink sink;
    sink.attach(router);
    QElapsedTimer t;
    t.start();
    for (int i = 0; i < n; ++i)
        router.startStream(req(QStringLiteral("s%1").arg(i), srv.url(), QStringLiteral("t%1").arg(i)));
    const bool allDone = waitUntil([&] { return sink.finishes.size() == n; }, 10000);
    srv.join();
    const qint64 ms = t.elapsed();

    const QByteArray label = QByteArray::number(n);
    check(allDone, ("router: all " + label + " concurrent streams finished").constData(),
          QStringLiteral("%1 of %2 in %3 ms").arg(sink.finishes.size()).arg(n).arg(ms));
    check(srv.barrierReached.load() && srv.peakOpen.load() == n,
          ("router: server saw " + label + " simultaneous open connections (barrier met)").constData(),
          QStringLiteral("peak %1, barrier at %2 ms").arg(srv.peakOpen.load()).arg(srv.barrierMs.load()));
    check(srv.barrierMs.load() >= 0 && srv.barrierMs.load() < 2000,
          ("router: barrier of " + label + " met fast, not by timeout").constData(),
          QString::number(srv.barrierMs.load()));
    bool content = true;
    QString bad;
    for (int i = 0; i < n; ++i) {
        const QString id = QStringLiteral("s%1").arg(i);
        const QString tag = QStringLiteral("t%1").arg(i);
        const bool ok = sink.error.value(id).isEmpty() && sink.text.value(id) == tag + ".1 " + tag + ".2 "
            && sink.thinking.value(id) == "think-" + tag && sink.finishes.value(id) == 1;
        if (!ok && bad.isEmpty())
            bad = id + ": text='" + sink.text.value(id) + "' think='" + sink.thinking.value(id) + "' err='"
                + sink.error.value(id) + "'";
        content = content && ok;
    }
    check(content, ("router: each of " + label + " streams got exactly its own deltas, clean finish").constData(), bad);
    check(router.peakConcurrentStreams() >= n, ("router: >= " + label + " streams receiving at once").constData(),
          QString::number(router.peakConcurrentStreams()));
    check(router.activeStreams() == 0 && waitUntil([&] { return router.handlesInMulti() == 0; }, 1000),
          "router: no streams or handles left after finish");
    std::printf("  %d concurrent streams: barrier met at %lld ms, all done in %lld ms\n", n,
                static_cast<long long>(srv.barrierMs.load()), static_cast<long long>(ms));
}

void cancelTest()
{
    MiniServer srv;
    if (!srv.listen()) {
        check(false, "cancel: server listen");
        return;
    }
    MiniServer::Config cfg;
    cfg.conns = 4;
    cfg.barrier = 4;
    cfg.rounds = 8;
    cfg.gapMs = 40;
    cfg.splitDelayMs = 3;  // also proves SSE lines split across TCP reads reassemble
    srv.start(cfg);

    LlmRouter router;
    Sink sink;
    sink.attach(router);
    for (int i = 0; i < 4; ++i)
        router.startStream(req(QStringLiteral("c%1").arg(i), srv.url(), QStringLiteral("c%1").arg(i)));
    const bool firstDelta = waitUntil([&] { return sink.deltas.value("c0") > 0; }, 5000);
    check(firstDelta, "cancel: c0 started streaming before the cancel");
    router.cancelStream(QStringLiteral("c0"));
    const int deltasAtCancel = sink.deltas.value("c0");
    check(!router.isActive("c0") && router.activeStreams() == 3, "cancel: c0 inactive at once, 3 siblings active");
    const bool siblings = waitUntil([&] { return sink.finishes.size() == 3; }, 8000);
    srv.join();
    waitUntil([] { return false; }, 50);  // drain any straggling events

    check(siblings, "cancel: the 3 sibling streams finished");
    bool clean = true;
    for (int i = 1; i < 4; ++i) {
        const QString id = QStringLiteral("c%1").arg(i);
        QString want;
        for (int r = 1; r < 8; ++r)
            want += id + "." + QString::number(r) + " ";
        clean = clean && sink.error.value(id).isEmpty() && sink.text.value(id) == want && sink.finishes.value(id) == 1;
    }
    check(clean, "cancel: siblings got every delta and a clean finish",
          sink.text.value("c1") + " | " + sink.error.value("c1"));
    check(!sink.finishes.contains("c0") && sink.deltas.value("c0") == deltasAtCancel,
          "cancel: no delta or finish for c0 after cancel",
          QStringLiteral("deltas %1 -> %2").arg(deltasAtCancel).arg(sink.deltas.value("c0")));
    check(srv.closedEarly.load() == 1, "cancel: the server saw exactly c0's socket close early",
          QString::number(srv.closedEarly.load()));
}

void restartTest()
{
    MiniServer srv;
    srv.listen();
    MiniServer::Config cfg;
    cfg.conns = 2;
    cfg.barrier = 1;
    cfg.rounds = 2;
    cfg.barrierTimeoutMs = 400;  // the replaced request may never connect
    srv.start(cfg);
    LlmRouter router;
    Sink sink;
    sink.attach(router);
    router.startStream(req(QStringLiteral("x"), srv.url(), QStringLiteral("old")));
    router.startStream(req(QStringLiteral("x"), srv.url(), QStringLiteral("new")));
    const bool done = waitUntil([&] { return sink.finishes.value("x") >= 1; }, 5000);
    srv.join();
    waitUntil([] { return false; }, 50);
    check(done && sink.finishes.value("x") == 1 && sink.text.value("x") == "new.1 " && sink.error.value("x").isEmpty(),
          "restart: same streamId replaces the old request; only the new one reports",
          sink.text.value("x") + " / " + sink.error.value("x"));
}

void warmAndErrorTests()
{
    MiniServer srv;
    srv.listen();
    MiniServer::Config cfg;
    cfg.rounds = 1;
    srv.start(cfg);
    LlmRouter router;
    Sink sink;
    sink.attach(router);
    const QByteArray payload = R"({"model":"m","max_tokens":1,"stream":false,"tag":"w"})";
    check(router.warmPrefix(srv.url(), QString(), payload, QStringLiteral("k1")), "warm: first warm sent");
    check(!router.warmPrefix(srv.url(), QString(), payload, QStringLiteral("k1")), "warm: duplicate in flight dropped");
    const bool done = waitUntil([&] { return sink.warm.contains("k1"); }, 5000);
    srv.join();
    check(done && sink.warm.value("k1"), "warm: finished ok");
    check(!router.warmPrefix(srv.url(), QString(), payload, QStringLiteral("k1")), "warm: duplicate after success dropped");
    check(srv.bodies().value(0).contains("\"max_tokens\":1"), "warm: payload reached the server", srv.bodies().value(0));
    check(router.warmsSent() == 1, "warm: exactly one request sent");

    // Connection refused: the warm fails and its key may be retried.
    const QString deadUrl = [] {
        MiniServer s;
        s.listen();
        return s.url();  // closed when s goes out of scope: nothing listens there now
    }();
    check(router.warmPrefix(deadUrl, QString(), payload, QStringLiteral("k2")), "warm: sent to a dead port");
    const bool failed = waitUntil([&] { return sink.warm.contains("k2"); }, 5000);
    check(failed && !sink.warm.value("k2"), "warm: refused connection reports failure");
    check(router.warmPrefix(deadUrl, QString(), payload, QStringLiteral("k2")), "warm: failed key may be retried");
    waitUntil([&] { return router.activeWarms() == 0; }, 5000);

    // HTTP 500 surfaces as a stream error with the body's message.
    MiniServer err;
    err.listen();
    MiniServer::Config ec;
    ec.status = 500;
    err.start(ec);
    router.startStream(req(QStringLiteral("e"), err.url(), QStringLiteral("e")));
    const bool errDone = waitUntil([&] { return sink.finishes.contains("e"); }, 5000);
    err.join();
    check(errDone && sink.error.value("e").startsWith("HTTP 500") && sink.error.value("e").contains("model overloaded"),
          "router: HTTP 500 -> error with status and body", sink.error.value("e"));
}

// --- Inkling request shape and Jarvis failover legs ----------------------------------------

QJsonObject jsonOf(const QByteArray &b)
{
    return QJsonDocument::fromJson(b).object();
}

void inklingTests()
{
    TieredPrompt p;
    p.model = QStringLiteral("Infatoshi/Inkling-2-Mini-heretic");
    p.systemPrompt = QStringLiteral("SYS");
    p.question = QStringLiteral("open x.com");
    for (const char *eff : {"", "fast", "none"}) {
        p.effort = QString::fromLatin1(eff);
        const QJsonObject o = jsonOf(buildTieredChatRequest(p));
        const QJsonObject k = o.value("chat_template_kwargs").toObject();
        check(k.size() == 1 && k.contains("reasoning_effort") && k.value("reasoning_effort").toDouble(-1) == 0.0
                  && !k.contains("enable_thinking") && !o.contains("reasoning_effort"),
              "inkling: no-think effort -> chat_template_kwargs.reasoning_effort 0.0, never enable_thinking",
              QString::fromUtf8(QJsonDocument(k).toJson(QJsonDocument::Compact)) + " effort=" + p.effort);
        check(o.value("temperature").toDouble(-1) == 0.0 && o.value("repeat_penalty").toDouble(-1) == 1.0
                  && o.value("max_tokens").toInt() == 384 && o.value("stream").toBool()
                  && o.value("continue_final_message").toBool() && !o.value("add_generation_prompt").toBool(true)
                  && o.value("messages").toArray().last().toObject().value("role").toString() == "assistant",
              "inkling: greedy, repeat_penalty 1.0, max_tokens 384, and continue_final_message prefill when not thinking");
    }
    const QHash<QString, double> want = {{"low", 0.2}, {"medium", 0.5}, {"high", 0.8}};
    for (auto it = want.cbegin(); it != want.cend(); ++it) {
        p.effort = it.key();
        const QJsonObject o = jsonOf(buildTieredChatRequest(p));
        check(o.value("chat_template_kwargs").toObject().value("reasoning_effort").toDouble(-1) == it.value()
                  && o.value("max_tokens").toInt() == 640 && o.value("repeat_penalty").toDouble(-1) == 1.0,
              "inkling: low/medium/high -> reasoning_effort 0.2/0.5/0.8, max_tokens 640", it.key());
    }
    // Inkling warm: the template renders the effort right after Tier 0, so the warm carries Turn 0's
    // effort and its messages stay a byte-identical prefix of the ask's.
    {
        TieredPrompt w = p;
        w.effort = QStringLiteral("fast");
        w.trunk = QStringLiteral("trunk");
        w.history = {ChatTurn{"user", "open x.com"}, ChatTurn{"assistant", "[[ACTION:cua_open:Chrome|https://x.com]]"}};
        const QJsonObject warm = jsonOf(buildTieredWarmRequest(w));
        const QJsonArray wm = warm.value("messages").toArray(), am = jsonOf(buildTieredChatRequest(w)).value("messages").toArray();
        bool prefix = wm.size() == 4 && am.size() == 6;
        for (qsizetype i = 0; prefix && i < wm.size(); ++i)
            prefix = wm[i] == am[i];
        check(prefix && warm.value("chat_template_kwargs").toObject().value("reasoning_effort").toDouble(-1) == 0.0
                  && !warm.value("add_generation_prompt").toBool(true) && warm.value("max_tokens").toInt() == 1,
              "inkling warm: Turn 0's effort kwarg, no generation prompt, a byte-identical prefix of the ask");
    }
    // Case-insensitive model match; a non-Inkling model keeps the Qwen no-think shape exactly.
    check(isInklingModel("thinkingmachines/inkling-2-mini") && isInklingModel("INKLING") && !isInklingModel("qwen3.8"),
          "inkling: model match is case-insensitive");
    TieredPrompt q = p;
    q.model = QStringLiteral("qwen3.8-flash-next");
    q.effort = QStringLiteral("fast");
    const QJsonObject qo = jsonOf(buildTieredChatRequest(q));
    check(qo.value("chat_template_kwargs").toObject() == QJsonObject{{"enable_thinking", false}} && !qo.contains("temperature")
              && !qo.contains("repeat_penalty") && !qo.contains("max_tokens"),
          "inkling: non-Inkling compat request unchanged (enable_thinking false only)");

    // OpenRouter: PrimeIntellect FP4 only, no fallbacks.
    TieredPrompt r = p;
    r.model = QStringLiteral("thinkingmachines/inkling-2-mini");
    r.effort = QStringLiteral("fast");
    const QJsonObject ro = jsonOf(buildOpenRouterChatRequest(r));
    const QJsonObject prov = ro.value("provider").toObject();
    check(prov.value("order").toArray() == QJsonArray{"PrimeIntellect"} && prov.value("quantizations").toArray() == QJsonArray{"fp4"}
              && prov.value("allow_fallbacks") == QJsonValue(false) && !prov.contains("sort"),
          "inkling openrouter: provider PrimeIntellect fp4, allow_fallbacks false",
          QString::fromUtf8(QJsonDocument(prov).toJson(QJsonDocument::Compact)));
    check(ro.value("temperature").toDouble(-1) == 0.0 && ro.value("max_tokens").toInt() == 384
              && ro.value("reasoning").toObject().value("effort") == "none" && !ro.contains("chat_template_kwargs"),
          "inkling openrouter: temperature 0, max_tokens 384 at effort none");
    r.effort = QStringLiteral("low");
    check(jsonOf(buildOpenRouterChatRequest(r)).value("max_tokens").toInt() == 640, "inkling openrouter: max_tokens 640 when thinking");
    TieredPrompt d = r;
    d.model = QStringLiteral("deepseek/deepseek-v4.1-flash-1017");
    const QJsonObject dp = jsonOf(buildOpenRouterChatRequest(d)).value("provider").toObject();
    check(dp.value("order").toArray() == QJsonArray{"Wafer", "InferenceNet", "Nebius"} && dp.value("allow_fallbacks").toBool()
              && dp.value("sort") == "latency" && !jsonOf(buildOpenRouterChatRequest(d)).contains("max_tokens"),
          "inkling openrouter: DeepSeek routing unchanged");
}

// A CUA follow-up's screenshot: an Inkling leaf becomes [image_url, text]; everything else
// (no image, a non-Inkling model, the prefix tiers, the OpenRouter legs) is byte-for-byte as before.
void multimodalTests()
{
    TieredPrompt p;
    p.model = QStringLiteral("Infatoshi/Inkling-2-Mini-heretic");
    p.systemPrompt = QStringLiteral("SYS");
    p.trunk = QStringLiteral("TRUNK");
    p.history = {{QStringLiteral("user"), QStringLiteral("open x.com")}, {QStringLiteral("assistant"), QStringLiteral("ok")}};
    p.question = QStringLiteral("[CUA observation after cua_open]");
    p.effort = QStringLiteral("low");
    const QByteArray plain = buildTieredChatRequest(p);
    const QByteArray jpeg("\xff\xd8\xff\xe0 fake jpeg bytes", 22);
    p.imageJpeg = jpeg;
    const QByteArray withImage = buildTieredChatRequest(p);
    const QJsonArray pm = jsonOf(plain).value("messages").toArray();
    const QJsonArray im = jsonOf(withImage).value("messages").toArray();
    const QJsonObject leaf = im.last().toObject();
    const QJsonArray parts = leaf.value("content").toArray();
    check(leaf.value("role") == "user" && parts.size() == 2 && parts[0].toObject().value("type") == "image_url"
              && parts[1].toObject().value("type") == "text",
          "multimodal: Inkling + image -> leaf content is [image_url, text]");
    check(parts[0].toObject().value("image_url").toObject().value("url").toString()
              == "data:image/jpeg;base64," + QString::fromLatin1(jpeg.toBase64()),
          "multimodal: image_url is a base64 JPEG data URL");
    check(parts[1].toObject().value("text").toString() == tieredLeafText(p)
              && pm.last().toObject().value("content").toString() == tieredLeafText(p),
          "multimodal: the text part is exactly the leaf a text turn sends");
    bool prefixSame = pm.size() == im.size();
    for (qsizetype i = 0; prefixSame && i + 1 < pm.size(); ++i)
        prefixSame = compact(pm[i]) == compact(im[i]);
    check(prefixSame && !compact(pm).contains("base64") && compact(QJsonArray{im[0], im[1], im[2], im[3]}).indexOf("base64") < 0,
          "multimodal: Tiers 0-2 byte-identical with and without the image (APC stays hot)");

    // Zero-think prefill still trails the multimodal leaf.
    p.effort = QStringLiteral("fast");
    const QJsonArray fm = jsonOf(buildTieredChatRequest(p)).value("messages").toArray();
    check(fm.last().toObject().value("role") == "assistant" && fm[fm.size() - 2].toObject().value("content").isArray(),
          "multimodal: zero-think prefill follows the image leaf");

    // Empty image or a non-Inkling model: the exact old string content.
    p.effort = QStringLiteral("low");
    p.imageJpeg.clear();
    check(buildTieredChatRequest(p) == plain && jsonOf(plain).value("messages").toArray().last().toObject().value("content").isString(),
          "multimodal: empty image -> the unchanged string request");
    TieredPrompt q = p;
    q.model = QStringLiteral("qwen3.8-flash-next");
    const QByteArray qPlain = buildTieredChatRequest(q);
    q.imageJpeg = jpeg;
    check(buildTieredChatRequest(q) == qPlain, "multimodal: non-Inkling model ignores the image (string content)");

    // Legs: only the self-hosted leg sends the screenshot.
    JarvisLegConfig cfg;
    cfg.jarvisEndpoint = QStringLiteral("http://127.0.0.1:9/v1");
    cfg.openrouterKey = QStringLiteral("k");
    const QVector<JarvisLeg> legs = planJarvisLegs(cfg);
    p.imageJpeg = jpeg;
    check(legs.size() == 3 && buildJarvisLegRequest(legs[0], p).contains("data:image/jpeg;base64,")
              && !buildJarvisLegRequest(legs[1], p).contains("base64") && !buildJarvisLegRequest(legs[2], p).contains("base64"),
          "multimodal: leg 1 (self) carries the image, the OpenRouter legs never do");
    check(kCompatSystemPrompt.contains("When a screenshot is attached to a follow-up turn, read visible text directly from the image"),
          "multimodal: system prompt tells the model to read the attached screenshot");
}

void jarvisLegTests()
{
    JarvisLegConfig cfg;
    check(planJarvisLegs(cfg).isEmpty(), "legs: nothing configured -> no legs");
    cfg.jarvisEndpoint = QStringLiteral("http://127.0.0.1:9/v1/");
    cfg.openrouterKey = QStringLiteral("k");
    QVector<JarvisLeg> legs = planJarvisLegs(cfg);
    QStringList names;
    for (const JarvisLeg &l : legs)
        names << l.name;
    check(names == QStringList({"self", "inkling", "deepseek"}), "legs: self, inkling, deepseek in order", names.join(','));
    if (legs.size() == 3) {
        check(legs[0].endpoint == "http://127.0.0.1:9/v1/chat/completions" && legs[0].model == kJarvisDefaultModel
                  && legs[0].connectTimeoutMs == 200 && legs[0].firstByteTimeoutMs == 1800 && !legs[0].openRouter
                  && legs[0].apiKey.isEmpty(),
              "legs: self leg 200 ms connect, 1.8 s first byte, default Inkling model, no key", legs[0].endpoint);
        check(legs[1].model == "thinkingmachines/inkling-2-mini" && legs[1].openRouter && legs[1].connectTimeoutMs == 3000
                  && legs[1].firstByteTimeoutMs == 2500 && legs[1].apiKey == "k"
                  && legs[1].endpoint == "https://openrouter.ai/api/v1/chat/completions",
              "legs: OpenRouter Inkling 3 s / 2.5 s");
        check(legs[2].model == "deepseek/deepseek-v4.1-flash-1017" && legs[2].connectTimeoutMs == 4000
                  && legs[2].firstByteTimeoutMs == 4000,
              "legs: OpenRouter DeepSeek 4 s / 4 s");
        TieredPrompt p;
        p.question = QStringLiteral("q");
        p.effort = QStringLiteral("fast");
        check(jsonOf(buildJarvisLegRequest(legs[0], p)).value("chat_template_kwargs").toObject().contains("reasoning_effort")
                  && jsonOf(buildJarvisLegRequest(legs[1], p)).value("provider").toObject().value("order").toArray()
                         == QJsonArray{"PrimeIntellect"}
                  && jsonOf(buildJarvisLegRequest(legs[2], p)).value("model") == "deepseek/deepseek-v4.1-flash-1017",
              "legs: each leg builds its own request shape");
    }
    cfg.jarvisModel = QStringLiteral("custom/model");
    check(planJarvisLegs(cfg).value(0).model == "custom/model", "legs: jarvisModel overrides the self model");
    cfg.selfBreakerOpen = true;
    check(planJarvisLegs(cfg).size() == 2 && planJarvisLegs(cfg).value(0).name == "inkling", "legs: open breaker skips self");
    cfg.openrouterKey.clear();
    check(planJarvisLegs(cfg).isEmpty(), "legs: open breaker and no key -> none (caller retries self)");

    check(jarvisLegRetryable("Failed to connect to 127.0.0.1 port 9") && jarvisLegRetryable("first byte timeout after 1800 ms")
              && jarvisLegRetryable("HTTP 503: overloaded") && jarvisLegRetryable("HTTP 429") && jarvisLegRetryable("HTTP 0")
              && !jarvisLegRetryable("HTTP 401: bad key") && !jarvisLegRetryable("HTTP 404") && !jarvisLegRetryable(QString()),
          "legs: transport / timeout / 5xx / 429 retry; auth and 404 do not");
    JarvisBreaker b;
    check(!b.isOpen(1000), "breaker: closed at start");
    b.trip(1000);
    check(b.isOpen(1000) && b.isOpen(15999) && !b.isOpen(16000), "breaker: open exactly 15 s after a trip");
}

// A socket that listens (the kernel completes the TCP handshake) but never answers: a hung model.
struct SilentListener {
    int fd = -1;
    quint16 port = 0;
    bool listen()
    {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        if (fd < 0 || ::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof addr) != 0 || ::listen(fd, 8) != 0)
            return false;
        socklen_t len = sizeof addr;
        ::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &len);
        port = ntohs(addr.sin_port);
        return true;
    }
    QString url() const { return QStringLiteral("http://127.0.0.1:%1/v1/chat/completions").arg(port); }
    ~SilentListener()
    {
        if (fd >= 0)
            ::close(fd);
    }
};

void firstByteAndFailoverTests()
{
    SilentListener hung;
    check(hung.listen(), "first byte: silent listener up");
    LlmRouter router;
    Sink sink;
    sink.attach(router);
    LlmStreamRequest r = req(QStringLiteral("fb"), hung.url(), QStringLiteral("fb"));
    r.firstByteTimeoutMs = 300;
    QElapsedTimer t;
    t.start();
    router.startStream(r);
    const bool done = waitUntil([&] { return sink.finishes.contains("fb"); }, 5000);
    const qint64 ms = t.elapsed();
    check(done && sink.error.value("fb") == "first byte timeout after 300 ms" && sink.deltas.value("fb") == 0,
          "first byte: a hung endpoint fails with a first byte timeout, no deltas", sink.error.value("fb"));
    check(ms >= 280 && ms < 1500, "first byte: fires near its 300 ms deadline", QString::number(ms));
    check(waitUntil([&] { return router.handlesInMulti() == 0; }, 1000), "first byte: the hung handle is removed");

    // Leg 1 (hung) -> leg 2 (answers) on one stream id, before any delta: the Backend's rule.
    MiniServer ok;
    check(ok.listen(), "failover: answering server up");
    MiniServer::Config c;
    c.rounds = 3;
    ok.start(c);
    QVector<JarvisLeg> legs = {{QStringLiteral("self"), hung.url(), QString(), QStringLiteral("inkling-local"), false, 200, 300},
                               {QStringLiteral("inkling"), ok.url(), QString(), QStringLiteral("inkling-or"), false, 3000, 2500}};
    int leg = 0;
    JarvisBreaker breaker;
    QStringList finishes;
    QString text;
    auto launch = [&] {
        LlmStreamRequest lr = req(QStringLiteral("spotlight"), legs[leg].endpoint, QStringLiteral("j%1").arg(leg));
        lr.connectTimeoutMs = legs[leg].connectTimeoutMs;
        lr.firstByteTimeoutMs = legs[leg].firstByteTimeoutMs;
        router.startStream(lr);
    };
    QObject::connect(&router, &LlmRouter::streamDelta, [&](const QString &id, const QString &tx, const QString &) {
        if (id == "spotlight")
            text += tx;
    });
    QObject::connect(&router, &LlmRouter::streamFinished, [&](const QString &id, const QString &e) {
        if (id != "spotlight")
            return;
        if (jarvisLegRetryable(e) && text.isEmpty() && leg + 1 < legs.size()) {
            if (legs[leg].name == "self")
                breaker.trip(QDateTime::currentMSecsSinceEpoch());
            ++leg;
            launch();
            return;
        }
        finishes << e;
    });
    launch();
    const bool fin = waitUntil([&] { return !finishes.isEmpty(); }, 6000);
    ok.join();
    check(fin && finishes == QStringList{QString()} && leg == 1 && !text.isEmpty()
              && breaker.isOpen(QDateTime::currentMSecsSinceEpoch()),
          "failover: hung leg 1 -> leg 2 answers on the same stream, one clean finish, breaker tripped",
          finishes.join(';') + " leg=" + QString::number(leg) + " text=" + text);
}

}  // namespace

int main(int argc, char **argv)
{
    std::signal(SIGPIPE, SIG_IGN);
    QCoreApplication app(argc, argv);
    prefixTests();
    tier0Tests();
    sseTests();
    concurrencyTest(10);
    concurrencyTest(16);
    cancelTest();
    restartTest();
    warmAndErrorTests();
    inklingTests();
    multimodalTests();
    jarvisLegTests();
    firstByteAndFailoverTests();
    std::printf("test_llm: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
