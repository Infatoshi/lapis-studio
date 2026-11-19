#include "llm/router.hpp"

#include "llm/sse.hpp"

#include <curl/curl.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace studio {

namespace {

std::once_flag g_curlInit;

constexpr long kMaxHostConnections = 32;  // >> 6: a dozen-plus parallel streams to one vLLM host
constexpr long kMaxConnects = 48;         // keep-alive pool size
constexpr int kIdlePollMs = 200;
constexpr int kOverflowPollMs = 2;
constexpr qsizetype kMaxErrorBody = 600;

QString oneLineSnippet(const QByteArray &body)
{
    QString s = QString::fromUtf8(body).simplified();
    if (s.size() > 200)
        s = s.left(200) + QChar(0x2026);
    return s;
}

}  // namespace

// --- network-thread state ------------------------------------------------------------------

struct LlmRouter::Net {
    struct Easy {
        CURL *h = nullptr;
        curl_slist *headers = nullptr;
        quint64 gen = 0;
        QString id;          // stream id, or the warm's dedupKey
        bool warm = false;
        bool receiving = false;
        long status = 0;
        QByteArray buf;      // bytes after the last complete SSE line
        QByteArray errBody;  // non-2xx body, for the error message
        QString pendText, pendThink;
        QString sseError;
        char errbuf[CURL_ERROR_SIZE] = {0};
        Net *net = nullptr;
    };

    LlmRouter *self = nullptr;
    CURLM *multi = nullptr;
    std::unordered_map<CURL *, std::unique_ptr<Easy>> easies;
    QHash<QString, Easy *> streams;  // active stream id -> its easy (warms are not listed)
    std::deque<Event> overflow;      // events the full ring could not take yet (order kept)
    int receiving = 0;

    void pushEvent(Event &&ev)
    {
        flushOverflow();
        if (overflow.empty() && self->m_events.tryPush(std::move(ev)))
            return;
        overflow.push_back(std::move(ev));
    }
    void flushOverflow()
    {
        while (!overflow.empty()) {
            if (!self->m_events.tryPush(std::move(overflow.front())))
                return;
            overflow.pop_front();
        }
    }

    static size_t onWrite(char *ptr, size_t size, size_t nmemb, void *userdata)
    {
        auto *e = static_cast<Easy *>(userdata);
        const size_t n = size * nmemb;
        if (!e->receiving) {
            e->receiving = true;
            Net *net = e->net;
            net->receiving += 1;
            int peak = net->self->m_peakReceiving.load(std::memory_order_relaxed);
            while (net->receiving > peak
                   && !net->self->m_peakReceiving.compare_exchange_weak(peak, net->receiving, std::memory_order_relaxed)) {
            }
        }
        if (e->status == 0)
            curl_easy_getinfo(e->h, CURLINFO_RESPONSE_CODE, &e->status);
        if (e->status >= 300) {
            if (e->errBody.size() < kMaxErrorBody)
                e->errBody.append(ptr, qsizetype(std::min<size_t>(n, size_t(kMaxErrorBody))));
            return n;
        }
        if (e->warm)
            return n;  // the one prefilled token is discarded
        e->buf.append(ptr, qsizetype(n));
        consumeLines(e, false);
        return n;
    }

    static void handleLine(Easy *e, const QByteArray &line)
    {
        const OpenAiSseEvent ev = parseOpenAiSseLine(line);
        if (ev.kind == OpenAiSseEvent::Delta) {
            e->pendText += ev.text;
            e->pendThink += ev.thinking;
        } else if (ev.kind == OpenAiSseEvent::Error) {
            e->sseError = ev.error.isEmpty() ? QStringLiteral("stream error") : ev.error;
        }
    }

    static void consumeLines(Easy *e, bool final)
    {
        qsizetype start = 0;
        qsizetype nl;
        while ((nl = e->buf.indexOf('\n', start)) >= 0) {
            handleLine(e, e->buf.mid(start, nl - start));
            start = nl + 1;
        }
        e->buf.remove(0, start);
        if (final && !e->buf.trimmed().isEmpty()) {
            handleLine(e, e->buf);
            e->buf.clear();
        }
    }

    void flushDelta(Easy *e)
    {
        if (e->warm || (e->pendText.isEmpty() && e->pendThink.isEmpty()))
            return;
        Event ev;
        ev.kind = Event::Delta;
        ev.gen = e->gen;
        ev.streamId = e->id;
        ev.text = std::move(e->pendText);
        ev.thinking = std::move(e->pendThink);
        e->pendText.clear();
        e->pendThink.clear();
        pushEvent(std::move(ev));
    }

    void removeEasy(CURL *h)
    {
        auto it = easies.find(h);
        if (it == easies.end())
            return;
        Easy *e = it->second.get();
        curl_multi_remove_handle(multi, h);  // a half-read response's connection is closed, not pooled
        if (e->receiving)
            receiving -= 1;
        if (!e->warm) {
            auto s = streams.find(e->id);
            if (s != streams.end() && s.value() == e)
                streams.erase(s);
        }
        curl_easy_cleanup(h);
        curl_slist_free_all(e->headers);
        easies.erase(it);
        self->m_handlesInMulti.store(int(easies.size()), std::memory_order_relaxed);
    }

    void cancel(const QString &id)
    {
        if (Easy *e = streams.value(id))
            removeEasy(e->h);
    }

    void add(const Command &c)
    {
        auto e = std::make_unique<Easy>();
        e->net = this;
        e->gen = c.gen;
        e->id = c.req.streamId;
        e->warm = c.kind == Command::Warm;
        CURL *h = curl_easy_init();
        if (!h) {
            Event ev;
            ev.kind = e->warm ? Event::WarmFinished : Event::Finished;
            ev.gen = e->gen;
            ev.streamId = e->id;
            ev.error = QStringLiteral("curl_easy_init failed");
            pushEvent(std::move(ev));
            return;
        }
        e->h = h;
        const QByteArray url = c.req.endpoint.toUtf8();
        e->headers = curl_slist_append(e->headers, "Content-Type: application/json");
        e->headers = curl_slist_append(e->headers, e->warm ? "Accept: application/json" : "Accept: text/event-stream");
        e->headers = curl_slist_append(e->headers, "Expect:");  // no 100-continue round trip on big prompts
        if (!c.req.apiKey.isEmpty()) {
            const QByteArray auth = QByteArray("Authorization: Bearer ") + c.req.apiKey.toUtf8();
            e->headers = curl_slist_append(e->headers, auth.constData());
        }
        curl_easy_setopt(h, CURLOPT_URL, url.constData());
        curl_easy_setopt(h, CURLOPT_HTTPHEADER, e->headers);
        curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE_LARGE, curl_off_t(c.req.payload.size()));
        curl_easy_setopt(h, CURLOPT_COPYPOSTFIELDS, c.req.payload.constData());
        curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &Net::onWrite);
        curl_easy_setopt(h, CURLOPT_WRITEDATA, e.get());
        curl_easy_setopt(h, CURLOPT_PRIVATE, e.get());
        curl_easy_setopt(h, CURLOPT_ERRORBUFFER, e->errbuf);
        curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
        curl_easy_setopt(h, CURLOPT_TCP_NODELAY, 1L);
        curl_easy_setopt(h, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, long(std::max(0, c.req.connectTimeoutMs)));
        curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, long(std::max(0, c.req.timeoutMs)));
        CURL *raw = h;
        easies.emplace(raw, std::move(e));
        Easy *ep = easies[raw].get();
        if (!ep->warm)
            streams.insert(ep->id, ep);
        curl_multi_add_handle(multi, raw);
        self->m_handlesInMulti.store(int(easies.size()), std::memory_order_relaxed);
    }

    void finish(CURL *h, CURLcode rc)
    {
        auto it = easies.find(h);
        if (it == easies.end())
            return;
        Easy *e = it->second.get();
        if (e->status == 0)
            curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &e->status);
        if (!e->warm) {
            consumeLines(e, true);
            flushDelta(e);
        }
        QString error;
        if (rc != CURLE_OK)
            error = e->errbuf[0] ? QString::fromUtf8(e->errbuf) : QString::fromUtf8(curl_easy_strerror(rc));
        else if (e->status >= 300 || e->status == 0)
            error = QStringLiteral("HTTP %1").arg(e->status)
                + (e->errBody.isEmpty() ? QString() : QStringLiteral(": ") + oneLineSnippet(e->errBody));
        else
            error = e->sseError;
        Event ev;
        ev.kind = e->warm ? Event::WarmFinished : Event::Finished;
        ev.gen = e->gen;
        ev.streamId = e->id;
        ev.error = error;
        removeEasy(h);
        pushEvent(std::move(ev));
    }

    ~Net()
    {
        while (!easies.empty())
            removeEasy(easies.begin()->first);
        if (multi)
            curl_multi_cleanup(multi);
    }
};

// --- Qt thread -----------------------------------------------------------------------------

LlmRouter::LlmRouter(QObject *parent) : QObject(parent)
{
    std::call_once(g_curlInit, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    m_net = std::make_unique<Net>();
    m_net->self = this;
    m_net->multi = curl_multi_init();
    curl_multi_setopt(m_net->multi, CURLMOPT_MAX_HOST_CONNECTIONS, kMaxHostConnections);
    curl_multi_setopt(m_net->multi, CURLMOPT_MAXCONNECTS, kMaxConnects);
    curl_multi_setopt(m_net->multi, CURLMOPT_MAX_TOTAL_CONNECTIONS, 0L);
    m_drain.setInterval(4);
    m_drain.setTimerType(Qt::PreciseTimer);
    connect(&m_drain, &QTimer::timeout, this, &LlmRouter::pump);
    m_thread = std::thread([this] { runNetwork(); });
}

LlmRouter::~LlmRouter()
{
    m_stop.store(true, std::memory_order_release);
    if (m_net && m_net->multi)
        curl_multi_wakeup(m_net->multi);
    if (m_thread.joinable())
        m_thread.join();
    m_net.reset();
}

void LlmRouter::enqueue(Command &&c)
{
    {
        std::lock_guard<std::mutex> lock(m_cmdMutex);
        m_cmds.push_back(std::move(c));
    }
    curl_multi_wakeup(m_net->multi);
}

void LlmRouter::ensureTimer()
{
    if (!m_drain.isActive())
        m_drain.start();
}

void LlmRouter::startStream(const LlmStreamRequest &req)
{
    if (req.streamId.isEmpty())
        return;
    Command c;
    c.kind = Command::Start;
    c.gen = m_nextGen++;
    c.req = req;
    m_streamGen.insert(req.streamId, c.gen);  // older generations' events are dropped from here on
    enqueue(std::move(c));
    ensureTimer();
}

void LlmRouter::cancelStream(const QString &streamId)
{
    if (!m_streamGen.remove(streamId))
        return;
    Command c;
    c.kind = Command::Cancel;
    c.req.streamId = streamId;
    enqueue(std::move(c));
}

bool LlmRouter::warmPrefix(const QString &endpoint, const QString &apiKey, const QByteArray &payload,
                           const QString &dedupKey)
{
    if (endpoint.isEmpty() || dedupKey.isEmpty() || m_warmInFlight.contains(dedupKey) || m_warmDone.contains(dedupKey))
        return false;
    if (m_warmDone.size() > 1024)
        m_warmDone.clear();
    m_warmInFlight.insert(dedupKey);
    ++m_warmsSent;
    Command c;
    c.kind = Command::Warm;
    c.gen = m_nextGen++;
    c.req.streamId = dedupKey;
    c.req.endpoint = endpoint;
    c.req.apiKey = apiKey;
    c.req.payload = payload;
    c.req.timeoutMs = 30000;
    enqueue(std::move(c));
    ensureTimer();
    return true;
}

void LlmRouter::pump()
{
    struct Pending {
        QString id;
        quint64 gen;
        QString text, thinking;
    };
    std::vector<Pending> pending;  // first-delta order
    QHash<QString, size_t> slot;
    auto flush = [this, &pending, &slot](const QString &id) {
        auto s = slot.find(id);
        if (s == slot.end())
            return;
        Pending &p = pending[s.value()];
        if (m_streamGen.value(p.id) == p.gen && (!p.text.isEmpty() || !p.thinking.isEmpty()))
            emit streamDelta(p.id, p.text, p.thinking);
        p.text.clear();
        p.thinking.clear();
        slot.erase(s);
    };

    Event ev;
    while (m_events.tryPop(ev)) {
        switch (ev.kind) {
        case Event::Delta: {
            if (m_streamGen.value(ev.streamId) != ev.gen)
                break;  // cancelled or restarted
            auto s = slot.find(ev.streamId);
            if (s == slot.end() || pending[s.value()].gen != ev.gen) {
                if (s != slot.end())
                    flush(ev.streamId);
                slot.insert(ev.streamId, pending.size());
                pending.push_back(Pending{ev.streamId, ev.gen, std::move(ev.text), std::move(ev.thinking)});
            } else {
                pending[s.value()].text += ev.text;
                pending[s.value()].thinking += ev.thinking;
            }
            break;
        }
        case Event::Finished:
            if (m_streamGen.value(ev.streamId) != ev.gen)
                break;
            flush(ev.streamId);
            if (m_streamGen.value(ev.streamId) != ev.gen)
                break;  // a delta slot restarted or cancelled it
            m_streamGen.remove(ev.streamId);
            emit streamFinished(ev.streamId, ev.error);
            break;
        case Event::WarmFinished:
            m_warmInFlight.remove(ev.streamId);
            if (ev.error.isEmpty())
                m_warmDone.insert(ev.streamId);
            emit warmFinished(ev.streamId, ev.error.isEmpty());
            break;
        }
    }
    for (size_t i = 0; i < pending.size(); ++i) {
        const auto s = slot.constFind(pending[i].id);
        if (s != slot.cend() && s.value() == i)
            flush(QString(pending[i].id));  // copy: flush() may drop the slot the id lives in
    }
    if (m_streamGen.isEmpty() && m_warmInFlight.isEmpty())
        m_drain.stop();
}

// --- network thread ------------------------------------------------------------------------

void LlmRouter::runNetwork()
{
    Net &net = *m_net;
    std::deque<Command> cmds;
    while (!m_stop.load(std::memory_order_acquire)) {
        {
            std::lock_guard<std::mutex> lock(m_cmdMutex);
            cmds.swap(m_cmds);
        }
        for (Command &c : cmds) {
            switch (c.kind) {
            case Command::Start:
                net.cancel(c.req.streamId);
                net.add(c);
                break;
            case Command::Cancel:
                net.cancel(c.req.streamId);
                break;
            case Command::Warm:
                net.add(c);
                break;
            }
        }
        cmds.clear();

        int running = 0;
        curl_multi_perform(net.multi, &running);
        for (auto &kv : net.easies)
            net.flushDelta(kv.second.get());
        int left = 0;
        while (CURLMsg *msg = curl_multi_info_read(net.multi, &left)) {
            if (msg->msg == CURLMSG_DONE)
                net.finish(msg->easy_handle, msg->data.result);
        }
        net.flushOverflow();
        curl_multi_poll(net.multi, nullptr, 0, net.overflow.empty() ? kIdlePollMs : kOverflowPollMs, nullptr);
    }
}

}  // namespace studio
