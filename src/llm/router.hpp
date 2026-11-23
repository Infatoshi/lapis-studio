// Multi-stream LLM router: every OpenAI-compatible SSE stream and prefix warm runs on one
// dedicated network thread driving a libcurl multi handle, so many HTTP/1.1 streams progress in
// parallel over pooled keep-alive connections (QNetworkAccessManager caps a host at 6).
//
// Threading: public methods are called on the owning (Qt) thread only. Commands reach the
// network thread through a short mutex-guarded queue plus curl_multi_wakeup; parsed SSE deltas
// come back through a lock-free SPSC ring that a 4 ms timer drains on the Qt thread while any
// stream or warm is active. Consecutive deltas of one stream are coalesced before emitting.
#pragma once

#include "core/ring_buffer.hpp"

#include <QByteArray>
#include <QHash>
#include <QObject>
#include <QSet>
#include <QString>
#include <QTimer>

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace studio {

struct LlmStreamRequest {
    QString streamId;
    QString endpoint;       // full URL, e.g. http://host:8000/v1/chat/completions
    QString apiKey;         // optional bearer token
    QByteArray payload;     // JSON body (stream: true)
    int timeoutMs = 0;      // whole-transfer timeout; 0 = none
    int connectTimeoutMs = 10000;
    // Fails the stream ("first byte timeout after N ms") when no response body byte has
    // arrived this long after the start; 0 = none. Lets a caller fail over before any delta.
    int firstByteTimeoutMs = 0;
};

class LlmRouter : public QObject {
    Q_OBJECT
public:
    explicit LlmRouter(QObject *parent = nullptr);
    ~LlmRouter() override;

    // Starts (or restarts: an active stream with the same id is cancelled first) a stream.
    void startStream(const LlmStreamRequest &req);
    // Removes the stream's handle from the multi at once; no further signals for it.
    void cancelStream(const QString &streamId);
    // Fire-and-forget max_tokens:1 prefill. Dropped while the same dedupKey is in flight or has
    // already succeeded; a failed warm forgets its key so the next call retries.
    // Returns false when deduplicated.
    bool warmPrefix(const QString &endpoint, const QString &apiKey, const QByteArray &payload,
                    const QString &dedupKey);
    void forgetWarm(const QString &dedupKey) { m_warmDone.remove(dedupKey); }

    bool isActive(const QString &streamId) const { return m_streamGen.contains(streamId); }
    int activeStreams() const { return int(m_streamGen.size()); }
    int activeWarms() const { return int(m_warmInFlight.size()); }
    int warmsSent() const { return m_warmsSent; }
    // Network-thread view: easy handles currently inside the multi (streams + warms).
    int handlesInMulti() const { return m_handlesInMulti.load(std::memory_order_relaxed); }
    // Most handles that were simultaneously receiving response bytes (proof of parallelism).
    int peakConcurrentStreams() const { return m_peakReceiving.load(std::memory_order_relaxed); }

    // Drain the event ring now (tests; normally the 4 ms timer does it).
    void pump();

    // One delta / finish crossing threads.
    struct Event {
        enum Kind : quint8 { Delta, Finished, WarmFinished };
        Kind kind = Delta;
        quint64 gen = 0;
        QString streamId;  // stream id, or the warm's dedupKey
        QString text;
        QString thinking;
        QString error;
    };

signals:
    void streamDelta(const QString &streamId, const QString &textDelta, const QString &thinkingDelta);
    // error is empty on a clean finish ([DONE] or a 2xx close).
    void streamFinished(const QString &streamId, const QString &error);
    void warmFinished(const QString &dedupKey, bool ok);

private:
    struct Command {
        enum Kind { Start, Cancel, Warm } kind = Start;
        quint64 gen = 0;
        LlmStreamRequest req;  // Warm: streamId holds the dedupKey
    };
    struct Net;  // network-thread state (curl handles), defined in router.cpp

    void enqueue(Command &&c);
    void ensureTimer();
    void runNetwork();

    // Qt-thread state
    QHash<QString, quint64> m_streamGen;  // active stream id -> generation
    QSet<QString> m_warmInFlight;
    QSet<QString> m_warmDone;
    quint64 m_nextGen = 1;
    int m_warmsSent = 0;
    QTimer m_drain;

    // cross-thread
    std::mutex m_cmdMutex;
    std::deque<Command> m_cmds;
    SpscRing<Event> m_events{4096};
    std::atomic<bool> m_stop{false};
    std::atomic<int> m_handlesInMulti{0};
    std::atomic<int> m_peakReceiving{0};
    std::unique_ptr<Net> m_net;
    std::thread m_thread;
};

}  // namespace studio
