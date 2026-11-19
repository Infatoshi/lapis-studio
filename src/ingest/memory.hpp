// Cross-session memory: completed (user, assistant) turns from past Claude Code, Codex,
// Antigravity and Phonon conversations, searched with a pure C++ hybrid retriever.
//
//   score(d) = 1 / (60 + rank_bm25(d)) + 1 / (60 + rank_semantic(d)) + 0.002 * exp(-ageDays / 7)
//
// rank_bm25 ranks Okapi BM25 over word tokens; rank_semantic ranks the cosine of a TF-IDF vector
// over word tokens plus character 3-grams (so "kublass" still finds "cuBLAS"). A term a document
// lacks contributes nothing; documents matching neither list never surface.
//
// The index is immutable once built: refreshAsync() builds a new one on a QtConcurrent thread
// and swaps it in on the owner's thread, so query() never locks and never blocks on I/O.
#pragma once

#include <QFutureWatcher>
#include <QHash>
#include <QObject>
#include <QSet>
#include <QString>
#include <QStringList>
#include <QVector>

#include <memory>

namespace studio {

struct MemoryTurn {
    QString harness;    // "claude" | "codex" | "agy" | "phonon"
    QString sessionId;  // the session uuid (Lapis SessionInfo::ident) or the Phonon meeting folder
    QString project;    // working-directory basename ("lapis-studio"), "" when unknown
    qint64 mtimeMs = 0;
    int turnIdx = 0;
    QString userText;
    QString assistantText;
};

struct MemoryHit {
    int turn = -1;  // index into turns()
    double score = 0;
    int bm25Rank = 0;      // 1-based, 0 = no BM25 match
    int semanticRank = 0;  // 1-based, 0 = below the cosine floor
};

// Where refresh() looks. Every path defaults to the real location under $HOME.
struct MemorySources {
    QString claudeProjects;  // ~/.claude/projects            (*/*.jsonl)
    QString codexSessions;   // ~/.codex/sessions             (**/*.jsonl)
    QString agyBrain;        // ~/.gemini/antigravity-cli/brain (*/.system_generated/logs/transcript.jsonl)
    QString phononMeetings;  // ~/Library/Application Support/Phonon/Meetings (meeting_*/transcript.txt)
    int maxAgeDays = 14;
    int maxTurns = 2000;
    qint64 tailBytes = 768 * 1024;  // per transcript: only its newest bytes are parsed
    // Case-insensitive substrings of a transcript path or its recorded cwd that are never read
    // (redacted categories). Default {"infinity"}.
    QStringList excludePathParts;
    // Session ids never indexed (the live registry's redacted sessions).
    QSet<QString> excludeSessionIds;
    static MemorySources defaults();
};

// One transcript's parsed turns, reused while its mtime and size are unchanged.
struct MemoryFileCache {
    qint64 mtimeMs = 0;
    qint64 size = 0;
    QVector<MemoryTurn> turns;
};
using MemoryCache = QHash<QString, MemoryFileCache>;

class MemoryIndex : public QObject {
    Q_OBJECT
public:
    struct Corpus;

    explicit MemoryIndex(QObject *parent = nullptr);
    ~MemoryIndex() override;

    // Replace the corpus synchronously (tests and fixtures). Turns keep their given order.
    void setTurns(const QVector<MemoryTurn> &turns);
    int size() const;
    QVector<MemoryTurn> turns() const;
    // Rough resident bytes of the corpus (texts + postings).
    qint64 memoryBytes() const;

    // Ranked hits, best first, never from excludeSessionId. nowMs <= 0: the current time.
    QVector<MemoryHit> search(const QString &text, const QString &excludeSessionId = QString(), int topK = 4,
                              qint64 nowMs = 0) const;
    // The Tier 3 block: top-K turns, "[harness session] project, age" headers, <= maxChars in
    // total. "" when nothing matches.
    QString query(const QString &text, const QString &excludeSessionId = QString(), int topK = 4,
                  int maxChars = 3200, qint64 nowMs = 0) const;

    // Background incremental rescan; emits refreshed() on this object's thread when swapped in.
    // A refresh already running makes this a no-op.
    void refreshAsync(const MemorySources &sources = MemorySources::defaults());
    bool refreshing() const { return m_watcher.isRunning(); }
    double lastRefreshMs() const { return m_lastRefreshMs; }

    // Synchronous scan (the worker body; also used by tests). `cache` is read and updated.
    static QVector<MemoryTurn> scan(const MemorySources &sources, qint64 nowMs, MemoryCache *cache);
    // Phonon "[mm:ss] SPEAKER: text" lines in windows of `linesPerTurn`.
    static QVector<MemoryTurn> phononTurns(const QString &transcript, const QString &meetingId, qint64 mtimeMs,
                                           int linesPerTurn = 10);

signals:
    void refreshed(int turns, double ms);

private:
    struct ScanOut {
        std::shared_ptr<const Corpus> corpus;
        MemoryCache cache;
        double ms = 0;
    };
    std::shared_ptr<const Corpus> m_corpus;
    MemoryCache m_cache;
    QFutureWatcher<ScanOut> m_watcher;
    double m_lastRefreshMs = 0;

    static std::shared_ptr<const Corpus> build(QVector<MemoryTurn> turns);
};

}  // namespace studio
