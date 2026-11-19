#include "ingest/memory.hpp"

#include "transcript.hpp"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QtConcurrent/QtConcurrentRun>

#include <algorithm>
#include <cmath>

namespace studio {

namespace {

constexpr int kUserChars = 700;        // stored (and indexed) per turn
constexpr int kAssistantChars = 1300;
constexpr double kBm25K1 = 1.2;
constexpr double kBm25B = 0.75;
constexpr double kRrfK = 60.0;
constexpr double kRecencyWeight = 0.002;
constexpr double kRecencyDays = 7.0;
// A cosine below this is shared-trigram noise, not a paraphrase: it earns no semantic rank.
constexpr double kCosineFloor = 0.06;
constexpr quint32 kWordSeed = 0x9e3779b9u;
constexpr quint32 kGramSeed = 0x85ebca6bu;

struct Posting {
    quint32 doc;
    float w;  // BM25: term frequency; semantic: tf-idf weight
};

const QSet<QString> &stopwords()
{
    static const QSet<QString> s = [] {
        QSet<QString> out;
        for (const char *w : {"a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "at",
                              "by", "for", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
                              "it", "its", "this", "that", "these", "those", "there", "here", "i", "me", "my", "we",
                              "our", "you", "your", "he", "she", "they", "them", "their", "what", "which", "who",
                              "whom", "how", "why", "when", "where", "do", "does", "did", "done", "can", "could",
                              "should", "would", "will", "just", "like", "not", "no", "yes", "all", "any", "some",
                              "about", "into", "out", "up", "down", "over", "than", "too", "very", "also", "have",
                              "has", "had", "get", "got", "let", "lets", "ok", "okay", "um", "uh", "yeah", "im",
                              "dont", "thats", "its", "we're", "you're", "one", "now"})
            out.insert(QString::fromLatin1(w));
        return out;
    }();
    return s;
}

// Lowercase alphanumeric runs ('_' joins), >= 2 chars, stopwords dropped, <= 40 chars each.
QStringList tokenize(const QString &text)
{
    QStringList out;
    const QString lower = text.toLower();
    QString cur;
    auto flush = [&] {
        if (cur.size() >= 2 && cur.size() <= 40 && !stopwords().contains(cur))
            out << cur;
        cur.clear();
    };
    for (QChar c : lower) {
        if (c.isLetterOrNumber() || c == QLatin1Char('_'))
            cur += c;
        else
            flush();
    }
    flush();
    return out;
}

quint32 featureHash(QStringView s, quint32 seed)
{
    return quint32(qHash(s, size_t(seed)));
}

// Word features plus the character 3-grams of each word padded with spaces (" cu", "cub", ...).
void semanticFeatures(const QStringList &words, QHash<quint32, int> *counts)
{
    QString padded;
    for (const QString &w : words) {
        ++(*counts)[featureHash(w, kWordSeed)];
        padded = QLatin1Char(' ') + w + QLatin1Char(' ');
        for (qsizetype i = 0; i + 3 <= padded.size(); ++i)
            ++(*counts)[featureHash(QStringView(padded).mid(i, 3), kGramSeed)];
    }
}

double tfWeight(int tf)
{
    return 1.0 + std::log(double(tf));
}

QString cut(const QString &s, int maxChars)
{
    if (maxChars <= 0)
        return {};
    if (s.size() <= maxChars)
        return s;
    if (maxChars == 1)
        return QString(QChar(0x2026));
    qsizetype at = s.lastIndexOf(QLatin1Char(' '), maxChars - 1);
    if (at < maxChars / 2)
        at = maxChars - 1;
    return s.left(at).trimmed() + QChar(0x2026);
}

QString ageLabel(qint64 ms)
{
    const qint64 m = std::max<qint64>(0, ms) / 60000;
    if (m < 60)
        return QStringLiteral("%1m ago").arg(std::max<qint64>(1, m));
    if (m < 48 * 60)
        return QStringLiteral("%1h ago").arg(m / 60);
    return QStringLiteral("%1d ago").arg(m / (24 * 60));
}

// --- scanning --------------------------------------------------------------------------------

struct Candidate {
    QString path;
    QString harness;
    QString sessionId;
    qint64 mtimeMs = 0;
    qint64 size = 0;
};

QByteArray readRange(const QString &path, qint64 from, qint64 maxBytes)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return {};
    if (from < 0)
        from = std::max<qint64>(0, f.size() - maxBytes);
    f.seek(from);
    return f.read(maxBytes);
}

QString cwdIn(const QByteArray &raw)
{
    static const QRegularExpression re(QStringLiteral("\"cwd\"\\s*:\\s*\"([^\"]+)\""));
    const QRegularExpressionMatch m = re.match(QString::fromUtf8(raw));
    return m.hasMatch() ? m.captured(1) : QString();
}

bool mentionsAny(const QString &hay, const QStringList &parts)
{
    for (const QString &p : parts)
        if (!p.isEmpty() && hay.contains(p, Qt::CaseInsensitive))
            return true;
    return false;
}

// Completed exchanges: a user turn that says something, then the assistant's prose up to the
// next such user turn (tool-result records carry no prose and do not split an exchange).
QVector<MemoryTurn> pairTurns(const QList<Turn> &turns, const Candidate &c, const QString &project)
{
    QVector<MemoryTurn> out;
    QString user, answer;
    auto emitPair = [&] {
        if (!user.isEmpty() && !answer.isEmpty()) {
            MemoryTurn t;
            t.harness = c.harness;
            t.sessionId = c.sessionId;
            t.project = project;
            t.mtimeMs = c.mtimeMs;
            t.turnIdx = int(out.size());
            t.userText = cut(user, kUserChars);
            t.assistantText = cut(answer, kAssistantChars);
            out << t;
        }
        user.clear();
        answer.clear();
    };
    // Harness-injected preambles recorded as user messages (Codex AGENTS.md, environment
    // context). The parser merges them with the request that follows, so cut them out.
    static const QRegularExpression preamble(
        QStringLiteral("# AGENTS\\.md instructions[^\\n]*|<(INSTRUCTIONS|environment_context|user_instructions)>.*?(</\\1>|$)"),
        QRegularExpression::DotMatchesEverythingOption);
    for (const Turn &t : turns) {
        QString text = t.text;
        if (text.trimmed().isEmpty())
            continue;
        if (t.role == QLatin1String("user"))
            text.remove(preamble);
        const QString clean = cleanTurnText(text.left(8000), 8000);
        if (clean.isEmpty())
            continue;
        if (t.role == QLatin1String("user")) {
            emitPair();
            user = clean;
        } else if (!user.isEmpty()) {
            // Keep the latest prose: a long agent turn ends with its answer.
            answer = answer.isEmpty() ? clean : answer + QLatin1Char('\n') + clean;
            if (answer.size() > 4 * kAssistantChars)
                answer = answer.right(2 * kAssistantChars);
        }
    }
    emitPair();
    return out;
}

void addCandidate(QVector<Candidate> *out, const QFileInfo &fi, const QString &harness, const QString &sessionId,
                  qint64 cutoff)
{
    const qint64 m = fi.lastModified().toMSecsSinceEpoch();
    if (m < cutoff || sessionId.isEmpty())
        return;
    out->append(Candidate{fi.absoluteFilePath(), harness, sessionId, m, fi.size()});
}

}  // namespace

struct MemoryIndex::Corpus {
    QVector<MemoryTurn> turns;
    QVector<float> docLen;   // BM25 token count
    QVector<float> docNorm;  // semantic vector L2 norm
    double avgLen = 1;
    QHash<quint32, QVector<Posting>> bm25;      // word -> (doc, tf)
    QHash<quint32, QVector<Posting>> semantic;  // feature -> (doc, tf-idf)
    QHash<quint32, float> semanticIdf;
};

MemorySources MemorySources::defaults()
{
    MemorySources s;
    const QString home = QDir::homePath();
    s.claudeProjects = home + QStringLiteral("/.claude/projects");
    s.codexSessions = home + QStringLiteral("/.codex/sessions");
    s.agyBrain = home + QStringLiteral("/.gemini/antigravity-cli/brain");
    s.phononMeetings = home + QStringLiteral("/Library/Application Support/Phonon/Meetings");
    s.excludePathParts = {QStringLiteral("infinity")};
    return s;
}

MemoryIndex::MemoryIndex(QObject *parent)
    : QObject(parent)
    , m_corpus(build({}))
{
    connect(&m_watcher, &QFutureWatcher<ScanOut>::finished, this, [this] {
        if (m_watcher.isCanceled() || !m_watcher.future().isValid() || m_watcher.future().resultCount() == 0)
            return;
        ScanOut out = m_watcher.result();
        m_corpus = out.corpus;
        m_cache = std::move(out.cache);
        m_lastRefreshMs = out.ms;
        emit refreshed(int(m_corpus->turns.size()), out.ms);
    });
}

MemoryIndex::~MemoryIndex()
{
    m_watcher.disconnect(this);
    m_watcher.waitForFinished();
}

void MemoryIndex::setTurns(const QVector<MemoryTurn> &turns)
{
    m_corpus = build(turns);
}

int MemoryIndex::size() const
{
    return int(m_corpus->turns.size());
}

QVector<MemoryTurn> MemoryIndex::turns() const
{
    return m_corpus->turns;
}

qint64 MemoryIndex::memoryBytes() const
{
    const Corpus &c = *m_corpus;
    qint64 b = 0;
    for (const MemoryTurn &t : c.turns)
        b += 2 * (t.userText.size() + t.assistantText.size() + t.harness.size() + t.sessionId.size() + t.project.size())
             + qint64(sizeof(MemoryTurn));
    for (const auto &p : c.bm25)
        b += p.size() * qint64(sizeof(Posting)) + 48;
    for (const auto &p : c.semantic)
        b += p.size() * qint64(sizeof(Posting)) + 48;
    b += c.semanticIdf.size() * 24 + c.docLen.size() * 8;
    return b;
}

std::shared_ptr<const MemoryIndex::Corpus> MemoryIndex::build(QVector<MemoryTurn> turns)
{
    auto c = std::make_shared<Corpus>();
    c->turns = std::move(turns);
    const int n = int(c->turns.size());
    c->docLen.resize(n);
    c->docNorm.resize(n);
    QVector<QHash<quint32, int>> features(n);
    QHash<quint32, int> df;
    double totalLen = 0;
    for (int d = 0; d < n; ++d) {
        const MemoryTurn &t = c->turns[d];
        const QStringList words = tokenize(t.userText + QLatin1Char('\n') + t.assistantText);
        QHash<quint32, int> tf;
        for (const QString &w : words)
            ++tf[featureHash(w, kWordSeed)];
        for (auto it = tf.cbegin(); it != tf.cend(); ++it)
            c->bm25[it.key()].append(Posting{quint32(d), float(it.value())});
        c->docLen[d] = float(words.size());
        totalLen += words.size();
        semanticFeatures(words, &features[d]);
        for (auto it = features[d].cbegin(); it != features[d].cend(); ++it)
            ++df[it.key()];
    }
    c->avgLen = n > 0 ? std::max(1.0, totalLen / n) : 1.0;
    for (auto it = df.cbegin(); it != df.cend(); ++it)
        c->semanticIdf.insert(it.key(), float(std::log((n + 1.0) / (it.value() + 1.0)) + 1.0));
    for (int d = 0; d < n; ++d) {
        double norm = 0;
        for (auto it = features[d].cbegin(); it != features[d].cend(); ++it) {
            const double w = tfWeight(it.value()) * c->semanticIdf.value(it.key());
            norm += w * w;
            c->semantic[it.key()].append(Posting{quint32(d), float(w)});
        }
        c->docNorm[d] = float(std::sqrt(norm));
    }
    return c;
}

QVector<MemoryHit> MemoryIndex::search(const QString &text, const QString &excludeSessionId, int topK,
                                       qint64 nowMs) const
{
    const std::shared_ptr<const Corpus> corpus = m_corpus;  // a swap mid-query keeps this one alive
    const Corpus &c = *corpus;
    const int n = int(c.turns.size());
    if (n == 0 || topK <= 0)
        return {};
    if (nowMs <= 0)
        nowMs = QDateTime::currentMSecsSinceEpoch();
    const QStringList words = tokenize(text);
    if (words.isEmpty())
        return {};
    auto excluded = [&](int d) {
        return !excludeSessionId.isEmpty() && c.turns[d].sessionId == excludeSessionId;
    };

    // BM25 over word tokens (each distinct query word once).
    QVector<double> bm(n, 0.0);
    QSet<quint32> seen;
    for (const QString &w : words) {
        const quint32 h = featureHash(w, kWordSeed);
        if (seen.contains(h))
            continue;
        seen.insert(h);
        const auto it = c.bm25.constFind(h);
        if (it == c.bm25.cend())
            continue;
        const double df = it->size();
        const double idf = std::log(1.0 + (n - df + 0.5) / (df + 0.5));
        for (const Posting &p : *it) {
            const double tf = p.w;
            const double norm = kBm25K1 * (1.0 - kBm25B + kBm25B * c.docLen[p.doc] / c.avgLen);
            bm[p.doc] += idf * tf * (kBm25K1 + 1.0) / (tf + norm);
        }
    }

    // TF-IDF cosine over words + character 3-grams.
    QHash<quint32, int> qf;
    semanticFeatures(words, &qf);
    QVector<double> dot(n, 0.0);
    double qnorm = 0;
    for (auto it = qf.cbegin(); it != qf.cend(); ++it) {
        const auto idf = c.semanticIdf.constFind(it.key());
        if (idf == c.semanticIdf.cend())
            continue;
        const double qw = tfWeight(it.value()) * *idf;
        qnorm += qw * qw;
        for (const Posting &p : c.semantic.value(it.key()))
            dot[p.doc] += qw * p.w;
    }
    qnorm = std::sqrt(qnorm);

    QVector<int> bmOrder, semOrder;
    QVector<double> cosine(n, 0.0);
    for (int d = 0; d < n; ++d) {
        if (excluded(d))
            continue;
        if (bm[d] > 0)
            bmOrder << d;
        if (qnorm > 0 && c.docNorm[d] > 0) {
            cosine[d] = dot[d] / (qnorm * c.docNorm[d]);
            if (cosine[d] >= kCosineFloor)
                semOrder << d;
        }
    }
    std::sort(bmOrder.begin(), bmOrder.end(), [&](int a, int b) { return bm[a] != bm[b] ? bm[a] > bm[b] : a < b; });
    std::sort(semOrder.begin(), semOrder.end(),
              [&](int a, int b) { return cosine[a] != cosine[b] ? cosine[a] > cosine[b] : a < b; });

    QHash<int, MemoryHit> fused;
    for (int r = 0; r < bmOrder.size(); ++r) {
        MemoryHit &h = fused[bmOrder[r]];
        h.turn = bmOrder[r];
        h.bm25Rank = r + 1;
        h.score += 1.0 / (kRrfK + r + 1);
    }
    for (int r = 0; r < semOrder.size(); ++r) {
        MemoryHit &h = fused[semOrder[r]];
        h.turn = semOrder[r];
        h.semanticRank = r + 1;
        h.score += 1.0 / (kRrfK + r + 1);
    }
    QVector<MemoryHit> hits;
    hits.reserve(fused.size());
    for (MemoryHit h : std::as_const(fused)) {
        const double ageDays = std::max<qint64>(0, nowMs - c.turns[h.turn].mtimeMs) / 86400000.0;
        h.score += kRecencyWeight * std::exp(-ageDays / kRecencyDays);
        hits << h;
    }
    std::sort(hits.begin(), hits.end(), [](const MemoryHit &a, const MemoryHit &b) {
        return a.score != b.score ? a.score > b.score : a.turn < b.turn;
    });
    if (hits.size() > topK)
        hits.resize(topK);
    return hits;
}

QString MemoryIndex::query(const QString &text, const QString &excludeSessionId, int topK, int maxChars,
                           qint64 nowMs) const
{
    const std::shared_ptr<const Corpus> corpus = m_corpus;
    if (nowMs <= 0)
        nowMs = QDateTime::currentMSecsSinceEpoch();
    const QVector<MemoryHit> hits = search(text, excludeSessionId, topK, nowMs);
    QString out;
    for (int i = 0; i < hits.size(); ++i) {
        const MemoryTurn &t = corpus->turns[hits[i].turn];
        const QString sep = out.isEmpty() ? QString() : QStringLiteral("\n\n");
        // An even share of what is left, so a long first hit never starves the rest.
        const int share = int((maxChars - out.size() - sep.size()) / (hits.size() - i));
        // uuids by their head; Phonon "meeting_<ms>" folders by their distinguishing tail.
        const QString shortId = t.sessionId.startsWith(QLatin1String("meeting_")) ? t.sessionId.right(8) : t.sessionId.left(8);
        QString header = QStringLiteral("[%1 %2]").arg(t.harness, shortId);
        if (!t.project.isEmpty())
            header += QLatin1Char(' ') + t.project;
        header += QStringLiteral(", ") + ageLabel(nowMs - t.mtimeMs);
        const QString userTag = QStringLiteral("\nUser: ");
        const QString asstTag = QStringLiteral("\nAssistant: ");
        const bool hasAnswer = !t.assistantText.isEmpty();
        const int body = share - int(header.size() + userTag.size() + (hasAnswer ? asstTag.size() : 0));
        if (body < 40)
            break;
        const int userBudget = hasAnswer ? std::min<int>(int(t.userText.size()), std::max(40, body * 35 / 100)) : body;
        QString block = header + userTag + cut(t.userText, userBudget);
        if (hasAnswer)
            block += asstTag + cut(t.assistantText, body - int(cut(t.userText, userBudget).size()));
        out += sep + block.left(share);
    }
    return out;
}

void MemoryIndex::refreshAsync(const MemorySources &sources)
{
    if (m_watcher.isRunning())
        return;
    const MemoryCache cache = m_cache;
    m_watcher.setFuture(QtConcurrent::run([sources, cache]() mutable {
        QElapsedTimer t;
        t.start();
        ScanOut out;
        out.cache = cache;
        QVector<MemoryTurn> turns = scan(sources, QDateTime::currentMSecsSinceEpoch(), &out.cache);
        out.corpus = build(std::move(turns));
        out.ms = t.nsecsElapsed() / 1e6;
        return out;
    }));
}

QVector<MemoryTurn> MemoryIndex::phononTurns(const QString &transcript, const QString &meetingId, qint64 mtimeMs,
                                             int linesPerTurn)
{
    QVector<MemoryTurn> out;
    QStringList window;
    auto flush = [&] {
        if (window.isEmpty())
            return;
        MemoryTurn t;
        t.harness = QStringLiteral("phonon");
        t.sessionId = meetingId;
        t.project = QStringLiteral("meeting");
        t.mtimeMs = mtimeMs;
        t.turnIdx = int(out.size());
        t.userText = cut(window.join(QLatin1Char('\n')), kUserChars + kAssistantChars);
        out << t;
        window.clear();
    };
    // Skip a "Status: ..." header block when the file has one.
    QString body = transcript;
    const qsizetype rule = body.indexOf(QLatin1String("\n---\n"));
    if (rule >= 0 && rule < 2000)
        body = body.mid(rule + 5);
    for (const QString &l : body.split(QLatin1Char('\n'))) {
        const QString line = l.trimmed();
        if (line.isEmpty())
            continue;
        window << line;
        if (window.size() >= linesPerTurn)
            flush();
    }
    flush();
    return out;
}

QVector<MemoryTurn> MemoryIndex::scan(const MemorySources &s, qint64 nowMs, MemoryCache *cache)
{
    const qint64 cutoff = nowMs - qint64(s.maxAgeDays) * 86400000;
    static const QRegularExpression uuidTail(QStringLiteral("([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"));
    QVector<Candidate> files;
    if (!s.claudeProjects.isEmpty()) {
        const QDir root(s.claudeProjects);
        for (const QString &d : root.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
            const QDir dir(root.filePath(d));
            for (const QFileInfo &fi : dir.entryInfoList({QStringLiteral("*.jsonl")}, QDir::Files))
                addCandidate(&files, fi, QStringLiteral("claude"), fi.completeBaseName(), cutoff);
        }
    }
    if (!s.codexSessions.isEmpty()) {
        QDirIterator it(s.codexSessions, {QStringLiteral("*.jsonl")}, QDir::Files, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            const QFileInfo fi(it.next());
            const QRegularExpressionMatch m = uuidTail.match(fi.completeBaseName());
            addCandidate(&files, fi, QStringLiteral("codex"), m.hasMatch() ? m.captured(1) : fi.completeBaseName(), cutoff);
        }
    }
    if (!s.agyBrain.isEmpty()) {
        const QDir root(s.agyBrain);
        for (const QString &d : root.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
            const QFileInfo fi(root.filePath(d) + QStringLiteral("/.system_generated/logs/transcript.jsonl"));
            if (fi.exists())
                addCandidate(&files, fi, QStringLiteral("agy"), d, cutoff);
        }
    }
    if (!s.phononMeetings.isEmpty()) {
        const QDir root(s.phononMeetings);
        for (const QString &d : root.entryList({QStringLiteral("meeting_*")}, QDir::Dirs | QDir::NoDotAndDotDot)) {
            const QFileInfo fi(root.filePath(d) + QStringLiteral("/transcript.txt"));
            if (fi.exists())
                addCandidate(&files, fi, QStringLiteral("phonon"), d, cutoff);
        }
    }
    std::sort(files.begin(), files.end(), [](const Candidate &a, const Candidate &b) {
        return a.mtimeMs != b.mtimeMs ? a.mtimeMs > b.mtimeMs : a.path < b.path;
    });

    MemoryCache next;
    QVector<MemoryTurn> out;
    for (const Candidate &c : files) {
        if (out.size() >= s.maxTurns)
            break;
        if (s.excludeSessionIds.contains(c.sessionId) || mentionsAny(c.path, s.excludePathParts))
            continue;
        QVector<MemoryTurn> turns;
        const auto hit = cache ? cache->constFind(c.path) : MemoryCache::const_iterator();
        if (cache && hit != cache->cend() && hit->mtimeMs == c.mtimeMs && hit->size == c.size) {
            turns = hit->turns;
        } else if (c.harness == QLatin1String("phonon")) {
            turns = phononTurns(QString::fromUtf8(readRange(c.path, -1, s.tailBytes)), c.sessionId, c.mtimeMs);
        } else {
            const QByteArray raw = readRange(c.path, -1, s.tailBytes);
            QString cwd = cwdIn(raw);
            if (cwd.isEmpty() && c.size > s.tailBytes)
                cwd = cwdIn(readRange(c.path, 0, 64 * 1024));
            // No recorded cwd (Antigravity): an excluded path anywhere in the head excludes it.
            const bool redacted = cwd.isEmpty()
                                      ? mentionsAny(QString::fromUtf8(readRange(c.path, 0, 64 * 1024)), s.excludePathParts)
                                      : mentionsAny(cwd, s.excludePathParts);
            if (!redacted) {
                const QList<Turn> parsed = parseTranscript(raw, harnessFromName(c.harness));
                turns = pairTurns(parsed, c, cwd.isEmpty() ? QString() : QFileInfo(cwd).fileName());
            }
        }
        if (cache)
            next.insert(c.path, MemoryFileCache{c.mtimeMs, c.size, turns});
        const qsizetype take = std::min<qsizetype>(turns.size(), s.maxTurns - out.size());
        for (qsizetype i = turns.size() - take; i < turns.size(); ++i)
            out << turns[i];
    }
    if (cache)
        *cache = std::move(next);
    return out;
}

}  // namespace studio
