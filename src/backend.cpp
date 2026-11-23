#include "backend.hpp"

#include "fixtures.hpp"
#include "ingest/ambient.hpp"
#include "mac/cua_executor.hpp"
#include "mac/cua_spotlight.hpp"
#include "pdfrender.hpp"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QThread>
#include <QUrl>
#include <QUuid>
#include <QtConcurrent>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>

namespace studio {

namespace {

QString home() { return QDir::homePath(); }
QString lapisPeers() { return home() + QStringLiteral("/.local/bin/lapis-peers"); }
QString claudeBin() { return home() + QStringLiteral("/.local/bin/claude"); }
QString agyBin() { return home() + QStringLiteral("/.local/bin/agy"); }
QString workspacePath() { return home() + QStringLiteral("/.lapis/runtime/workspace.json"); }
// The quick ask's router stream id (askStream() refuses it, so the two never collide).
constexpr char kQuickStreamId[] = "__quick__";
constexpr char kReflectionStreamId[] = "reflection:1";
constexpr char kAmbientStreamId[] = "__ambient__";
// The MacBook-wide Spotlight bar's conversation stream (multi-turn: Tier 2 history per stream).
constexpr char kSpotlightStreamId[] = "spotlight";
// How long the Phonon fn key-down waits after the bar had to activate this app, so Phonon's
// frontmost-app / focused-field snapshot lands on the bar's prompt, not the app behind it.
constexpr int kSpotlightVoiceDelayMs = 45;

// The Jarvis interrupt's cancel latch. studio::cua::cancelFlag aborts an in-flight step within
// milliseconds, but it is one process-wide flag: left set, it would kill the next turn's steps
// too. Each interrupt bumps the epoch under the lock; a queued step from an older epoch never
// runs, and the first step of the new epoch clears a latch whose step has already returned (the
// pool runs one step at a time). onCuaActionDone clears it as soon as the queue drains.
std::mutex g_cuaEpochMx;
std::atomic<quint64> g_cuaEpoch{0};
void latchCuaCancel()
{
    const std::lock_guard<std::mutex> lock(g_cuaEpochMx);
    studio::cua::cancelFlag.store(true);
    g_cuaEpoch.fetch_add(1);
}
// Worker thread, before a step runs: false when an interrupt superseded it.
bool claimCuaEpoch(quint64 epoch)
{
    const std::lock_guard<std::mutex> lock(g_cuaEpochMx);
    if (epoch != g_cuaEpoch.load())
        return false;
    studio::cua::cancelFlag.store(false);
    return true;
}
void releaseCuaCancel()
{
    const std::lock_guard<std::mutex> lock(g_cuaEpochMx);
    studio::cua::cancelFlag.store(false);
}
// Tier 2 cap per conversation stream, in messages (user + assistant pairs).
constexpr int kMaxStreamHistory = 24;

QString expandHome(const QString &p)
{
    if (p == QLatin1String("~"))
        return home();
    if (p.startsWith(QLatin1String("~/")))
        return home() + p.mid(1);
    return p;
}

QString tildePath(const QString &p)
{
    if (p == home())
        return QStringLiteral("~");
    if (p.startsWith(home() + QLatin1Char('/')))
        return QStringLiteral("~") + p.mid(home().size());
    return p;
}

QString cswapHookBin() { return home() + QStringLiteral("/dev/claude-swap-personal/.venv/bin/cswap-private"); }
QString codexBin()
{
    const QString local = home() + QStringLiteral("/.local/bin/codex");
    return QFileInfo(local).isExecutable() ? local : QStringLiteral("/opt/homebrew/bin/codex");
}

const char *kQuickSystemPrompt =
    "You answer quick questions about what is on the user's screen in lapis studio. "
    "Reply with one crisp line under 20 words: the answer first, numbers when they help. "
    "No preamble, no lists, no TLDR line.";

// With --setting-sources "" Claude skips ~/.claude/settings.json entirely, including the env block
// that routes it through the local cswap proxy. Carry over only those two routing variables, so the
// helper still authenticates without loading any hooks, plugins, MCP servers or permissions.
QProcessEnvironment quickHelperEnv()
{
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    QFile f(home() + QStringLiteral("/.claude/settings.json"));
    if (f.open(QIODevice::ReadOnly)) {
        const QJsonObject e = QJsonDocument::fromJson(f.readAll()).object().value(QLatin1String("env")).toObject();
        for (const char *k : {"ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN"})
            if (e.contains(QLatin1String(k)))
                env.insert(QString::fromLatin1(k), e.value(QLatin1String(k)).toString());
    }
    // A helper launched from inside a Claude Code session must not look like its child.
    for (const char *k : {"CLAUDE_CODE_CHILD_SESSION", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_MESSAGING_SOCKET",
                          "CLAUDE_CODE_MESSAGING_TOKEN", "CLAUDE_CODE_ENTRYPOINT"})
        env.remove(QString::fromLatin1(k));
    return env;
}

int statusRank(const QString &s)
{
    if (s == QLatin1String("needs-ok")) return 4;
    if (s == QLatin1String("working")) return 3;
    if (s == QLatin1String("finished")) return 2;
    if (s == QLatin1String("idle")) return 1;
    return 0;
}

QString oneLine(QString s, int maxWords)
{
    QStringList keep;
    for (const QString &l : s.split(QLatin1Char('\n'))) {
        const QString t = l.trimmed();
        if (t.isEmpty() || t.startsWith(QLatin1String("TLDR")))
            continue;
        keep << t;
    }
    return clampWords(keep.join(QLatin1Char(' ')), maxWords);
}

// A quick answer as the answer strip renders it: markdown kept (line breaks, code fences),
// tool noise gone, capped so a runaway reply never floods the strip.
QString answerBody(const QString &s)
{
    QString out = stripToolNoise(s).trimmed();
    if (out.size() > 6000)
        out = out.left(6000) + QStringLiteral("…");
    return out;
}

// --- transcript discovery (worker thread, live mode only) -----------------------

QString findTranscript(const SessionInfo &s)
{
    static const QRegularExpression uuid(QStringLiteral("^[0-9a-fA-F-]{36}$"));
    if (!uuid.match(s.ident).hasMatch())
        return {};
    if (s.harness == QLatin1String("agy")) {
        const QString p = home() + QStringLiteral("/.gemini/antigravity-cli/brain/") + s.ident
            + QStringLiteral("/.system_generated/logs/transcript.jsonl");
        return QFileInfo::exists(p) ? p : QString();
    }
    if (s.harness == QLatin1String("codex")) {
        QDirIterator it(home() + QStringLiteral("/.codex/sessions"),
                        {QStringLiteral("rollout-*") + s.ident + QStringLiteral(".jsonl")}, QDir::Files,
                        QDirIterator::Subdirectories);
        QString best;
        qint64 bestM = 0;
        while (it.hasNext()) {
            const QString p = it.next();
            const qint64 m = QFileInfo(p).lastModified().toMSecsSinceEpoch();
            if (m > bestM) {
                bestM = m;
                best = p;
            }
        }
        return best;
    }
    const QDir projects(home() + QStringLiteral("/.claude/projects"));
    QString best;
    qint64 bestM = 0;
    for (const QString &d : projects.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        const QString p = projects.filePath(d) + QLatin1Char('/') + s.ident + QStringLiteral(".jsonl");
        const QFileInfo fi(p);
        if (fi.exists() && fi.lastModified().toMSecsSinceEpoch() > bestM) {
            bestM = fi.lastModified().toMSecsSinceEpoch();
            best = p;
        }
    }
    return best;
}

QByteArray tailFile(const QString &path, qint64 maxBytes)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return {};
    const qint64 size = f.size();
    if (size > maxBytes)
        f.seek(size - maxBytes);
    return f.readAll();
}

struct ScanResult {
    QList<SessionInfo> sessions;
    QHash<QString, ParsedCache> cache;
    QStringList read;
    double ms = 0;  // scan wall time
    bool onGuiThread = false;
};

ScanResult scanWorker(QList<SessionInfo> sessions, QHash<QString, ParsedCache> cache)
{
    QElapsedTimer clock;
    clock.start();
    ScanResult r;
    for (SessionInfo &s : sessions) {
        // infinity sessions are parsed too: this is the user's own local window, not a board broadcast,
        // so their charts, tables and prose show when he zooms in. Only output that leaves the
        // process (--probe-live prints) keeps them redacted.
        if (s.ident.isEmpty() && s.harness == QLatin1String("claude")) {
            QProcess p;  // read-only lookup of a brand-new Claude session id
            p.start(lapisPeers(), {QStringLiteral("session-of"), s.id});
            if (p.waitForFinished(4000))
                s.ident = QString::fromUtf8(p.readAllStandardOutput()).trimmed();
        }
        if (s.transcript.isEmpty() || !QFileInfo::exists(s.transcript))
            s.transcript = findTranscript(s);
        if (s.transcript.isEmpty())
            continue;
        const QFileInfo fi(s.transcript);
        s.mtime = fi.lastModified().toMSecsSinceEpoch();
        ParsedCache c = cache.value(s.transcript);
        if (c.mtime != s.mtime || c.size != fi.size()) {
            const QList<Turn> turns = parseTranscript(tailFile(s.transcript, 3 * 1024 * 1024), harnessFromName(s.harness));
            const QList<uint> before = c.ex.tailSigs;
            c.mtime = s.mtime;
            c.size = fi.size();
            // The artifact deck, params and job name are re-synthesized here, on the worker.
            c.ex = extractArtifacts(turns);
            c.ex.jobName = synthesizeJobName(turns, c.ex.goal, s.title);  // the tab title can name the job
            c.ex.added = newTurnsSince(before, c.ex.tailSigs);
            r.read << s.transcript;
        }
        r.cache.insert(s.transcript, c);
    }
    r.sessions = sessions;
    r.ms = clock.nsecsElapsed() / 1e6;
    r.onGuiThread = QThread::currentThread() == QCoreApplication::instance()->thread();
    return r;
}

}  // namespace

// --- project grouping -----------------------------------------------------------------

bool isScratchDir(const QString &dir)
{
    const QString d = QDir::cleanPath(expandHome(dir.trimmed()));
    if (d.isEmpty() || d == QLatin1String(".") || d == QDir::cleanPath(home()) || d == QLatin1String("/"))
        return true;
    const QString tmp = QDir::cleanPath(QDir::tempPath());
    return d == QLatin1String("/tmp") || d == QLatin1String("/private/tmp") || d == tmp
        || d == QFileInfo(tmp).canonicalFilePath();
}

QString slugFromTitle(const QString &title)
{
    // Filler that never names a project: articles, prepositions, and generic task verbs.
    static const QSet<QString> stop = {
        "a", "an", "the", "to", "of", "for", "on", "in", "at", "by", "with", "and", "or", "from", "into", "about",
        "is", "are", "be", "it", "this", "that", "my", "our", "your", "me", "send", "sending", "sent", "run", "runs",
        "running", "review", "reviews", "setup", "set", "up", "test", "tests", "testing", "fix", "fixes", "check",
        "look", "new", "done", "can", "close", "please", "quick", "help", "vs", "via", "using", "some", "more", "do",
        "make", "get", "try", "blind", "development", "dev", "work", "task", "session", "chat", "notes"};
    static const QRegularExpression wordRe(QStringLiteral("[A-Za-z0-9]+(?:\\.[0-9]+)*"));
    struct W {
        QString orig;
        QString low;
        int pos;
    };
    QList<W> all;
    for (auto it = wordRe.globalMatch(title); it.hasNext();) {
        const QString w = it.next().captured(0);
        all << W{w, w.toLower(), int(all.size())};
    }
    // Sentence case ("Photos to send Alex") marks a capitalised later word as a name, which
    // leads the slug; title case ("Anthropic Interview Practice") capitalises everything, so there
    // the words keep their order.
    static const QRegularExpression alpha(QStringLiteral("^[A-Za-z]+$"));
    static const QRegularExpression name(QStringLiteral("^[A-Z][a-z]+$"));
    int lower = 0, upper = 0;
    for (const W &w : all) {
        if (w.pos == 0 || !alpha.match(w.orig).hasMatch())
            continue;
        (w.orig.at(0).isUpper() ? upper : lower) += 1;
    }
    const bool sentenceCase = lower > upper;
    QList<W> names, rest;
    for (const W &w : all) {
        if (stop.contains(w.low))
            continue;
        if (sentenceCase && w.pos > 0 && name.match(w.orig).hasMatch())
            names << w;
        else
            rest << w;
    }
    const QList<W> words = names + rest;
    QStringList out;
    for (const W &w : words) {
        // Two words; a bare version number ("GPT 6.2 ...") earns a third.
        const bool version = !out.isEmpty() && QRegularExpression(QStringLiteral("^[0-9.]+$")).match(out.last()).hasMatch();
        if (out.size() >= 2 && !(out.size() == 2 && version))
            break;
        if (!out.contains(w.low))
            out << w.low;
    }
    return out.join(QLatin1Char('-'));
}

ProjectId projectIdFor(const QString &directory, const QString &title, const QString &category)
{
    const QString d = QDir::cleanPath(expandHome(directory.trimmed()));
    if (!isScratchDir(d)) {
        const QString base = QFileInfo(d).fileName();
        return ProjectId{base, base, false};
    }
    QString slug = slugFromTitle(title);
    // A tab still named after the folder ("user") says nothing; its Lapis category does. Live
    // sessions never land here: sessionProjectId() gives each such tab its own island.
    if (isGenericTabTitle(title))
        slug = category.toLower().split(QRegularExpression(QStringLiteral("[^a-z0-9]+")), Qt::SkipEmptyParts)
                   .mid(0, 3).join(QLatin1Char('-'));
    if (slug.isEmpty())
        slug = QStringLiteral("scratch");
    return ProjectId{QStringLiteral("~") + slug, slug, true};
}

bool isGenericTabTitle(const QString &title)
{
    const QString slug = slugFromTitle(title);
    return slug.isEmpty() || slug == QFileInfo(home()).fileName().toLower() || slug == QLatin1String("tmp");
}

ProjectId sessionProjectId(const SessionInfo &s, const Extracted &ex, const QString &boardSnippet)
{
    ProjectId pid = projectIdFor(s.directory, s.title, s.category);
    if (!pid.fromTitle || !isGenericTabTitle(s.title) || s.id.isEmpty())
        return pid;
    pid.key = QStringLiteral("~tab:") + s.id.left(8);
    QString name = ex.promptName;
    if (name.isEmpty())
        name = ex.jobName;
    if (name.isEmpty() && !boardSnippet.isEmpty()) {
        // The agent's latest reply is a statement, not a request: its opening words, clipped.
        static const QRegularExpression firstSentence(QStringLiteral("^(.+?)([.!?:;](\\s|$)|$)"));
        static const QRegularExpression noise(QStringLiteral("[`*_#>\\[\\]()~]"));
        QString first = firstSentence.match(boardSnippet.simplified()).captured(1);
        first.replace(noise, QStringLiteral(" "));
        name = clampWords(first.simplified(), 5);
    }
    pid.name = name.isEmpty() ? QStringLiteral("tab ") + s.id.left(8) : name;
    return pid;
}

QList<BoardRow> parseBoardJson(const QByteArray &json)
{
    QList<BoardRow> out;
    const QJsonObject b = QJsonDocument::fromJson(json).object();
    for (const QJsonValue &c : b.value(QLatin1String("categories")).toArray())
        for (const QJsonValue &a : c.toObject().value(QLatin1String("agents")).toArray()) {
            const QJsonObject ao = a.toObject();
            BoardRow r;
            r.id = ao.value(QLatin1String("id")).toString();
            r.status = ao.value(QLatin1String("status")).toString();
            // Never keep text the board marks as withheld (infinity).
            if (!ao.value(QLatin1String("infinityRedacted")).toBool()) {
                r.snippet = ao.value(QLatin1String("lastReplySnippet")).toString();
                if (r.snippet.isEmpty())
                    r.snippet = ao.value(QLatin1String("snippet")).toString();
                r.snippet = cleanTurnText(r.snippet, 400);
            }
            if (!r.id.isEmpty())
                out << r;
        }
    return out;
}

QString elapsedLabel(qint64 ms)
{
    const qint64 m = std::max<qint64>(0, ms) / 60000;
    if (m < 1)
        return QStringLiteral("just now");
    if (m < 60)
        return QStringLiteral("%1m").arg(m);
    const qint64 h = m / 60;
    if (h < 24)
        return m % 60 ? QStringLiteral("%1h %2m").arg(h).arg(m % 60) : QStringLiteral("%1h").arg(h);
    return h % 24 ? QStringLiteral("%1d %2h").arg(h / 24).arg(h % 24) : QStringLiteral("%1d").arg(h / 24);
}

namespace {
// Sentences of prose, in order: code fences, markdown tables, headings and TLDR lines dropped;
// a blank line, list item or line ending in punctuation ends a sentence.
QStringList sentencesOf(const QString &raw)
{
    QString s = raw;
    static const QRegularExpression fence(QStringLiteral("```[\\s\\S]*?(```|$)"));
    s.remove(fence);
    static const QRegularExpression noise(QStringLiteral("[*`]+|^>\\s*"));
    static const QRegularExpression bullet(QStringLiteral("^([-+]|\\d+[.)])\\s+"));
    static const QRegularExpression split(QStringLiteral("(?<=[.!?])\\s+(?=[A-Z0-9\"'(])"));
    QStringList paras;
    QString cur;
    auto flush = [&] {
        if (!cur.trimmed().isEmpty())
            paras << cur.simplified();
        cur.clear();
    };
    for (QString line : s.split(QLatin1Char('\n'))) {
        line = line.trimmed();
        if (line.startsWith(QLatin1Char('|')) || line.startsWith(QLatin1Char('#')) || line.startsWith(QLatin1String("TLDR"))
            || line.startsWith(QLatin1String("---"))) {
            flush();
            continue;
        }
        if (line.isEmpty() || bullet.match(line).hasMatch())
            flush();
        line.remove(bullet);
        line = stripMarkdownLinks(line);
        if (line.contains(QLatin1Char('<')))
            line = stripMarkup(line);
        line.replace(noise, QString());
        cur += QLatin1Char(' ') + line;
    }
    flush();
    QStringList out;
    for (const QString &p : std::as_const(paras))
        for (const QString &part : p.split(split, Qt::SkipEmptyParts)) {
            const QString t = part.trimmed();
            if (t.size() > 3)
                out << t;
        }
    return out;
}
// One short line: <= maxWords words, a trailing period dropped, an ellipsis when cut.
QString shortLine(const QString &s, int maxWords)
{
    const QString c = clampWords(s, maxWords);
    QString out = c.size() < s.simplified().size() ? c + QChar(0x2026) : c;
    if (out.endsWith(QLatin1Char('.')) && !out.endsWith(QLatin1String("..")))
        out.chop(1);
    return out;
}
QString lastAssistantTurn(const QStringList &recent)
{
    for (qsizetype i = recent.size() - 1; i >= 0; --i)
        if (recent[i].startsWith(QLatin1String("Assistant: ")))
            return recent[i].mid(11);
    return QString();
}
}  // namespace

IslandState islandState(const Extracted &ex, const QString &status)
{
    IslandState st;
    if (!ex.jobName.isEmpty())
        st.now = stripMarkdownLinks(ex.jobName);
    else if (!ex.goal.isEmpty())
        st.now = shortLine(sentencesOf(ex.goal).value(0, ex.goal), 9);
    // The latest assistant word: the full write-up when there is one, else the clipped turn.
    const QString latest = !ex.prose.isEmpty() ? ex.prose : lastAssistantTurn(ex.recentTurns);
    const QStringList sents = sentencesOf(latest);
    QString question;
    for (qsizetype i = sents.size() - 1; i >= 0 && question.isEmpty(); --i)
        if (sents[i].endsWith(QLatin1Char('?')))
            question = sents[i];
    // A blocker only when the newest turn is the agent's (a question already answered is not one).
    const bool agentSpokeLast = ex.recentTurns.isEmpty() || ex.recentTurns.last().startsWith(QLatin1String("Assistant: "));
    // The island stays sparse (<= 9 words a line); the briefing gets the longer cut.
    QString full;
    if (status == QLatin1String("needs-ok") || (!question.isEmpty() && agentSpokeLast)) {
        st.blocker = true;
        full = !question.isEmpty() ? question : QStringLiteral("Waiting on your OK to continue");
    } else if (!sents.isEmpty()) {
        full = sents.first();
    } else if (!ex.toolCalls.isEmpty()) {
        full = QStringLiteral("Last step: ") + ex.toolCalls.last();
    }
    st.last = shortLine(full, 9);
    st.lastLong = shortLine(full, 16);
    if (st.now.isEmpty() && !st.blocker) {
        st.now = st.last;
        st.last.clear();
    }
    return st;
}

QVariantList localBriefing(const QVariantList &projects)
{
    struct Row {
        int rank;
        int fresh;
        int index;
        QVariantMap m;
    };
    QList<Row> rows;
    for (int i = 0; i < projects.size(); ++i) {
        const QVariantMap p = projects[i].toMap();
        const QString status = p.value(QStringLiteral("status")).toString();
        const int fresh = p.value(QStringLiteral("fresh")).toInt();
        const bool blocker = p.value(QStringLiteral("blocker")).toBool() || status == QLatin1String("needs-ok");
        const bool active = p.value(QStringLiteral("active")).toBool();
        if (!blocker && !active && fresh <= 0)
            continue;  // unchanged and idle: not in the briefing
        const QString name = p.value(QStringLiteral("name")).toString();
        const QString now = p.value(QStringLiteral("nowLine")).toString();
        const QString last = p.contains(QStringLiteral("lastLong")) ? p.value(QStringLiteral("lastLong")).toString()
                                                                     : p.value(QStringLiteral("lastLine")).toString();
        const QString happened = !blocker && !last.isEmpty() ? last : (!now.isEmpty() ? now : QStringLiteral("Work in progress"));
        // Proof: the metric the transcript states (parts the line already says are dropped), else
        // the newest artifact, plus the new-turn count.
        QStringList parts;
        for (const QString &h : p.value(QStringLiteral("headline")).toString().split(QStringLiteral(" · "), Qt::SkipEmptyParts))
            if (!happened.contains(h.trimmed()))
                parts << h.trimmed();
        QString proof = parts.join(QStringLiteral(" · "));
        const QVariantList arts = p.value(QStringLiteral("artifacts")).toList();
        if (proof.isEmpty() && p.value(QStringLiteral("headline")).toString().isEmpty() && !arts.isEmpty())
            proof = arts.first().toMap().value(QStringLiteral("title")).toString();
        if (fresh > 0)
            proof += (proof.isEmpty() ? QString() : QStringLiteral(" · ")) + QStringLiteral("%1 new turn%2").arg(fresh).arg(fresh == 1 ? "" : "s");
        QString line1 = proof.isEmpty() ? happened : happened + QStringLiteral(" · ") + proof;
        QString line2;
        if (blocker)
            line2 = QStringLiteral("Decide: ") + (last.isEmpty() ? QStringLiteral("give the OK to continue") : last);
        else if (!arts.isEmpty())
            line2 = QStringLiteral("Review the ") + arts.first().toMap().value(QStringLiteral("title")).toString().toLower();
        else if (active)
            line2 = QStringLiteral("Check in on ") + (now.isEmpty() ? name : now.left(1).toLower() + now.mid(1));
        else
            line2 = QStringLiteral("Read the latest turn");
        QVariantMap m;
        m[QStringLiteral("key")] = p.value(QStringLiteral("key"));
        m[QStringLiteral("name")] = name;
        m[QStringLiteral("index")] = i;
        m[QStringLiteral("line1")] = line1;
        m[QStringLiteral("line2")] = line2;
        m[QStringLiteral("blocker")] = blocker;
        rows.append({blocker ? 0 : (fresh > 0 ? 1 : 2), fresh, i, m});
    }
    std::stable_sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
        return a.rank != b.rank ? a.rank < b.rank : a.fresh > b.fresh;
    });
    QVariantList out;
    for (const Row &r : rows)
        out << r.m;
    return out;
}

int mergeBriefingReply(QVariantList &rows, const QString &reply)
{
    int merged = 0;
    for (const QString &raw : reply.split(QLatin1Char('\n'), Qt::SkipEmptyParts)) {
        QString l = raw.trimmed();
        while (l.startsWith(QLatin1Char('-')) || l.startsWith(QLatin1Char('*')))
            l = l.mid(1).trimmed();
        const QStringList parts = l.split(QLatin1Char('|'));
        if (parts.size() < 3)
            continue;
        const QString key = parts[0].trimmed();
        const QString l1 = shortLine(parts[1].trimmed(), 22);
        const QString l2 = shortLine(parts.mid(2).join(QLatin1Char('|')).trimmed(), 16);
        if (key.isEmpty() || l1.isEmpty() || l2.isEmpty())
            continue;
        for (QVariant &v : rows) {
            QVariantMap m = v.toMap();
            if (m.value(QStringLiteral("key")).toString() != key)
                continue;
            m[QStringLiteral("line1")] = l1;
            m[QStringLiteral("line2")] = l2;
            v = m;
            ++merged;
            break;
        }
    }
    return merged;
}

// --- helper command lines -----------------------------------------------------------------

// --- chat model catalog ----------------------------------------------------------------------

const QList<ModelSpec> &modelCatalog()
{
    // The curated lineup, in picker order. Efforts per model: agy's own `agy models` list for
    // Gemini (hydrogen and neon have xhigh, carbon stops at high); Claude Code's --effort levels
    // for Claude; codex reasoning efforts for GPT-6.
    static const QStringList lmh = {QStringLiteral("low"), QStringLiteral("medium"), QStringLiteral("high")};
    static const QStringList all = lmh + QStringList{QStringLiteral("xhigh")};
    // Speed classes are what a quick ask feels like: Fast = first token in ~1-3 s; Deep Think =
    // a heavy thinking model (gemini-4-hydrogen: 31.8 s median at high, 4.3 s cold at low), whose
    // agy stream sends no thinking text, only step events, so the UI shows a live phase instead.
    static const QList<ModelSpec> c = {
        // "fast" = reasoning effort "none" on OpenRouter, the default; low..high opt into thinking.
        {QStringLiteral("deepseek/deepseek-v4.1-flash-1017"), QStringLiteral("openrouter"), QStringLiteral("DeepSeek V4.1 Flash"),
         QStringList{QStringLiteral("fast")} + lmh, QStringLiteral("Fast")},
        // "fast" = no thinking (enable_thinking false), the default; low / medium opt into it.
        {QStringLiteral("qwen3.8-flash-next"), QStringLiteral("openai-compat"), QStringLiteral("Qwen3.8 Flash Next"),
         QStringList{QStringLiteral("fast"), QStringLiteral("low"), QStringLiteral("medium")}, QStringLiteral("Fast")},
        {QStringLiteral("claude-fable-5-2"), QStringLiteral("anthropic"), QStringLiteral("Fable 5.2"), all, QStringLiteral("Frontier")},
        {QStringLiteral("claude-opus-5-5"), QStringLiteral("anthropic"), QStringLiteral("Opus 5.5"), all, QStringLiteral("Frontier")},
        {QStringLiteral("claude-sonnet-5-5"), QStringLiteral("anthropic"), QStringLiteral("Sonnet 5.5"), all, QStringLiteral("Fast")},
        {QStringLiteral("gpt-6-astra"), QStringLiteral("openai"), QStringLiteral("GPT 6.2 Astra"), all, QStringLiteral("Frontier")},
        {QStringLiteral("gpt-6-sol"), QStringLiteral("openai"), QStringLiteral("GPT 6.2 Sol"), all, QStringLiteral("Balanced")},
        {QStringLiteral("gemini-4-carbon"), QStringLiteral("google"), QStringLiteral("Gemini 4 Carbon"), lmh, QStringLiteral("Teach")},
        {QStringLiteral("gemini-4-neon"), QStringLiteral("google"), QStringLiteral("Gemini 4 Neon"), all, QStringLiteral("Balanced")},
        {QStringLiteral("gemini-4-hydrogen"), QStringLiteral("google"), QStringLiteral("Gemini 4 Hydrogen"), all, QStringLiteral("Deep Think")},
    };
    return c;
}

QString modelDisplayLabel(const QString &model)
{
    for (const ModelSpec &m : modelCatalog())
        if (m.id == model)
            return m.speed.isEmpty() ? m.label : m.label + QStringLiteral(" · ") + m.speed;
    return model;
}

QString tailWords(const QString &s, int maxWords)
{
    const QStringList w = s.simplified().split(QLatin1Char(' '), Qt::SkipEmptyParts);
    if (w.size() <= maxWords)
        return w.join(QLatin1Char(' '));
    return QStringLiteral("… ") + w.mid(w.size() - maxWords).join(QLatin1Char(' '));
}

StreamProgress agyProgress(const QJsonObject &o)
{
    StreamProgress p;
    const QString ev = o.value(QLatin1String("event")).toString();
    if (ev == QLatin1String("step_update")) {
        const QJsonObject su = o.value(QLatin1String("step_update")).toObject();
        const QString type = su.value(QLatin1String("step_type")).toString();
        const bool active = su.value(QLatin1String("state")).toString() == QLatin1String("ACTIVE");
        for (const char *k : {"thinking_delta", "thinking"})
            p.thinking += su.value(QLatin1String(k)).toString();
        p.text = su.value(QLatin1String("text_delta")).toString();
        if (type == QLatin1String("user_input"))
            p.phase = QStringLiteral("thinking");  // the prompt is in: the model is on it
        else if (type == QLatin1String("tool") && active)
            p.phase = QStringLiteral("running ") + su.value(QLatin1String("tool_name")).toString(QStringLiteral("a tool"));
        else if (type == QLatin1String("tool"))
            p.phase = QStringLiteral("thinking");  // tool done: back to the model
        else if (type == QLatin1String("agent_response") && !p.text.trimmed().isEmpty())
            p.phase = QStringLiteral("writing");
        const QJsonValue tt = su.value(QLatin1String("usage")).toObject().value(QLatin1String("thinking_tokens"));
        if (tt.isDouble())
            p.thinkingTokens = tt.toInt();
    } else if (ev == QLatin1String("result")) {
        const QJsonValue tt = o.value(QLatin1String("result")).toObject().value(QLatin1String("usage")).toObject()
                                  .value(QLatin1String("thinking_tokens"));
        if (tt.isDouble())
            p.thinkingTokens = tt.toInt();
    }
    return p;
}

StreamProgress claudeProgress(const QJsonObject &o)
{
    StreamProgress p;
    const QString type = o.value(QLatin1String("type")).toString();
    if (type == QLatin1String("system") && o.value(QLatin1String("subtype")).toString() == QLatin1String("init"))
        return p;
    if (type != QLatin1String("stream_event"))
        return p;
    const QJsonObject e = o.value(QLatin1String("event")).toObject();
    const QString et = e.value(QLatin1String("type")).toString();
    if (et == QLatin1String("message_start")) {
        p.phase = QStringLiteral("thinking");
    } else if (et == QLatin1String("content_block_start")) {
        const QString bt = e.value(QLatin1String("content_block")).toObject().value(QLatin1String("type")).toString();
        p.phase = bt == QLatin1String("text") ? QStringLiteral("writing") : QStringLiteral("thinking");
    } else if (et == QLatin1String("content_block_delta")) {
        const QJsonObject d = e.value(QLatin1String("delta")).toObject();
        const QString dt = d.value(QLatin1String("type")).toString();
        if (dt == QLatin1String("thinking_delta"))
            p.thinking = d.value(QLatin1String("thinking")).toString();
        else if (dt == QLatin1String("text_delta"))
            p.text = d.value(QLatin1String("text")).toString();
    }
    return p;
}

namespace {
const ModelSpec *specFor(const QString &model)
{
    for (const ModelSpec &m : modelCatalog())
        if (m.id == model)
            return &m;
    return nullptr;
}
int effortRank(const QString &e)
{
    static const QStringList order = {QStringLiteral("fast"), QStringLiteral("low"), QStringLiteral("medium"),
                                      QStringLiteral("high"), QStringLiteral("xhigh")};
    return int(order.indexOf(e));
}
}  // namespace

QString providerFor(const QString &model)
{
    const ModelSpec *m = specFor(model);
    return m ? m->provider : QString();
}

QStringList effortsFor(const QString &model)
{
    const ModelSpec *m = specFor(model);
    return m ? m->efforts : QStringList();
}

QString clampEffort(const QString &model, const QString &effort)
{
    const QStringList e = effortsFor(model);
    if (e.isEmpty() || e.contains(effort))
        return effort;
    const int want = std::max(0, effortRank(effort));
    QString best = e.first();
    int bestD = 99;
    for (const QString &x : e) {
        const int d = std::abs(effortRank(x) - want);
        if (d < bestD) {
            bestD = d;
            best = x;
        }
    }
    return best;
}

QString agyModelId(const QString &model, const QString &effort)
{
    return model + QLatin1Char('-') + clampEffort(model, effort);
}

QString channelKeyFor(const QString &model, const QString &effort)
{
    const QString p = providerFor(model);
    if (p == QLatin1String("openai"))
        return QStringLiteral("openai");  // one app-server: model and effort ride on each turn/start
    if (p == QLatin1String("google"))
        return QStringLiteral("google:") + agyModelId(model, effort);
    if (p == QLatin1String("anthropic"))
        return QStringLiteral("anthropic:%1:%2").arg(model, clampEffort(model, effort));
    if (p == QLatin1String("openai-compat") || p == QLatin1String("openrouter"))
        return p + QLatin1Char(':') + model;
    return {};
}

// kCompatSystemPrompt, the tiered request builders and parseOpenAiSseLine live in lapis_llm.

QByteArray buildFilterPrompt(const QString &model, const QString &goal, const QVariantList &params,
                             const QStringList &recentActions, const QString &latestTurn,
                             const QStringList &tasteRules)
{
    static const QString kSys = QStringLiteral(
        "You filter a coding agent's latest turn for the human who directs it. Drop boilerplate, routine "
        "steps (reading files, running builds, re-stating the plan) and anything off the human's direction. "
        "Keep only what matches the human's goal, their pinned parameters and terms, and what they are "
        "inspecting right now. Reply in exactly this format, plain English, no other text:\n"
        "STATE: one line under 14 words, what matters now\n"
        "FORM: chart if the turn holds a genuine comparison of two or more measured arms, else summary\n"
        "POINTS:\n- up to 5 key points, each under 25 words, concrete (numbers, files, decisions)");
    QString ctx;
    ctx += QStringLiteral("Human direction (goal): ") + (goal.trimmed().isEmpty() ? QStringLiteral("(none stated)") : goal.trimmed());
    QStringList pinned;
    for (const QVariant &v : params) {
        const QVariantMap m = v.toMap();
        const QString name = m.value(QStringLiteral("name")).toString();
        if (name.isEmpty())
            continue;
        const QString value = m.value(QStringLiteral("value")).toString();
        pinned << (value.isEmpty() ? name : name + QLatin1Char('=') + value);
    }
    ctx += QStringLiteral("\nPinned params and terms: ") + (pinned.isEmpty() ? QStringLiteral("(none)") : pinned.join(QStringLiteral(", ")));
    ctx += QStringLiteral("\nWhat the human is inspecting now (recent actions, newest last):");
    if (recentActions.isEmpty())
        ctx += QStringLiteral(" (none)");
    for (const QString &a : recentActions)
        ctx += QStringLiteral("\n- ") + a;
    if (!tasteRules.isEmpty()) {
        ctx += QStringLiteral("\nThe human's standing taste rules (filter by them):");
        for (const QString &r : tasteRules)
            ctx += QStringLiteral("\n- ") + r;
    }
    QString turn = latestTurn.trimmed();
    if (turn.size() > 6000)
        turn = turn.left(6000) + QChar(0x2026);
    ctx += QStringLiteral("\nLatest turn:\n") + turn;
    return buildOpenAiCompatRequest(model, kSys, ctx, QStringLiteral("Filter the latest turn for this human."),
                                    QStringLiteral("low"));
}

HumanFilterReply parseFilterReply(const QString &reply)
{
    HumanFilterReply r;
    QStringList points;
    bool inPoints = false;
    static const QRegularExpression label(QStringLiteral("^[*_#\\s]*(STATE|FORM|POINTS)[*_\\s]*:[*_\\s]*(.*)$"),
                                          QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression bullet(QStringLiteral("^\\s*(?:[-*•]|\\d+[.)])\\s+"));
    for (const QString &raw : reply.split(QLatin1Char('\n'))) {
        const QString l = raw.trimmed();
        if (l.isEmpty())
            continue;
        const auto m = label.match(l);
        if (m.hasMatch()) {
            const QString key = m.captured(1).toUpper();
            const QString val = m.captured(2).trimmed().remove(QStringLiteral("**"));
            inPoints = key == QLatin1String("POINTS");
            if (key == QLatin1String("STATE"))
                r.stateLine = stripMarkdownLinks(val).simplified();
            else if (key == QLatin1String("FORM"))
                r.form = val.toLower().contains(QLatin1String("chart")) ? QStringLiteral("chart") : QStringLiteral("summary");
            else if (!val.isEmpty())
                points << val;
            continue;
        }
        if (inPoints) {
            QString p = l;
            p.remove(bullet);
            p = p.remove(QStringLiteral("**")).simplified();
            if (!p.isEmpty() && points.size() < 5)
                points << p;
        }
    }
    r.summary = points.join(QLatin1Char('\n'));
    return r;
}

QVariantList stageArtifacts(const QVariantList &artifacts, bool showDiffs)
{
    QVariantList out, diffs;
    for (const QVariant &v : artifacts) {
        if (v.toMap().value(QStringLiteral("kind")).toString() == QLatin1String("diff"))
            diffs << v;
        else
            out << v;
    }
    if (showDiffs)
        out << diffs;
    return out;
}

bool projectNeedsMe(const QString &status, bool blocker)
{
    return blocker || status == QLatin1String("needs-ok") || status == QLatin1String("needs-user")
           || status == QLatin1String("waiting") || status == QLatin1String("gate");
}

bool isSteerTurn(const QString &text)
{
    static const QRegularExpression steer(
        QStringLiteral("^(no\\b|nope\\b|not\\b|stop\\b|wait\\b|actually\\b|instead\\b|wrong\\b|rather\\b|"
                       "don'?t\\b|do not\\b|never\\b|always\\b|please don'?t\\b)|"
                       "\\b(instead of|rather than|don'?t|do not|never|stop|that's wrong|not what i)\\b"),
        QRegularExpression::CaseInsensitiveOption);
    return steer.match(text.simplified()).hasMatch();
}

namespace {
// "don't lead with the diff; show the chart" -> "Don't lead with the diff; show the chart."
QString asRule(const QString &text)
{
    QString s = sentencesOf(text).value(0, text.simplified());
    static const QRegularExpression lead(QStringLiteral("^(no|nope|wait|actually|hmm|ok|okay)[,.!]*\\s+"),
                                         QRegularExpression::CaseInsensitiveOption);
    s.remove(lead);
    s = shortLine(s, 16);
    if (s.isEmpty())
        return s;
    s[0] = s[0].toUpper();
    if (!s.endsWith(QChar(0x2026)) && !s.endsWith(QLatin1Char('.')) && !s.endsWith(QLatin1Char('!')))
        s += QLatin1Char('.');
    return s;
}
}  // namespace

SessionReflection localReflection(const QVariantList &sessions)
{
    SessionReflection r;
    QStringList steers;
    QStringList steered, quiet;  // intent lines: sessions the human corrected lead
    for (const QVariant &v : sessions) {
        const QVariantMap s = v.toMap();
        const QString name = s.value(QStringLiteral("name")).toString();
        const QString goal = s.value(QStringLiteral("goal")).toString().simplified();
        const QString last = s.value(QStringLiteral("last")).toString().simplified();
        const QStringList users = s.value(QStringLiteral("userTurns")).toStringList();
        bool corrected = false;
        for (qsizetype i = 1; i < users.size(); ++i)  // the first request is the intent, not a steer
            if (isSteerTurn(users[i])) {
                steers << users[i];
                corrected = true;
            }
        if (goal.isEmpty())
            continue;
        // The request verbatim (its casing keeps "65B", "DCP=4"), then where the work landed.
        QString line = name + QStringLiteral(": asked “") + shortLine(sentencesOf(goal).value(0, goal), 10)
                       + QStringLiteral("”");
        if (!last.isEmpty())
            line += QStringLiteral(", landed on ") + shortLine(last, 10);
        (corrected ? steered : quiet) << line;
    }
    const QStringList intents = (steered + quiet).mid(0, 2);
    r.intent = intents.join(QLatin1Char('\n'));
    QStringList quoted;
    for (qsizetype i = std::max<qsizetype>(0, steers.size() - 3); i < steers.size(); ++i)
        quoted << QStringLiteral("“") + shortLine(sentencesOf(steers[i]).value(0, steers[i]), 12) + QStringLiteral("”");
    r.corrections = quoted.isEmpty() ? QStringLiteral("No steers spotted.") : quoted.join(QStringLiteral("  "));
    for (qsizetype i = steers.size() - 1; i >= 0 && r.rules.size() < 3; --i) {
        const QString rule = asRule(steers[i]);
        if (!rule.isEmpty() && !r.rules.contains(rule, Qt::CaseInsensitive))
            r.rules << rule;
    }
    return r;
}

SessionReflection parseReflectionReply(const QString &reply)
{
    SessionReflection r;
    QString section;
    static const QRegularExpression label(
        QStringLiteral("^[*_#\\s\\d.)]*(INTENT|CORRECTIONS|STEERS|RULES)[*_\\s]*:[*_\\s]*(.*)$"),
        QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression bullet(QStringLiteral("^\\s*(?:[-*•]|\\d+[.)])\\s+"));
    QStringList intent, corrections;
    for (const QString &raw : reply.split(QLatin1Char('\n'))) {
        QString l = raw.trimmed();
        if (l.isEmpty())
            continue;
        const auto m = label.match(l);
        if (m.hasMatch()) {
            section = m.captured(1).toUpper();
            if (section == QLatin1String("STEERS"))
                section = QStringLiteral("CORRECTIONS");
            l = m.captured(2).trimmed();
            if (l.isEmpty())
                continue;
        }
        l = l.remove(QStringLiteral("**")).simplified();
        if (section == QLatin1String("INTENT"))
            intent << l.remove(bullet);
        else if (section == QLatin1String("CORRECTIONS"))
            corrections << l.remove(bullet);
        else if (section == QLatin1String("RULES") && r.rules.size() < 3) {
            l.remove(bullet);
            if (!l.isEmpty() && l.compare(QLatin1String("none"), Qt::CaseInsensitive) != 0)
                r.rules << l;
        }
    }
    r.intent = intent.join(QLatin1Char('\n'));
    r.corrections = corrections.join(QLatin1Char(' '));
    return r;
}

QStringList quickHelperArgs(const QString &sessionId, const QString &model, const QString &effort)
{
    return {QStringLiteral("-p"),
            QStringLiteral("--setting-sources"), QString(),
            QStringLiteral("--strict-mcp-config"),
            QStringLiteral("--session-id"), sessionId,
            QStringLiteral("--input-format"), QStringLiteral("stream-json"),
            QStringLiteral("--output-format"), QStringLiteral("stream-json"),
            QStringLiteral("--verbose"),
            QStringLiteral("--include-partial-messages"),
            QStringLiteral("--model"), model,
            QStringLiteral("--effort"), clampEffort(model, effort),
            QStringLiteral("--tools"), QString(),
            QStringLiteral("--no-session-persistence"),
            QStringLiteral("--append-system-prompt"), QString::fromLatin1(kQuickSystemPrompt)};
}

QStringList agyChatArgs(const QString &model, const QString &effort)
{
    // `-p ""`: print mode with an empty prompt as its own argv entry, so every turn arrives as a
    // stdin stream-json message and the process stays up between turns.
    return {QStringLiteral("-p"), QString(),
            QStringLiteral("--input-format"), QStringLiteral("stream-json"),
            QStringLiteral("--output-format"), QStringLiteral("stream-json"),
            QStringLiteral("--disable-slash-commands"),
            QStringLiteral("--sandbox"),
            QStringLiteral("--model"), agyModelId(model, effort)};
}

QByteArray cswapHookPayload(const QString &sessionId)
{
    const QJsonObject o{{QStringLiteral("hook_event_name"), QStringLiteral("SessionStart")},
                        {QStringLiteral("session_id"), sessionId},
                        {QStringLiteral("cwd"), QStringLiteral("/tmp")},
                        {QStringLiteral("source"), QStringLiteral("startup")}};
    return QJsonDocument(o).toJson(QJsonDocument::Compact);
}

namespace {
QJsonObject userMessage(const QString &text)
{
    return QJsonObject{{QStringLiteral("role"), QStringLiteral("user")},
                       {QStringLiteral("content"), QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("text")},
                                                                         {QStringLiteral("text"), text}}}}};
}
}  // namespace

QByteArray claudeUserLine(const QString &text)
{
    const QJsonObject o{{QStringLiteral("type"), QStringLiteral("user")}, {QStringLiteral("message"), userMessage(text)}};
    return QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n';
}

QStringList traceHelperArgs()
{
    // agy's -p takes the prompt as its value, so a bare "-p" would swallow the next flag; "-p="
    // is print mode with an empty prompt, and every turn then arrives as a stdin message.
    // (--mode plan is ignored once slash commands are off, so it is not passed.)
    return {QStringLiteral("--input-format"), QStringLiteral("stream-json"),
            QStringLiteral("--output-format"), QStringLiteral("stream-json"),
            QStringLiteral("--model"), QStringLiteral("gemini-3.8-flash-low"),
            QStringLiteral("--sandbox"),
            QStringLiteral("--disable-slash-commands"),
            QStringLiteral("-p=")};
}

QByteArray agyUserLine(const QString &text)
{
    const QJsonObject o{{QStringLiteral("event"), QStringLiteral("user")}, {QStringLiteral("message"), userMessage(text)}};
    return QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n';
}

QStringList codexAppServerArgs()
{
    return {QStringLiteral("app-server"),
            QStringLiteral("-c"), QStringLiteral("model=\"gpt-6-astra\""),
            QStringLiteral("-c"), QStringLiteral("model_reasoning_effort=\"low\""),
            QStringLiteral("-c"), QStringLiteral("mcp_servers={}")};
}

QByteArray codexRpc(int id, const QString &method, const QJsonObject &params)
{
    QJsonObject o{{QStringLiteral("method"), method}, {QStringLiteral("params"), params}};
    if (id >= 0)
        o.insert(QStringLiteral("id"), id);
    return QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n';
}

QJsonObject codexThreadStartParams(const QString &instructions)
{
    return QJsonObject{{QStringLiteral("cwd"), QDir::tempPath()},
                       {QStringLiteral("approvalPolicy"), QStringLiteral("never")},
                       {QStringLiteral("sandbox"), QStringLiteral("read-only")},
                       {QStringLiteral("ephemeral"), true},
                       {QStringLiteral("developerInstructions"), instructions}};
}

QJsonObject codexTurnParams(const QString &threadId, const QString &text, const QString &model, const QString &effort)
{
    return QJsonObject{{QStringLiteral("threadId"), threadId},
                       {QStringLiteral("input"), QJsonArray{QJsonObject{{QStringLiteral("type"), QStringLiteral("text")},
                                                                        {QStringLiteral("text"), text}}}},
                       {QStringLiteral("model"), model},
                       {QStringLiteral("effort"), clampEffort(model, effort)},
                       {QStringLiteral("summary"), QStringLiteral("concise")}};
}

CodexEvent parseCodexLine(const QByteArray &line)
{
    CodexEvent ev;
    QJsonParseError err{};
    const QJsonObject o = QJsonDocument::fromJson(line, &err).object();
    if (err.error != QJsonParseError::NoError || o.isEmpty())
        return ev;
    const QString method = o.value(QLatin1String("method")).toString();
    const QJsonObject params = o.value(QLatin1String("params")).toObject();
    ev.threadId = params.value(QLatin1String("threadId")).toString();
    if (method.isEmpty()) {
        ev.id = o.value(QLatin1String("id")).toInt(-1);
        if (o.contains(QLatin1String("error"))) {
            ev.kind = CodexEvent::Error;
            ev.text = o.value(QLatin1String("error")).toObject().value(QLatin1String("message")).toString();
        } else if (o.contains(QLatin1String("result"))) {
            ev.kind = CodexEvent::Response;
            ev.threadId = o.value(QLatin1String("result")).toObject().value(QLatin1String("thread")).toObject()
                              .value(QLatin1String("id")).toString();
        }
    } else if (method == QLatin1String("item/agentMessage/delta")) {
        ev.kind = CodexEvent::AgentDelta;
        ev.text = params.value(QLatin1String("delta")).toString();
    } else if (method == QLatin1String("item/reasoning/summaryTextDelta")
               || method == QLatin1String("item/reasoning/textDelta")) {
        ev.kind = CodexEvent::ReasoningDelta;
        ev.text = params.value(QLatin1String("delta")).toString();
    } else if (method == QLatin1String("item/started")) {
        ev.kind = CodexEvent::ItemStarted;
        ev.text = params.value(QLatin1String("item")).toObject().value(QLatin1String("type")).toString();
    } else if (method == QLatin1String("turn/completed")) {
        ev.kind = CodexEvent::TurnCompleted;
        if (ev.threadId.isEmpty())
            ev.threadId = params.value(QLatin1String("turn")).toObject().value(QLatin1String("threadId")).toString();
    } else if (method == QLatin1String("error")) {
        ev.kind = CodexEvent::Error;
        ev.text = params.value(QLatin1String("error")).toObject().value(QLatin1String("message")).toString();
    }
    return ev;
}

// --- construction -------------------------------------------------------------------

Backend::Backend(Mode mode, QObject *parent, bool warmHelpers)
    : QObject(parent)
    , m_mode(mode)
{
    m_prefsSave.setSingleShot(true);
    m_prefsSave.setInterval(400);  // a wheel zoom fires dozens of changes; write once it settles
    connect(&m_prefsSave, &QTimer::timeout, this, [this] { savePrefs(); });
    // One source of sound at a time: Spotify starting pauses slowmix, slowmix starting pauses Spotify.
    connect(&m_spotify, &SpotifyController::playbackStarted, this, [this] { m_slowmix.pause(); });
    connect(&m_slowmix, &SlowmixController::stateChanged, this, [this] {
        const bool now = m_slowmix.playing();
        if (now && !m_slowmixWasPlaying && m_spotify.playing())
            m_spotify.pause();
        m_slowmixWasPlaying = now;
    });
    m_ambientDebounce.setSingleShot(true);
    m_ambientDebounce.setInterval(800);
    connect(&m_ambientDebounce, &QTimer::timeout, this, &Backend::flushAmbient);
    m_proposalTimer.setSingleShot(true);
    m_proposalTimer.setInterval(15000);
    connect(&m_proposalTimer, &QTimer::timeout, this, &Backend::dismissPendingProposal);
    // L3 + R3 anywhere (background monitoring is on) runs the staged Ambient proposal.
    connect(&m_gamepad, &Gamepad::confirmProposalRequested, this, [this] { confirmPendingProposal(); });
    // L2 routing (in-Studio ask card vs the MacBook-wide Spotlight bar) runs before the Phonon
    // key goes out. While the bar is up it owns R2 / ✕ (send) and ○ (close); queued, so QML's
    // handlers for the same press still see spotlightOpen and stand aside.
    m_gamepad.setPrepareVoiceTarget([this](bool active) { return prepareVoiceTarget(active); });
    m_gamepad.setOverlayActive([this] { return spotlightOpen(); });
    m_gamepad.setSummonStudio([this] { summonStudio(); });
    connect(&m_gamepad, &Gamepad::sendPressed, this, [this] { if (spotlightOpen()) submitSpotlight(); }, Qt::QueuedConnection);
    connect(&m_gamepad, &Gamepad::confirm, this, [this] { if (spotlightOpen()) submitSpotlight(); }, Qt::QueuedConnection);
    connect(&m_gamepad, &Gamepad::back, this, [this] {
        // Jarvis: ○ cancels listening, or stops a running turn (bar open, closed or ghosted).
        if (jarvisOverlay() && (m_awaitingAutoSend || m_gamepad.voiceActive()
                                || (spotlightOpen() && m_spotlight->state().voice != studio::cua::SpotlightVoice::Idle)
                                || m_spotlightTurn || m_cuaPending > 0)) {
            handleJarvisEscape();
            return;
        }
        if (spotlightOpen())
            dismissSpotlight(true);
    }, Qt::QueuedConnection);
    // Jarvis: the live Studio defaults to the unified overlay; every mock / offscreen mode keeps
    // the split routing its tests were written for (a test opts in with setJarvisBarMode).
    m_jarvisBarMode = mode == Mode::Live ? QStringLiteral("overlay") : QStringLiteral("split");
    m_gamepad.setVoiceTapToggle(jarvisOverlay());
    connect(&m_gamepad, &Gamepad::voiceToggledChanged, this, [this] {
        if (m_voiceTarget == VoiceTarget::Spotlight && spotlightOpen() && m_gamepad.voiceToggled())
            m_spotlight->setVoice(studio::cua::SpotlightVoice::Toggled);
    });
    for (QTimer *t : {&m_autoSendDebounce, &m_autoSendNoPaste, &m_autoSendGiveUp})
        t->setSingleShot(true);
    m_autoSendDebounce.setInterval(kAutoSendDebounceMs);
    m_autoSendNoPaste.setInterval(kAutoSendNoPasteMs);
    m_autoSendGiveUp.setInterval(kAutoSendGiveUpMs);
    connect(&m_autoSendDebounce, &QTimer::timeout, this, [this] {
        if (m_awaitingAutoSend && m_spotlight && !m_spotlight->inputText().trimmed().isEmpty())
            autoSendNow();  // the paste has landed
    });
    connect(&m_autoSendNoPaste, &QTimer::timeout, this, [this] {
        // Text was already there and no paste came: send what the field holds.
        if (m_awaitingAutoSend && m_spotlight && !m_spotlight->inputText().trimmed().isEmpty()
            && m_spotlight->inputText() == m_autoSendBaseline && !m_autoSendDebounce.isActive())
            autoSendNow();
    });
    connect(&m_autoSendGiveUp, &QTimer::timeout, this, [this] {
        if (!m_awaitingAutoSend || !m_spotlight)
            return;
        if (!m_spotlight->inputText().trimmed().isEmpty()) {
            autoSendNow();
            return;
        }
        // Nothing was said: close the empty bar quietly.
        disarmAutoSend();
        dismissSpotlight(!m_spotlight->state().overStudio);
    });
    connect(this, &Backend::streamDelta, this, [this](const QString &id, const QString &text, const QString &) {
        if (id == QLatin1String(kSpotlightStreamId)) {
            m_spotlightRaw += text;
            onSpotlightDelta();
        }
    });
    connect(this, &Backend::streamFinished, this, [this](const QString &id, const QString &err) {
        if (id == QLatin1String(kSpotlightStreamId))
            onSpotlightFinished(err);
    });
    if (m_mode == Mode::Mock) {
        ensureFixtureFiles();
        const QString clips = ensureSlowmixFixture();
        m_slowmix.setLibrary(clips, clips + QStringLiteral("/.plays.tsv"));
        loadFixtures();
        m_folders = fixtureFolders();
        m_defaultFolder = m_folders.value(0);
        connect(&m_autoTimer, &QTimer::timeout, this, [this] {
            const double dt = m_autoClock.restart() / 1000.0;
            advance(dt);
        });
        publish();
        return;
    }
    loadFolders();
    // Tier 0 tail: identity + workstation notes, byte-stable so the prefix stays hot in vLLM.
    m_studioContext = loadLocalStudioContext();
    // Read-only until the overlay asks (refresh); spawning stays off unless main() allows it.
    m_slowmix.setLibrary(SlowmixController::defaultClipsDir(),
                         SlowmixController::defaultClipsDir() + QStringLiteral("/.plays.tsv"));
    // The quick answer line shows live seconds while a model thinks, so a heavy model never
    // looks hung. Its own signal: a tick re-runs only the bindings that show the clock.
    m_quickTicker.setInterval(100);
    connect(&m_quickTicker, &QTimer::timeout, this, [this] {
        if (!m_quickBusy) {
            m_quickTicker.stop();
            return;
        }
        m_quickElapsed = m_quickClock.elapsed() / 1000.0;
        emit quickTick();
    });
    m_rescanDebounce.setSingleShot(true);
    m_rescanDebounce.setInterval(600);
    connect(&m_rescanDebounce, &QTimer::timeout, this, &Backend::refresh);
    connect(&m_watcher, &QFileSystemWatcher::fileChanged, this, [this](const QString &p) {
        if (!m_watcher.files().contains(p) && QFileInfo::exists(p))
            m_watcher.addPath(p);  // atomic rewrites drop the watch
        m_rescanDebounce.start();
    });
    if (QFileInfo::exists(workspacePath()))
        m_watcher.addPath(workspacePath());
    m_boardTimer.setInterval(5000);
    connect(&m_boardTimer, &QTimer::timeout, this, &Backend::pollBoard);
    connect(&m_boardTimer, &QTimer::timeout, this, &Backend::pollPhonon);
    m_boardTimer.start();
    refresh();
    pollBoard();
    pollPhonon();
    m_ageTimer.setInterval(30000);
    connect(&m_ageTimer, &QTimer::timeout, this, &Backend::publish);
    m_ageTimer.start();
    // Cross-session memory: first scan once the registry has named its redacted sessions, then
    // an incremental rescan (unchanged transcripts come from the cache) every 5 minutes.
    connect(&m_memory, &MemoryIndex::refreshed, this, [](int turns, double ms) {
        std::fprintf(stderr, "lapis_studio: memory index %d turns in %.0f ms\n", turns, ms);
    });
    m_memoryTimer.setInterval(5 * 60 * 1000);
    connect(&m_memoryTimer, &QTimer::timeout, this, &Backend::refreshMemory);
    m_memoryTimer.start();
    QTimer::singleShot(1500, this, &Backend::refreshMemory);
    if (warmHelpers) {
        // The all-Spaces HUD bar (real app only: it opens a window). Holding voice shows
        // "Listening…"; on release it says "Thinking…" and fades unless a cua_* action takes over.
        // The executor drives it during actions; a finished queue always fades it out.
        connect(&m_gamepad, &Gamepad::voiceActiveChanged, this, [this] {
            // The in-Studio ask card and the Spotlight bar show their own "Listening…".
            if (m_voiceTarget != VoiceTarget::None)
                return;
            if (m_gamepad.voiceActive()) {
                studio::cua::showCuaHud(QStringLiteral("Listening\u2026"), studio::cua::HudTone::Listening);
            } else if (!m_cuaPending) {
                studio::cua::showCuaHud(QStringLiteral("Thinking\u2026"), studio::cua::HudTone::Working);
                studio::cua::hideCuaHud(4.0);
            }
        });
        connect(this, &Backend::cuaActionFinished, this, [this] {
            if (!m_cuaPending && !m_gamepad.voiceActive())
                studio::cua::hideCuaHud(studio::cua::kHudLingerSec);
        });
        m_cuaHud = true;
        // Human-click shadow recorder: the real window only (warm helpers), never Mock or an
        // offscreen mode. setPrefsPath (right after construction) and this timer start it.
        {
            const QString env = qEnvironmentVariable("LAPIS_SHADOW_RECORD").trimmed().toLower();
            m_shadowAllowed = !(env == QLatin1String("0") || env == QLatin1String("off") || env == QLatin1String("false")
                                || env == QLatin1String("no"));
        }
        QTimer::singleShot(1500, this, [this] {
            syncShadowRecorder();
            refreshShadowExcludes();
        });
        // The overlay windows (reticle, HUD bar) appear lazily: refresh what frames exclude.
        connect(&m_gamepad, &Gamepad::voiceActiveChanged, this, &Backend::refreshShadowExcludes);
        connect(this, &Backend::cuaActionFinished, this, &Backend::refreshShadowExcludes);
        // The default model's daemon is the one the first Enter will use; every other provider
        // channel starts lazily the first time the picker sends to it, then stays up.
        QTimer::singleShot(400, this, &Backend::warmChannel);
        QTimer::singleShot(900, this, &Backend::ensureTraceHelper);  // warm the agy trace helper
    }
}

Backend::~Backend()
{
    m_spotlight.reset();
    if (m_shadow)
        m_shadow->stop();  // removes the tap, joins the worker (a capture is bounded to 5 s)
    if (m_cuaPool) {
        // A running action may be blocked on a reticle call queued to this (main) thread: cancel
        // its input, keep serving the event loop until it returns, then drop the latch.
        studio::cua::cancelFlag.store(true);
        m_cuaPool->clear();
        while (!m_cuaPool->waitForDone(10))
            QCoreApplication::processEvents(QEventLoop::AllEvents, 10);
        studio::cua::cancelFlag.store(false);
    }
    if (m_prefsSave.isActive())
        savePrefs();
    if (m_router)
        m_router->disconnect(this);  // an injected router may outlive this backend
    if (m_suggestReply) {
        m_suggestReply->disconnect(this);
        m_suggestReply->abort();
        m_suggestReply->deleteLater();
        m_suggestReply.clear();
    }
    shutdownChannels();
    if (m_traceProc && m_traceProc->state() != QProcess::NotRunning) {
        m_traceProc->kill();
        m_traceProc->waitForFinished(500);
    }
}

QList<qint64> Backend::shutdownChannels()
{
    // EOF first: claude, agy and codex app-server all exit on a closed stdin, which lets them
    // reap their own children. SIGTERM, then SIGKILL, only for a daemon that hangs on.
    QList<qint64> pids;
    QList<QProcess *> procs;
    for (Channel *c : std::as_const(m_channels)) {
        if (c->proc && c->proc->state() != QProcess::NotRunning) {
            pids << c->proc->processId();
            c->proc->disconnect(this);
            c->proc->closeWriteChannel();
            procs << c->proc;
        }
        delete c;
    }
    m_channels.clear();
    QElapsedTimer t;
    t.start();
    for (QProcess *p : procs)
        p->waitForFinished(std::max<qint64>(1, 600 - t.elapsed()));
    for (QProcess *p : procs)
        if (p->state() != QProcess::NotRunning) {
            p->terminate();
            if (!p->waitForFinished(400)) {
                p->kill();
                p->waitForFinished(400);
            }
        }
    return pids;
}

QVariantMap Backend::harnessModels() const
{
    QVariantMap m;
    m[QStringLiteral("claude")] = QStringList{QStringLiteral("claude-opus-5-5"), QStringLiteral("claude-fable-5-2[1m]")};
    m[QStringLiteral("codex")] = QStringList{QStringLiteral("gpt-6-astra")};
    m[QStringLiteral("agy")] = QStringList{QStringLiteral("gemini-4-carbon-medium")};
    return m;
}

QProcess *Backend::spawn(const QString &program, const QStringList &args, const QString &cwd,
                         const QProcessEnvironment &env)
{
    if (m_mode != Mode::Live) {
        ++m_blockedSpawns;  // the mock backend never reaches Lapis or a model CLI
        return nullptr;
    }
    ++m_spawnCount;
    auto *p = new QProcess(this);
    p->setProgram(program);
    p->setArguments(args);
    p->setWorkingDirectory(cwd.isEmpty() ? home() : cwd);
    if (!env.isEmpty())
        p->setProcessEnvironment(env);
    connect(p, &QProcess::finished, p, &QObject::deleteLater);
    p->start();
    return p;
}

// --- preferences ----------------------------------------------------------------------

double Backend::clampScale(double s)
{
    if (!std::isfinite(s))
        return 1.0;
    return std::clamp(std::round(s * 100.0) / 100.0, kMinScale, kMaxScale);
}

QString Backend::defaultPrefsPath()
{
    return home() + QStringLiteral("/.lapis-next/studio_prefs.json");
}

void Backend::setUiScale(double s)
{
    s = clampScale(s);
    if (qFuzzyCompare(s, m_uiScale))
        return;
    m_uiScale = s;
    emit prefsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::setNavMode(const QString &mode)
{
    QString m = mode.trimmed().toLower();
    if (m == QLatin1String("arrows"))
        m = QStringLiteral("wasd");  // arrow-key mode was gaming mode; it is not its own option
    if (!navModes().contains(m) || m == m_navMode)
        return;
    m_navMode = m;
    emit prefsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::setTheme(const QString &theme)
{
    const QString t = theme.trimmed().toLower();
    if (!themes().contains(t) || t == m_theme)
        return;
    m_theme = t;
    if (m_mode == Mode::Live)
        studio::cua::setOverlayTheme(m_theme == QLatin1String("lapis"));
    emit prefsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::cycleNavMode()
{
    const QStringList modes = navModes();
    setNavMode(modes.value((modes.indexOf(m_navMode) + 1) % modes.size()));
}

void Backend::completeOnboarding(const QString &mode)
{
    setNavMode(mode);
    m_firstLaunch = false;
    emit prefsChanged();
    if (!m_prefsPath.isEmpty())
        savePrefs();  // the file now exists, so the modal never shows again
}

void Backend::showOnboarding()
{
    m_firstLaunch = true;
    emit prefsChanged();
}

QVariantList Backend::models() const
{
    QVariantList out;
    for (const ModelSpec &m : modelCatalog())
        out << QVariantMap{{QStringLiteral("id"), m.id},
                           {QStringLiteral("provider"), m.provider},
                           {QStringLiteral("label"), m.label},
                           {QStringLiteral("speed"), m.speed},
                           {QStringLiteral("display"), modelDisplayLabel(m.id)},
                           {QStringLiteral("efforts"), m.efforts}};
    return out;
}

void Backend::setFocusedProject(const QString &key)
{
    if (key == m_focusedProject && !m_fresh.contains(key))
        return;
    m_focusedProject = key;
    m_fresh.remove(key);  // opening a project reads its new turns
    publish();
    emit projectsChanged();  // focusedProject itself changed, even when the list did not
    if (!key.isEmpty()) {
        warmProjectPrefix(key);  // both no-ops unless live with an OpenAI-compatible endpoint
        filterForHuman(key);
    }
}

void Backend::mockAppendTurn(const QString &key, const Turn &t)
{
    if (m_mode != Mode::Mock)
        return;
    Project *p = find(key);
    if (!p)
        return;
    p->mockTurns << t;
    p->ex = extractArtifacts(p->mockTurns);
    if (key != m_focusedProject)
        m_fresh[key] += 1;
    publish();
}

QString Backend::shortModelLabel(const QString &model) const
{
    const ModelSpec *m = specFor(model);
    return m ? m->label : model;
}

QString Backend::modelSpeed(const QString &model) const
{
    const ModelSpec *m = specFor(model);
    return m ? m->speed : QString();
}

void Backend::setChatModel(const QString &model)
{
    if (!specFor(model) || model == m_chatModel)
        return;
    m_chatModel = model;
    // A no-think default ("fast") wins on the way in; thinking stays opt-in via the effort strip.
    // Leaving it restores the thinking level the user had before, so the picker never loses it.
    const QStringList e = effortsFor(model);
    if (!e.isEmpty() && e.first() == QLatin1String("fast")) {
        if (m_chatEffort != QLatin1String("fast"))
            m_effortBeforeFast = m_chatEffort;
        m_chatEffort = e.first();
    } else {
        const QString want = m_chatEffort == QLatin1String("fast") && !m_effortBeforeFast.isEmpty() ? m_effortBeforeFast
                                                                                                  : m_chatEffort;
        m_chatEffort = clampEffort(model, want);
    }
    emit prefsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::setChatEffort(const QString &effort)
{
    if (!effortsFor(m_chatModel).contains(effort) || effort == m_chatEffort)
        return;
    m_chatEffort = effort;
    emit prefsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::cycleModel(int delta)
{
    const QList<ModelSpec> &c = modelCatalog();
    int i = 0;
    for (int k = 0; k < c.size(); ++k)
        if (c[k].id == m_chatModel)
            i = k;
    const int n = int(c.size());
    setChatModel(c[((i + delta) % n + n) % n].id);
}

void Backend::cycleEffort(int delta)
{
    const QStringList e = effortsFor(m_chatModel);
    const int i = std::max(0, int(e.indexOf(m_chatEffort)));
    setChatEffort(e.value(std::clamp(i + delta, 0, int(e.size()) - 1)));
}

void Backend::setQuickHarness(const QString &h)
{
    if (h == QLatin1String("codex")) {
        m_chatModel = QStringLiteral("gpt-6-astra");
        m_chatEffort = QStringLiteral("low");
        emit prefsChanged();
    }
}

void Backend::setModelSuggestions(bool on)
{
    if (on == m_modelSuggestions)
        return;
    m_modelSuggestions = on;
    emit prefsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::setJarvisBarMode(const QString &mode)
{
    const QString m = mode.trimmed().toLower();
    if ((m != QLatin1String("overlay") && m != QLatin1String("split")) || m == m_jarvisBarMode)
        return;
    m_jarvisBarMode = m;
    m_gamepad.setVoiceTapToggle(jarvisOverlay());
    if (!jarvisOverlay())
        disarmAutoSend();
    emit jarvisBarModeChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::setShowCodeDiffs(bool on)
{
    if (on == m_showCodeDiffs)
        return;
    m_showCodeDiffs = on;
    emit prefsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
    publish();  // the Stage deck gains or drops its diffs
}

void Backend::setCardPosition(const QString &projectKey, double x, double y)
{
    if (projectKey.isEmpty() || !std::isfinite(x) || !std::isfinite(y))
        return;
    const QVariantMap c{{QStringLiteral("x"), std::clamp(x, 0.0, 1.0)}, {QStringLiteral("y"), std::clamp(y, 0.0, 1.0)}};
    if (m_cardPositions.value(projectKey).toMap() == c)
        return;
    m_cardPositions.insert(projectKey, c);
    emit cardPositionsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::resetCardPositions()
{
    if (m_cardPositions.isEmpty())
        return;
    m_cardPositions.clear();
    emit cardPositionsChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

void Backend::setMapSort(const QString &sort)
{
    const QString s = sort == QLatin1String("recent") || sort == QLatin1String("activity") ? sort : QStringLiteral("category");
    if (s == m_mapSort)
        return;
    m_mapSort = s;
    emit mapLayoutChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

QStringList Backend::presentClusters() const
{
    QSet<QString> have;
    for (const QVariant &v : m_projectList)
        have.insert(v.toMap().value(QStringLiteral("cluster")).toString());
    QStringList out;
    for (const QString &c : m_clusterOrder)
        if (have.contains(c))
            out << c;
    return out;
}

bool Backend::moveCluster(const QString &cluster, const QString &dir)
{
    const QStringList next = moveClusterInOrder(m_clusterOrder, presentClusters(), cluster, dir);
    if (next == m_clusterOrder)
        return false;
    m_clusterOrder = next;
    emit mapLayoutChanged();
    resetCardPositions();  // after the order: the map has its glide on by now, dragged cards spring too
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
    return true;
}

void Backend::mockSetCategory(const QString &key, const QString &category)
{
    if (m_mode != Mode::Mock)
        return;
    Project *p = find(key);
    if (!p)
        return;
    for (SessionInfo &s : p->sessions)
        s.category = category;
    publish();
}

void Backend::mockAddProject(const QString &key, const QString &name, const QString &dir, const QString &category,
                             const QString &status)
{
    if (m_mode != Mode::Mock || find(key))
        return;
    Project p;
    p.key = key;
    p.name = name;
    p.dir = dir;
    p.mtime = 500 - m_projects.size();
    SessionInfo s;
    s.id = QStringLiteral("f1x7%1-0000-0000-0000-000000000000").arg(qHash(key) % 10000, 4, 10, QLatin1Char('0'));
    s.title = name;
    s.harness = QStringLiteral("claude");
    s.status = status;
    s.category = category;
    p.sessions << s;
    m_projects << p;
    publish();
}

void Backend::setPrefsPath(const QString &path)
{
    if (m_prefsSave.isActive()) {
        m_prefsSave.stop();
        savePrefs();  // flush pending changes to the old file first
    }
    m_prefsPath = path;
    m_firstLaunch = !path.isEmpty() && !QFileInfo::exists(path);
    loadPrefs();
    if (m_mode == Mode::Live)
        studio::cua::setOverlayTheme(m_theme == QLatin1String("lapis"));  // desktop overlays match Theme.qml
    emit prefsChanged();
    syncShadowRecorder();  // the prefs toggle is known now
}

namespace {
// Only known lenses, a non-negative artifact index and a real bool survive into prefs.
QVariantMap normalizedViewState(const QVariantMap &in)
{
    static const QStringList lenses = {QStringLiteral("constellation"), QStringLiteral("ladder"), QStringLiteral("stage")};
    const QString lens = in.value(QStringLiteral("lens")).toString();
    if (!lenses.contains(lens))
        return {};
    return {{QStringLiteral("lens"), lens},
            {QStringLiteral("focusedProject"), in.value(QStringLiteral("focusedProject")).toString()},
            {QStringLiteral("artifactIndex"), std::max(0, in.value(QStringLiteral("artifactIndex")).toInt())},
            {QStringLiteral("needsMeOnly"), in.value(QStringLiteral("needsMeOnly")).toBool()}};
}
}  // namespace

void Backend::loadPrefs()
{
    if (m_prefsPath.isEmpty())
        return;
    QFile f(m_prefsPath);
    if (!f.open(QIODevice::ReadOnly))
        return;
    const QJsonObject o = QJsonDocument::fromJson(f.read(64 * 1024)).object();
    m_uiScale = clampScale(o.value(QLatin1String("uiScale")).toDouble(1.0));
    QString nav = o.value(QLatin1String("navMode")).toString();
    if (nav == QLatin1String("arrows"))
        nav = QStringLiteral("wasd");  // files written before arrows mode was dropped
    if (navModes().contains(nav))
        m_navMode = nav;
    const QString theme = o.value(QLatin1String("theme")).toString().trimmed().toLower();
    m_theme = themes().contains(theme) ? theme : QStringLiteral("gold");
    const QString model = o.value(QLatin1String("defaultModel")).toString();
    if (specFor(model))
        m_chatModel = model;
    m_chatEffort = clampEffort(m_chatModel, o.value(QLatin1String("defaultEffort")).toString(m_chatEffort));
    // Files from before "fast" existed saved "low" as the floor: those start at no-think once.
    if (o.value(QLatin1String("effortSchema")).toInt(1) < 2 && effortsFor(m_chatModel).value(0) == QLatin1String("fast"))
        m_chatEffort = QStringLiteral("fast");
    m_modelSuggestions = o.value(QLatin1String("modelSuggestions")).toBool(false);
    {
        const QString jm = o.value(QLatin1String("jarvisBarMode")).toString().trimmed().toLower();
        if ((jm == QLatin1String("overlay") || jm == QLatin1String("split")) && jm != m_jarvisBarMode) {
            m_jarvisBarMode = jm;
            m_gamepad.setVoiceTapToggle(jarvisOverlay());
            emit jarvisBarModeChanged();
        }
    }
    m_openaiCompatUrl = o.value(QLatin1String("openaiCompatUrl")).toString().trimmed();
    m_compatUrlCleared = false;
    const QString compatModel = o.value(QLatin1String("openaiCompatModel")).toString().trimmed();
    if (!compatModel.isEmpty())
        m_openaiCompatModel = compatModel;
    m_openaiCompatKey = o.value(QLatin1String("openaiCompatKey")).toString().trimmed();
    m_openrouterKey = o.value(QLatin1String("openrouterApiKey")).toString().trimmed();
    m_viewState = normalizedViewState(o.value(QLatin1String("viewState")).toObject().toVariantMap());
    emit viewStateChanged();
    m_showCodeDiffs = o.value(QLatin1String("showCodeDiffs")).toBool(false);
    if (m_shadowPref != o.value(QLatin1String("shadowRecordEnabled")).toBool(true)) {
        m_shadowPref = !m_shadowPref;
        emit shadowRecordChanged();
    }
    m_windowState = WindowState::fromJson(o.value(QLatin1String("windowState")).toObject());
    m_cardPositions.clear();
    const QJsonObject cards = o.value(QLatin1String("cardPositions")).toObject();
    for (auto it = cards.begin(); it != cards.end(); ++it) {
        const QJsonObject c = it.value().toObject();
        if (it.key().isEmpty() || !c.value(QLatin1String("x")).isDouble() || !c.value(QLatin1String("y")).isDouble())
            continue;
        m_cardPositions.insert(it.key(), QVariantMap{{QStringLiteral("x"), std::clamp(c.value(QLatin1String("x")).toDouble(), 0.0, 1.0)},
                                                     {QStringLiteral("y"), std::clamp(c.value(QLatin1String("y")).toDouble(), 0.0, 1.0)}});
    }
    emit cardPositionsChanged();
    {
        const QString sort = o.value(QLatin1String("mapSort")).toString();
        m_mapSort = sort == QLatin1String("recent") || sort == QLatin1String("activity") ? sort : QStringLiteral("category");
        // A saved order counts only as a full permutation of the four clusters.
        QStringList order;
        for (const QJsonValue &v : o.value(QLatin1String("clusterOrder")).toArray()) {
            const QString c = normalizeClusterName(v.toString());
            if (!c.isEmpty() && !order.contains(c))
                order << c;
        }
        m_clusterOrder = order.size() == defaultClusterOrder().size() ? order : defaultClusterOrder();
        emit mapLayoutChanged();
    }
    QHash<QString, QString> padMap = GamepadState::defaultBindings();
    const QJsonObject padObj = o.value(QLatin1String("gamepadBindings")).toObject();
    for (auto it = padObj.begin(); it != padObj.end(); ++it)
        padMap.insert(it.key(), it.value().toString());
    m_gamepad.setBindingsMap(padMap);
    emit prefsChanged();
}

void Backend::setGamepadBinding(const QString &inputId, const QString &actionId)
{
    if (!GamepadState::validInputs().contains(inputId) || !GamepadState::validActions().contains(actionId))
        return;
    QHash<QString, QString> map = m_gamepad.bindingsMap();
    if (map.value(inputId) == actionId)
        return;
    map.insert(inputId, actionId);
    m_gamepad.setBindingsMap(map);
    emit prefsChanged();
    m_prefsSave.start();
}

void Backend::resetGamepadBindings()
{
    m_gamepad.setBindingsMap(GamepadState::defaultBindings());
    emit prefsChanged();
    m_prefsSave.start();
}

void Backend::recordAction(const QString &action)
{
    const QString a = action.simplified();
    if (a.isEmpty())
        return;
    m_recentActions << a;
    while (m_recentActions.size() > 12)
        m_recentActions.removeFirst();
}

void Backend::pollPhonon()
{
    if (m_mode != Mode::Live)
        return;
    const QString path = phononLatestPath();
    const QFileInfo fi(path);
    const QString target = fi.canonicalFilePath();
    const qint64 mtime = target.isEmpty() ? 0 : QFileInfo(target).lastModified().toMSecsSinceEpoch();
    if (mtime == m_phononMtime && mtime != 0) {
        const qint64 ageMs = QDateTime::currentMSecsSinceEpoch() - mtime;
        if (ageMs <= 10 * 60 * 1000)
            return;  // unchanged and still fresh
    }
    m_phononMtime = mtime;
    const PhononSnapshot snap = readPhononSnapshot(path);
    if (snap.tail != m_phononTail || snap.live != m_phononLive) {
        m_phononTail = snap.tail;
        m_phononLive = snap.live;
        emit phononTailChanged();
        feedAmbientTail(snap.tail);
    }
}

// --- Human-click shadow recorder -------------------------------------------------------------

void Backend::setShadowRecordEnabled(bool on)
{
    if (!m_shadowAllowed || on == m_shadowPref)
        return;  // Mock / offscreen / LAPIS_SHADOW_RECORD=0: stays false
    m_shadowPref = on;
    syncShadowRecorder();
    emit shadowRecordChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();
}

bool Backend::shadowRecording() const
{
    return m_shadow && m_shadow->isRunning();
}

void Backend::syncShadowRecorder()
{
    const bool want = m_mode == Mode::Live && shadowRecordEnabled();
    if (!want) {
        if (m_shadow)
            m_shadow->stop();
        return;
    }
    if (shadowRecording())
        return;
    if (!m_shadow)
        m_shadow = std::make_unique<studio::cua::CuaShadowRecorder>();
    const QString dir = home() + QStringLiteral("/.lapis-next/cua_traces");
    if (m_shadow->start(dir))
        std::fprintf(stderr, "lapis_studio: shadow recorder on (%s/human_traces.jsonl)\n", qPrintable(dir));
}

void Backend::refreshShadowExcludes()
{
    if (shadowRecording())
        m_shadow->setExcludeWindowIds(studio::cua::overlayWindowIds());
}

// --- Ambient voice -----------------------------------------------------------------------------

void Backend::setAmbientVoiceEnabled(bool on)
{
    if (on == m_ambientEnabled)
        return;
    m_ambientEnabled = on;
    // Only what is said from now on counts: the tail already seen (fed while off, or on disk) is old news.
    if (m_ambientPrevTail.isEmpty())
        m_ambientPrevTail = m_phononTail;
    m_ambientPending.clear();
    m_ambientDebounce.stop();
    if (!on && m_ambientBusy) {
        if (m_router)
            m_router->cancelStream(QString::fromLatin1(kAmbientStreamId));
        m_ambientBusy = false;
    }
    if (!on)
        dismissPendingProposal();
    emit ambientChanged();
}

void Backend::clearAmbientAnswer()
{
    if (m_ambientAnswer.isEmpty() && m_ambientHeard.isEmpty())
        return;
    m_ambientAnswer.clear();
    m_ambientHeard.clear();
    emit ambientChanged();
}

void Backend::feedAmbientTail(const QString &tail)
{
    const QString prev = m_ambientPrevTail;
    m_ambientPrevTail = tail;
    if (!m_ambientEnabled)
        return;
    const QStringList said = extractNewUserUtterances(prev, tail);
    if (said.isEmpty())
        return;
    m_ambientPending << said;
    m_ambientDebounce.start();  // 800 ms after the last new line, the batch goes out
}

void Backend::flushAmbient()
{
    m_ambientDebounce.stop();
    if (m_ambientPending.isEmpty())
        return;
    const QString spoken = m_ambientPending.join(QLatin1Char(' ')).simplified();
    m_ambientPending.clear();
    if (spoken.isEmpty() || !m_ambientEnabled)
        return;
    runAmbient(spoken);
}

void Backend::runAmbient(const QString &spoken)
{
    // A plain UI command needs no model: "open music", "pause spotify", "volume 40".
    QString reply;
    if (handleStudioCommand(spoken, m_focusedProject, &reply)) {
        m_ambientHeard = spoken;
        m_ambientAnswer = reply;
        emit ambientChanged();
        return;
    }
    LlmRouter *router = llmRouter();
    const QString endpoint = compatEndpoint();
    if (!router || endpoint.isEmpty())
        return;  // headless or no local model: banter stays banter
    const QString project = m_focusedProject;
    // Redacted (infinity) projects never leave the process: no trunk for them.
    const Project *p = find(project);
    const bool redacted = p && std::any_of(p->sessions.cbegin(), p->sessions.cend(),
                                           [](const SessionInfo &s) { return s.redacted; });
    QString hover;
    if (m_targets && m_targets->hasActiveTarget())
        hover = formatTargetSlice({m_targets->active()}, 1500);
    LlmStreamRequest req;
    req.streamId = QString::fromLatin1(kAmbientStreamId);
    req.endpoint = endpoint;
    req.apiKey = compatKey();
    req.payload = buildAmbientIntentRequest(compatModel(), kCompatSystemPrompt, m_studioContext,
                                            redacted ? QString() : contextTrunk(project), studioLiveState(), hover,
                                            spoken);
    m_ambientRaw.clear();
    m_ambientProject = project;
    m_ambientHeard = spoken;
    m_ambientBusy = true;
    emit ambientChanged();
    router->startStream(req);  // replaces an ambient ask still streaming
}

void Backend::setOpenaiCompatUrl(const QString &url)
{
    const QString trimmed = url.trimmed();
    if (trimmed == m_openaiCompatUrl)
        return;
    m_openaiCompatUrl = trimmed;
    m_compatUrlCleared = trimmed.isEmpty();  // an explicit clear is the only way the file loses it
    emit prefsChanged();
    m_prefsSave.start();
}

void Backend::setOpenaiCompatModel(const QString &model)
{
    const QString trimmed = model.trimmed();
    if (trimmed.isEmpty() || trimmed == m_openaiCompatModel)
        return;
    m_openaiCompatModel = trimmed;
    emit prefsChanged();
    m_prefsSave.start();
}

void Backend::setViewState(const QVariantMap &state)
{
    const QVariantMap v = normalizedViewState(state);
    if (v.isEmpty() || v == m_viewState)
        return;
    m_viewState = v;
    emit viewStateChanged();
    if (!m_prefsPath.isEmpty())
        m_prefsSave.start();  // debounced 400 ms; savePrefs never lets a mock write the real file
}

bool Backend::savePrefs()
{
    m_prefsSave.stop();
    if (m_prefsPath.isEmpty())
        return false;
    // Only the live window may write the real ~/.lapis-next/studio_prefs.json. Mock backends
    // (--test, --snap-*, the demo) write only to a test's explicit temp path, never the real one.
    if (m_mode != Mode::Live
        && QFileInfo(m_prefsPath).absoluteFilePath() == QFileInfo(defaultPrefsPath()).absoluteFilePath())
        return false;
    const QFileInfo fi(m_prefsPath);
    if (!QFileInfo::exists(fi.absolutePath())) {
        if (!QDir().mkpath(fi.absolutePath()))
            return false;
        QFile::setPermissions(fi.absolutePath(),
                              QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner);
    }
    // Merge over what is on disk: keys this build does not know (or another writer added since
    // we loaded) survive. The endpoint keys survive unless they were cleared here on purpose.
    QJsonObject o;
    {
        QFile old(m_prefsPath);
        if (old.open(QIODevice::ReadOnly))
            o = QJsonDocument::fromJson(old.read(256 * 1024)).object();
    }
    static const char *const kOwnedOptional[] = {"windowState", "cardPositions", "gamepadBindings", "openaiCompatModel",
                                                 "mapSort", "clusterOrder", "viewState", "shadowRecordEnabled"};
    for (const char *k : kOwnedOptional)
        o.remove(QLatin1String(k));
    if (m_compatUrlCleared)
        o.remove(QStringLiteral("openaiCompatUrl"));
    o.insert(QStringLiteral("uiScale"), m_uiScale);
    o.insert(QStringLiteral("navMode"), m_navMode);
    o.insert(QStringLiteral("theme"), m_theme);
    o.insert(QStringLiteral("defaultModel"), m_chatModel);
    o.insert(QStringLiteral("defaultEffort"), m_chatEffort);
    o.insert(QStringLiteral("effortSchema"), 2);
    o.insert(QStringLiteral("modelSuggestions"), m_modelSuggestions);
    o.insert(QStringLiteral("jarvisBarMode"), m_jarvisBarMode);
    if (!m_openaiCompatUrl.isEmpty())
        o.insert(QStringLiteral("openaiCompatUrl"), m_openaiCompatUrl);
    if (!m_openaiCompatModel.isEmpty() && m_openaiCompatModel != QLatin1String("qwen3.8-flash-next"))
        o.insert(QStringLiteral("openaiCompatModel"), m_openaiCompatModel);
    if (!m_openaiCompatKey.isEmpty())
        o.insert(QStringLiteral("openaiCompatKey"), m_openaiCompatKey);
    o.insert(QStringLiteral("showCodeDiffs"), m_showCodeDiffs);
    if (!m_shadowPref)
        o.insert(QStringLiteral("shadowRecordEnabled"), false);
    if (m_windowState.hasGeometry() || m_windowState.fullscreen || m_windowState.maximized)
        o.insert(QStringLiteral("windowState"), m_windowState.toJson());
    if (!m_cardPositions.isEmpty()) {
        QJsonObject cards;
        for (auto it = m_cardPositions.cbegin(); it != m_cardPositions.cend(); ++it) {
            const QVariantMap c = it.value().toMap();
            cards.insert(it.key(), QJsonObject{{QStringLiteral("x"), c.value(QStringLiteral("x")).toDouble()},
                                               {QStringLiteral("y"), c.value(QStringLiteral("y")).toDouble()}});
        }
        o.insert(QStringLiteral("cardPositions"), cards);
    }
    if (m_mapSort != QLatin1String("category"))
        o.insert(QStringLiteral("mapSort"), m_mapSort);
    if (m_clusterOrder != defaultClusterOrder())
        o.insert(QStringLiteral("clusterOrder"), QJsonArray::fromStringList(m_clusterOrder));
    if (!m_viewState.isEmpty())
        o.insert(QStringLiteral("viewState"), QJsonObject::fromVariantMap(m_viewState));
    {
        const QHash<QString, QString> padMap = m_gamepad.bindingsMap();
        const QHash<QString, QString> defMap = GamepadState::defaultBindings();
        if (padMap != defMap) {
            QJsonObject padObj;
            for (const QString &k : GamepadState::validInputs())
                padObj.insert(k, padMap.value(k));
            o.insert(QStringLiteral("gamepadBindings"), padObj);
        }
    }
    QSaveFile f(m_prefsPath);  // temp file + rename: a crash never leaves half a file
    if (!f.open(QIODevice::WriteOnly))
        return false;
    f.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
    f.write(QJsonDocument(o).toJson(QJsonDocument::Indented));
    if (!f.commit())
        return false;
    return QFile::setPermissions(m_prefsPath, QFileDevice::ReadOwner | QFileDevice::WriteOwner);
}

void Backend::setToast(const QString &t)
{
    m_toast = t;
    emit toastChanged();
}

Backend::Project *Backend::find(const QString &key)
{
    for (Project &p : m_projects)
        if (p.key == key)
            return &p;
    return nullptr;
}

QString Backend::activeSessionId(const QString &projectKey) const
{
    for (const Project &p : m_projects) {
        if (p.key != projectKey)
            continue;
        const SessionInfo *best = nullptr;
        for (const SessionInfo &s : p.sessions)
            if ((!s.ident.isEmpty() || !s.id.isEmpty()) && (!best || s.mtime > best->mtime))
                best = &s;
        // The harness uuid names the transcript the memory index read; a session without one
        // (a fixture, a fresh kickoff) falls back to its Lapis id.
        return best ? (best->ident.isEmpty() ? best->id : best->ident) : QString();
    }
    return {};
}

QString Backend::memoryContext(const QString &projectKey, const QString &question) const
{
    if (m_memory.size() == 0 || question.trimmed().isEmpty())
        return {};
    return m_memory.query(question, activeSessionId(projectKey), 4, 3200);
}

void Backend::refreshMemory()
{
    if (m_mode != Mode::Live)
        return;
    MemorySources src = MemorySources::defaults();
    for (const Project &p : m_projects)
        for (const SessionInfo &s : p.sessions)
            if (s.redacted && !s.ident.isEmpty())
                src.excludeSessionIds.insert(s.ident);  // a redacted category is never read
    m_memory.refreshAsync(src);
}

void Backend::setWindowState(const WindowState &state)
{
    if (state == m_windowState)
        return;
    m_windowState = state;
    if (!m_prefsPath.isEmpty())
        savePrefs();  // now, not debounced: build scripts pkill the app right after a resize
}

int Backend::pdfPages(const QString &path) const
{
    // Opening a PDFDocument costs milliseconds; QML re-asks on every artifact refresh.
    const qint64 mtime = QFileInfo(path).lastModified().toMSecsSinceEpoch();
    const auto it = m_pdfPages.constFind(path);
    if (it != m_pdfPages.cend() && it->first == mtime)
        return it->second;
    const int n = pdfPageCount(path);
    m_pdfPages.insert(path, {mtime, n});
    return n;
}

bool Backend::isRedactedCategory(const QString &category)
{
    return category.compare(QLatin1String("infinity"), Qt::CaseInsensitive) == 0;
}

int Backend::projectIndex(const QString &key) const
{
    for (int i = 0; i < m_projectList.size(); ++i)
        if (m_projectList[i].toMap().value(QStringLiteral("key")).toString() == key)
            return i;
    return -1;
}

// --- publishing -----------------------------------------------------------------------

QVariantMap Backend::projectMap(const Project &p) const
{
    QVariantMap m;
    m[QStringLiteral("key")] = p.key;
    m[QStringLiteral("name")] = p.name;
    m[QStringLiteral("dir")] = tildePath(expandHome(p.dir));
    QString status = QStringLiteral("idle");
    QVariantList sessions;
    QVariantList dots;
    // Any infinity session marks the project: its transcripts feed this window, but nothing that
    // prints or leaves the process (--probe-live) may show their contents.
    bool redacted = false;
    for (const SessionInfo &s : p.sessions) {
        const QString st = m_statusById.value(s.id, s.status.isEmpty() ? QStringLiteral("idle") : s.status);
        if (statusRank(st) > statusRank(status))
            status = st;
        redacted = redacted || s.redacted;
        QVariantMap sm;
        sm[QStringLiteral("id")] = s.id;
        sm[QStringLiteral("shortId")] = s.id.left(8);
        sm[QStringLiteral("title")] = s.title;
        sm[QStringLiteral("harness")] = s.harness;
        sm[QStringLiteral("status")] = st;
        sessions << sm;
        dots << statusRank(st);
    }
    if (p.pending && statusRank(status) < statusRank(QStringLiteral("working")))
        status = QStringLiteral("working");
    m[QStringLiteral("status")] = status;
    m[QStringLiteral("active")] = status == QLatin1String("working") || status == QLatin1String("needs-ok");
    m[QStringLiteral("sessions")] = sessions;
    m[QStringLiteral("dots")] = dots;
    m[QStringLiteral("sessionCount")] = int(p.sessions.size());
    m[QStringLiteral("redacted")] = redacted;
    QString primaryCat;
    bool activeInLapis = false;
    for (const SessionInfo &s : p.sessions) {
        if (primaryCat.isEmpty() && !s.category.isEmpty())
            primaryCat = s.category;
        if (m_selectedAgentIds.contains(s.id))
            activeInLapis = true;
    }
    m[QStringLiteral("category")] = primaryCat;
    m[QStringLiteral("cluster")] = mapClusterFor(p.key, primaryCat, p.dir);
    m[QStringLiteral("mtime")] = p.mtime;
    m[QStringLiteral("isActiveInLapis")] = activeInLapis;
    // The metric stays in the map (Stage, briefing proof) but the island shows it only when pinned;
    // the island itself speaks plain English: what the agent is on, and its latest step or blocker.
    m[QStringLiteral("headline")] = p.ex.headline;
    m[QStringLiteral("metricPinned")] = m_pinnedMetrics.contains(p.key);
    const IslandState is = islandState(p.ex, status);
    m[QStringLiteral("nowLine")] = is.now;
    // A tab island named from its request ("Add support to PS4 controller") would repeat itself
    // as "Adding support to PS4 controller": the job line drops when it restates the name.
    if (p.key.startsWith(QLatin1String("~tab:")) && !is.blocker) {
        auto tail = [](const QString &s) { return s.section(QLatin1Char(' '), 1).simplified().toLower(); };
        if (!is.now.isEmpty() && tail(is.now) == tail(p.name))
            m[QStringLiteral("nowLine")] = QString();
    }
    m[QStringLiteral("lastLine")] = is.last;
    m[QStringLiteral("lastLong")] = is.lastLong;
    m[QStringLiteral("blocker")] = is.blocker;
    m[QStringLiteral("needsMe")] = projectNeedsMe(status, is.blocker);
    // Plain-English job name: for pushes running over 15 min, or a folder / tab name that does
    // not read as English, the island and Stage header say what the agent is doing right now.
    qint64 started = 0;
    for (const SessionInfo &s : p.sessions)
        started = std::max(started, s.started);
    const qint64 runMs = started > 0 ? QDateTime::currentMSecsSinceEpoch() - started : 0;
    const bool longRunning = runMs > 15 * 60 * 1000;
    m[QStringLiteral("longRunning")] = longRunning;
    const bool showJob = !p.ex.jobName.isEmpty() && (longRunning || isCrypticName(p.name))
        && p.ex.jobName.compare(p.name, Qt::CaseInsensitive) != 0;
    m[QStringLiteral("jobName")] = showJob ? p.ex.jobName : QString();
    m[QStringLiteral("params")] = p.ex.params;
    m[QStringLiteral("fresh")] = m_fresh.value(p.key);
    m[QStringLiteral("spark")] = p.ex.spark;
    m[QStringLiteral("sparkBest")] = p.ex.sparkBest;
    // The human-interest pass, while it still describes this transcript: its state line leads the
    // island (a blocker question still wins) and its key points replace the local summary text.
    QVariantList artifacts = p.ex.artifacts;
    QString summary;
    QString stateLine;
    if (const auto f = m_humanFilter.constFind(p.key); f != m_humanFilter.cend() && f->mtime == p.mtime) {
        summary = f->summary;
        stateLine = stripMarkdownLinks(f->stateLine);
        if (!stateLine.isEmpty() && !is.blocker)
            m[QStringLiteral("lastLine")] = stateLine;
        if (!summary.isEmpty())
            for (QVariant &v : artifacts) {
                QVariantMap a = v.toMap();
                if (a.value(QStringLiteral("kind")).toString() != QLatin1String("summary"))
                    continue;
                a[QStringLiteral("text")] = summary;
                a[QStringLiteral("points")] = summary.split(QLatin1Char('\n'), Qt::SkipEmptyParts);
                a[QStringLiteral("filtered")] = true;
                v = a;
                break;
            }
        m[QStringLiteral("formHint")] = f->form;
    }
    m[QStringLiteral("summary")] = summary;
    m[QStringLiteral("stateLine")] = stateLine;
    // Visual first: the deck leads with charts, tables, media and text; diffs only on request.
    int diffCount = 0;
    for (const QVariant &v : std::as_const(artifacts))
        diffCount += v.toMap().value(QStringLiteral("kind")).toString() == QLatin1String("diff") ? 1 : 0;
    m[QStringLiteral("diffCount")] = diffCount;
    m[QStringLiteral("artifacts")] = stageArtifacts(artifacts, m_showCodeDiffs);
    m[QStringLiteral("prose")] = p.ex.prose;
    m[QStringLiteral("thinking")] = p.ex.thinking;
    // The conversation (Ladder). A session with no local transcript (remote over SSH) still has
    // its newest lapis-peers board reply: it fills the island's latest line and the thread.
    QVariantList turns = p.ex.threadTurns;
    QString boardSnippet;
    for (const SessionInfo &s : p.sessions)  // newest session first
        if (boardSnippet.isEmpty())
            boardSnippet = m_snippetById.value(s.id);
    const bool noLocalText = turns.isEmpty() && p.ex.prose.isEmpty() && p.ex.recentTurns.isEmpty();
    if (noLocalText && !boardSnippet.isEmpty()) {
        turns << QVariantMap{{QStringLiteral("role"), QStringLiteral("assistant")},
                             {QStringLiteral("text"), boardSnippet},
                             {QStringLiteral("source"), QStringLiteral("board")}};
        const QString first = sentencesOf(boardSnippet).value(0, boardSnippet.simplified());
        if (!is.blocker && stateLine.isEmpty()) {
            m[QStringLiteral("lastLine")] = shortLine(first, 9);
            m[QStringLiteral("lastLong")] = shortLine(first, 16);
        }
    }
    m[QStringLiteral("turns")] = turns;
    m[QStringLiteral("boardSnippet")] = boardSnippet;

    QVariantMap ladder;
    ladder[QStringLiteral("goal")] = p.ladder.goal;
    ladder[QStringLiteral("assumptions")] = p.ladder.assumptions;
    ladder[QStringLiteral("arms")] = p.ladder.arms;
    ladder[QStringLiteral("armsLabel")] = p.ladder.armsLabel;
    ladder[QStringLiteral("winner")] = p.ladder.winner;
    m[QStringLiteral("ladder")] = ladder;
    return m;
}

void Backend::publish()
{
    QElapsedTimer clock;
    clock.start();
    // Lock existing card slots so transcript/mtime updates never shuffle the user's muscle memory.
    // On first publish (empty m_projectOrder), seed order by pending then recency; after that,
    // existing keys keep their exact indices and only newly discovered projects append at the end.
    if (m_projectOrder.isEmpty()) {
        std::stable_sort(m_projects.begin(), m_projects.end(), [](const Project &a, const Project &b) {
            if (a.pending != b.pending)
                return a.pending;
            return a.mtime > b.mtime;
        });
        for (const Project &p : std::as_const(m_projects))
            m_projectOrder << p.key;
    } else {
        for (const Project &p : std::as_const(m_projects)) {
            if (!m_projectOrder.contains(p.key)) {
                if (p.pending)
                    m_projectOrder.prepend(p.key);
                else
                    m_projectOrder << p.key;
            }
        }
        std::stable_sort(m_projects.begin(), m_projects.end(), [this](const Project &a, const Project &b) {
            return m_projectOrder.indexOf(a.key) < m_projectOrder.indexOf(b.key);
        });
    }
    QVariantList list;
    for (const Project &p : m_projects) {
        if (list.size() >= 12)
            break;
        list << projectMap(p);
    }
    // Board polls, the age timer and transcript rescans often rebuild identical data. Emitting it
    // anyway re-runs every QML binding and rebuilds delegates for nothing.
    if (list != m_projectList) {
        m_projectList = list;
        emit projectsChanged();
        m_stall.emits += 1;
    }
    const double ms = clock.nsecsElapsed() / 1e6;  // includes the synchronous QML rebinding
    m_stall.publishMs += ms;
    m_stall.publishes += 1;
    noteStall(ms);
}

namespace {

// The first `frac` of `s`, cut at a token boundary (a word plus its trailing space or newline),
// so mock replies arrive the way a model streams them: whole tokens, never half a letter.
QString tokenPrefix(const QString &s, double frac)
{
    if (frac >= 1.0)
        return s;
    if (frac <= 0.0)
        return {};
    static const QRegularExpression tok(QStringLiteral("\\S+\\s*"));
    QList<qsizetype> ends;
    for (auto it = tok.globalMatch(s); it.hasNext();)
        ends << it.next().capturedEnd(0);
    if (ends.isEmpty())
        return {};
    const qsizetype n = qsizetype(frac * ends.size());
    return n <= 0 ? QString() : s.left(ends.value(n - 1));
}

void ladderFromExtracted(const Extracted &ex, const QString &fallbackGoal, QString *goal, QVariantList *assumptions,
                         QVariantList *arms, QString *armsLabel, QString *winner)
{
    // A Lapis tab title ("Phonon fn key toggle not working") states the goal better than a
    // transcript's first line; a title that is just the folder name does not.
    const bool titleUseful = !fallbackGoal.trimmed().isEmpty() && fallbackGoal.contains(QLatin1Char(' '));
    *goal = titleUseful ? clampWords(fallbackGoal, 7) : (ex.goal.isEmpty() ? clampWords(fallbackGoal, 7) : ex.goal);
    assumptions->clear();
    for (const QString &a : ex.assumptions)
        *assumptions << QVariantMap{{QStringLiteral("text"), a}, {QStringLiteral("pinned"), false}};
    *arms = ex.arms;
    *armsLabel = ex.arms.isEmpty() ? QStringLiteral("No arms yet") : QStringLiteral("%1 arms tested").arg(ex.arms.size());
    *winner = ex.winner;
}

}  // namespace

// --- mock -------------------------------------------------------------------------------

void Backend::loadFixtures()
{
    qint64 t = 1000;
    for (const FixtureProject &f : fixtureProjects()) {
        Project p;
        p.key = f.key;
        p.name = f.key;
        p.dir = f.dir;
        p.mockTurns = f.turns;
        p.ex = extractArtifacts(f.turns);
        p.mtime = t--;
        SessionInfo s;
        s.id = QStringLiteral("f1x7%1-0000-0000-0000-000000000000").arg(qHash(f.key) % 10000, 4, 10, QLatin1Char('0'));
        s.title = f.key;
        s.harness = QStringLiteral("claude");
        s.status = f.status;
        // Working fixtures have been running for a while (long-horizon pushes get a job name).
        if (f.status == QLatin1String("working"))
            s.started = QDateTime::currentMSecsSinceEpoch() - 47 * 60 * 1000;
        p.sessions << s;
        ladderFromExtracted(p.ex, f.key, &p.ladder.goal, &p.ladder.assumptions, &p.ladder.arms, &p.ladder.armsLabel,
                            &p.ladder.winner);
        m_projects << p;
    }
}

void Backend::setAutoAdvance(bool on)
{
    if (on) {
        m_autoClock.start();
        m_autoTimer.start(16);
    } else {
        m_autoTimer.stop();
    }
}

void Backend::advanceKickoff(Project &p, double t)
{
    static const char *names[4] = {"baseline", "128x256", "dcp4-split", "128-reg"};
    static const char *values[4] = {"11.3", "17.9", "22.4", "29.6"};
    static const double durations[4] = {2.0, 2.5, 2.9, 3.4};
    if (t >= 0.8 && p.ladder.assumptions.isEmpty()) {
        p.ladder.assumptions << QVariantMap{{QStringLiteral("text"), QStringLiteral("DCP=4")}, {QStringLiteral("pinned"), false}}
                             << QVariantMap{{QStringLiteral("text"), QStringLiteral("MTP=3")}, {QStringLiteral("pinned"), false}};
    }
    if (t >= 1.6) {
        QVariantList arms;
        bool allDone = true;
        for (int i = 0; i < 4; ++i) {
            const double prog = std::clamp((t - 1.6) / durations[i], 0.0, 1.0);
            allDone = allDone && prog >= 1.0;
            arms << QVariantMap{{QStringLiteral("name"), QLatin1String(names[i])},
                                {QStringLiteral("value"), prog >= 1.0 ? QLatin1String(values[i]) : QLatin1String("")},
                                {QStringLiteral("progress"), prog * (i == 3 ? 1.0 : (11.25 + i * 5.9) / 29.6)},
                                {QStringLiteral("best"), i == 3 && prog >= 1.0}};
        }
        p.ladder.arms = arms;
        p.ladder.armsLabel = allDone ? QStringLiteral("4 arms tested") : QStringLiteral("4 arms running");
    }
    if (t >= 5.2 && p.ladder.winner.isEmpty()) {
        p.ladder.winner = QStringLiteral("128-reg · 29.6 tok/s");
        p.ex = extractArtifacts(kickoffTurns());
        p.pending = false;
        p.mtime = 2000;
        if (!p.sessions.isEmpty())
            p.sessions.first().status = QStringLiteral("finished");
    }
}

void Backend::advance(double dt)
{
    if (m_mode != Mode::Mock)
        return;
    if (m_kickoff.t >= 0) {
        m_kickoff.t += dt;
        if (Project *p = find(m_kickoff.projectKey)) {
            advanceKickoff(*p, m_kickoff.t);
            publish();
        }
        if (m_kickoff.t > 6.0)
            m_kickoff.t = -1;
    }
    if (m_mockQuick.t >= 0) {
        m_mockQuick.t += dt;
        // Thinking streams dim from 0.1 s, the first answer token lands at 0.6 s and replaces it,
        // then the rest arrives token by token, like a warm thinking model.
        const double t = m_mockQuick.t, d = m_mockQuick.duration;
        constexpr double think0 = 0.1, answer0 = 0.6;
        const QString thought = t < think0 ? QString()
            : tokenPrefix(m_mockQuick.thinking, std::min(1.0, (t - think0) / (answer0 - think0)));
        const QString shown = t < answer0 ? QString() : tokenPrefix(m_mockQuick.answer, std::min(1.0, (t - answer0) / (d - answer0)));
        m_quickElapsed = std::min(t, d);
        emit quickTick();
        if (t >= d) {
            m_quickAnswer = m_mockQuick.answer;
            m_quickBusy = false;
            m_quickPhase.clear();
            m_quickSeconds = d;
            m_mockQuick.t = -1;
            noteExchange(m_commandProject, m_quickModel, m_quickQuestion, m_quickAnswer);
            emit quickChanged();
        } else if (shown != m_quickAnswer || thought != m_quickThinking) {
            if (thought != m_quickThinking)
                m_quickThinkDeltas += 1;
            m_quickThinking = thought;
            m_quickAnswer = shown;
            m_quickPhase = !shown.isEmpty() ? QStringLiteral("writing") : QStringLiteral("thinking");
            emit quickChanged();
        }
    }
    if (m_mockSuggestT >= 0) {
        m_mockSuggestT += dt;
        if (m_mockSuggestT >= 0.4) {  // the warm suggestion lands a beat after the heuristic one
            m_mockSuggestT = -1;
            m_suggestion = QStringLiteral("Why did 128-reg beat the baseline?");
            m_suggestionKey = m_suggestPendingKey;
            emit suggestionChanged();
        }
    }
    if (m_mockTrace.t >= 0) {
        m_mockTrace.t += dt;
        const double t = m_mockTrace.t;
        const double d = m_mockTrace.duration;
        if (t < d) {
            m_traceThinking = tokenPrefix(m_mockTrace.thinking, std::min(1.0, std::max(0.0, t - 0.25) / (d * 0.9 - 0.25)));
        } else {
            m_traceThinking = m_mockTrace.thinking;
            m_traceBusy = false;
            m_traceSeconds = d;
            const double at = std::min(1.0, (t - d) / 0.5);
            m_traceAnswer = tokenPrefix(m_mockTrace.answer, at);
            if (at >= 1.0)
                m_mockTrace.t = -1;
        }
        emit traceChanged();
    }
}

// --- actions ------------------------------------------------------------------------------

QString Backend::contextFor(const QString &projectKey, int artifactIndex) const
{
    QString head, slice, rest;
    if (!contextPieces(projectKey, artifactIndex, &head, &slice, &rest))
        return {};
    return head + slice + rest;
}

QString Backend::contextTrunk(const QString &projectKey) const
{
    QString head, rest;
    if (!contextPieces(projectKey, -1, &head, nullptr, &rest))
        return {};
    return head + rest;
}

QString Backend::contextSlice(const QString &projectKey, int artifactIndex) const
{
    QString slice;
    if (!contextPieces(projectKey, artifactIndex, nullptr, &slice, nullptr))
        return {};
    return slice.trimmed();
}

bool Backend::contextPieces(const QString &projectKey, int artifactIndex, QString *head, QString *slice,
                            QString *rest) const
{
    for (const Project &p : m_projects) {
        if (p.key != projectKey)
            continue;
        if (head)
            *head = QStringLiteral("Project: %1 (%2).").arg(p.name, tildePath(expandHome(p.dir)));
        QString ctx;  // the on-screen slice
        // Indexed like the Stage deck the user sees (projectMap), not the extracted order.
        const QVariantMap a = artifactIndex >= 0 ? stageArtifacts(p.ex.artifacts, m_showCodeDiffs).value(artifactIndex).toMap()
                                                 : QVariantMap();
        if (slice && !a.isEmpty()) {
            const QString kind = a.value(QStringLiteral("kind")).toString();
            ctx += QStringLiteral(" On screen: %1 \"%2\" (%3).")
                       .arg(kind, a.value(QStringLiteral("title")).toString(),
                            a.value(QStringLiteral("metric")).toString());
            const QStringList labels = a.value(QStringLiteral("labels")).toStringList();
            const QStringList vt = a.value(QStringLiteral("valueText")).toStringList();
            for (int i = 0; i < labels.size() && i < 12; ++i)
                ctx += QStringLiteral(" %1=%2").arg(labels[i], vt.value(i));
            if (kind == QLatin1String("table")) {
                const QStringList hdr = a.value(QStringLiteral("header")).toStringList();
                const QVariantList rows = a.value(QStringLiteral("rows")).toList();
                if (!hdr.isEmpty())
                    ctx += QStringLiteral("\nTable:\n| ") + hdr.join(QStringLiteral(" | ")) + QStringLiteral(" |");
                for (int r = 0; r < rows.size() && r < 40; ++r)
                    ctx += QStringLiteral("\n| ") + rows[r].toStringList().join(QStringLiteral(" | ")) + QStringLiteral(" |");
            } else if (kind == QLatin1String("summary")) {
                const auto f = m_humanFilter.constFind(p.key);
                const QString text = f != m_humanFilter.cend() && f->mtime == p.mtime && !f->summary.isEmpty()
                    ? f->summary
                    : a.value(QStringLiteral("text")).toString();
                ctx += QStringLiteral("\nKey points:\n") + text.left(2500);
            } else if (kind == QLatin1String("diff")) {
                const QVariantList lines = a.value(QStringLiteral("lines")).toList();
                QStringList out;
                for (int i = 0; i < lines.size() && i < 60; ++i)
                    out << lines[i].toMap().value(QStringLiteral("text")).toString();
                if (!out.isEmpty())
                    ctx += QStringLiteral("\nDiff:\n") + out.join(QLatin1Char('\n')).left(2500);
            } else if (a.contains(QLatin1String("path"))) {
                ctx += QStringLiteral("\nFile: ") + tildePath(a.value(QStringLiteral("path")).toString());
            }
        }
        if (slice)
            *slice = ctx;
        if (!rest)
            return true;
        ctx.clear();  // from here on: the stable trunk after the header
        if (!p.ex.artifacts.isEmpty()) {
            QStringList all;
            for (const QVariant &v : p.ex.artifacts) {
                const QVariantMap m = v.toMap();
                all << QStringLiteral("%1 \"%2\"").arg(m.value(QStringLiteral("kind")).toString(),
                                                       m.value(QStringLiteral("title")).toString());
            }
            ctx += QStringLiteral("\nArtifacts the session produced (newest first): ") + all.join(QStringLiteral(", "));
        }
        if (!p.ex.goal.isEmpty())
            ctx += QStringLiteral("\nFirst request: ") + p.ex.goal.simplified().left(400);
        if (!p.ex.recentTurns.isEmpty())
            ctx += QStringLiteral("\nLast %1 turns:\n").arg(p.ex.recentTurns.size()) + p.ex.recentTurns.join(QLatin1Char('\n'));
        if (!p.ex.files.isEmpty())
            ctx += QStringLiteral("\nFiles touched: ") + p.ex.files.join(QStringLiteral(", "));
        if (!p.ex.toolCalls.isEmpty())
            ctx += QStringLiteral("\nRecent tool calls:\n- ") + p.ex.toolCalls.join(QStringLiteral("\n- "));
        if (!p.ex.prose.isEmpty())
            ctx += QStringLiteral("\nAgent's latest write-up:\n") + p.ex.prose.left(1500);
        *rest = ctx;
        return true;
    }
    return false;
}

void Backend::clearAnswers()
{
    resetCuaLoop();  // the user dismissed the answer: no more automatic turns
    m_quickQuestion.clear();
    m_quickAnswer.clear();
    m_quickBusy = false;
    m_traceQuestion.clear();
    m_traceThinking.clear();
    m_traceAnswer.clear();
    m_traceBusy = false;
    emit quickChanged();
    emit traceChanged();
}

void Backend::ask(const QString &projectKey, int artifactIndex, const QString &question)
{
    const QString q = question.trimmed();
    if (q.isEmpty())
        return;
    m_commandProject = projectKey;
    const bool followup = m_cuaFollowupAsk;
    if (!followup) {
        // A new question from the user: any closed CUA loop still running for the last one stops.
        resetCuaLoop();
        m_quickArtifact = artifactIndex;
        m_quickNote = q;
    }
    QString commandReply;
    if (!followup && handleStudioCommand(q, projectKey, &commandReply)) {
        // A UI command: it already ran; the answer strip shows its one-line confirmation.
        m_quickQuestion = q;
        m_quickAnswer = m_quickAnswerRaw = commandReply;
        m_quickThinking.clear();
        m_quickThinkRaw.clear();
        m_quickPhase.clear();
        m_quickBusy = false;
        m_quickSeconds = m_quickElapsed = 0;
        m_quickThinkTokens = -1;
        m_quickModel = QStringLiteral("studio");
        m_mockQuick.t = -1;
        emit quickChanged();
        emit quickTick();
        return;
    }
    if (!followup)
        m_quickQuestion = q;  // a follow-up keeps showing the user's own question
    m_quickAnswer.clear();
    m_quickThinking.clear();
    m_quickAnswerRaw.clear();
    m_quickThinkRaw.clear();
    m_quickPhase = QStringLiteral("sending");  // something is on screen from the first frame
    m_quickBusy = true;
    m_quickSeconds = 0;
    m_quickElapsed = 0;
    m_quickThinkTokens = -1;
    m_quickFirstEventMs = m_quickFirstThinkMs = m_quickFirstTokenMs = -1;
    m_quickThinkDeltas = 0;
    m_quickModel = m_chatModel;
    m_lastAskModel = m_chatModel;
    m_lastAskEffort = m_chatEffort;
    m_lastAskChannel = channelKeyFor(m_chatModel, m_chatEffort);
    emit quickChanged();
    emit quickTick();
    if (m_mode == Mode::Mock) {
        static const QStringList canned = {QStringLiteral("128-reg: dequant stays in registers, zero spills."),
                                           QStringLiteral("Baseline spilled 212 registers per thread.")};
        // Thinking streams first (dim), then the answer replaces it, like a real thinking model.
        // A parameter-scoped question ("About MTP = on: ...") gets an answer about that parameter.
        const bool aboutMtp = q.startsWith(QLatin1String("About MTP"));
        m_mockQuick = MockReply{0.0, 1.3, q,
                                aboutMtp ? QStringLiteral("Weighing the MTP draft acceptance against the DCP=4 baseline at c1")
                                         : QStringLiteral("Comparing spills and registers across the four arms before answering"),
                                aboutMtp ? QStringLiteral("Yes: MTP adds 15 tok/s at c1, 241 vs 226 on DCP=4.")
                                         : canned.value(m_mockQuickCount++ % canned.size())};
        return;
    }
    // CLI channels (claude / codex / agy) get the same shared conversation and live screen state
    // the OpenAI-compatible tiers carry, so a model switch never loses the thread.
    QString prompt = contextFor(projectKey, artifactIndex);
    const QString live = studioLiveState();
    if (!live.isEmpty())
        prompt += QStringLiteral("\n\nStudio live state:\n") + live;
    const QString recent = recentConversationBlock();
    if (!recent.isEmpty())
        prompt += QStringLiteral("\n\n") + recent;
    prompt += QStringLiteral("\n\n") + currentTimeLine(QDateTime::currentDateTime());
    prompt += QStringLiteral("\n\nQuestion: ") + q;
    m_quickClock.start();
    m_quickTicker.start();
    if (providerFor(m_chatModel) == QLatin1String("openai-compat")) {
        askOpenAiCompat(projectKey, artifactIndex, q);
        return;
    }
    if (providerFor(m_chatModel) == QLatin1String("openrouter")) {
        askOpenRouter(projectKey, artifactIndex, q);
        return;
    }
    Channel *c = ensureChannel(m_chatModel, m_chatEffort);
    if (!c) {
        m_quickBusy = false;
        m_quickAnswer = providerFor(m_chatModel) + QStringLiteral(" helper unavailable");
        emit quickChanged();
        return;
    }
    m_quickChannel = c->key;
    c->inFlight += 1;
    c->lastUsed = QDateTime::currentMSecsSinceEpoch();
    if (c->provider == QLatin1String("openai")) {
        c->queue << codexTurnParams(QString(), prompt, m_chatModel, m_chatEffort);
        flushCodex(c);
        return;
    }
    const QByteArray line = c->provider == QLatin1String("google") ? agyUserLine(prompt) : claudeUserLine(prompt);
    if (c->proc && c->proc->state() == QProcess::Running)
        c->proc->write(line);
    else
        c->pending += line;  // written the moment the process is up
}

// --- persistent provider channels ----------------------------------------------------------------

void Backend::warmChannel()
{
    if (m_mode == Mode::Live)
        ensureChannel(m_chatModel, m_chatEffort);
}

Backend::Channel *Backend::ensureChannel(const QString &model, const QString &effort)
{
    const QString prov = providerFor(model);
    if (m_mode != Mode::Live || prov == QLatin1String("openai-compat") || prov == QLatin1String("openrouter"))
        return nullptr;
    const QString key = channelKeyFor(model, effort);
    if (key.isEmpty())
        return nullptr;
    if (Channel *c = m_channels.value(key)) {
        if (c->starting || (c->proc && c->proc->state() != QProcess::NotRunning))
            return c;
        m_channels.remove(key);  // a dead process: start a fresh one below
        delete c;
    }
    evictChannels();
    auto *c = new Channel;
    c->key = key;
    c->provider = providerFor(model);
    c->model = model;
    c->effort = clampEffort(model, effort);
    c->lastUsed = QDateTime::currentMSecsSinceEpoch();
    m_channels.insert(key, c);
    if (c->provider == QLatin1String("anthropic")) {
        // A fresh session id, announced to the cswap router first so the helper's requests land in
        // the right account pool even though it loads no hooks of its own.
        c->sessionId = QUuid::createUuid().toString(QUuid::WithoutBraces);
        if (QFileInfo(cswapHookBin()).isExecutable()) {
            if (QProcess *hook = spawn(cswapHookBin(), {QStringLiteral("hook")}, QDir::tempPath())) {
                c->starting = true;
                hook->write(cswapHookPayload(c->sessionId));
                hook->closeWriteChannel();
                auto started = std::make_shared<bool>(false);
                auto go = [this, started, key] {
                    if (*started)
                        return;
                    *started = true;
                    if (Channel *ch = m_channels.value(key)) {
                        ch->starting = false;
                        spawnChannel(ch);
                    }
                };
                connect(hook, &QProcess::finished, this, go);
                connect(hook, &QProcess::errorOccurred, this, go);
                QTimer::singleShot(3000, this, go);  // never let a slow hook hold the helper back
                return c;
            }
        }
    }
    spawnChannel(c);
    return m_channels.value(key);
}

qint64 Backend::channelPid(const QString &key) const
{
    const Channel *c = m_channels.value(key);
    return c && c->proc ? c->proc->processId() : 0;
}

void Backend::spawnChannel(Channel *c)
{
    const QString key = c->key;
    c->buf.clear();
    if (c->provider == QLatin1String("anthropic"))
        c->proc = spawn(claudeBin(), quickHelperArgs(c->sessionId, c->model, c->effort), QDir::tempPath(), quickHelperEnv());
    else if (c->provider == QLatin1String("google"))
        c->proc = spawn(agyBin(), agyChatArgs(c->model, c->effort), QDir::tempPath());
    else
        c->proc = spawn(codexBin(), codexAppServerArgs(), QDir::tempPath());
    if (!c->proc) {
        if (m_quickBusy && m_quickChannel == key) {
            m_quickBusy = false;
            m_quickAnswer = c->provider + QStringLiteral(" helper unavailable");
            emit quickChanged();
        }
        m_channels.remove(key);
        delete c;
        return;
    }
    QProcess *proc = c->proc;
    connect(proc, &QProcess::readyReadStandardOutput, this, [this, key] { onChannelOutput(key); });
    connect(proc, &QProcess::started, this, [this, key] {
        Channel *ch = m_channels.value(key);
        if (ch && ch->proc && !ch->pending.isEmpty()) {
            ch->proc->write(ch->pending);
            ch->pending.clear();
        }
    });
    connect(proc, &QProcess::finished, this, [this, key, proc] {
        Channel *ch = m_channels.value(key);
        if (!ch || ch->proc != proc)
            return;
        // The daemon died: keep what streamed; the next ask to this model starts a fresh one.
        if (m_quickBusy && m_quickChannel == key)
            finishQuick();
        m_channels.remove(key);
        delete ch;
    });
    if (c->provider == QLatin1String("openai")) {
        // JSON-RPC handshake, then one thread for answers (and one for suggestions on demand).
        proc->write(codexRpc(c->nextId++, QStringLiteral("initialize"),
                             QJsonObject{{QStringLiteral("clientInfo"),
                                          QJsonObject{{QStringLiteral("name"), QStringLiteral("lapis_studio")},
                                                      {QStringLiteral("version"), QStringLiteral("1.2")}}}}));
        proc->write(codexRpc(-1, QStringLiteral("initialized"), {}));
        c->threadReq = c->nextId++;
        proc->write(codexRpc(c->threadReq, QStringLiteral("thread/start"),
                             codexThreadStartParams(QString::fromLatin1(kQuickSystemPrompt))));
    }
}

void Backend::evictChannels()
{
    // At most four live daemons: the least recently used idle one goes first.
    while (m_channels.size() >= 4) {
        Channel *victim = nullptr;
        for (Channel *c : std::as_const(m_channels))
            if (c->inFlight == 0 && c->key != m_quickChannel && (!victim || c->lastUsed < victim->lastUsed))
                victim = c;
        if (!victim)
            return;
        m_channels.remove(victim->key);
        if (victim->proc) {
            victim->proc->disconnect(this);
            victim->proc->kill();
        }
        delete victim;
    }
}

void Backend::flushCodex(Channel *c)
{
    if (!c->proc)
        return;
    while (!c->thread.isEmpty() && !c->queue.isEmpty()) {
        QJsonObject t = c->queue.takeFirst();
        t.insert(QStringLiteral("threadId"), c->thread);
        c->proc->write(codexRpc(c->nextId++, QStringLiteral("turn/start"), t));
    }
    while (!c->suggestThread.isEmpty() && !c->suggestQueue.isEmpty()) {
        QJsonObject t = c->suggestQueue.takeFirst();
        t.insert(QStringLiteral("threadId"), c->suggestThread);
        c->proc->write(codexRpc(c->nextId++, QStringLiteral("turn/start"), t));
    }
}

void Backend::finishQuick(bool ok)
{
    if (ok && m_quickBusy) {
        m_cuaAnswerStream.clear();
        runAnswerTags(&m_quickAnswer);  // a model's [[ACTION:..]] runs, and never shows as text
        noteExchange(m_commandProject, m_quickModel, m_quickNote.isEmpty() ? m_quickQuestion : m_quickNote, m_quickAnswer);
    }
    m_quickBusy = false;
    m_quickPhase.clear();
    m_quickSeconds = m_quickClock.isValid() ? m_quickClock.elapsed() / 1000.0 : 0;
    m_quickElapsed = m_quickSeconds;
    m_quickTicker.stop();
    emit quickChanged();
    emit quickTick();
}

void Backend::askOpenAiCompat(const QString &projectKey, int artifactIndex, const QString &question)
{
    QString base = m_openaiCompatUrl;
    if (base.isEmpty())
        base = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_URL").trimmed();
    if (base.isEmpty()) {
        m_quickAnswer = QStringLiteral("Set openaiCompatUrl in studio_prefs.json or LAPIS_OPENAI_COMPAT_URL");
        finishQuick(false);
        return;
    }
    while (base.endsWith(QLatin1Char('/')))
        base.chop(1);
    const QString endpoint = base.endsWith(QLatin1String("/chat/completions"))
                                 ? base
                                 : base + QStringLiteral("/chat/completions");
    const QString key = compatKey();

    if (m_suggestReply) {
        m_suggestReply->disconnect(this);
        m_suggestReply->abort();
        m_suggestReply->deleteLater();
        m_suggestReply.clear();
    }
    m_quickChannel = channelKeyFor(m_chatModel, m_chatEffort);
    LlmRouter *router = llmRouter();
    if (!router) {
        m_quickAnswer = QStringLiteral("openai-compat: no router");
        finishQuick(false);
        return;
    }

    // Tier 0 system + Tier 1 trunk are byte-identical to what warmProjectPrefix() prefilled when
    // the island opened; the on-screen slice, recent actions, Phonon tail and question ride last.
    const TieredPrompt p = tieredPrompt(projectKey, contextSlice(projectKey, artifactIndex), question, m_sharedHistory);
    LlmStreamRequest req;
    req.streamId = QString::fromLatin1(kQuickStreamId);
    req.endpoint = endpoint;
    req.apiKey = key;
    req.payload = buildTieredChatRequest(p);
    router->startStream(req);  // replaces any quick ask still streaming
}

namespace {
QString readEnvVarsFileKey(const QString &varName)
{
    QFile f(QDir::homePath() + QStringLiteral("/.env_vars"));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text))
        return {};
    QTextStream in(&f);
    const QString prefix = varName + QLatin1Char('=');
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty() || line.startsWith(QLatin1Char('#')))
            continue;
        if (line.startsWith(QLatin1String("export ")))
            line = line.mid(7).trimmed();
        if (!line.startsWith(prefix))
            continue;
        QString val = line.mid(prefix.size()).trimmed();
        if (val.size() >= 2 && ((val.front() == QLatin1Char('"') && val.back() == QLatin1Char('"'))
                                || (val.front() == QLatin1Char('\'') && val.back() == QLatin1Char('\'')))) {
            val = val.mid(1, val.size() - 2);
        }
        return val.trimmed();
    }
    return {};
}
} // namespace

QString Backend::openrouterKey() const
{
    QString key = m_openrouterKey;
    if (key.isEmpty())
        key = qEnvironmentVariable("OPENROUTER_API_KEY").trimmed();
    if (key.isEmpty() && m_mode == Mode::Live)
        key = readEnvVarsFileKey(QStringLiteral("OPENROUTER_API_KEY"));
    return key;
}

void Backend::askOpenRouter(const QString &projectKey, int artifactIndex, const QString &question)
{
    const QString key = openrouterKey();
    if (key.isEmpty()) {
        m_quickAnswer = QStringLiteral("Set OPENROUTER_API_KEY (env) or openrouterApiKey in studio_prefs.json");
        finishQuick(false);
        return;
    }
    if (m_suggestReply) {
        m_suggestReply->disconnect(this);
        m_suggestReply->abort();
        m_suggestReply->deleteLater();
        m_suggestReply.clear();
    }
    m_quickChannel = channelKeyFor(m_chatModel, m_chatEffort);
    LlmRouter *router = llmRouter();
    if (!router) {
        m_quickAnswer = QStringLiteral("openrouter: no router");
        finishQuick(false);
        return;
    }
    // Same tiers as the compat ask (the shared ring rides in Tier 2), but the picked model and
    // effort, and OpenRouter's reasoning / provider routing instead of chat_template_kwargs.
    TieredPrompt p = tieredPrompt(projectKey, contextSlice(projectKey, artifactIndex), question, m_sharedHistory);
    p.model = m_chatModel;
    p.effort = m_chatEffort;
    LlmStreamRequest req;
    req.streamId = QString::fromLatin1(kQuickStreamId);
    req.endpoint = QStringLiteral("https://openrouter.ai/api/v1/chat/completions");
    req.apiKey = key;
    req.payload = buildOpenRouterChatRequest(p);
    router->startStream(req);
}

// --- LlmRouter streams ----------------------------------------------------------------------

LlmRouter *Backend::llmRouter()
{
    if (!m_router && m_mode == Mode::Live) {
        m_ownedRouter = std::make_unique<LlmRouter>();
        setLlmRouter(m_ownedRouter.get());
    }
    return m_router.data();
}

void Backend::setLlmRouter(LlmRouter *router)
{
    if (m_router == router)
        return;
    if (m_router)
        m_router->disconnect(this);
    m_router = router;
    if (!router)
        return;
    connect(router, &LlmRouter::streamDelta, this, &Backend::onRouterDelta);
    connect(router, &LlmRouter::streamFinished, this, &Backend::onRouterFinished);
}

QString Backend::compatModel() const
{
    QString model = m_openaiCompatModel;
    if (model.isEmpty())
        model = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_MODEL").trimmed();
    if (model.isEmpty())
        model = m_chatModel;
    return model;
}

QString Backend::compatEffort() const
{
    // The picker's effort applies only when the picked model is the OpenAI-compatible one; card
    // streams under a CLI model (Sonnet, GPT-6, ...) still go to the compat endpoint, no thinking.
    if (providerFor(m_chatModel) != QLatin1String("openai-compat"))
        return QStringLiteral("fast");
    return m_chatEffort.isEmpty() ? QStringLiteral("fast") : m_chatEffort;
}

QString Backend::compatKey() const
{
    QString key = m_openaiCompatKey;
    if (key.isEmpty())
        key = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_KEY").trimmed();
    if (key.isEmpty()) {
        const QString url = !m_openaiCompatUrl.isEmpty()
                                ? m_openaiCompatUrl
                                : qEnvironmentVariable("LAPIS_OPENAI_COMPAT_URL").trimmed();
        if (url.contains(QLatin1String("openrouter.ai"), Qt::CaseInsensitive))
            key = openrouterKey();
    }
    return key;
}

void Backend::onRouterDelta(const QString &streamId, const QString &text, const QString &thinking)
{
    if (streamId == QLatin1String(kReflectionStreamId)) {
        if (m_reflectionBusy)
            m_reflectionRaw += text;
        return;
    }
    if (streamId == QLatin1String(kAmbientStreamId)) {
        if (m_ambientBusy)
            m_ambientRaw += text;  // thinking is off; any that arrives is dropped
        return;
    }
    if (streamId == QLatin1String(kQuickStreamId)) {
        if (!thinking.isEmpty())
            noteThinking(thinking);
        if (!text.isEmpty())
            appendAnswer(text);
        emit quickChanged();
        return;
    }
    auto it = m_streamAsks.find(streamId);
    if (it == m_streamAsks.end())
        return;  // not ours (a shared, injected router)
    it->answer += text;
    emit streamDelta(streamId, text, thinking);
}

void Backend::onRouterFinished(const QString &streamId, const QString &error)
{
    if (streamId == QLatin1String(kReflectionStreamId)) {
        if (!m_reflectionBusy)
            return;
        m_reflectionBusy = false;
        // A dead endpoint or a mangled reply keeps the local lines; never an empty card.
        const SessionReflection r = error.isEmpty() ? parseReflectionReply(m_reflectionRaw) : SessionReflection{};
        if (!r.intent.isEmpty() || !r.rules.isEmpty()) {
            if (r.intent.isEmpty())
                m_reflection.rules = r.rules;
            else
                m_reflection = r;
            if (m_reflection.corrections.isEmpty())
                m_reflection.corrections = QStringLiteral("No steers spotted.");
            m_reflectionSource = compatModel();
        }
        m_reflectionRaw.clear();
        emit reflectionChanged();
        return;
    }
    if (streamId == QLatin1String(kAmbientStreamId)) {
        if (!m_ambientBusy)
            return;
        m_ambientBusy = false;
        const QString raw = m_ambientRaw;
        m_ambientRaw.clear();
        if (!error.isEmpty() || !m_ambientEnabled) {
            emit ambientChanged();
            return;
        }
        const ParsedModelOutput out = parseStudioModelOutput(raw);
        if (out.ignore) {
            m_ambientHeard.clear();  // banter: say nothing, show nothing
            emit ambientChanged();
            return;
        }
        // Overheard speech never acts on its own: known action tags become a proposal on the
        // all-Spaces HUD bar that L3 + R3 runs (or 15 s expires). Plain prose shows as before.
        const QStringList known = studioActionNames();
        QVector<StudioAction> actions;
        for (const StudioAction &c : out.actions)
            if (known.contains(c.name.toLower()))
                actions << c;
        if (!actions.isEmpty()) {
            QString summary = out.cleanText.simplified();
            if (summary.isEmpty()) {
                QStringList parts;
                for (const StudioAction &c : actions)
                    parts << (c.arg.isEmpty() ? c.name : c.name + QLatin1Char(' ') + c.arg);
                summary = parts.join(QStringLiteral(", "));
            }
            stageProposal(actions, clampWords(summary, 10), m_ambientProject);
            m_ambientAnswer = m_pendingProposalSummary + QStringLiteral("  ·  L3+R3 Run");
            emit ambientChanged();
            return;
        }
        m_ambientAnswer = out.cleanText.trimmed();
        if (!m_ambientAnswer.isEmpty())
            noteExchange(m_ambientProject, compatModel(), m_ambientHeard, m_ambientAnswer);
        emit ambientChanged();
        return;
    }
    if (streamId == QLatin1String(kQuickStreamId)) {
        if (!error.isEmpty())
            m_quickAnswer = QStringLiteral("openai-compat: ") + oneLine(error, 12);
        if (m_quickBusy)
            finishQuick(error.isEmpty());
        return;
    }
    auto it = m_streamAsks.find(streamId);
    if (it == m_streamAsks.end())
        return;
    const StreamAsk done = it.value();
    m_streamAsks.erase(it);
    if (error.isEmpty() && !done.answer.trimmed().isEmpty()) {
        QString answer = done.answer;
        const QString keep = m_commandProject;
        m_commandProject = done.projectKey;
        m_cuaAnswerStream = streamId;
        m_cuaAnswerGoal = done.goal;
        runAnswerTags(&answer);  // the card strips the tag itself; the action runs here once
        m_cuaAnswerStream.clear();
        m_cuaAnswerGoal.clear();
        m_commandProject = keep;
        const QString userTurn = done.note.isEmpty() ? done.question : done.note;
        QVector<ChatTurn> &h = m_streamHistory[streamId];
        h << ChatTurn{QStringLiteral("user"), userTurn} << ChatTurn{QStringLiteral("assistant"), answer};
        while (h.size() > kMaxStreamHistory)
            h.remove(0, 2);  // drop the oldest exchange (re-prefills Tier 2 once)
        noteExchange(done.projectKey, compatModel(), userTurn, answer);
    }
    emit streamFinished(streamId, error);
}

void Backend::askStream(const QString &streamId, const QString &projectKey, int artifactIndex,
                        const QString &question)
{
    startConversation(streamId, projectKey, contextSlice(projectKey, artifactIndex), question);
}

void Backend::askStreamWithSlice(const QString &streamId, const QString &projectKey, int artifactIndex,
                                 const QString &targetSlice, const QString &question)
{
    const QString slice = targetSlice.trimmed();
    startConversation(streamId, projectKey,
                      slice.isEmpty() && artifactIndex >= 0 ? contextSlice(projectKey, artifactIndex) : slice, question);
}

TargetRegistry *Backend::targets()
{
    if (!m_targets)
        m_targets = std::make_unique<TargetRegistry>();
    return m_targets.get();
}

void Backend::startConversation(const QString &streamId, const QString &projectKey, const QString &slice,
                                const QString &question)
{
    const QString q = question.trimmed();
    if (streamId.isEmpty() || streamId == QLatin1String(kQuickStreamId) || q.isEmpty())
        return;
    const bool followup = m_cuaFollowupAsk;
    if (!followup)
        resetCuaLoop();  // the user asked something new: a running CUA loop stops
    QString commandReply;
    if (!followup && handleStudioCommand(q, projectKey, &commandReply)) {
        // A UI command: done already; its confirmation is the whole answer, in any mode.
        emit streamDelta(streamId, commandReply, QString());
        emit streamFinished(streamId, QString());
        return;
    }
    LlmRouter *router = llmRouter();
    if (streamId == QLatin1String(kSpotlightStreamId))
        m_spotlightRaw.clear();  // each model reply (a follow-up's too) replaces the shown one
    if (!router && m_testStreamCapture) {
        // Tests: register the ask exactly as a real stream would; the test plays the reply.
        m_streamAsks.insert(streamId, StreamAsk{q, QString(), projectKey, followup ? m_cuaFollowupNote : QString(),
                                                followup ? m_cuaFollowupGoal : q});
        m_testStreamStarts << streamId + QLatin1Char('|') + q;
        m_testLastSlice = slice;
        return;
    }
    if (!router) {
        emit streamFinished(streamId, QStringLiteral("no LLM router (mock mode)"));
        return;
    }
    const QString endpoint = compatEndpoint();
    if (endpoint.isEmpty()) {
        emit streamFinished(streamId, QStringLiteral("Set openaiCompatUrl in studio_prefs.json or LAPIS_OPENAI_COMPAT_URL"));
        return;
    }
    // Tier 2 is the shared ring: every Studio exchange, whichever model or card answered it.
    const TieredPrompt p = tieredPrompt(projectKey, slice, q, m_sharedHistory);
    m_streamAsks.insert(streamId, StreamAsk{q, QString(), projectKey, followup ? m_cuaFollowupNote : QString(),
                                            followup ? m_cuaFollowupGoal : q});
    LlmStreamRequest req;
    req.streamId = streamId;
    req.endpoint = endpoint;
    req.apiKey = compatKey();
    req.payload = buildTieredChatRequest(p);
    router->startStream(req);
}

TieredPrompt Backend::tieredPrompt(const QString &projectKey, const QString &slice, const QString &question,
                                   const QVector<ChatTurn> &history) const
{
    TieredPrompt p;
    p.model = compatModel();
    p.effort = compatEffort();             // "fast" -> enable_thinking false
    p.systemPrompt = kCompatSystemPrompt;  // Tier 0
    p.extraSystemContext = m_studioContext;  // Tier 0 tail, same bytes as the warm
    p.trunk = contextTrunk(projectKey);    // Tier 1: byte-identical to warmProjectPrefix()
    p.history = history;                   // Tier 2
    p.targetSlice = slice;                 // Tier 3 from here on
    p.liveState = studioLiveState();       // what is on screen right now ("" headless)
    p.recentActions = m_recentActions;
    p.phononTail = m_phononTail;
    p.memory = memoryContext(projectKey, question);  // recall rides in Tier 3 only: the prefix stays hot
    p.currentTime = currentTimeLine(QDateTime::currentDateTime());  // Tier 3 only: Tiers 0/1 stay APC-hot
    p.question = question;
    return p;
}

void Backend::cancelStream(const QString &streamId)
{
    if (streamId.isEmpty() || streamId == QLatin1String(kQuickStreamId))
        return;
    if (m_cuaBatch.streamId == streamId)
        resetCuaLoop();  // the user stopped this conversation: no more automatic turns
    m_streamAsks.remove(streamId);
    if (m_router)
        m_router->cancelStream(streamId);
}

void Backend::clearStreamHistory(const QString &streamId)
{
    m_streamHistory.remove(streamId);
}

void Backend::notePhase(const QString &phase)
{
    if (phase.isEmpty() || phase == m_quickPhase)
        return;
    if (m_quickFirstEventMs < 0 && m_quickClock.isValid())
        m_quickFirstEventMs = m_quickClock.elapsed();
    m_quickPhase = phase;
}

void Backend::noteThinking(const QString &delta)
{
    if (delta.isEmpty())
        return;
    if (m_quickClock.isValid()) {
        if (m_quickFirstEventMs < 0)
            m_quickFirstEventMs = m_quickClock.elapsed();
        if (m_quickFirstThinkMs < 0)
            m_quickFirstThinkMs = m_quickClock.elapsed();
    }
    m_quickThinkDeltas += 1;
    // The newest words, so the dim line keeps moving for as long as the model thinks. The raw
    // stream is kept whole: a delta's trailing space must survive until the next delta lands.
    m_quickThinkRaw += delta;
    m_quickThinking = tailWords(stripToolNoise(m_quickThinkRaw), 40);
    if (m_quickPhase != QLatin1String("writing"))
        m_quickPhase = QStringLiteral("thinking");
}

void Backend::noteAnswer(const QString &answer)
{
    if (m_quickClock.isValid() && m_quickFirstTokenMs < 0 && !answer.trimmed().isEmpty()) {
        m_quickFirstTokenMs = m_quickClock.elapsed();
        if (m_quickFirstEventMs < 0)
            m_quickFirstEventMs = m_quickFirstTokenMs;
    }
    m_quickAnswer = answer;
    if (!answer.trimmed().isEmpty())
        m_quickPhase = QStringLiteral("writing");
}

void Backend::appendAnswer(const QString &delta)
{
    // Concatenate raw deltas, then clamp: trimming each delta first glued "No, " + "1001" into "No,1001".
    m_quickAnswerRaw += delta;
    noteAnswer(answerBody(m_quickAnswerRaw));
}

void Backend::onChannelOutput(const QString &key)
{
    Channel *c = m_channels.value(key);
    if (!c || !c->proc)
        return;
    c->buf += c->proc->readAllStandardOutput();
    bool changed = false;
    bool done = false;
    qsizetype nl;
    while ((nl = c->buf.indexOf('\n')) >= 0) {
        const QByteArray line = c->buf.left(nl);
        c->buf.remove(0, nl + 1);
        // Turns run in order; while an older question is still answering, its tokens are skipped.
        const bool current = m_quickChannel == key && c->inFlight <= 1 && m_quickBusy;
        if (c->provider == QLatin1String("openai")) {
            const CodexEvent ev = parseCodexLine(line);
            const bool suggest = !c->suggestThread.isEmpty() && ev.threadId == c->suggestThread;
            if (ev.kind == CodexEvent::Response && ev.id == c->threadReq) {
                c->thread = ev.threadId;
            } else if (ev.kind == CodexEvent::Response && ev.id == c->suggestReq) {
                c->suggestThread = ev.threadId;
            } else if (suggest && ev.kind == CodexEvent::AgentDelta) {
                m_suggestion = oneLine(m_suggestion + ev.text, 12);
            } else if (suggest && ev.kind == CodexEvent::TurnCompleted) {
                m_suggestionKey = m_suggestPendingKey;
                emit suggestionChanged();
            } else if (ev.kind == CodexEvent::AgentDelta && current) {
                appendAnswer(ev.text);
                changed = true;
            } else if (ev.kind == CodexEvent::ReasoningDelta && current) {
                noteThinking(ev.text);
                changed = true;
            } else if (ev.kind == CodexEvent::ItemStarted && current && !suggest) {
                // item/started: "reasoning" -> thinking, "commandExecution" -> running a command.
                const QString it = ev.text;
                notePhase(it == QLatin1String("reasoning") ? QStringLiteral("thinking")
                          : it == QLatin1String("agentMessage") ? QStringLiteral("writing")
                          : it == QLatin1String("commandExecution") ? QStringLiteral("running a command")
                          : it == QLatin1String("userMessage") ? QStringLiteral("thinking")
                          : QString());
                changed = true;
            } else if (ev.kind == CodexEvent::TurnCompleted) {
                c->inFlight = std::max(0, c->inFlight - 1);
                done = done || current;
            } else if (ev.kind == CodexEvent::Error && current) {
                c->inFlight = std::max(0, c->inFlight - 1);
                m_quickAnswer = QStringLiteral("codex: ") + oneLine(ev.text, 8);
                done = true;
            }
            continue;
        }
        const QJsonObject o = QJsonDocument::fromJson(line).object();
        if (c->provider == QLatin1String("google")) {
            const QString ev = o.value(QLatin1String("event")).toString();
            if (ev == QLatin1String("step_update") && current) {
                // agy sends thinking text only on some models; heavy ones (gemini-4-hydrogen) send
                // step events alone, so the phase ("thinking", "running run_command") is the
                // live signal, and any thinking text streams dim when it does come.
                const StreamProgress sp = agyProgress(o);
                notePhase(sp.phase);
                noteThinking(sp.thinking);
                if (!sp.text.isEmpty())
                    appendAnswer(sp.text);
                if (sp.thinkingTokens >= 0)
                    m_quickThinkTokens = sp.thinkingTokens;
                changed = true;
            } else if (ev == QLatin1String("result")) {
                c->inFlight = std::max(0, c->inFlight - 1);
                if (current) {
                    const QString resp = o.value(QLatin1String("result")).toObject().value(QLatin1String("response")).toString();
                    if (!resp.isEmpty())
                        noteAnswer(answerBody(resp));
                    const StreamProgress sp = agyProgress(o);
                    if (sp.thinkingTokens >= 0)
                        m_quickThinkTokens = sp.thinkingTokens;
                    done = true;
                }
            }
            continue;
        }
        // anthropic: claude stream-json
        const QString type = o.value(QLatin1String("type")).toString();
        if (type == QLatin1String("stream_event") && current) {
            // Token by token: thinking deltas stream dim, then every text_delta repaints the answer.
            const StreamProgress sp = claudeProgress(o);
            notePhase(sp.phase);
            noteThinking(sp.thinking);
            if (!sp.text.isEmpty())
                appendAnswer(sp.text);
            changed = changed || !sp.phase.isEmpty() || !sp.thinking.isEmpty() || !sp.text.isEmpty();
        } else if (type == QLatin1String("assistant") && current) {
            QString text;
            for (const QJsonValue &v : o.value(QLatin1String("message")).toObject().value(QLatin1String("content")).toArray())
                if (v.toObject().value(QLatin1String("type")).toString() == QLatin1String("text"))
                    text += v.toObject().value(QLatin1String("text")).toString();
            if (!text.isEmpty()) {
                noteAnswer(answerBody(text));
                changed = true;
            }
        } else if (type == QLatin1String("result")) {
            c->inFlight = std::max(0, c->inFlight - 1);
            if (current) {
                const QString r = o.value(QLatin1String("result")).toString();
                if (!r.isEmpty())
                    noteAnswer(answerBody(r));
                done = true;
            }
        }
    }
    if (done)
        finishQuick();
    else if (changed)
        emit quickChanged();
    if (c->provider == QLatin1String("openai"))
        flushCodex(c);
}

void Backend::requestSuggestion(const QString &projectKey, int artifactIndex)
{
    if (!m_modelSuggestions || projectKey.isEmpty())
        return;
    const QString key = QStringLiteral("%1:%2").arg(projectKey).arg(artifactIndex);
    if (key == m_suggestionKey || key == m_suggestPendingKey)
        return;
    m_suggestPendingKey = key;
    m_suggestion.clear();
    if (m_mode == Mode::Mock) {
        m_mockSuggestT = 0;
        return;
    }
    // Try fast OpenAI-compatible provider first (mirroring Lapis IDE's provider chain) before CLI.
    QString base = m_openaiCompatUrl;
    if (base.isEmpty())
        base = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_URL").trimmed();
    if (!base.isEmpty()) {
        while (base.endsWith(QLatin1Char('/')))
            base.chop(1);
        const QString endpoint = base.endsWith(QLatin1String("/chat/completions"))
                                     ? base
                                     : base + QStringLiteral("/chat/completions");
        QString model = m_openaiCompatModel;
        if (model.isEmpty())
            model = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_MODEL").trimmed();
        if (model.isEmpty())
            model = QStringLiteral("qwen3.8-flash-next");
        QString apiKey = m_openaiCompatKey;
        if (apiKey.isEmpty())
            apiKey = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_KEY").trimmed();
        if (m_suggestReply) {
            m_suggestReply->disconnect(this);
            m_suggestReply->abort();
            m_suggestReply->deleteLater();
            m_suggestReply.clear();
        }
        static const QString kSuggestSys = QStringLiteral(
            "Predict the user's most likely next prompt in one short sentence under 12 words, "
            "matching their current mode (question, approval, correction, or next step).");
        QString ctx = contextFor(projectKey, artifactIndex);
        if (!m_recentActions.isEmpty())
            ctx += QStringLiteral("\nRecent user actions:\n- ") + m_recentActions.join(QStringLiteral("\n- "));
        const QByteArray payload = buildOpenAiCompatRequest(
            model, kSuggestSys, ctx,
            QStringLiteral("Suggest the next prompt only."), QStringLiteral("low"));
        QNetworkRequest req{QUrl(endpoint)};
        req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
        req.setRawHeader("Accept", "text/event-stream");
        if (!apiKey.isEmpty())
            req.setRawHeader("Authorization", QByteArray("Bearer ") + apiKey.toUtf8());
        QNetworkReply *reply = m_nam.post(req, payload);
        m_suggestReply = reply;
        auto buf = std::make_shared<QByteArray>();
        auto acc = std::make_shared<QString>();
        connect(reply, &QNetworkReply::readyRead, this, [this, reply, buf, acc, key] {
            if (reply != m_suggestReply)
                return;
            *buf += reply->readAll();
            qsizetype nl;
            while ((nl = buf->indexOf('\n')) >= 0) {
                const QByteArray line = buf->left(nl);
                buf->remove(0, nl + 1);
                const OpenAiSseEvent ev = parseOpenAiSseLine(line);
                if (ev.kind == OpenAiSseEvent::Delta && !ev.text.isEmpty())
                    *acc += ev.text;
            }
        });
        connect(reply, &QNetworkReply::finished, this, [this, reply, acc, key] {
            if (reply == m_suggestReply)
                m_suggestReply.clear();
            const QString s = oneLine(*acc, 12);
            reply->deleteLater();
            if (!s.isEmpty() && m_suggestPendingKey == key) {
                m_suggestion = s;
                m_suggestionKey = key;
                emit suggestionChanged();
            }
        });
        return;
    }
    // A second thread on the warm codex app-server, so a suggestion never interleaves with an answer.
    // gpt-6-sol at low: the lightest GPT-6 left in the lineup (Luna is retired).
    Channel *c = ensureChannel(QStringLiteral("gpt-6-sol"), QStringLiteral("low"));
    if (!c || !c->proc)
        return;
    if (c->suggestReq < 0) {
        c->suggestReq = c->nextId++;
        c->proc->write(codexRpc(c->suggestReq, QStringLiteral("thread/start"),
                                codexThreadStartParams(QStringLiteral(
                                    "Suggest the single most useful next question the user could ask about what is on "
                                    "screen. Reply with only the question, under 9 words."))));
    }
    c->suggestQueue = {codexTurnParams(QString(), contextFor(projectKey, artifactIndex), QStringLiteral("gpt-6-sol"),
                                       QStringLiteral("low"))};
    flushCodex(c);
}

// --- warm agy trace (Opt+Enter) ----------------------------------------------------------------

void Backend::ensureTraceHelper()
{
    if (m_mode != Mode::Live || (m_traceProc && m_traceProc->state() != QProcess::NotRunning))
        return;
    m_traceBuf.clear();
    m_traceInFlight = 0;
    m_traceProc = spawn(agyBin(), traceHelperArgs(), QDir::tempPath());
    if (!m_traceProc)
        return;
    connect(m_traceProc, &QProcess::readyReadStandardOutput, this, &Backend::onTraceOutput);
    QProcess *proc = m_traceProc;
    connect(proc, &QProcess::finished, this, [this, proc] {
        if (proc != m_traceProc)
            return;
        m_traceInFlight = 0;
        if (m_traceBusy)
            finishTrace();  // the helper died mid-turn: keep what streamed; the next ask restarts it
    });
}

void Backend::think(const QString &projectKey, int artifactIndex, const QString &question)
{
    const QString q = question.trimmed();
    if (q.isEmpty())
        return;
    m_traceQuestion = q;
    m_traceThinking.clear();
    m_traceAnswer.clear();
    m_traceBusy = true;
    m_traceSeconds = 0;
    emit traceChanged();
    if (m_mode == Mode::Mock) {
        m_mockTrace = MockReply{
            0.0, 2.4, q,
            QStringLiteral("decode at batch 1 is bandwidth bound\n"
                           "baseline: 64-wide acc spills 212 regs to local memory\n"
                           "128-reg tile keeps fp4 dequant + scale in registers\n"
                           "barrier after the smem scale lookup is gone\n"
                           "unroll 128 hides the packed KV load latency"),
            QStringLiteral("Registers hold the dequant: no spills, no barrier, 2.63x.")};
        return;
    }
    ensureTraceHelper();
    if (!m_traceProc) {
        m_traceBusy = false;
        emit traceChanged();
        return;
    }
    const QString prompt = contextFor(projectKey, artifactIndex)
        + QStringLiteral("\n\nThink it through, then answer in one crisp line under 20 words.\nQuestion: ") + q;
    m_traceRaw.clear();
    m_traceSawThinking = false;
    m_traceClock.start();
    m_traceInFlight += 1;
    m_traceProc->write(agyUserLine(prompt));  // buffered by QProcess until the helper has started
}

void Backend::onTraceOutput()
{
    if (!m_traceProc)
        return;
    m_traceBuf += m_traceProc->readAllStandardOutput();
    bool changed = false;
    qsizetype nl;
    while ((nl = m_traceBuf.indexOf('\n')) >= 0) {
        const QJsonObject o = QJsonDocument::fromJson(m_traceBuf.left(nl)).object();
        m_traceBuf.remove(0, nl + 1);
        const QString ev = o.value(QLatin1String("event")).toString();
        // Turns run in order; while an older question is still answering, its tokens are skipped.
        const bool current = m_traceInFlight <= 1;
        if (ev == QLatin1String("step_update") && current && m_traceBusy) {
            const QJsonObject su = o.value(QLatin1String("step_update")).toObject();
            for (const char *k : {"thinking_delta", "thinking"}) {
                const QString t = su.value(QLatin1String(k)).toString();
                if (!t.isEmpty()) {
                    if (!m_traceSawThinking)
                        m_traceThinking.clear();
                    m_traceSawThinking = true;
                    m_traceThinking += t;
                    changed = true;
                }
            }
            const QString d = su.value(QLatin1String("text_delta")).toString();
            if (!d.isEmpty()) {
                m_traceRaw += d;
                if (!m_traceSawThinking)
                    m_traceThinking = m_traceRaw;  // no thinking channel: stream the draft dim
                changed = true;
            }
        } else if (ev == QLatin1String("result")) {
            m_traceInFlight = std::max(0, m_traceInFlight - 1);
            if (current && m_traceBusy) {
                const QJsonObject r = o.value(QLatin1String("result")).toObject();
                const QString resp = r.value(QLatin1String("response")).toString();
                if (!resp.isEmpty())
                    m_traceRaw = resp;
                m_traceSeconds = r.value(QLatin1String("duration_seconds")).toDouble(m_traceClock.elapsed() / 1000.0);
                finishTrace();
                changed = false;  // finishTrace emitted
            }
        }
    }
    if (changed)
        emit traceChanged();
}

void Backend::finishTrace()
{
    m_traceBusy = false;
    if (m_traceSeconds <= 0)
        m_traceSeconds = m_traceClock.isValid() ? m_traceClock.elapsed() / 1000.0 : 0;
    m_traceAnswer = oneLine(stripToolNoise(m_traceRaw), 28);
    emit traceChanged();
}

void Backend::steer(const QString &projectKey, const QString &text)
{
    const QString t = text.trimmed();
    Project *p = find(projectKey);
    if (t.isEmpty() || !p)
        return;
    const SessionInfo *target = nullptr;
    for (const SessionInfo &s : p->sessions)
        if (!target || s.mtime > target->mtime)
            target = &s;
    if (!target) {
        setToast(QStringLiteral("No session"));
        return;
    }
    const QString title = clampWords(target->title, 4);
    if (m_mode == Mode::Mock) {
        setToast(QStringLiteral("→ ") + title);
        return;
    }
    QProcess *proc = spawn(lapisPeers(), {QStringLiteral("send"), QStringLiteral("--raw"), target->id, t});
    if (!proc)
        return;
    setToast(QStringLiteral("→ ") + title);
    connect(proc, &QProcess::finished, this, [this, proc, title](int code) {
        if (code != 0)
            setToast(QStringLiteral("send failed: ") + oneLine(QString::fromUtf8(proc->readAllStandardError()), 8));
    });
}

void Backend::sendWithRetry(const QString &agent, const QString &text, int attempt)
{
    QProcess *proc = spawn(lapisPeers(), {QStringLiteral("send"), QStringLiteral("--raw"), agent, text});
    if (!proc)
        return;
    connect(proc, &QProcess::finished, this, [this, proc, agent, text, attempt](int code) {
        if (code == 0) {
            setToast(QStringLiteral("Goal sent"));
            return;
        }
        if (attempt < 4) {
            QTimer::singleShot(2000, this, [this, agent, text, attempt] { sendWithRetry(agent, text, attempt + 1); });
            return;
        }
        setToast(QStringLiteral("send failed: ") + oneLine(QString::fromUtf8(proc->readAllStandardError()), 8));
    });
}

QString Backend::startSession(const QString &folder, const QString &harness, const QString &model, const QString &goal)
{
    const QString g = goal.trimmed();
    const QString dir = QDir::cleanPath(expandHome(folder.trimmed().isEmpty() ? QStringLiteral("~/dev") : folder.trimmed()));
    const ProjectId pid = projectIdFor(dir, clampWords(g, 5), QString());
    const QString key = pid.key;
    Project *p = find(key);
    if (!p) {
        Project np;
        np.key = key;
        np.name = pid.name;
        np.dir = dir;
        m_projects.prepend(np);
        p = &m_projects.first();
    }
    p->pending = true;
    p->ladderOverride = true;
    p->ladder = Ladder{};
    p->ladder.goal = clampWords(g, 7);
    static const QRegularExpression assumeRe(QStringLiteral("(?<![\\w-])([A-Z][A-Z0-9_]{1,15})=([\\w.]{1,12})"));
    auto am = assumeRe.globalMatch(g);
    while (am.hasNext())
        p->ladder.assumptions << QVariantMap{{QStringLiteral("text"), am.next().captured(0)}, {QStringLiteral("pinned"), false}};
    p->ladder.armsLabel = QStringLiteral("Arms starting");
    m_kickoffs.insert(key, p->ladder);
    const QString title = clampWords(g, 5);

    if (m_mode == Mode::Mock) {
        SessionInfo s;
        s.id = QStringLiteral("f1x70000-0000-0000-0000-00000000c0de");
        s.title = title;
        s.harness = harness;
        s.status = QStringLiteral("working");
        p->sessions = {s};
        p->ex = Extracted{};
        m_kickoff = MockScript{key, 0.0};
        publish();
        emit sessionStarted(key);
        emit uiActionRequested(QStringLiteral("agent_spawned"), s.id.left(8));
        return key;
    }

    QStringList args{QStringLiteral("new"), dir, QStringLiteral("--harness"), harness};
    if (!model.isEmpty())
        args << QStringLiteral("--model") << model;
    if (!title.isEmpty())
        args << QStringLiteral("--title") << title;
    QProcess *proc = spawn(lapisPeers(), args);
    setToast(QStringLiteral("Opening ") + harness + QStringLiteral(" · ") + studio::shortModel(model));
    if (proc) {
        connect(proc, &QProcess::finished, this, [this, proc, g](int code) {
            const QString out = QString::fromUtf8(proc->readAllStandardOutput());
            static const QRegularExpression opened(QStringLiteral("opened ([0-9a-f]{8})"));
            const auto m = opened.match(out);
            if (code != 0 || !m.hasMatch()) {
                setToast(QStringLiteral("new failed: ") + oneLine(QString::fromUtf8(proc->readAllStandardError()) + out, 8));
                return;
            }
            setToast(QStringLiteral("Agent %1 opened").arg(m.captured(1)));
            emit uiActionRequested(QStringLiteral("agent_spawned"), m.captured(1));
            if (!g.isEmpty())
                QTimer::singleShot(1500, this, [this, id = m.captured(1), g] { sendWithRetry(id, g, 0); });
            m_rescanDebounce.start();
        });
    }
    publish();
    emit sessionStarted(key);
    return key;
}

void Backend::editAssumption(const QString &projectKey, int index, const QString &text)
{
    Project *p = find(projectKey);
    const QString t = text.trimmed();
    if (!p || index < 0 || index >= p->ladder.assumptions.size() || t.isEmpty())
        return;
    QVariantMap a = p->ladder.assumptions[index].toMap();
    const QString old = a.value(QStringLiteral("text")).toString();
    if (old == t)
        return;
    a[QStringLiteral("text")] = t;
    a[QStringLiteral("pinned")] = true;
    p->ladder.assumptions[index] = a;
    p->ladderOverride = true;
    m_kickoffs[projectKey].assumptions = p->ladder.assumptions;
    publish();
    if (m_mode == Mode::Live) {
        steer(projectKey, QStringLiteral("Assumption change: %1 -> %2").arg(old, t));
    }
}

void Backend::togglePin(const QString &projectKey, int index)
{
    Project *p = find(projectKey);
    if (!p || index < 0 || index >= p->ladder.assumptions.size())
        return;
    QVariantMap a = p->ladder.assumptions[index].toMap();
    a[QStringLiteral("pinned")] = !a.value(QStringLiteral("pinned")).toBool();
    p->ladder.assumptions[index] = a;
    publish();
}

// --- live scanning ---------------------------------------------------------------------------

void Backend::loadFolders()
{
    QStringList ordered;
    auto addFolder = [&](const QString &raw) {
        const QString t = tildePath(expandHome(raw));
        if (!t.isEmpty() && t != QLatin1String("~") && t != QLatin1String("/tmp") && !ordered.contains(t))
            ordered << t;
    };
    for (const Project &p : std::as_const(m_projects))
        if (!isScratchDir(p.dir))
            addFolder(p.dir);
    QDir dev(home() + QStringLiteral("/dev"));
    const QFileInfoList dirs = dev.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
    for (const QFileInfo &fi : dirs) {
        if (ordered.size() >= 18)
            break;
        if (QFileInfo::exists(fi.absoluteFilePath() + QStringLiteral("/.git")))
            addFolder(fi.absoluteFilePath());
    }
    if (ordered.isEmpty())
        ordered << QStringLiteral("~/dev");
    m_folders = ordered;
    m_defaultFolder = m_folders.first();
    emit foldersChanged();
}

void Backend::refresh()
{
    if (m_mode != Mode::Live)
        return;
    if (m_scanning) {
        m_rescanQueued = true;
        return;
    }
    QElapsedTimer clock;
    clock.start();
    QFile f(workspacePath());
    if (!f.open(QIODevice::ReadOnly))
        return;
    const QJsonObject ws = QJsonDocument::fromJson(f.readAll()).object();
    QHash<QString, QString> categories;
    const QString activeCatId = ws.value(QLatin1String("activeCategory")).toString();
    QStringList selectedIds;
    for (const QJsonValue &c : ws.value(QLatin1String("categories")).toArray()) {
        const QJsonObject co = c.toObject();
        const QString cid = co.value(QLatin1String("id")).toString();
        const QString cname = co.value(QLatin1String("name")).toString();
        categories.insert(cid, cname);
        if (cid == activeCatId)
            m_activeCategory = cname;
        const QString sel = co.value(QLatin1String("selected")).toString();
        if (!sel.isEmpty())
            selectedIds << sel;
    }
    m_selectedAgentIds = selectedIds;
    QHash<QString, SessionInfo> previous;
    for (const Project &p : m_projects)
        for (const SessionInfo &s : p.sessions)
            previous.insert(s.id, s);
    QList<SessionInfo> sessions;
    static const QRegularExpression uuidArg(
        QStringLiteral("\\bs=([0-9a-f-]{36})|--(?:session-id|resume|conversation)[ =]([0-9a-f-]{36})"));
    for (const QJsonValue &v : ws.value(QLatin1String("agents")).toArray()) {
        const QJsonObject a = v.toObject();
        SessionInfo s = previous.value(a.value(QLatin1String("id")).toString());
        s.id = a.value(QLatin1String("id")).toString();
        s.title = a.value(QLatin1String("title")).toString();
        s.harness = a.value(QLatin1String("harness")).toString(QStringLiteral("claude"));
        s.directory = a.value(QLatin1String("directory")).toString(home());
        s.category = categories.value(a.value(QLatin1String("category")).toString());
        s.redacted = isRedactedCategory(s.category);
        // Uptime from the Lapis endpoint socket's birth time: file metadata only.
        if (s.started == 0) {
            const QFileInfo ep(a.value(QLatin1String("endpoint")).toString());
            if (ep.exists()) {
                const QDateTime b = ep.birthTime().isValid() ? ep.birthTime() : ep.lastModified();
                s.started = b.toMSecsSinceEpoch();
            }
        }
        QString ident = a.value(QLatin1String("managedResume")).toObject().value(QLatin1String("identity")).toString();
        if (ident.isEmpty())
            ident = a.value(QLatin1String("resumeThread")).toString();
        if (ident.isEmpty() && s.harness == QLatin1String("agy"))
            ident = a.value(QLatin1String("conversation")).toString();
        if (ident.isEmpty()) {
            QStringList argv;
            for (const QJsonValue &x : a.value(QLatin1String("arguments")).toArray())
                argv << x.toString();
            const auto m = uuidArg.match(argv.join(QLatin1Char(' ')));
            if (m.hasMatch())
                ident = m.captured(1).isEmpty() ? m.captured(2) : m.captured(1);
        }
        if (!ident.isEmpty() && ident != s.ident) {
            s.ident = ident;
            s.transcript.clear();
        }
        if (s.id.isEmpty())
            continue;
        sessions << s;
    }
    m_scanning = true;
    auto *watcher = new QFutureWatcher<ScanResult>(this);
    connect(watcher, &QFutureWatcher<ScanResult>::finished, this, [this, watcher] {
        const ScanResult r = watcher->result();
        watcher->deleteLater();
        m_transcriptsRead += r.read;
        (r.onGuiThread ? m_stall.guiParseMs : m_stall.workerMs) += r.ms;
        m_stall.parsed += int(r.read.size());
        QHash<QString, int> added;
        for (const QString &path : r.read)
            if (const int n = r.cache.value(path).ex.added; n > 0)
                added.insert(path, n);
        applyScan(r.sessions, r.cache, added);
        m_scanning = false;
        if (m_rescanQueued) {
            m_rescanQueued = false;
            m_rescanDebounce.start();
        }
    });
    watcher->setFuture(QtConcurrent::run(scanWorker, sessions, m_cache));
    const double ms = clock.nsecsElapsed() / 1e6;
    m_stall.refreshMs += ms;
    m_stall.refreshes += 1;
    noteStall(ms);
}

void Backend::benchReapply()
{
    if (m_mode == Mode::Live)
        applyScan(m_lastSessions, m_cache);
    else
        publish();  // mock: the same projects, rebuilt and re-emitted
}

void Backend::applyScan(const QList<SessionInfo> &sessions, const QHash<QString, ParsedCache> &cache,
                        const QHash<QString, int> &added)
{
    QElapsedTimer clock;
    clock.start();
    const double publishBefore = m_stall.publishMs;
    m_lastSessions = sessions;
    m_cache = cache;
    QHash<QString, Project> byKey;
    QStringList order;
    QSet<QString> folderNames;
    // One watcher update per scan: on macOS every addPath() restarts the FSEvents stream, which
    // cost ~5 ms per transcript on the GUI thread at cold start.
    const QStringList watchedList = m_watcher.files();
    const QSet<QString> watched(watchedList.cbegin(), watchedList.cend());
    QStringList toWatch;
    for (const SessionInfo &s : sessions) {
        const ProjectId pid = sessionProjectId(s, cache.value(s.transcript).ex, m_snippetById.value(s.id));
        const QString key = pid.key;
        if (!pid.fromTitle)
            folderNames.insert(pid.name);
        if (!byKey.contains(key)) {
            Project p;
            p.key = key;
            p.name = pid.name;
            p.dir = s.directory;
            byKey.insert(key, p);
            order << key;
        }
        Project &p = byKey[key];
        p.sessions << s;
        p.mtime = std::max(p.mtime, s.mtime);
        if (!s.transcript.isEmpty() && !watched.contains(s.transcript) && !toWatch.contains(s.transcript))
            toWatch << s.transcript;
    }
    if (!toWatch.isEmpty())
        m_watcher.addPaths(toWatch);
    // A scratch tab titled like a real project folder ("Infinity" in $HOME) keeps its own island,
    // named apart so the two never read as one.
    for (auto it = byKey.begin(); it != byKey.end(); ++it)
        if (it->key.startsWith(QLatin1Char('~')) && folderNames.contains(it->name))
            it->name += QLatin1Char('-') + it->sessions.value(0).harness;
    QList<Project> next;
    for (const QString &key : order) {
        Project p = byKey.take(key);
        std::stable_sort(p.sessions.begin(), p.sessions.end(),
                         [](const SessionInfo &a, const SessionInfo &b) { return a.mtime > b.mtime; });
        Extracted merged;
        for (const SessionInfo &s : p.sessions) {
            if (!cache.contains(s.transcript))
                continue;
            const Extracted &e = cache.value(s.transcript).ex;
            for (const QVariant &a : e.artifacts)
                if (merged.artifacts.size() < 12)
                    merged.artifacts << a;
            if (merged.headline.isEmpty()) merged.headline = e.headline;
            if (merged.spark.isEmpty()) merged.spark = e.spark;
            if (merged.prose.isEmpty()) merged.prose = e.prose;
            if (merged.thinking.isEmpty()) merged.thinking = e.thinking;
            if (merged.goal.isEmpty()) merged.goal = e.goal;
            if (merged.assumptions.isEmpty()) merged.assumptions = e.assumptions;
            if (merged.recentTurns.isEmpty()) merged.recentTurns = e.recentTurns;  // newest session first
            if (merged.threadTurns.isEmpty()) merged.threadTurns = e.threadTurns;
            if (merged.promptName.isEmpty()) merged.promptName = e.promptName;
            if (merged.files.isEmpty()) merged.files = e.files;
            if (merged.toolCalls.isEmpty()) merged.toolCalls = e.toolCalls;
            if (merged.params.isEmpty()) merged.params = e.params;
            if (merged.jobName.isEmpty()) merged.jobName = e.jobName;
            // New turns while the project is off screen raise its island's +N badge.
            if (const int n = added.value(s.transcript); n > 0 && key != m_focusedProject)
                m_fresh[key] += n;
            if (merged.arms.isEmpty()) {
                merged.arms = e.arms;
                merged.winner = e.winner;
            }
        }
        p.ex = merged;
        ladderFromExtracted(p.ex, p.sessions.value(0).title, &p.ladder.goal, &p.ladder.assumptions, &p.ladder.arms,
                            &p.ladder.armsLabel, &p.ladder.winner);
        if (m_kickoffs.contains(key)) {
            const Ladder &k = m_kickoffs.value(key);
            p.ladder.goal = k.goal;
            if (p.ladder.assumptions.isEmpty() || !k.assumptions.isEmpty())
                p.ladder.assumptions = k.assumptions.isEmpty() ? p.ladder.assumptions : k.assumptions;
            if (p.ladder.arms.isEmpty())
                p.ladder.armsLabel = QStringLiteral("Arms starting");
        }
        next << p;
    }
    // Keep a just-started project visible until its tab shows up in the registry.
    for (const Project &old : m_projects)
        if (old.pending && std::none_of(next.begin(), next.end(), [&](const Project &n) { return n.key == old.key; }))
            next.prepend(old);
    m_projects = next;
    publish();
    // The human-interest pass for the project on screen (else the newest one); async, deduped.
    if (m_mode == Mode::Live && !m_projects.isEmpty())
        filterForHuman(m_focusedProject.isEmpty() ? m_projects.first().key : m_focusedProject);
    // applyMs excludes the publish() slice, which is accounted on its own.
    const double ms = clock.nsecsElapsed() / 1e6;
    m_stall.applyMs += ms - (m_stall.publishMs - publishBefore);
    m_stall.applies += 1;
    noteStall(ms);
}

void Backend::pollBoard()
{
    if (m_mode != Mode::Live || m_boardBusy)
        return;
    // The read-only `lapis-peers board` poll runs entirely on a worker thread: forking the
    // process cost ~3 ms of GUI-thread time every 5 s, and its JSON is parsed there too.
    QElapsedTimer clock;
    clock.start();
    m_boardBusy = true;
    ++m_spawnCount;
    using Rows = QList<BoardRow>;
    auto *watcher = new QFutureWatcher<std::optional<Rows>>(this);
    connect(watcher, &QFutureWatcher<std::optional<Rows>>::finished, this, [this, watcher] {
        QElapsedTimer apply;
        apply.start();
        const double publishBefore = m_stall.publishMs;
        const double applyBefore = m_stall.applyMs;
        const std::optional<Rows> r = watcher->result();
        watcher->deleteLater();
        m_boardBusy = false;
        if (!r)
            return;
        applyBoard(*r);
        const double ms = apply.nsecsElapsed() / 1e6;
        m_stall.boardMs += ms - (m_stall.publishMs - publishBefore) - (m_stall.applyMs - applyBefore);
        m_stall.boards += 1;
        noteStall(ms);
    });
    watcher->setFuture(QtConcurrent::run([program = lapisPeers(), cwd = home()]() -> std::optional<Rows> {
        QProcess p;
        p.setWorkingDirectory(cwd);
        // Snippets too: a remote (SSH) session has no local transcript, so the board's latest
        // reply is all the context its island and conversation can show.
        p.start(program, {QStringLiteral("board"), QStringLiteral("--json"), QStringLiteral("--snippet-chars"),
                          QStringLiteral("400")});
        if (!p.waitForFinished(15000)) {
            p.kill();
            p.waitForFinished(1000);
            return std::nullopt;
        }
        if (p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0)
            return std::nullopt;
        return parseBoardJson(p.readAllStandardOutput());
    }));
    const double ms = clock.nsecsElapsed() / 1e6;  // dispatch only
    m_stall.boardMs += ms;
    noteStall(ms);
}

void Backend::applyBoard(const QList<BoardRow> &rows)
{
    bool changed = false;
    bool rename = false;  // a transcript-less generic tab takes its island name from its snippet
    for (const BoardRow &r : rows) {
        if (m_statusById.value(r.id) != r.status) {
            m_statusById.insert(r.id, r.status);
            changed = true;
        }
        if (r.snippet.isEmpty() || m_snippetById.value(r.id) == r.snippet)
            continue;
        m_snippetById.insert(r.id, r.snippet);
        changed = true;
        for (const Project &p : std::as_const(m_projects))
            if (p.key.startsWith(QLatin1String("~tab:")) && p.ex.promptName.isEmpty() && p.ex.jobName.isEmpty()
                && std::any_of(p.sessions.cbegin(), p.sessions.cend(), [&](const SessionInfo &s) { return s.id == r.id; }))
                rename = true;
    }
    if (rename && !m_lastSessions.isEmpty())
        applyScan(m_lastSessions, m_cache);
    else if (changed)
        publish();
}

// --- Human-interest filter -------------------------------------------------------------------

QString Backend::compatEndpoint() const
{
    QString base = m_openaiCompatUrl;
    if (base.isEmpty() && m_mode == Mode::Live)
        base = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_URL").trimmed();
    if (base.isEmpty())
        return {};
    while (base.endsWith(QLatin1Char('/')))
        base.chop(1);
    return base.endsWith(QLatin1String("/chat/completions")) ? base : base + QStringLiteral("/chat/completions");
}

void Backend::filterForHuman(const QString &projectKey)
{
    if (m_mode != Mode::Live || projectKey.isEmpty())
        return;
    const QString endpoint = compatEndpoint();
    if (endpoint.isEmpty())
        return;
    const Project *p = find(projectKey);
    if (!p || (p->ex.prose.trimmed().isEmpty() && p->ex.recentTurns.isEmpty()))
        return;
    // Redacted (infinity) projects never leave the process.
    if (std::any_of(p->sessions.cbegin(), p->sessions.cend(), [](const SessionInfo &s) { return s.redacted; }))
        return;
    const QString tag = QStringLiteral("%1@%2").arg(projectKey).arg(p->mtime);
    if (tag == m_filterInFlight || m_filterDone.contains(tag))
        return;
    if (m_filterReply) {  // concurrency 1: wait for the slot
        m_filterQueue.removeAll(projectKey);
        m_filterQueue << projectKey;
        return;
    }
    QString model = m_openaiCompatModel;
    if (model.isEmpty())
        model = QStringLiteral("qwen3.8-flash-next");
    QString apiKey = m_openaiCompatKey;
    if (apiKey.isEmpty())
        apiKey = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_KEY").trimmed();
    QString latest = p->ex.prose.trimmed();
    if (!p->ex.recentTurns.isEmpty()) {
        // The human's last words frame the turn; fall back to the turn summaries without prose.
        for (qsizetype i = p->ex.recentTurns.size() - 1; i >= 0; --i)
            if (p->ex.recentTurns[i].startsWith(QLatin1String("User: "))) {
                latest = p->ex.recentTurns[i] + QStringLiteral("\n\nAssistant: ") + latest;
                break;
            }
        if (p->ex.prose.trimmed().isEmpty())
            latest = p->ex.recentTurns.join(QLatin1Char('\n'));
    }
    const QByteArray payload = buildFilterPrompt(model, p->ex.goal, p->ex.params, m_recentActions, latest, m_tasteRules);
    if (m_filterDone.size() > 512)
        m_filterDone.clear();
    m_filterDone.insert(tag);
    m_filterInFlight = tag;
    ++m_filterRequests;
    const qint64 mtime = p->mtime;

    QNetworkRequest req{QUrl(endpoint)};
    req.setTransferTimeout(20000);
    req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    req.setRawHeader("Accept", "text/event-stream");
    if (!apiKey.isEmpty())
        req.setRawHeader("Authorization", QByteArray("Bearer ") + apiKey.toUtf8());
    QNetworkReply *reply = m_nam.post(req, payload);
    m_filterReply = reply;
    auto buf = std::make_shared<QByteArray>();
    auto acc = std::make_shared<QString>();
    connect(reply, &QNetworkReply::readyRead, this, [this, reply, buf, acc] {
        if (reply != m_filterReply)
            return;
        *buf += reply->readAll();
        qsizetype nl;
        while ((nl = buf->indexOf('\n')) >= 0) {
            const QByteArray line = buf->left(nl);
            buf->remove(0, nl + 1);
            const OpenAiSseEvent ev = parseOpenAiSseLine(line);
            if (ev.kind == OpenAiSseEvent::Delta && !ev.text.isEmpty())
                *acc += ev.text;
        }
    });
    connect(reply, &QNetworkReply::finished, this, [this, reply, acc, projectKey, mtime] {
        if (reply == m_filterReply)
            m_filterReply.clear();
        m_filterInFlight.clear();
        reply->deleteLater();
        if (reply->error() == QNetworkReply::NoError) {
            const HumanFilterReply r = parseFilterReply(*acc);
            if (!r.stateLine.isEmpty() || !r.summary.isEmpty()) {
                m_humanFilter.insert(projectKey, HumanFilter{mtime, r.stateLine, r.summary, r.form});
                publish();
                // A summary artifact's context now carries the filtered key points: re-warm.
                if (projectKey == m_focusedProject)
                    warmProjectPrefix(projectKey);
            }
        }
        // Next project waiting for the slot, if any.
        while (!m_filterQueue.isEmpty() && !m_filterReply)
            filterForHuman(m_filterQueue.takeFirst());
    });
}

// --- KV-cache pre-warm -----------------------------------------------------------------------

void Backend::warmProjectPrefix(const QString &projectKey)
{
    if (m_mode != Mode::Live || projectKey.isEmpty())
        return;
    const QString endpoint = compatEndpoint();
    if (endpoint.isEmpty())
        return;
    const Project *p = find(projectKey);
    if (!p)
        return;
    // Redacted (infinity) projects never leave the process.
    if (std::any_of(p->sessions.cbegin(), p->sessions.cend(), [](const SessionInfo &s) { return s.redacted; }))
        return;
    const QString trunk = contextTrunk(projectKey);
    if (trunk.isEmpty())
        return;
    QString model = m_openaiCompatModel;
    if (model.isEmpty())
        model = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_MODEL").trimmed();
    if (model.isEmpty())
        model = QStringLiteral("qwen3.8-flash-next");
    LlmRouter *router = llmRouter();
    if (!router)
        return;
    // key@mtime, plus the trunk hash: the trunk can change without the transcript mtime moving.
    const QString tag = QStringLiteral("%1@%2@%3").arg(projectKey).arg(p->mtime).arg(qHash(trunk));
    TieredPrompt tp;
    tp.model = model;
    tp.systemPrompt = kCompatSystemPrompt;
    tp.extraSystemContext = m_studioContext;
    tp.trunk = trunk;
    // Fire-and-forget: the server keeps the prefilled KV; the one token is discarded. The router
    // drops a repeat of the same tag and forgets a failed one so the next focus retries.
    if (router->warmPrefix(endpoint, compatKey(), buildTieredWarmRequest(tp), tag))
        ++m_warmRequests;
}

// --- Morning Briefing ------------------------------------------------------------------------

void Backend::toggleMetricPin(const QString &projectKey)
{
    if (m_pinnedMetrics.contains(projectKey))
        m_pinnedMetrics.remove(projectKey);
    else
        m_pinnedMetrics.insert(projectKey);
    publish();
}

void Backend::requestBriefing()
{
    if (m_briefingReply) {
        m_briefingReply->disconnect(this);
        m_briefingReply->abort();
        m_briefingReply->deleteLater();
        m_briefingReply.clear();
    }
    // The local lines land at once, so the overlay never opens empty or waits on the network.
    m_briefing = localBriefing(m_projectList);
    m_briefingSource = QStringLiteral("local");
    QString base = m_openaiCompatUrl;
    if (base.isEmpty() && m_mode == Mode::Live)
        base = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_URL").trimmed();
    if (m_mode != Mode::Live || base.isEmpty() || m_briefing.isEmpty()) {
        emit briefingChanged();
        return;
    }
    while (base.endsWith(QLatin1Char('/')))
        base.chop(1);
    const QString endpoint = base.endsWith(QLatin1String("/chat/completions")) ? base : base + QStringLiteral("/chat/completions");
    QString model = m_openaiCompatModel;
    if (model.isEmpty())
        model = QStringLiteral("qwen3.8-flash-next");
    QString apiKey = m_openaiCompatKey;
    if (apiKey.isEmpty())
        apiKey = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_KEY").trimmed();

    // Context: per changed project, its state lines, proof and latest turn. Redacted (infinity)
    // projects never leave the process: they keep their local lines.
    QString ctx;
    for (const QVariant &v : std::as_const(m_briefing)) {
        const QVariantMap row = v.toMap();
        const QVariantMap p = m_projectList.value(row.value(QStringLiteral("index")).toInt()).toMap();
        if (p.value(QStringLiteral("redacted")).toBool())
            continue;
        const QString key = row.value(QStringLiteral("key")).toString();
        ctx += QStringLiteral("\n### key: %1 (%2), status %3\n").arg(key, p.value(QStringLiteral("name")).toString(),
                                                                     p.value(QStringLiteral("status")).toString());
        ctx += QStringLiteral("Now: %1\nLatest: %2\nMetric: %3\nNew turns: %4\n")
                   .arg(p.value(QStringLiteral("nowLine")).toString(), p.value(QStringLiteral("lastLine")).toString(),
                        p.value(QStringLiteral("headline")).toString())
                   .arg(p.value(QStringLiteral("fresh")).toInt());
        for (const Project &pr : std::as_const(m_projects))
            if (pr.key == key && !pr.ex.recentTurns.isEmpty())
                ctx += QStringLiteral("Last turns:\n") + pr.ex.recentTurns.mid(std::max<qsizetype>(0, pr.ex.recentTurns.size() - 2)).join(QLatin1Char('\n')) + QLatin1Char('\n');
    }
    if (ctx.isEmpty()) {
        emit briefingChanged();
        return;
    }
    static const QString kSys = QStringLiteral(
        "You write a morning briefing for an engineer returning to several coding-agent projects. "
        "For each project output exactly one line: key | what happened, with concrete proof (a number, "
        "file or artifact) | the single next action, imperative, doable by controller or voice. "
        "Each part under 18 words, plain English, no markdown, no extra lines.");
    const QByteArray payload = buildOpenAiCompatRequest(model, kSys, ctx, QStringLiteral("Write the briefing."), QStringLiteral("low"));
    QNetworkRequest req{QUrl(endpoint)};
    req.setTransferTimeout(20000);  // a dead endpoint leaves the local lines, never a spinner
    req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    req.setRawHeader("Accept", "text/event-stream");
    if (!apiKey.isEmpty())
        req.setRawHeader("Authorization", QByteArray("Bearer ") + apiKey.toUtf8());
    QNetworkReply *reply = m_nam.post(req, payload);
    m_briefingReply = reply;
    auto buf = std::make_shared<QByteArray>();
    auto acc = std::make_shared<QString>();
    connect(reply, &QNetworkReply::readyRead, this, [this, reply, buf, acc] {
        if (reply != m_briefingReply)
            return;
        *buf += reply->readAll();
        qsizetype nl;
        while ((nl = buf->indexOf('\n')) >= 0) {
            const QByteArray line = buf->left(nl);
            buf->remove(0, nl + 1);
            const OpenAiSseEvent ev = parseOpenAiSseLine(line);
            if (ev.kind == OpenAiSseEvent::Delta && !ev.text.isEmpty())
                *acc += ev.text;
        }
    });
    connect(reply, &QNetworkReply::finished, this, [this, reply, acc, model] {
        if (reply != m_briefingReply)
            return;
        m_briefingReply.clear();
        reply->deleteLater();
        if (reply->error() == QNetworkReply::NoError && mergeBriefingReply(m_briefing, *acc) > 0)
            m_briefingSource = model;
        emit briefingChanged();
    });
    emit briefingChanged();  // local lines now, busy until the model answers
}

// --- End-of-session reflection ------------------------------------------------------------------

QVariantList Backend::reflectionSessions() const
{
    // Newest work first; infinity (redacted) projects never leave the process or show up here.
    QList<const Project *> order;
    for (const Project &p : m_projects) {
        const bool redacted = std::any_of(p.sessions.cbegin(), p.sessions.cend(), [](const SessionInfo &s) { return s.redacted; });
        if (!redacted)
            order << &p;
    }
    std::stable_sort(order.begin(), order.end(), [](const Project *a, const Project *b) { return a->mtime > b->mtime; });
    QVariantList out;
    for (const Project *p : std::as_const(order)) {
        QStringList users;
        QStringList turns;
        for (const QVariant &v : p->ex.threadTurns) {
            const QVariantMap t = v.toMap();
            const QString role = t.value(QStringLiteral("role")).toString();
            const QString text = t.value(QStringLiteral("text")).toString().simplified();
            if (text.isEmpty())
                continue;
            if (role == QLatin1String("user"))
                users << text;
            turns << (role == QLatin1String("user") ? QStringLiteral("User: ") : QStringLiteral("Assistant: ")) + text.left(400);
        }
        if (users.isEmpty())
            continue;
        const QVariantMap pm = m_projectList.value(projectIndex(p->key)).toMap();
        out << QVariantMap{{QStringLiteral("key"), p->key},
                           {QStringLiteral("name"), p->name},
                           {QStringLiteral("goal"), p->ex.goal.isEmpty() ? users.first() : p->ex.goal},
                           {QStringLiteral("last"), pm.value(QStringLiteral("lastLong")).toString()},
                           {QStringLiteral("userTurns"), users},
                           {QStringLiteral("turns"), turns.mid(std::max<qsizetype>(0, turns.size() - 8))}};
        if (out.size() >= 6)
            break;
    }
    return out;
}

QVariantMap Backend::reflection() const
{
    return {{QStringLiteral("intent"), m_reflection.intent},
            {QStringLiteral("corrections"), m_reflection.corrections},
            {QStringLiteral("rules"), m_reflection.rules}};
}

void Backend::runSessionReflection()
{
    const QVariantList sessions = reflectionSessions();
    // The local lines land at once, so the overlay never opens empty or waits on the network.
    m_reflection = localReflection(sessions);
    if (m_reflection.intent.isEmpty())
        m_reflection.intent = QStringLiteral("No session turns to reflect on yet.");
    m_reflectionSource = QStringLiteral("local");
    m_reflectionRaw.clear();
    m_reflectionBusy = false;
    const QString endpoint = compatEndpoint();
    LlmRouter *router = endpoint.isEmpty() || sessions.isEmpty() ? nullptr : llmRouter();
    if (!router) {
        emit reflectionChanged();
        return;
    }
    QString ctx;
    for (const QVariant &v : sessions) {
        const QVariantMap s = v.toMap();
        ctx += QStringLiteral("\n### %1\nOriginal request: %2\nLatest step: %3\nTurns (oldest first):\n")
                   .arg(s.value(QStringLiteral("name")).toString(), s.value(QStringLiteral("goal")).toString().left(600),
                        s.value(QStringLiteral("last")).toString());
        ctx += s.value(QStringLiteral("turns")).toStringList().join(QLatin1Char('\n')) + QLatin1Char('\n');
        if (ctx.size() > 9000)
            break;
    }
    if (!m_tasteRules.isEmpty())
        ctx += QStringLiteral("\nRules the human already keeps (do not repeat them):\n- ") + m_tasteRules.join(QStringLiteral("\n- "));
    static const QString kSys = QStringLiteral(
        "You review a human's session directing coding agents. Be concise and concrete, plain English, no markdown. "
        "Reply in exactly this format and nothing else:\n"
        "INTENT: one or two lines, what they originally set out to do vs what actually matched it\n"
        "CORRECTIONS: one line naming the human's corrections or steers you spotted, or none\n"
        "RULES:\n- 1 to 3 crisp personal taste rules worth keeping, each an imperative under 16 words");
    LlmStreamRequest req;
    req.streamId = QString::fromLatin1(kReflectionStreamId);
    req.endpoint = endpoint;
    req.apiKey = compatKey();
    req.payload = buildOpenAiCompatRequest(compatModel(), kSys, ctx, QStringLiteral("Reflect on this session."),
                                           QStringLiteral("low"));
    req.timeoutMs = 30000;  // a dead endpoint leaves the local lines, never a spinner
    m_reflectionBusy = true;
    emit reflectionChanged();  // local lines now, busy until the model answers
    router->startStream(req);
}

QString Backend::defaultReviewsPath()
{
    return home() + QStringLiteral("/.lapis-next/session_reviews.jsonl");
}

void Backend::setReviewsPath(const QString &path)
{
    m_reviewsPath = path;
    loadTasteRules();
}

void Backend::loadTasteRules()
{
    QStringList rules;
    QFile f(m_reviewsPath);
    if (!m_reviewsPath.isEmpty() && f.open(QIODevice::ReadOnly)) {
        // Newest line per rule wins: a later rating <= 0 retires an earlier keep.
        while (!f.atEnd()) {
            const QJsonObject o = QJsonDocument::fromJson(f.readLine(16 * 1024)).object();
            const QString rule = o.value(QLatin1String("rule")).toString().simplified();
            if (rule.isEmpty())
                continue;
            for (qsizetype i = rules.size() - 1; i >= 0; --i)
                if (rules[i].compare(rule, Qt::CaseInsensitive) == 0)
                    rules.removeAt(i);
            if (o.value(QLatin1String("rating")).toInt(1) > 0)
                rules << rule;
        }
    }
    while (rules.size() > 12)
        rules.removeFirst();  // the filter prompt carries the newest dozen
    if (rules != m_tasteRules) {
        m_tasteRules = rules;
        emit tasteRulesChanged();
    }
}

bool Backend::saveTasteRule(const QString &rule, int rating)
{
    const QString r = rule.simplified();
    if (r.isEmpty())
        return false;
    QStringList rules = m_tasteRules;
    for (qsizetype i = rules.size() - 1; i >= 0; --i)
        if (rules[i].compare(r, Qt::CaseInsensitive) == 0)
            rules.removeAt(i);
    if (rating > 0)
        rules << r;
    while (rules.size() > 12)
        rules.removeFirst();
    if (rules != m_tasteRules) {
        m_tasteRules = rules;
        emit tasteRulesChanged();
    }
    if (m_reviewsPath.isEmpty())
        return false;
    const QFileInfo fi(m_reviewsPath);
    if (!QFileInfo::exists(fi.absolutePath())) {
        if (!QDir().mkpath(fi.absolutePath()))
            return false;
        QFile::setPermissions(fi.absolutePath(), QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner);
    }
    QFile f(m_reviewsPath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Append))
        return false;
    f.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
    const QJsonObject o{{QStringLiteral("ts"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
                        {QStringLiteral("rule"), r},
                        {QStringLiteral("rating"), rating},
                        {QStringLiteral("source"), m_reflectionSource.isEmpty() ? QStringLiteral("manual") : m_reflectionSource}};
    const QByteArray line = QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n';
    return f.write(line) == line.size();
}

// --- Studio UI commands (natural language + model [[ACTION:..]] tags) ------------------------

namespace {

// Lower case, punctuation as spaces, single-spaced: "Can you open the Slowmix tab?" ->
// "can you open the slowmix tab".
QString cmdNorm(const QString &s)
{
    QString out;
    out.reserve(s.size());
    for (const QChar c : s.toLower())
        out += c.isLetterOrNumber() ? c : QLatin1Char(' ');
    return out.simplified();
}

QString squashAlnum(const QString &s)
{
    QString out;
    for (const QChar c : s.toLower())
        if (c.isLetterOrNumber())
            out += c;
    return out;
}

// Words a voice-typed command may trail with that never change its meaning.
bool onlyFiller(const QString &rest)
{
    static const QSet<QString> filler{
        QStringLiteral("inside"), QStringLiteral("of"), QStringLiteral("this"), QStringLiteral("here"),
        QStringLiteral("for"), QStringLiteral("me"), QStringLiteral("now"), QStringLiteral("please"),
        QStringLiteral("app"), QStringLiteral("studio"), QStringLiteral("lapis"), QStringLiteral("up"),
        QStringLiteral("in"), QStringLiteral("the"), QStringLiteral("tab"), QStringLiteral("overlay"),
        QStringLiteral("panel"), QStringLiteral("card"), QStringLiteral("window"), QStringLiteral("view"),
        QStringLiteral("screen"), QStringLiteral("right"), QStringLiteral("quickly"), QStringLiteral("thanks"),
        QStringLiteral("thank"), QStringLiteral("you"), QStringLiteral("again"), QStringLiteral("page"),
        QStringLiteral("menu"), QStringLiteral("pls"), QStringLiteral("real"), QStringLiteral("quick"),
        QStringLiteral("player"), QStringLiteral("again")};
    for (const QString &w : rest.split(QLatin1Char(' '), Qt::SkipEmptyParts))
        if (!filler.contains(w))
            return false;
    return true;
}

// Leading politeness and wake words: "hey lapis can you please open music" -> "open music".
QString stripLead(QString n)
{
    static const QRegularExpression lead(QStringLiteral(
        "^(?:hey|ok|okay|yo|so|lapis|studio|please|pls|can you|could you|would you|will you|can u|could u|"
        "would you mind|i want to|i d like to|i would like to|i want you to|let s|lets|go ahead and|just|now|and)\\s+"));
    for (int i = 0; i < 8; ++i) {
        const auto m = lead.match(n);
        if (!m.hasMatch())
            break;
        n = n.mid(m.capturedLength());
    }
    return n;
}

// "^(?:VERBS)\s+(?:the |my )?(?:OBJECTS)" with only filler after it.
bool verbObject(const QString &n, const QString &verbs, const QString &objects)
{
    const QRegularExpression re(QStringLiteral("^(?:%1)\\s+(?:the\\s+|my\\s+)?(?:%2)\\b(.*)$").arg(verbs, objects));
    const auto m = re.match(n);
    return m.hasMatch() && onlyFiller(m.captured(1));
}

const QString kOpenVerbs = QStringLiteral(
    "open|show|show me|bring up|pull up|launch|go to|switch to|take me to|display|pop up|pop open|start");
const QString kCloseVerbs = QStringLiteral("close|hide|dismiss|exit|leave|get rid of");
const QString kMusicWords = QStringLiteral(
    "music player|slowmix player|music|slowmix|slow mix|slowmix music|songs|song list|player");

}  // namespace

QStringList studioActionNames()
{
    return {QStringLiteral("open_music"),     QStringLiteral("close_music"),   QStringLiteral("play_track"),
            QStringLiteral("pause_music"),    QStringLiteral("resume_music"),  QStringLiteral("toggle_music"),
            QStringLiteral("open_review"),    QStringLiteral("filter_needs_me"), QStringLiteral("filter_all"),
            QStringLiteral("reset_layout"),   QStringLiteral("open_settings"), QStringLiteral("open_briefing"),
            QStringLiteral("set_theme"),      QStringLiteral("show_diffs"),    QStringLiteral("spawn_agent"),
            QStringLiteral("steer_agent"),    QStringLiteral("set_volume"),    QStringLiteral("open_spotify"),  QStringLiteral("spotify_play"),
            QStringLiteral("spotify_pause"),  QStringLiteral("spotify_next"),  QStringLiteral("spotify_prev"),
            QStringLiteral("spotify_volume"), QStringLiteral("focus_project"), QStringLiteral("open_stage"),
            QStringLiteral("open_ladder"),    QStringLiteral("go_home"),       QStringLiteral("sort_map"),
            QStringLiteral("move_cluster"),   QStringLiteral("cua_click"),     QStringLiteral("cua_type"),
            QStringLiteral("cua_scroll"),     QStringLiteral("cua_observe"),   QStringLiteral("cua_hotkey"),
            QStringLiteral("cua_drag"),       QStringLiteral("cua_open")};
}

QString cuaObservationPrompt(const QString &action, const QString &tier, bool ok, const QString &detail,
                             const QString &observation)
{
    QString obs = observation.trimmed();
    if (obs.isEmpty())
        obs = QStringLiteral("(no accessibility nodes visible)");
    return QStringLiteral("[CUA observation after %1 (%2, ok=%3, detail=%4)]:\n%5")
        .arg(action, tier.isEmpty() ? QStringLiteral("ERR") : tier, ok ? QStringLiteral("1") : QStringLiteral("0"),
             detail.simplified(), obs);
}

bool cuaShouldContinue(const QVector<CuaStepResult> &steps, int depth)
{
    if (depth >= kMaxCuaLoopDepth || steps.isEmpty())
        return false;
    return true;  // feed post-action observation back until the model finishes without cua_* tags or hits depth cap
}

QString cuaFollowupPrompt(const QVector<CuaStepResult> &steps)
{
    QStringList blocks;
    for (const CuaStepResult &s : steps)
        blocks << cuaObservationPrompt(s.action + (s.target.isEmpty() ? QString() : QLatin1Char(':') + s.target), s.tier,
                                       s.ok, s.detail, s.observation);
    return blocks.join(QStringLiteral("\n\n"))
           + QStringLiteral("\nContinue only if more actions are needed; otherwise give a concise final answer without cua_* tags.");
}

CuaCommand parseCuaAction(const QString &action, const QString &arg)
{
    CuaCommand c;
    const QString a = action.trimmed().toLower();
    if (a == QLatin1String("cua_click")) {
        c.kind = QStringLiteral("click");
        c.target = arg.simplified();
        c.valid = !c.target.isEmpty();
    } else if (a == QLatin1String("cua_type")) {
        c.kind = QStringLiteral("type");
        const qsizetype bar = arg.indexOf(QLatin1Char('|'));
        if (bar < 0)
            return c;
        c.target = arg.left(bar).simplified();
        c.text = arg.mid(bar + 1);  // typed verbatim, spaces included
        c.valid = !c.target.isEmpty() && !c.text.isEmpty();
    } else if (a == QLatin1String("cua_scroll")) {
        c.kind = QStringLiteral("scroll");
        const qsizetype bar = arg.lastIndexOf(QLatin1Char('|'));
        const QString amount = (bar < 0 ? arg : arg.mid(bar + 1)).trimmed().toLower();
        c.target = bar < 0 ? QString() : arg.left(bar).simplified();
        bool ok = false;
        const int n = amount.toInt(&ok);
        // Notches (scrollAt moves kScrollPxPerNotch points each): positive up, negative down.
        if (ok)
            c.deltaY = std::clamp(n, -100, 100);
        else if (amount == QLatin1String("up"))
            c.deltaY = 5;
        else if (amount == QLatin1String("down"))
            c.deltaY = -5;
        else if (amount == QLatin1String("pageup") || amount == QLatin1String("page up"))
            c.deltaY = 12;
        else if (amount == QLatin1String("pagedown") || amount == QLatin1String("page down"))
            c.deltaY = -12;
        c.valid = c.deltaY != 0;
    } else if (a == QLatin1String("cua_open")) {
        // "<app>", "<url>", or "<app>|<url>"; the URL gets https:// when it has no scheme.
        c.kind = QStringLiteral("open");
        const qsizetype bar = arg.indexOf(QLatin1Char('|'));
        QString app = (bar < 0 ? arg : arg.left(bar)).simplified();
        QString url = bar < 0 ? QString() : arg.mid(bar + 1).trimmed();
        if (bar < 0 && !app.endsWith(QLatin1String(".app"), Qt::CaseInsensitive)
            && !studio::cua::normalizeOpenUrl(app).isEmpty()) {
            url = app;  // a bare URL: the default browser opens it
            app.clear();
        }
        if (!url.isEmpty()) {
            c.text = studio::cua::normalizeOpenUrl(url);
            if (c.text.isEmpty())
                return c;  // a URL half that is not a URL
        }
        c.target = app;
        c.valid = !c.target.isEmpty() || !c.text.isEmpty();
    } else if (a == QLatin1String("cua_observe")) {
        c.kind = QStringLiteral("observe");
        c.target = arg.simplified();  // the app to read; empty reads whatever is in front
        c.valid = true;
    } else if (a == QLatin1String("cua_hotkey")) {
        c.kind = QStringLiteral("hotkey");
        const qsizetype bar = arg.indexOf(QLatin1Char('|'));
        c.target = bar < 0 ? QString() : arg.left(bar).simplified();  // the app to press it in; empty = front app
        c.text = (bar < 0 ? arg : arg.mid(bar + 1)).simplified().remove(QLatin1Char(' '));
        c.valid = studio::cua::parseHotkeyCombo(c.text).valid;
    } else if (a == QLatin1String("cua_drag")) {
        c.kind = QStringLiteral("drag");
        const qsizetype bar = arg.indexOf(QLatin1Char('|'));
        if (bar < 0)
            return c;
        c.target = arg.left(bar).simplified();
        c.text = arg.mid(bar + 1).simplified();
        c.valid = !c.target.isEmpty() && !c.text.isEmpty();
    }
    return c;
}

QStringList defaultClusterOrder()
{
    return {QStringLiteral("Work"), QStringLiteral("Projects"), QStringLiteral("Apps"), QStringLiteral("Quick Qs")};
}

QString normalizeClusterName(const QString &name)
{
    const QString n = name.toLower().remove(QRegularExpression(QStringLiteral("[^a-z]")));
    if (n == QLatin1String("work") || n == QLatin1String("infinity"))
        return QStringLiteral("Work");
    if (n == QLatin1String("app") || n == QLatin1String("apps"))
        return QStringLiteral("Apps");
    if (n == QLatin1String("project") || n == QLatin1String("projects") || n == QLatin1String("repos"))
        return QStringLiteral("Projects");
    if (n == QLatin1String("quickqs") || n == QLatin1String("quickq") || n == QLatin1String("quickquestions")
        || n == QLatin1String("quickquestion") || n == QLatin1String("questions") || n == QLatin1String("scratch")
        || n == QLatin1String("tabs"))
        return QStringLiteral("Quick Qs");
    return {};
}

QString mapClusterFor(const QString &key, const QString &category, const QString &dir)
{
    const QString cat = category.trimmed().toLower();
    if (cat == QLatin1String("infinity"))
        return QStringLiteral("Work");
    if (cat == QLatin1String("apps") || cat == QLatin1String("app"))
        return QStringLiteral("Apps");
    if (key.startsWith(QLatin1String("~tab:")) || normalizeClusterName(cat) == QLatin1String("Quick Qs"))
        return QStringLiteral("Quick Qs");
    const QString d = dir.trimmed();
    if (!d.isEmpty() && isScratchDir(QDir::cleanPath(expandHome(d))))
        return QStringLiteral("Quick Qs");
    return QStringLiteral("Projects");
}

QStringList moveClusterInOrder(const QStringList &order, const QStringList &present, const QString &cluster,
                               const QString &dir)
{
    const QString c = normalizeClusterName(cluster);
    QStringList shown;
    for (const QString &o : order)
        if (present.contains(o))
            shown << o;
    const int i = int(shown.indexOf(c));
    if (i < 0)
        return order;
    const int n = int(shown.size());
    const int cols = n <= 1 ? 1 : 2;
    const QString d = dir.trimmed().toLower();
    int j = -1;
    if (d == QLatin1String("left"))
        j = (i % cols) > 0 ? i - 1 : -1;
    else if (d == QLatin1String("right"))
        j = (i % cols) < cols - 1 && i + 1 < n ? i + 1 : -1;
    else if (d == QLatin1String("top") || d == QLatin1String("up"))
        j = i - cols >= 0 ? i - cols : -1;
    else if (d == QLatin1String("bottom") || d == QLatin1String("down"))
        j = i + cols < n ? i + cols : -1;
    if (j < 0)
        return order;
    std::swap(shown[i], shown[j]);
    // The shown clusters take the cells they held, in their new order; empty ones keep their place.
    QStringList out = order;
    int k = 0;
    for (QString &o : out)
        if (present.contains(o))
            o = shown[k++];
    return out;
}

StudioCommand parseStudioCommand(const QString &text)
{
    const QString n = stripLead(cmdNorm(text));
    if (n.isEmpty() || n.split(QLatin1Char(' ')).size() > 40)
        return {};
    auto cmd = [](const char *a, const QString &arg = QString()) { return StudioCommand{QString::fromLatin1(a), arg}; };

    // Agent dispatch first: its task text is free-form and may mention anything below.
    {
        static const QRegularExpression origLead(
            QStringLiteral("^(?:hey|ok|okay|yo|so|lapis|studio|please|pls|can you|could you|would you|will you|"
                           "can u|i want to|i'?d like to|i would like to|i want you to|let'?s|go ahead and|just|now)[,\\s]+"),
            QRegularExpression::CaseInsensitiveOption);
        QString o = text.simplified();
        for (int i = 0; i < 8; ++i) {
            const auto m = origLead.match(o);
            if (!m.hasMatch())
                break;
            o = o.mid(m.capturedLength());
        }
        while (!o.isEmpty() && QStringLiteral("?.!").contains(o.back()))
            o.chop(1);
        static const QRegularExpression spawnRe(
            QStringLiteral("^(?:fire off|spin up|spawn|launch|start|kick off|send|dispatch|get|open|put)\\s+"
                           "(?:a\\s+|an\\s+|another\\s+|the\\s+|my\\s+)?(?:new\\s+)?(?:coding\\s+)?"
                           "(?:agent|session|claude|codex|peer|worker)\\s+"
                           "(?:to\\s+|that\\s+will\\s+|and\\s+(?:have\\s+it\\s+)?|on\\s+|for\\s+|working\\s+on\\s+)(.+)$"),
            QRegularExpression::CaseInsensitiveOption);
        static const QRegularExpression tellRe(
            QStringLiteral("^(?:tell|ask|have|get)\\s+(?:the\\s+|my\\s+|an\\s+|a\\s+|this\\s+)?"
                           "(?:agent|session|claude|codex|peer|worker)\\s+(?:to\\s+)?(.+)$"),
            QRegularExpression::CaseInsensitiveOption);
        auto m = spawnRe.match(o);
        if (m.hasMatch())
            return StudioCommand{QStringLiteral("spawn_agent"), m.captured(1).trimmed()};
        m = tellRe.match(o);
        if (m.hasMatch()) {
            StudioCommand c{QStringLiteral("spawn_agent"), m.captured(1).trimmed()};
            c.steer = true;
            return c;
        }
    }

    // Theme: "switch to lapis blue", "use the gold theme", "blue mode".
    {
        static const QRegularExpression themeRe(QStringLiteral(
            "^(?:(?:switch|change|set|use|go|make it|turn it|flip|put it)\\s+(?:back\\s+)?(?:to\\s+|the\\s+theme\\s+to\\s+|"
            "theme\\s+to\\s+|it\\s+to\\s+|on\\s+)?(?:the\\s+)?)?(lapis blue|blue|gold|golden|amber)"
            "(?:\\s+(?:theme|mode|accent|color|colour|look))?(.*)$"));
        const auto m = themeRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(2))) {
            const bool explicitVerb = !n.startsWith(m.captured(1));
            const bool themeWord = n.contains(QLatin1String("theme")) || n.contains(QLatin1String("mode"))
                                   || n.contains(QLatin1String("accent")) || n.contains(QLatin1String("lapis blue"));
            if (explicitVerb || themeWord)
                return cmd("set_theme", m.captured(1) == QLatin1String("gold") || m.captured(1) == QLatin1String("golden")
                                                || m.captured(1) == QLatin1String("amber")
                                            ? QStringLiteral("gold")
                                            : QStringLiteral("blue"));
        }
    }

    // Code diffs on the Stage.
    {
        static const QRegularExpression diffRe(QStringLiteral(
            "^(show|hide|turn on|turn off|enable|disable|display)\\s+(?:the\\s+|me\\s+the\\s+|me\\s+)?(?:code\\s+)?"
            "(?:diffs?|code changes|code)\\b(.*)$"));
        auto m = diffRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(2))) {
            const QString v = m.captured(1);
            const bool on = v == QLatin1String("show") || v == QLatin1String("turn on") || v == QLatin1String("enable")
                            || v == QLatin1String("display");
            return cmd("show_diffs", on ? QStringLiteral("1") : QStringLiteral("0"));
        }
        static const QRegularExpression diffOnOff(
            QStringLiteral("^turn\\s+(?:the\\s+)?(?:code\\s+)?diffs?\\s+(on|off)\\b(.*)$"));
        m = diffOnOff.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(2)))
            return cmd("show_diffs", m.captured(1) == QLatin1String("on") ? QStringLiteral("1") : QStringLiteral("0"));
    }

    // Spotify (spotify_player), before the generic music rules so "pause spotify" never means slowmix.
    if (n.contains(QLatin1String("spotify"))) {
        if (verbObject(n, kOpenVerbs, QStringLiteral("spotify|spotify tab|spotify player")))
            return cmd("open_spotify");
        static const QRegularExpression pauseRe(
            QStringLiteral("^(?:pause|stop|mute|halt)\\s+(?:the\\s+|my\\s+)?(?:music\\s+on\\s+|song\\s+on\\s+)?spotify(.*)$"));
        auto m = pauseRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(1)))
            return cmd("spotify_pause");
        static const QRegularExpression resumeRe(
            QStringLiteral("^(?:resume|unpause|play|start|continue)\\s+(?:the\\s+|my\\s+)?(?:music\\s+on\\s+)?spotify(.*)$"));
        m = resumeRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(1)))
            return cmd("spotify_play");
        static const QRegularExpression nextRe(QStringLiteral(
            "^(?:next|skip)(?:\\s+(?:this|the))?(?:\\s+(?:song|track))?\\s+(?:on\\s+|in\\s+)?spotify(.*)$|^spotify\\s+(?:next|skip)(.*)$"));
        m = nextRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(1) + m.captured(2)))
            return cmd("spotify_next");
        static const QRegularExpression prevRe(QStringLiteral(
            "^(?:previous|prev|last|go back)(?:\\s+(?:a|the))?(?:\\s+(?:song|track))?\\s+(?:on\\s+|in\\s+)?spotify(.*)$|"
            "^spotify\\s+(?:previous|prev|back)(.*)$"));
        m = prevRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(1) + m.captured(2)))
            return cmd("spotify_prev");
        static const QRegularExpression playOnRe(QStringLiteral(
            "^(?:play|put on|queue up|queue|throw on|listen to|start playing)\\s+(.+?)\\s+(?:on|in|from|with|via)\\s+spotify(.*)$"));
        m = playOnRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(2)))
            return cmd("spotify_play", m.captured(1).trimmed());
    }
    {
        // "volume 60", "set the volume to 60 percent", "spotify volume 40", "volume 40 on spotify".
        static const QRegularExpression volRe(QStringLiteral(
            "^(?:set|turn|put|change|make)?\\s*(?:the\\s+|my\\s+)?(spotify\\s+|slowmix\\s+|music\\s+)?volume\\s+"
            "(?:to\\s+|at\\s+)?(\\d{1,3})(?:\\s+percent)?(?:\\s+(?:on|in|for)\\s+(spotify|slowmix|music))?(.*)$"));
        const auto m = volRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(4))) {
            const bool spotify = m.captured(1).startsWith(QLatin1String("spotify"))
                                 || m.captured(3) == QLatin1String("spotify");
            return cmd(spotify ? "spotify_volume" : "set_volume", m.captured(2));
        }
    }

    // Music overlay and playback.
    if (verbObject(n, kOpenVerbs, kMusicWords))
        return cmd("open_music");
    if (verbObject(n, kCloseVerbs, kMusicWords))
        return cmd("close_music");
    {
        static const QRegularExpression pauseRe(QStringLiteral(
            "^(?:pause|stop|mute|halt|silence)(?:\\s+(?:the\\s+|my\\s+)?(?:music|song|track|playback|slowmix|it|audio|player))?(.*)$"));
        const auto m = pauseRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(1)))
            return cmd("pause_music");
        static const QRegularExpression resumeRe(QStringLiteral(
            "^(?:resume|unpause|un pause|continue|keep playing|unmute)(?:\\s+(?:the\\s+|my\\s+)?(?:music|song|track|playback|slowmix|it|audio|player|playing))?(.*)$"));
        const auto r = resumeRe.match(n);
        if (r.hasMatch() && onlyFiller(r.captured(1)))
            return cmd("resume_music");
        static const QRegularExpression toggleRe(QStringLiteral(
            "^(?:toggle|play pause|play or pause)(?:\\s+(?:the\\s+)?(?:music|playback|slowmix|song|track))?(.*)$"));
        const auto t = toggleRe.match(n);
        if (t.hasMatch() && onlyFiller(t.captured(1)))
            return cmd("toggle_music");
        static const QRegularExpression playRe(QStringLiteral(
            "^(?:play|put on|queue up|queue|start playing|listen to|throw on)(?:\\s+(.*))?$"));
        const auto p = playRe.match(n);
        if (p.hasMatch()) {
            QString what = p.captured(1).trimmed();
            static const QRegularExpression lead(QStringLiteral("^(?:the|my|some|a|song|track|tune)\\s+"));
            static const QRegularExpression tail(QStringLiteral(
                "\\s+(?:on slowmix|in slowmix|on the music|please|for me|now|again|song|track)$"));
            for (int i = 0; i < 4; ++i) {
                what = what.remove(lead).remove(tail).trimmed();
            }
            static const QSet<QString> generic{QStringLiteral("music"), QStringLiteral("something"), QStringLiteral("song"),
                                               QStringLiteral("slowmix"), QStringLiteral("it"), QStringLiteral("songs"),
                                               QStringLiteral("tunes"), QStringLiteral("anything"), QStringLiteral("")};
            if (generic.contains(what))
                return cmd("resume_music");
            return cmd("play_track", what);
        }
    }

    if (verbObject(n, kOpenVerbs, QStringLiteral("session review|end of session review|review|reflection|session reflection")))
        return cmd("open_review");
    if (verbObject(n, kOpenVerbs, QStringLiteral("settings|preferences|prefs|controller settings|controller map")))
        return cmd("open_settings");
    if (verbObject(n, kOpenVerbs, QStringLiteral("morning briefing|briefing|brief|daily briefing")))
        return cmd("open_briefing");
    {
        static const QRegularExpression needsRe(QStringLiteral(
            "^(?:show|filter|filter to|show only|only show|just show|switch to|go to|open)\\s+(?:me\\s+)?(?:only\\s+)?(?:the\\s+)?"
            "(?:what\\s+|cards\\s+that\\s+|projects\\s+that\\s+|sessions\\s+that\\s+|ones\\s+that\\s+|agents\\s+that\\s+)?"
            "(?:needs? me|need my attention|needs my attention|waiting on me|blocked on me)(.*)$"));
        const auto m = needsRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(1).replace(QLatin1String("filter"), QString())
                                           .replace(QLatin1String("cards"), QString())))
            return cmd("filter_needs_me");
        static const QRegularExpression allRe(QStringLiteral(
            "^(?:show|show me|see|go back to)\\s+(?:all|every|everything)\\s*(?:of\\s+)?(?:the\\s+|my\\s+)?(cards|projects|islands|sessions|agents)?(.*)$"));
        const auto a = allRe.match(n);
        if (a.hasMatch() && onlyFiller(a.captured(2)))
            return cmd("filter_all");
        static const QRegularExpression clearRe(QStringLiteral("^(?:clear|remove|reset|drop)\\s+(?:the\\s+)?filters?(.*)$"));
        const auto c = clearRe.match(n);
        if (c.hasMatch() && onlyFiller(c.captured(1)))
            return cmd("filter_all");
    }
    {
        static const QRegularExpression layoutRe(QStringLiteral(
            "^(?:reset|restore|tidy|tidy up|rearrange|re arrange|clean up|snap back)\\s+(?:the\\s+|my\\s+)?(?:card\\s+|cards\\s+)?"
            "(?:layout|cards|positions|card positions|constellation)(.*)$"));
        const auto m = layoutRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(1)))
            return cmd("reset_layout");
    }
    {
        // "sort the map by recent", "group the cards by category", "order by activity".
        static const QRegularExpression sortRe(QStringLiteral(
            "^(?:sort|order|arrange|organi[sz]e|group|re ?sort|re ?order)\\s+(?:the\\s+|my\\s+)?"
            "(?:(?:map|cards|constellation|islands|projects|sessions)\\s+)?by\\s+(?:the\\s+|most\\s+)?"
            "(category|categories|type|recent|recency|recently used|latest|newest|activity|active|attention)(.*)$"));
        const auto m = sortRe.match(n);
        if (m.hasMatch() && onlyFiller(m.captured(2))) {
            const QString w = m.captured(1);
            const QString mode = w.startsWith(QLatin1String("categor")) || w == QLatin1String("type") ? QStringLiteral("category")
                                 : (w.startsWith(QLatin1String("activ")) || w == QLatin1String("attention"))
                                     ? QStringLiteral("activity")
                                     : QStringLiteral("recent");
            return cmd("sort_map", mode);
        }
        // "move the work cluster to the left", "put apps on top", "move quick qs down".
        static const QRegularExpression moveRe(QStringLiteral(
            "^(?:move|put|shift|place|slide|drag|push)\\s+(?:the\\s+|my\\s+)?"
            "(work|apps?|projects?|repos|quick ?qs?|quick questions|scratch|tabs)\\s+"
            "(?:(?:cluster|group|cards|section|zone|islands)\\s+)?(?:to\\s+|on\\s+|at\\s+|over\\s+to\\s+)?(?:the\\s+)?"
            "(left|right|top|bottom|up|down)(.*)$"));
        const auto mv = moveRe.match(n);
        if (mv.hasMatch() && onlyFiller(mv.captured(3))) {
            QString d = mv.captured(2);
            if (d == QLatin1String("up"))
                d = QStringLiteral("top");
            else if (d == QLatin1String("down"))
                d = QStringLiteral("bottom");
            return cmd("move_cluster", normalizeClusterName(mv.captured(1)) + QLatin1Char(':') + d);
        }
    }
    return {};
}

QList<StudioCommand> extractActionTags(QString *answer)
{
    QList<StudioCommand> out;
    if (!answer)
        return out;
    static const QRegularExpression tag(QStringLiteral("^\\s*\\[\\[ACTION:([A-Za-z_]+)(?::([^\\]]*))?\\]\\]\\s*"));
    const QStringList known = studioActionNames();
    for (int i = 0; i < 6; ++i) {
        const auto m = tag.match(*answer);
        if (!m.hasMatch())
            break;
        const QString name = m.captured(1).toLower();
        if (known.contains(name))
            out << StudioCommand{name, m.captured(2).trimmed()};
        answer->remove(0, m.capturedLength());
    }
    return out;
}

QString answerDisplayText(const QString &raw, bool streaming)
{
    if (raw.contains(QLatin1String("[[IGNORE]]")))
        return {};
    QString src = raw;
    if (streaming) {
        // A tag still streaming in never flashes: cut an unclosed "[[" and a lone trailing "[".
        const qsizetype open = src.lastIndexOf(QLatin1String("[["));
        if (open >= 0 && src.indexOf(QLatin1String("]]"), open) < 0)
            src.truncate(open);
        else if (src.endsWith(QLatin1Char('[')))
            src.chop(1);
    }
    static const QRegularExpression kTag(QStringLiteral(R"(\[\[ACTION:[^\]]*\]\])"));
    QString out;
    qsizetype last = 0;
    auto it = kTag.globalMatch(src);
    while (it.hasNext()) {
        const QRegularExpressionMatch m = it.next();
        out += src.mid(last, m.capturedStart() - last);
        last = m.capturedEnd();
        QString head = out;
        while (head.endsWith(u' ') || head.endsWith(u'\t'))
            head.chop(1);
        const QChar next = last < src.size() ? src.at(last) : QChar(u'\n');
        const bool proseBefore = !head.isEmpty() && !head.endsWith(u'\n');
        if (proseBefore && !next.isSpace() && next != u'[') {
            out = head + QStringLiteral("\n\n");  // "bar. [[..]]x.com" -> two paragraphs
        } else if (next.isSpace() || next.isPunct()) {
            out = head;  // "a [[..]] b" -> "a b"
        }
    }
    out += src.mid(last);
    static const QRegularExpression kTrailing(QStringLiteral("[ \\t]+\\n"));
    static const QRegularExpression kBlankRuns(QStringLiteral("\\n{3,}"));
    out.replace(kTrailing, QStringLiteral("\n"));
    out.replace(kBlankRuns, QStringLiteral("\n\n"));
    return out.trimmed();
}

QString cuaFollowupBreak(const QString &shownSoFar)
{
    QString s = shownSoFar;
    while (s.endsWith(u' ') || s.endsWith(u'\t'))
        s.chop(1);
    if (s.trimmed().isEmpty() || s.endsWith(QLatin1String("\n\n")))
        return {};
    return s.endsWith(u'\n') ? QStringLiteral("\n") : QStringLiteral("\n\n");
}

int matchSlowmixTrack(const QVariantList &tracks, const QString &query)
{
    static const QSet<QString> stop{QStringLiteral("the"), QStringLiteral("song"), QStringLiteral("track"),
                                    QStringLiteral("a"),   QStringLiteral("by"),   QStringLiteral("please"),
                                    QStringLiteral("for"), QStringLiteral("me"),   QStringLiteral("on"),
                                    QStringLiteral("slowmix"), QStringLiteral("music"), QStringLiteral("again")};
    QStringList words;
    for (const QString &w : cmdNorm(query).split(QLatin1Char(' '), Qt::SkipEmptyParts))
        if (!stop.contains(w))
            words << w;
    const QString q = words.join(QString());
    if (q.size() < 2)
        return -1;
    int best = -1, bestScore = 0;
    for (int i = 0; i < tracks.size(); ++i) {
        const QVariantMap t = tracks[i].toMap();
        const QString name = squashAlnum(t.value(QStringLiteral("name")).toString());
        const QString file = squashAlnum(QFileInfo(t.value(QStringLiteral("fileName")).toString()).completeBaseName());
        int score = 0;
        for (const QString &cand : {name, file}) {
            if (cand.isEmpty())
                continue;
            if (cand == q)
                score = std::max(score, 100);
            else if (cand.startsWith(q))
                score = std::max(score, 90);
            else if (cand.contains(q))
                score = std::max(score, 80);
            else if (words.size() > 1) {
                bool all = true;
                for (const QString &w : words)
                    all = all && cand.contains(squashAlnum(w));
                if (all)
                    score = std::max(score, 60);
            }
        }
        if (score > bestScore) {  // ties keep the earlier (more-played) track
            bestScore = score;
            best = i;
        }
    }
    return best;
}

QString Backend::defaultHistoryPath()
{
    return home() + QStringLiteral("/.lapis-next/studio_history.jsonl");
}

bool Backend::tryHandleStudioCommand(const QString &text)
{
    QString reply;
    return handleStudioCommand(text, m_focusedProject, &reply);
}

bool Backend::handleStudioCommand(const QString &text, const QString &projectKey, QString *reply)
{
    const StudioCommand c = parseStudioCommand(text);
    if (c.isEmpty())
        return false;
    const QString keep = m_commandProject;
    m_commandProject = projectKey;
    const QString r = c.action == QLatin1String("spawn_agent") ? spawnAgent(projectKey, c.arg, c.steer)
                                                               : dispatchStudioAction(c.action, c.arg);
    m_commandProject = keep;
    if (r.isEmpty())
        return false;  // e.g. "play <something that is not a song>": the model answers it
    if (reply)
        *reply = r;
    noteExchange(projectKey, QStringLiteral("studio"), text.trimmed(), r);
    return true;
}

QString Backend::dispatchStudioAction(const QString &action, const QString &arg)
{
    const QString a = action.trimmed().toLower();
    auto speedLabel = [](double s) {
        return std::abs(s - std::round(s)) < 0.001 ? QString::number(s, 'f', 1) + QLatin1Char('x')
                                                   : QString::number(s, 'g', 3) + QLatin1Char('x');
    };
    QString reply;
    QString emitArg = arg;
    if (a == QLatin1String("open_music")) {
        reply = QStringLiteral("Opened Music (Slowmix).");
    } else if (a == QLatin1String("close_music")) {
        reply = QStringLiteral("Closed Music; playback keeps going.");
    } else if (a == QLatin1String("play_track")) {
        m_slowmix.refresh();
        QVariantList pool = m_slowmix.tracks();
        int i = matchSlowmixTrack(pool, arg);
        if (i < 0) {
            // Not in the 30-day rotation: look through every song in the clips folder.
            pool.clear();
            for (const SlowmixTrack &t : curateSlowmixTracks(m_slowmix.clipsDir(), m_slowmix.playsPath(), true))
                pool << QVariantMap{{QStringLiteral("name"), t.name},
                                    {QStringLiteral("fileName"), t.fileName},
                                    {QStringLiteral("path"), t.path}};
            i = matchSlowmixTrack(pool, arg);
        }
        if (i < 0)
            return {};
        const QVariantMap t = pool[i].toMap();
        m_slowmix.selectTrack(t.value(QStringLiteral("path")).toString());
        emitArg = m_slowmix.currentTrack();
        reply = QStringLiteral("Playing %1 (%2 %3).").arg(m_slowmix.currentTrack(), speedLabel(m_slowmix.speed()),
                                                           m_slowmix.reverb());
    } else if (a == QLatin1String("pause_music")) {
        if (m_slowmix.playing())
            m_slowmix.togglePlay();
        reply = m_slowmix.currentTrack().isEmpty() ? QStringLiteral("Nothing is playing.")
                                                   : QStringLiteral("Paused %1.").arg(m_slowmix.currentTrack());
    } else if (a == QLatin1String("resume_music") || a == QLatin1String("toggle_music")) {
        if (a == QLatin1String("toggle_music") && m_slowmix.playing()) {
            m_slowmix.togglePlay();
            reply = QStringLiteral("Paused %1.").arg(m_slowmix.currentTrack());
        } else if (!m_slowmix.playing()) {
            if (m_slowmix.currentPath().isEmpty()) {
                m_slowmix.refresh();
                const QVariantList tracks = m_slowmix.tracks();
                if (tracks.isEmpty())
                    return QStringLiteral("No songs in the clips folder.");
                m_slowmix.selectTrack(tracks.first().toMap().value(QStringLiteral("path")).toString());
            } else {
                m_slowmix.togglePlay();
            }
        }
        if (reply.isEmpty())
            reply = QStringLiteral("Playing %1.").arg(m_slowmix.currentTrack());
    } else if (a == QLatin1String("set_volume") || a == QLatin1String("spotify_volume")) {
        // "60", "60%", "0.6" -> 60.
        QString v = arg.trimmed();
        v.remove(QLatin1Char('%'));
        bool ok = false;
        double x = v.toDouble(&ok);
        if (!ok)
            return {};
        if (x > 0 && x <= 1.0 && v.contains(QLatin1Char('.')))
            x *= 100.0;
        const int pct = std::clamp(int(std::lround(x)), 0, 100);
        emitArg = QString::number(pct);
        if (a == QLatin1String("set_volume")) {
            m_slowmix.setVolume(pct / 100.0);
            reply = QStringLiteral("Slowmix volume %1%.").arg(pct);
        } else {
            m_spotify.setVolume(pct);
            reply = QStringLiteral("Spotify volume %1%.").arg(pct);
        }
    } else if (a == QLatin1String("open_spotify")) {
        reply = QStringLiteral("Opened Music (Spotify).");
    } else if (a == QLatin1String("spotify_play")) {
        const QString q = arg.simplified();
        if (q.isEmpty()) {
            m_spotify.play();
            reply = QStringLiteral("Playing Spotify.");
        } else {
            m_spotify.searchAndPlay(q);
            reply = QStringLiteral("Playing %1 on Spotify.").arg(q);
        }
    } else if (a == QLatin1String("spotify_pause")) {
        m_spotify.pause();
        reply = QStringLiteral("Paused Spotify.");
    } else if (a == QLatin1String("spotify_next")) {
        m_spotify.nextTrack();
        reply = QStringLiteral("Next track on Spotify.");
    } else if (a == QLatin1String("spotify_prev")) {
        m_spotify.prevTrack();
        reply = QStringLiteral("Previous track on Spotify.");
    } else if (a == QLatin1String("go_home")) {
        reply = QStringLiteral("Back to the Constellation.");
    } else if (a == QLatin1String("focus_project") || a == QLatin1String("open_stage")
               || a == QLatin1String("open_ladder")) {
        // Best project name match: exact, prefix, then substring, punctuation ignored.
        const QString q = squashAlnum(arg);
        QString key, name;
        int best = 0;
        for (const QVariant &v : std::as_const(m_projectList)) {
            const QVariantMap p = v.toMap();
            const QString n = p.value(QStringLiteral("name")).toString();
            const QString k = p.value(QStringLiteral("key")).toString();
            int score = 0;
            if (q.isEmpty())
                score = k == m_focusedProject ? 1 : 0;
            else
                for (const QString &c : {squashAlnum(n), squashAlnum(k)})
                    score = std::max(score, c == q ? 3 : c.startsWith(q) ? 2 : c.contains(q) ? 1 : 0);
            if (score > best) {
                best = score;
                key = k;
                name = n;
            }
        }
        if (key.isEmpty())
            return {};
        emitArg = key;
        reply = a == QLatin1String("focus_project") ? QStringLiteral("Focused %1.").arg(name)
                : a == QLatin1String("open_stage")  ? QStringLiteral("Opened %1's Stage.").arg(name)
                                                    : QStringLiteral("Opened %1's Ladder.").arg(name);
    } else if (a == QLatin1String("open_review")) {
        reply = QStringLiteral("Opened Session Review.");
    } else if (a == QLatin1String("filter_needs_me")) {
        reply = QStringLiteral("Showing only the cards that need you.");
    } else if (a == QLatin1String("filter_all")) {
        reply = QStringLiteral("Showing all cards.");
    } else if (a == QLatin1String("reset_layout")) {
        resetCardPositions();
        reply = QStringLiteral("Reset the card layout.");
    } else if (a == QLatin1String("sort_map")) {
        const QString v = arg.trimmed().toLower();
        const QString mode = v.startsWith(QLatin1String("rec")) ? QStringLiteral("recent")
                             : v.startsWith(QLatin1String("act")) ? QStringLiteral("activity")
                                                                  : QStringLiteral("category");
        setMapSort(mode);
        resetCardPositions();  // a re-sort re-flows every card, dragged ones included
        emitArg = mode;
        reply = mode == QLatin1String("recent")     ? QStringLiteral("Sorted the map by most recent.")
                : mode == QLatin1String("activity") ? QStringLiteral("Sorted the map by activity.")
                                                    : QStringLiteral("Sorted the map by category.");
    } else if (a == QLatin1String("move_cluster")) {
        const QString cluster = normalizeClusterName(arg.section(QLatin1Char(':'), 0, 0));
        QString dir = arg.section(QLatin1Char(':'), 1, 1).trimmed().toLower();
        if (dir == QLatin1String("up"))
            dir = QStringLiteral("top");
        else if (dir == QLatin1String("down"))
            dir = QStringLiteral("bottom");
        if (cluster.isEmpty() || !QStringList{QStringLiteral("left"), QStringLiteral("right"), QStringLiteral("top"),
                                              QStringLiteral("bottom")}
                                      .contains(dir))
            return {};
        if (!presentClusters().contains(cluster))
            return QStringLiteral("There are no %1 cards on the map.").arg(cluster);
        emitArg = cluster + QLatin1Char(':') + dir;
        reply = moveCluster(cluster, dir) ? QStringLiteral("Moved %1 to the %2.").arg(cluster, dir)
                                          : QStringLiteral("%1 is already at the %2.").arg(cluster, dir);
    } else if (a == QLatin1String("open_settings")) {
        reply = QStringLiteral("Opened Settings.");
    } else if (a == QLatin1String("open_briefing")) {
        reply = QStringLiteral("Opened the Morning Briefing.");
    } else if (a == QLatin1String("set_theme")) {
        const QString v = arg.trimmed().toLower();
        const bool gold = v == QLatin1String("gold") || v == QLatin1String("golden") || v == QLatin1String("amber");
        setTheme(gold ? QStringLiteral("gold") : QStringLiteral("lapis"));
        emitArg = gold ? QStringLiteral("gold") : QStringLiteral("blue");
        reply = gold ? QStringLiteral("Switched to the Gold theme.") : QStringLiteral("Switched to Lapis Blue.");
    } else if (a == QLatin1String("show_diffs")) {
        const QString v = arg.trimmed().toLower();
        const bool on = !(v == QLatin1String("0") || v == QLatin1String("off") || v == QLatin1String("false"));
        setShowCodeDiffs(on);
        emitArg = on ? QStringLiteral("1") : QStringLiteral("0");
        reply = on ? QStringLiteral("Code diffs are on.") : QStringLiteral("Code diffs are hidden.");
    } else if (a == QLatin1String("spawn_agent")) {
        return spawnAgent(m_commandProject.isEmpty() ? m_focusedProject : m_commandProject, arg, false);
    } else if (a == QLatin1String("steer_agent")) {
        QString targetKey = m_commandProject.isEmpty() ? m_focusedProject : m_commandProject;
        QString msg = arg.trimmed();
        const qsizetype bar = msg.indexOf(QLatin1Char('|'));
        if (bar > 0) {
            const QString projQuery = squashAlnum(msg.left(bar));
            const QString rest = msg.mid(bar + 1).trimmed();
            if (!projQuery.isEmpty() && !rest.isEmpty()) {
                int best = 0;
                QString matchedKey;
                for (const Project &p : m_projects) {
                    int score = 0;
                    for (const QString &c : {squashAlnum(p.name), squashAlnum(p.key)})
                        score = std::max(score, c == projQuery ? 3 : c.startsWith(projQuery) ? 2 : c.contains(projQuery) ? 1 : 0);
                    if (score > best) {
                        best = score;
                        matchedKey = p.key;
                    }
                }
                if (!matchedKey.isEmpty()) {
                    targetKey = matchedKey;
                    msg = rest;
                }
            }
        }
        return spawnAgent(targetKey, msg, true);
    } else if (a.startsWith(QLatin1String("cua_"))) {
        const CuaCommand c = parseCuaAction(a, arg);
        if (!c.valid)
            return {};
        reply = startCuaAction(a, c);
        emitArg = c.target;
    } else {
        return {};
    }
    emit uiActionRequested(a, emitArg);
    return reply;
}

QString Backend::startCuaAction(const QString &action, const CuaCommand &c)
{
    const QString verb = c.kind == QLatin1String("type")      ? QStringLiteral("Typing into %1").arg(c.target)
                         : c.kind == QLatin1String("scroll")  ? (c.target.isEmpty() ? QStringLiteral("Scrolling")
                                                                                     : QStringLiteral("Scrolling %1").arg(c.target))
                         : c.kind == QLatin1String("observe") ? (c.target.isEmpty() ? QStringLiteral("Reading the screen")
                                                                                     : QStringLiteral("Reading %1").arg(c.target))
                         : c.kind == QLatin1String("hotkey")
                             ? QStringLiteral("Pressing %1%2")
                                   .arg(studio::cua::parseHotkeyCombo(c.text).display,
                                        c.target.isEmpty() ? QString() : QStringLiteral(" in %1").arg(c.target))
                         : c.kind == QLatin1String("drag") ? QStringLiteral("Dragging %1 to %2").arg(c.target, c.text)
                         : c.kind == QLatin1String("open")
                             ? (c.text.isEmpty() ? QStringLiteral("Opening %1").arg(c.target)
                                                 : QStringLiteral("Opening %1%2")
                                                       .arg(c.text, c.target.isEmpty() ? QString() : QStringLiteral(" in %1").arg(c.target)))
                             : QStringLiteral("Clicking %1").arg(c.target);
    // What the log, the finished signal and the follow-up turn name as the step's target.
    const bool hotkey = c.kind == QLatin1String("hotkey");
    const bool open = c.kind == QLatin1String("open");
    const QString shownTarget = hotkey ? (c.target.isEmpty() ? c.text : c.target + QLatin1Char('|') + c.text)
                                : open ? (c.text.isEmpty() ? c.target : c.target.isEmpty() ? c.text : c.target + QLatin1Char('|') + c.text)
                                : c.kind == QLatin1String("drag") ? c.target + QLatin1Char('|') + c.text
                                                                  : c.target;
    const QString logLine = action + QLatin1Char(':') + shownTarget
                            + (c.kind == QLatin1String("type")     ? QLatin1Char('|') + c.text
                               : c.kind == QLatin1String("scroll") ? QLatin1Char('|') + QString::number(c.deltaY)
                                                                   : QString());
    // Inside a model answer's batch, the result feeds the closed loop (maybeFinishCuaBatch).
    const bool inBatch = m_cuaBatch.open;
    const quint64 batchId = m_cuaBatchId;
    if (inBatch)
        ++m_cuaBatch.outstanding;
    // A Spotlight turn acts on the app behind the bar: the bar ghosts out while it runs, and a
    // step naming no app goes to that app.
    const bool spot = inBatch && m_cuaBatch.streamId == QLatin1String(kSpotlightStreamId) && m_spotlight;
    const QString defaultApp = spot ? m_spotlight->targetApp() : QString();
    if (spot) {
        m_spotlightTurnRanCua = true;
        m_spotlight->setActionRunning(true, (defaultApp.isEmpty() ? QString() : defaultApp + QStringLiteral(" \u00B7 "))
                                                + verb + QStringLiteral("\u2026"));
    }
    if (m_mode != Mode::Live) {
        // --test and every mock backend: record only, never capture, click or type.
        m_cuaLog << logLine;
        emit cuaActionFinished(action, shownTarget, true, QStringLiteral("MOCK"), QStringLiteral("recorded, not run"));
        if (spot)
            noteSpotlightAction(true, verb);
        if (inBatch)
            noteCuaResult(batchId, CuaStepResult{action, shownTarget, true, QStringLiteral("MOCK"), QStringLiteral("recorded, not run"),
                                                 c.kind == QLatin1String("observe") ? QStringLiteral("(mock: no screen in --test)")
                                                                                    : QString()});
        return verb + QStringLiteral(" (recorded, not run).");
    }
    if (!m_cuaPool) {
        m_cuaPool = new QThreadPool(this);
        m_cuaPool->setMaxThreadCount(1);  // one visible action at a time, in order
    }
    ++m_cuaPending;
    syncEscapeWatch();
    if (m_cuaHud)
        studio::cua::showCuaHud(verb + QStringLiteral("\u2026"), studio::cua::HudTone::Working);
    const quint64 epoch = g_cuaEpoch.load();
    // The trace log (training data) learns who asked and whether that project is redacted; the
    // executor skips the step entirely for redacted / infinity work.
    studio::cua::TraceContext trace;
    trace.projectKey = inBatch ? m_cuaBatch.projectKey : m_commandProject;
    trace.goal = inBatch ? m_cuaBatch.goal : m_quickQuestion;
    if (const Project *p = find(trace.projectKey))
        trace.redacted = std::any_of(p->sessions.cbegin(), p->sessions.cend(), [](const SessionInfo &s) { return s.redacted; })
                         || studio::cua::traceBlocked(false, {p->name, p->dir});
    QPointer<Backend> self(this);
    m_cuaPool->start([self, action, c, shownTarget, logLine, inBatch, batchId, trace, spot, defaultApp, verb, epoch] {
        // An interrupt since this step was queued: it never runs. A cancel latched by an older
        // turn (its last step has returned: the pool runs one at a time) is cleared first.
        if (!claimCuaEpoch(epoch)) {
            QMetaObject::invokeMethod(
                QCoreApplication::instance(),
                [self, action, target = shownTarget, logLine] {
                    if (!self)
                        return;
                    const QString why = QStringLiteral("interrupted by the user");
                    self->m_cuaPending = std::max(0, self->m_cuaPending - 1);
                    self->m_cuaLog << QStringLiteral("%1 -> ERR %2").arg(logLine, why);
                    emit self->cuaActionFinished(action, target, false, QStringLiteral("ERR"), why);
                    self->onCuaActionDone();
                    self->syncSpotlightTurn();  // a new turn may be waiting on the queue to drain
                },
                Qt::QueuedConnection);
            return;
        }
        studio::cua::CuaExecutor ex;
        ex.setTraceContext(trace);
        ex.setDefaultAppHint(defaultApp);
        const studio::cua::CuaExecResult r = c.kind == QLatin1String("type")      ? ex.typeIntoTarget(c.target, c.text)
                                             : c.kind == QLatin1String("scroll")  ? ex.scrollTarget(c.target, c.deltaY)
                                             : c.kind == QLatin1String("observe") ? ex.observeApp(c.target)
                                             : c.kind == QLatin1String("hotkey")  ? ex.hotkeyAction(c.target, c.text)
                                             : c.kind == QLatin1String("drag")    ? ex.dragTarget(c.target, c.text)
                                             : c.kind == QLatin1String("open")    ? ex.openAction(c.target, c.text)
                                                                                  : ex.clickTarget(c.target);
        QMetaObject::invokeMethod(
            QCoreApplication::instance(),
            [self, action, target = shownTarget, logLine, r, inBatch, batchId, spot, verb] {
                if (!self)
                    return;
                self->m_cuaPending = std::max(0, self->m_cuaPending - 1);
                self->m_cuaLog << QStringLiteral("%1 -> %2 %3").arg(logLine, r.ok ? r.tier : QStringLiteral("ERR"), r.detail);
                while (self->m_cuaLog.size() > 50)
                    self->m_cuaLog.removeFirst();
                emit self->cuaActionFinished(action, target, r.ok, r.tier, r.detail);
                if (spot)
                    self->noteSpotlightAction(r.ok, r.ok ? verb : r.detail);
                if (inBatch)
                    self->noteCuaResult(batchId, CuaStepResult{action, target, r.ok, r.ok ? r.tier : QStringLiteral("ERR"),
                                                               r.detail, r.observation});
                self->onCuaActionDone();
                if (spot)
                    self->syncSpotlightTurn();  // the last step of a turn closes the bar
            },
            Qt::QueuedConnection);
    });
    return verb + QStringLiteral("...");
}

QString Backend::spawnAgent(const QString &projectKey, const QString &task, bool steerLive)
{
    const QString t = task.simplified();
    if (t.isEmpty())
        return QStringLiteral("Say what the agent should do.");
    Project *p = find(projectKey);
    if (!p)
        return QStringLiteral("Open a project first, so the agent knows which folder to work in.");
    if (steerLive && !p->sessions.isEmpty()) {
        steer(p->key, t);
        emit uiActionRequested(QStringLiteral("spawn_agent"), t);
        return QStringLiteral("Sent to the %1 agent: %2").arg(p->name, clampWords(t, 14));
    }
    // A $HOME scratch tab ("~tab:*", or any session living in $HOME) spawns in $HOME; only an
    // empty or temp folder (/tmp, the system temp dir, /) is refused.
    const QString homeDir = QDir::cleanPath(home());
    QString dir = QDir::cleanPath(expandHome(p->dir));
    if (p->dir.trimmed().isEmpty() && p->key.startsWith(QLatin1String("~tab:")))
        dir = homeDir;
    if (dir != homeDir && (p->dir.trimmed().isEmpty() || isScratchDir(dir)))
        return QStringLiteral("%1 has no project folder (scratch or /tmp); open a real project to spawn an agent.").arg(p->name);
    emit uiActionRequested(QStringLiteral("spawn_agent"), t);
    startSession(dir, QStringLiteral("claude"), QStringLiteral("claude-opus-5-5"), t);
    return QStringLiteral("Spawning an Opus 5.5 agent in %1: %2").arg(tildePath(dir), clampWords(t, 14));
}

void Backend::runAnswerTags(QString *answer)
{
    if (!answer || !answer->contains(QLatin1String("[[")))
        return;
    // Every tag anywhere in the reply runs, in order, in this one turn (compound batches).
    const ParsedModelOutput parsed = parseStudioModelOutput(*answer);
    if (parsed.ignore) {
        answer->clear();
        return;
    }
    if (parsed.actions.isEmpty())
        return;
    // Every cua_* action this answer starts joins one batch; when the last finishes, the batch
    // may earn a follow-up turn (maybeFinishCuaBatch).
    m_cuaBatch = CuaBatch{};
    m_cuaBatch.open = true;
    m_cuaBatch.streamId = m_cuaAnswerStream;
    m_cuaBatch.projectKey = m_commandProject;
    m_cuaBatch.goal = m_cuaAnswerStream.isEmpty() ? m_quickQuestion : m_cuaAnswerGoal;
    m_cuaBatch.shownRaw = *answer;
    ++m_cuaBatchId;
    const QStringList replies = runStudioActions(parsed.actions);
    m_cuaBatch.open = false;
    *answer = parsed.cleanText.isEmpty() ? replies.join(QLatin1Char(' ')) : parsed.cleanText;
    maybeFinishCuaBatch();  // Mock (and an all-invalid batch) finished synchronously
}

QString Backend::runModelAnswer(const QString &answer)
{
    QString a = answer;
    m_cuaAnswerStream.clear();
    runAnswerTags(&a);
    return a;
}

void Backend::resetCuaLoop()
{
    m_cuaBatch = CuaBatch{};
    ++m_cuaBatchId;  // results of actions still running belong to nothing now
    m_cuaLoopDepth = 0;
    m_cuaFollowups.clear();
}

void Backend::noteCuaResult(quint64 batchId, const CuaStepResult &step)
{
    if (batchId != m_cuaBatchId)
        return;  // a new prompt or a cancel came in while it ran
    m_cuaBatch.steps << step;
    m_cuaBatch.outstanding = std::max(0, m_cuaBatch.outstanding - 1);
    maybeFinishCuaBatch();
}

void Backend::maybeFinishCuaBatch()
{
    if (m_cuaBatch.open || m_cuaBatch.outstanding > 0 || m_cuaBatch.steps.isEmpty())
        return;
    const QVector<CuaStepResult> steps = std::exchange(m_cuaBatch.steps, {});
    if (studio::cua::cancelFlag.load()) {
        resetCuaLoop();  // the user aborted computer use: stop here
        return;
    }
    if (!cuaShouldContinue(steps, m_cuaLoopDepth))
        return;
    ++m_cuaLoopDepth;
    const QString prompt = cuaFollowupPrompt(steps);
    m_cuaFollowups << prompt;
    if (m_mode != Mode::Live)
        return;  // --test: recorded only; no model call, no network
    QStringList heads;
    for (const CuaStepResult &s : steps)
        heads << cuaObservationPrompt(s.action + (s.target.isEmpty() ? QString() : QLatin1Char(':') + s.target), s.tier,
                                      s.ok, s.detail, QStringLiteral("-"))
                     .section(QLatin1Char('\n'), 0, 0);
    const QString note = heads.join(QLatin1Char(' '));  // history keeps the headers, never the AX dump
    const QString streamId = m_cuaBatch.streamId, project = m_cuaBatch.projectKey, goal = m_cuaBatch.goal;
    const QString shown = m_cuaBatch.shownRaw;
    const quint64 id = m_cuaBatchId;
    m_cuaFollowupScheduled = true;
    // Next event-loop turn: the answer that started this batch has been recorded by then.
    QTimer::singleShot(0, this, [this, id, streamId, project, prompt, note, goal, shown] {
        m_cuaFollowupScheduled = false;
        if (id != m_cuaBatchId) {
            syncSpotlightTurn();
            return;  // reset (new question, cancel) in between
        }
        m_cuaFollowupAsk = true;
        m_cuaFollowupNote = note;
        m_cuaFollowupGoal = goal;
        if (streamId.isEmpty()) {
            m_quickNote = note;
            ask(project, m_quickArtifact, prompt);
        } else {
            // The card keeps appending on this stream: the follow-up's prose opens a fresh
            // paragraph instead of running into the last turn ("...bar.x.com is loaded").
            // The Spotlight bar shows only the newest reply, so it needs no break.
            const QString gap = cuaFollowupBreak(shown);
            if (!gap.isEmpty() && streamId != QLatin1String(kSpotlightStreamId))
                emit streamDelta(streamId, gap, QString());
            startConversation(streamId, project, QString(), prompt);
        }
        m_cuaFollowupAsk = false;
        m_cuaFollowupNote.clear();
        m_cuaFollowupGoal.clear();
        syncSpotlightTurn();
    });
}

// --- L2 routing and the MacBook-wide Spotlight bar ---------------------------------------------

studio::cua::CuaSpotlight *Backend::spotlight()
{
    if (!m_spotlight) {
        // A real panel only in the interactive live window (warm helpers); Offscreen everywhere
        // else, so --test and every snapshot mode open no window.
        const auto mode = m_mode == Mode::Live && m_cuaHud ? studio::cua::CuaSpotlight::Mode::Visible
                                                           : studio::cua::CuaSpotlight::Mode::Offscreen;
        m_spotlight = std::make_unique<studio::cua::CuaSpotlight>(mode);
        m_spotlight->onSubmit = [this](const QString &) { submitSpotlight(); };
        m_spotlight->onDismiss = [this] {
            if (jarvisOverlay())
                handleJarvisEscape();
            else
                dismissSpotlight(studio::cua::studioAppIsActive());
        };
        m_spotlight->onInputEdited = [this] { onSpotlightInputEdited(); };
        m_spotlight->onGlobalEscape = [this] { if (jarvisOverlay()) handleJarvisEscape(); };
    }
    return m_spotlight.get();
}

bool Backend::spotlightOpen() const
{
    return m_spotlight && m_spotlight->isOpen();
}

void Backend::syncSpotlightOpen()
{
    const bool now = spotlightOpen();
    if (now == m_spotlightOpenSeen)
        return;
    m_spotlightOpenSeen = now;
    if (now && m_cuaHud)
        refreshShadowExcludes();  // the panel's window exists now: keep it out of frames
    emit spotlightOpenChanged();
}

int Backend::prepareVoiceTarget(bool active)
{
    if (!active) {
        if (m_spotlight && m_spotlight->isOpen()) {
            if (jarvisOverlay() && m_voiceTarget == VoiceTarget::Spotlight && !m_voiceCancelling) {
                // Listening stopped (release after a hold, or the second tap): Phonon pastes
                // in ~185 ms, and that paste sends itself.
                m_spotlight->setVoice(studio::cua::SpotlightVoice::Transcribing);
                armAutoSend();
            } else {
                m_spotlight->setListening(false);
            }
        }
        return 0;
    }
    const bool studioActive = m_studioActiveProbe ? m_studioActiveProbe()
                                                  : (m_cuaHud ? studio::cua::studioAppIsActive() : true);
    if (jarvisOverlay()) {
        // Instant interrupt: L2 while anything runs is the harness's Esc, then a fresh turn.
        if (jarvisBusy())
            interruptJarvis(true);
        disarmAutoSend();
        m_voiceTarget = VoiceTarget::Spotlight;
        studio::cua::CuaSpotlight *sp = spotlight();
        // The bar already up keeps its context; else Studio itself, or the app in front.
        const bool overStudio = sp->isOpen() ? sp->state().overStudio : studioActive;
        const bool wasKey = overStudio ? sp->openOrFocusOverStudio(jarvisStudioLabel()) : sp->openOrFocus();
        sp->setListening(true);
        syncSpotlightOpen();
        return wasKey ? 0 : kSpotlightVoiceDelayMs;
    }
    if (studioActive && !spotlightOpen()) {
        // Inside Studio: QML's padVoiceHold focuses the ask card; the key goes out at once.
        m_voiceTarget = VoiceTarget::Studio;
        return 0;
    }
    m_voiceTarget = VoiceTarget::Spotlight;
    studio::cua::CuaSpotlight *sp = spotlight();
    const bool wasKey = sp->openOrFocus();
    sp->setListening(true);
    syncSpotlightOpen();
    return wasKey ? 0 : kSpotlightVoiceDelayMs;
}

void Backend::submitSpotlight()
{
    if (!m_spotlight || !m_spotlight->isOpen())
        return;
    const QString q = m_spotlight->takeInputText().trimmed();
    disarmAutoSend();  // R2 / Return beat the paste timer, or the timer fired: either way, once
    if (q.isEmpty())
        return;
    if (m_streamAsks.contains(QLatin1String(kSpotlightStreamId)))
        cancelStream(QLatin1String(kSpotlightStreamId));  // a new prompt replaces a running one
    // Tier 3: the app right behind the bar, with a fast bounded AX read of it (live only).
    const QString app = m_spotlight->targetApp();
    const int pid = m_spotlight->targetPid();
    QString slice;
    if (m_spotlight->state().overStudio) {
        // The Jarvis bar over Studio itself: the hovered / selected island is the target (the
        // live state rides in Tier 3 as always), and both Studio UI actions and desktop cua_*
        // actions are fair game.
        slice = QStringLiteral("The user is in Lapis Studio itself (%1); the Jarvis bar is open over it. Studio UI actions "
                               "(open_stage, focus_project, steer_agent, spawn_agent, open_music, ...) and desktop cua_* "
                               "actions are both available.\n")
                    .arg(jarvisStudioLabel());
        if (m_targets && m_targets->hasActiveTarget())
            slice += formatTargetSlice({m_targets->active()}, 1500);
    } else if (!app.isEmpty() || pid > 0) {
        slice = QStringLiteral("Active macOS app behind Spotlight: %1\n").arg(app.isEmpty() ? QStringLiteral("unknown") : app);
        if (m_mode == Mode::Live && m_cuaHud && pid > 0)
            slice += studio::cua::formatNodesForModel(studio::cua::inspectApp(pid, 500, 32, 0.35),
                                                      studio::cua::kPostActionChars);
    }
    m_spotlight->beginTurn(q);
    m_spotlightTurn = true;
    m_spotlightTurnRanCua = false;
    m_spotlightLastAction.clear();
    startConversation(QLatin1String(kSpotlightStreamId), focusedProject(), slice, q);
    syncSpotlightTurn();  // a UI command (or an error) may already have finished it
}

void Backend::dismissSpotlight(bool restoreTarget)
{
    if (!m_spotlight)
        return;
    disarmAutoSend();
    if (jarvisOverlay() && m_voiceTarget == VoiceTarget::Spotlight && (m_gamepad.voiceActive() || m_gamepad.voiceToggled())) {
        m_voiceCancelling = true;  // never leave Phonon's fn key down behind a closed bar
        m_gamepad.cancelVoice();
        m_voiceCancelling = false;
    }
    if (m_streamAsks.contains(QLatin1String(kSpotlightStreamId)))
        cancelStream(QLatin1String(kSpotlightStreamId));
    m_spotlightTurn = false;
    m_spotlight->setBusy(false);
    m_spotlight->dismiss(restoreTarget);
    syncSpotlightOpen();
}

// --- Jarvis: hold / tap voice, auto-send, instant interrupt ---------------------------------

QString Backend::jarvisStudioLabel()
{
    const Project *p = m_focusedProject.isEmpty() ? nullptr : find(m_focusedProject);
    const QString name = p ? p->name.simplified() : QString();
    return name.isEmpty() ? QStringLiteral("Lapis") : QStringLiteral("Lapis \u00B7 ") + name;
}

bool Backend::jarvisBusy() const
{
    return !m_streamAsks.isEmpty() || m_cuaPending > 0 || m_cuaBatch.open || m_cuaBatch.outstanding > 0
           || m_cuaFollowupScheduled || m_spotlightTurn;
}

bool Backend::interruptJarvis(bool superseded)
{
    if (!jarvisBusy())
        return false;
    ++m_jarvisInterrupts;
    // What was being done, for the model's next turn: the bar's request first, then any card's.
    QString goal;
    const auto spotIt = m_streamAsks.constFind(QLatin1String(kSpotlightStreamId));
    if (spotIt != m_streamAsks.cend())
        goal = spotIt->goal.isEmpty() ? spotIt->question : spotIt->goal;
    for (auto it = m_streamAsks.cbegin(); goal.isEmpty() && it != m_streamAsks.cend(); ++it)
        goal = it->goal.isEmpty() ? it->question : it->goal;
    if (goal.isEmpty())
        goal = m_cuaBatch.goal;
    if (goal.isEmpty() && m_spotlightTurn && m_spotlight)
        goal = m_spotlight->state().lastQuestion;
    // 1. Abort the in-flight step (glide, click, type, hotkey, drag, AX scan) and drop the queue.
    latchCuaCancel();  // also bumps the epoch: nothing queued by the old turn will run
    if (m_cuaPending == 0)
        releaseCuaCancel();  // nothing in flight: never leave the latch set for the next turn
    // 2. Every conversation stream and the closed CUA loop.
    const QStringList ids = m_streamAsks.keys();
    for (const QString &id : ids) {
        cancelStream(id);
        if (id != QLatin1String(kSpotlightStreamId))
            emit streamFinished(id, QStringLiteral("Interrupted"));  // the card stops its busy state
    }
    resetCuaLoop();
    m_spotlightTurn = false;
    m_spotlightRaw.clear();
    if (m_spotlight) {
        m_spotlight->setActionRunning(false);
        m_spotlight->setBusy(false);
        m_spotlight->setStatusLine(superseded ? QStringLiteral("Interrupted") : QStringLiteral("Stopped"));
    }
    // 3. Nothing of the old turn stays on screen.
    if (m_cuaHud) {
        studio::cua::hideCuaReticle(0);
        studio::cua::hideCuaHud(0);
    }
    // 4. The model sees what was cut off, so "no, in a new tab instead" lands in context.
    if (!goal.trimmed().isEmpty()) {
        m_sharedHistory << ChatTurn{QStringLiteral("user"), goal.trimmed()}
                        << ChatTurn{QStringLiteral("assistant"),
                                    superseded ? QStringLiteral("[Interrupted by user \u2014 superseded by the next instruction]")
                                               : QStringLiteral("[Stopped by user]")};
        while (m_sharedHistory.size() > 20)
            m_sharedHistory.remove(0, 2);
    }
    syncEscapeWatch();
    return true;
}

void Backend::cancelJarvisListening()
{
    disarmAutoSend();
    if (m_gamepad.voiceActive() || m_gamepad.voiceToggled()) {
        m_voiceCancelling = true;
        m_gamepad.cancelVoice();  // fn key-up (when it went down), toggle cleared
        m_voiceCancelling = false;
    }
    if (!m_spotlight)
        return;
    m_spotlight->setInputText(QString());  // the partial draft goes too
    m_spotlight->setVoice(studio::cua::SpotlightVoice::Idle);
    if (m_spotlight->isOpen())
        dismissSpotlight(!m_spotlight->state().overStudio);
}

void Backend::handleJarvisEscape()
{
    const bool listening = m_awaitingAutoSend || m_gamepad.voiceActive() || m_gamepad.voiceToggled()
                           || (m_spotlight && m_spotlight->state().voice != studio::cua::SpotlightVoice::Idle);
    if (listening) {
        cancelJarvisListening();
        return;
    }
    if (m_spotlightTurn || m_cuaPending > 0 || m_cuaBatch.outstanding > 0 || m_cuaFollowupScheduled)
        interruptJarvis(false);
    if (spotlightOpen())
        dismissSpotlight(!m_spotlight->state().overStudio);
}

void Backend::armAutoSend()
{
    if (!m_spotlight)
        return;
    m_awaitingAutoSend = true;
    m_autoSendBaseline = m_spotlight->inputText();
    m_autoSendDebounce.stop();
    m_autoSendNoPaste.start();
    m_autoSendGiveUp.start();
}

void Backend::disarmAutoSend()
{
    const bool was = m_awaitingAutoSend;
    m_awaitingAutoSend = false;
    m_autoSendDebounce.stop();
    m_autoSendNoPaste.stop();
    m_autoSendGiveUp.stop();
    if (was && m_spotlight && m_spotlight->state().voice == studio::cua::SpotlightVoice::Transcribing)
        m_spotlight->setVoice(studio::cua::SpotlightVoice::Idle);
}

void Backend::autoSendNow()
{
    if (!m_awaitingAutoSend)
        return;
    disarmAutoSend();
    submitSpotlight();
}

void Backend::onSpotlightInputEdited()
{
    if (!m_awaitingAutoSend)
        return;
    // Phonon's Cmd+V: wait for the paste to finish (it can arrive as more than one edit).
    m_autoSendNoPaste.stop();
    m_autoSendDebounce.start();
}

void Backend::syncEscapeWatch()
{
    if (!m_spotlight)
        return;
    const bool on = jarvisOverlay() && (m_cuaPending > 0 || m_cuaFollowupScheduled || (m_spotlightTurn && m_spotlightTurnRanCua));
    if (on != m_spotlight->state().escapeWatch)
        m_spotlight->setEscapeWatch(on);
}

void Backend::onCuaActionDone()
{
    if (m_cuaPending > 0)
        return;
    // The queue drained: an interrupt's latch has done its job.
    if (studio::cua::cancelFlag.load())
        releaseCuaCancel();
    // No step left: the cursor pops out (the last step's own 0.45 s fade, never left parked).
    if (m_cuaHud && !m_cuaFollowupScheduled)
        studio::cua::hideCuaReticle(studio::cua::kStepHideSec);
    syncEscapeWatch();
}

void Backend::testBeginCuaAction()
{
    ++m_cuaPending;
    syncEscapeWatch();
}

void Backend::testFinishCuaAction()
{
    m_cuaPending = std::max(0, m_cuaPending - 1);
    onCuaActionDone();
    syncSpotlightTurn();
}

void Backend::summonStudio()
{
    // A turn still streaming keeps filling the (closed) bar's memory; nothing is cancelled.
    spotlight()->summonStudio();
    syncSpotlightOpen();
}

void Backend::onSpotlightDelta()
{
    if (!m_spotlight)
        return;
    // Every tag anywhere is hidden, and one still streaming in ("[[ACTION:cua_cl") never flashes.
    m_spotlight->setAnswer(answerDisplayText(m_spotlightRaw, true));
}

void Backend::onSpotlightFinished(const QString &error)
{
    if (!m_spotlight)
        return;
    if (!error.isEmpty()) {
        m_spotlight->setStatusLine(QStringLiteral("Error \u00B7 ") + oneLine(error, 14));
    } else {
        QString clean = answerDisplayText(m_spotlightRaw);
        // A tag-only reply: the action's own confirmation (what the history recorded).
        if (clean.isEmpty()) {
            const QVector<ChatTurn> h = m_streamHistory.value(QLatin1String(kSpotlightStreamId));
            if (!h.isEmpty() && h.last().role == QLatin1String("assistant") && !m_spotlightRaw.isEmpty())
                clean = h.last().content.trimmed();
        }
        if (!clean.isEmpty())
            m_spotlight->setAnswer(clean);
    }
    syncSpotlightTurn();
}

void Backend::noteSpotlightAction(bool ok, const QString &detail)
{
    m_spotlightLastAction = (ok ? QStringLiteral("Done \u00B7 ") : QStringLiteral("Failed \u00B7 ")) + oneLine(detail, 14);
    if (m_spotlight)
        m_spotlight->setStatusLine(m_spotlightLastAction);
}

void Backend::syncSpotlightTurn()
{
    if (!m_spotlightTurn || !m_spotlight)
        return;
    const bool streaming = m_streamAsks.contains(QLatin1String(kSpotlightStreamId));
    if (streaming || m_cuaPending > 0 || m_cuaFollowupScheduled)
        return;  // still answering, acting, or about to take a follow-up turn
    m_spotlightTurn = false;
    syncEscapeWatch();
    if (m_cuaHud)
        studio::cua::hideCuaReticle(0);  // the turn is over: no cursor left on screen
    if (m_spotlightTurnRanCua) {
        // Acted on the app behind: leave that app frontmost; the exchange waits for the next L2.
        m_spotlight->finishAfterAction(m_spotlightLastAction.isEmpty() ? QStringLiteral("Done") : m_spotlightLastAction);
    } else {
        // Pure Q&A: stay up showing the answer, prompt focused for the next turn.
        m_spotlight->setBusy(false);
    }
    syncSpotlightOpen();
}

QStringList Backend::runStudioActions(const QVector<StudioAction> &actions)
{
    const QStringList known = studioActionNames();
    QStringList replies;
    for (const StudioAction &c : actions) {
        const QString name = c.name.toLower();
        if (!known.contains(name))
            continue;
        const QString r = name == QLatin1String("spawn_agent")
                              ? spawnAgent(m_commandProject.isEmpty() ? m_focusedProject : m_commandProject, c.arg, false)
                              : dispatchStudioAction(name, c.arg);
        if (!r.isEmpty())
            replies << r;
    }
    return replies;
}

void Backend::stageProposal(const QVector<StudioAction> &actions, const QString &summary, const QString &project)
{
    m_pendingProposalActions = actions;  // a newer proposal replaces one still waiting
    m_pendingProposalSummary = summary;
    m_pendingProposalProject = project;
    m_proposalTimer.start();
    if (m_cuaHud)
        studio::cua::showCuaHud(summary + QStringLiteral("  ·  L3+R3 Run"), studio::cua::HudTone::Listening);
    emit pendingProposalChanged();
}

bool Backend::confirmPendingProposal()
{
    if (m_pendingProposalActions.isEmpty())
        return false;
    const QVector<StudioAction> actions = m_pendingProposalActions;
    const QString summary = m_pendingProposalSummary;
    const QString project = m_pendingProposalProject;
    m_pendingProposalActions.clear();
    m_pendingProposalSummary.clear();
    m_pendingProposalProject.clear();
    m_proposalTimer.stop();
    emit pendingProposalChanged();
    if (m_cuaHud)
        studio::cua::showCuaHud(summary + QStringLiteral("…"), studio::cua::HudTone::Working);
    const QString keep = m_commandProject;
    m_commandProject = project;
    const QStringList replies = runStudioActions(actions);
    m_commandProject = keep;
    // cua_* actions drive the HUD themselves and fade it when their queue drains.
    if (m_cuaHud && !m_cuaPending)
        studio::cua::hideCuaHud(studio::cua::kHudLingerSec);
    m_ambientAnswer = replies.isEmpty() ? summary : replies.join(QLatin1Char(' '));
    noteExchange(project, compatModel(), m_ambientHeard, m_ambientAnswer);
    emit ambientChanged();
    return true;
}

void Backend::dismissPendingProposal()
{
    m_proposalTimer.stop();
    if (m_pendingProposalActions.isEmpty())
        return;
    m_pendingProposalActions.clear();
    m_pendingProposalSummary.clear();
    m_pendingProposalProject.clear();
    if (m_cuaHud)
        studio::cua::hideCuaHud(0.0);
    if (m_ambientAnswer.endsWith(QStringLiteral("L3+R3 Run")))
        m_ambientAnswer.clear();
    emit pendingProposalChanged();
    emit ambientChanged();
}

void Backend::noteExchange(const QString &projectKey, const QString &model, const QString &prompt, const QString &answer)
{
    const QString q = prompt.trimmed(), a = answer.trimmed();
    if (q.isEmpty() || a.isEmpty())
        return;
    m_sharedHistory << ChatTurn{QStringLiteral("user"), q} << ChatTurn{QStringLiteral("assistant"), a};
    while (m_sharedHistory.size() > 20)
        m_sharedHistory.remove(0, 2);  // the last 10 exchanges, across every model
    if (m_historyPath.isEmpty())
        return;
    // The real log is the live window's alone; a mock backend writes only a test's temp path.
    if (m_mode != Mode::Live
        && QFileInfo(m_historyPath).absoluteFilePath() == QFileInfo(defaultHistoryPath()).absoluteFilePath())
        return;
    const QFileInfo fi(m_historyPath);
    if (!QFileInfo::exists(fi.absolutePath())) {
        if (!QDir().mkpath(fi.absolutePath()))
            return;
        QFile::setPermissions(fi.absolutePath(), QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner);
    }
    QFile f(m_historyPath);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Append))
        return;
    f.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner);
    const QJsonObject o{{QStringLiteral("ts"), QDateTime::currentDateTimeUtc().toString(Qt::ISODate)},
                        {QStringLiteral("project"), projectKey},
                        {QStringLiteral("model"), model},
                        {QStringLiteral("prompt"), q},
                        {QStringLiteral("answer"), a}};
    f.write(QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n');
}

QString Backend::recentConversationBlock() const
{
    if (m_sharedHistory.isEmpty())
        return {};
    QString out = QStringLiteral("Recent Studio conversation (every model, oldest first):");
    for (const ChatTurn &t : m_sharedHistory) {
        QString c = t.content.simplified();
        if (c.size() > 600)
            c = c.left(600) + QStringLiteral("…");
        out += (t.role == QLatin1String("user") ? QStringLiteral("\nUser: ") : QStringLiteral("\nAssistant: ")) + c;
    }
    return out;
}

void Backend::setUiState(const QVariantMap &state)
{
    m_uiState = state;
}

QString Backend::studioLiveState() const
{
    if (m_uiState.isEmpty())
        return {};
    QStringList lines;
    const QString view = m_uiState.value(QStringLiteral("view")).toString();
    const QString project = m_uiState.value(QStringLiteral("project")).toString();
    QString viewLine = QStringLiteral("View: ") + (view.isEmpty() ? QStringLiteral("Constellation") : view);
    if (!project.isEmpty() && view != QLatin1String("Constellation"))
        viewLine += QStringLiteral(" of ") + project;
    const QString artifact = m_uiState.value(QStringLiteral("artifact")).toString();
    if (!artifact.isEmpty())
        viewLine += QStringLiteral(", showing ") + artifact;
    const QString filter = m_uiState.value(QStringLiteral("filter")).toString();
    if (filter == QLatin1String("needs"))
        viewLine += QStringLiteral(" (Needs Me filter on)");
    lines << viewLine;
    QString hl = m_uiState.value(QStringLiteral("highlighted")).toString();
    if (hl.isEmpty() && m_targets && m_targets->hasActiveTarget())
        hl = m_targets->active().label;
    lines << QStringLiteral("Highlighted: ") + (hl.isEmpty() ? QStringLiteral("nothing") : hl);
    const QStringList overlays = m_uiState.value(QStringLiteral("overlays")).toStringList();
    lines << QStringLiteral("Open overlays: ") + (overlays.isEmpty() ? QStringLiteral("none") : overlays.join(QStringLiteral(", ")));
    if (m_slowmix.currentTrack().isEmpty())
        lines << QStringLiteral("Music: idle");
    else
        lines << QStringLiteral("Music: %1 %2 (%3x, %4 reverb)")
                     .arg(m_slowmix.playing() ? QStringLiteral("playing") : QStringLiteral("paused"), m_slowmix.currentTrack(),
                          QString::number(m_slowmix.speed(), 'g', 3), m_slowmix.reverb());
    lines << QStringLiteral("Cards (%1):").arg(m_projectList.size());
    int n = 0;
    for (const QVariant &v : m_projectList) {
        if (++n > 14) {
            lines << QStringLiteral("- … %1 more").arg(m_projectList.size() - 14);
            break;
        }
        const QVariantMap p = v.toMap();
        QString state = p.value(QStringLiteral("stateLine")).toString();
        if (state.isEmpty())
            state = p.value(QStringLiteral("lastLine")).toString();
        lines << QStringLiteral("- %1 [%2%3]: %4")
                     .arg(p.value(QStringLiteral("name")).toString(), p.value(QStringLiteral("status")).toString(),
                          p.value(QStringLiteral("needsMe")).toBool() ? QStringLiteral(", needs you") : QString(),
                          clampWords(state.simplified(), 16));
    }
    QString out = lines.join(QLatin1Char('\n'));
    if (out.size() > 2600)
        out = out.left(2600) + QStringLiteral("…");
    return out;
}

}  // namespace studio
