#include "backend.hpp"

#include "fixtures.hpp"
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
#include <cmath>
#include <memory>
#include <optional>

namespace studio {

namespace {

QString home() { return QDir::homePath(); }
QString lapisPeers() { return home() + QStringLiteral("/.local/bin/lapis-peers"); }
QString claudeBin() { return home() + QStringLiteral("/.local/bin/claude"); }
QString agyBin() { return home() + QStringLiteral("/.local/bin/agy"); }
QString workspacePath() { return home() + QStringLiteral("/.lapis/runtime/workspace.json"); }

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
    // A tab still named after the folder ("user") says nothing; its Lapis category does.
    if (slug.isEmpty() || slug == QFileInfo(home()).fileName().toLower() || slug == QLatin1String("tmp"))
        slug = category.toLower().split(QRegularExpression(QStringLiteral("[^a-z0-9]+")), Qt::SkipEmptyParts)
                   .mid(0, 3).join(QLatin1Char('-'));
    if (slug.isEmpty())
        slug = QStringLiteral("scratch");
    return ProjectId{QStringLiteral("~") + slug, slug, true};
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

QString statusMetric(const QString &status, qint64 elapsedMs)
{
    const QString word = status == QLatin1String("working") || status == QLatin1String("needs-ok")
        ? QStringLiteral("live")
        : status == QLatin1String("finished") ? QStringLiteral("done") : QStringLiteral("idle");
    return elapsedMs > 0 ? word + QStringLiteral(" · ") + elapsedLabel(elapsedMs) : word;
}

// --- helper command lines -----------------------------------------------------------------

// --- chat model catalog ----------------------------------------------------------------------

const QList<ModelSpec> &modelCatalog()
{
    // Efforts per model: agy's own `agy models` list for Gemini (hydrogen is the only one with
    // xhigh); Claude Code's --effort levels for Claude, minus xhigh on Haiku; codex reasoning
    // efforts for GPT-6, minus xhigh on the small Luna.
    static const QStringList lmh = {QStringLiteral("low"), QStringLiteral("medium"), QStringLiteral("high")};
    static const QStringList all = lmh + QStringList{QStringLiteral("xhigh")};
    // Speed classes are what a quick ask feels like: Fast = first token in ~1-3 s; Deep Think =
    // a heavy thinking model (gemini-4-hydrogen: 31.8 s median at high, 4.3 s cold at low), whose
    // agy stream sends no thinking text, only step events, so the UI shows a live phase instead.
    static const QList<ModelSpec> c = {
        {QStringLiteral("claude-sonnet-5-5"), QStringLiteral("anthropic"), QStringLiteral("Sonnet 5.5"), all, QStringLiteral("Fast")},
        {QStringLiteral("claude-opus-5-5"), QStringLiteral("anthropic"), QStringLiteral("Opus 5.5"), all, QStringLiteral("Frontier")},
        {QStringLiteral("claude-fable-5-2"), QStringLiteral("anthropic"), QStringLiteral("Fable 5.2"), all, QStringLiteral("Frontier")},
        {QStringLiteral("claude-haiku-5-5"), QStringLiteral("anthropic"), QStringLiteral("Haiku 5.5"), lmh, QStringLiteral("Fast")},
        {QStringLiteral("gpt-6-astra"), QStringLiteral("openai"), QStringLiteral("GPT-6 Astra"), all, QStringLiteral("Frontier")},
        {QStringLiteral("gpt-6-sol"), QStringLiteral("openai"), QStringLiteral("GPT-6 Sol"), all, QStringLiteral("Balanced")},
        {QStringLiteral("gpt-6-luna"), QStringLiteral("openai"), QStringLiteral("GPT-6 Luna"), lmh, QStringLiteral("Fast")},
        {QStringLiteral("gemini-3.8-flash"), QStringLiteral("google"), QStringLiteral("Gemini 3.8 Flash"), lmh, QStringLiteral("Fast")},
        {QStringLiteral("gemini-4-carbon"), QStringLiteral("google"), QStringLiteral("Gemini 4 Carbon"), lmh, QStringLiteral("Teach")},
        {QStringLiteral("gemini-4-hydrogen"), QStringLiteral("google"), QStringLiteral("Gemini 4 Hydrogen"), all, QStringLiteral("Deep Think")},
        {QStringLiteral("gemini-4-argon"), QStringLiteral("google"), QStringLiteral("Gemini 4 Argon"), lmh, QStringLiteral("Balanced")},
        {QStringLiteral("qwen3.8-flash-next"), QStringLiteral("openai-compat"), QStringLiteral("Qwen3.8 Flash"), lmh, QStringLiteral("Fast")},
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
    static const QStringList order = {QStringLiteral("low"), QStringLiteral("medium"), QStringLiteral("high"),
                                      QStringLiteral("xhigh")};
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
    if (p == QLatin1String("openai-compat"))
        return QStringLiteral("openai-compat:") + model;
    return {};
}

QByteArray buildOpenAiCompatRequest(const QString &model, const QString &systemPrompt,
                                      const QString &context, const QString &question,
                                      const QString &effort)
{
    // Keep the shared system prompt and pinned repo/artifact context at the front of `messages`
    // so prefix-caching backends (e.g. 1600-token KV blocks) reuse prefilled KV across turns.
    QJsonArray msgs;
    if (!systemPrompt.isEmpty())
        msgs << QJsonObject{{QStringLiteral("role"), QStringLiteral("system")},
                            {QStringLiteral("content"), systemPrompt}};
    if (!context.isEmpty())
        msgs << QJsonObject{{QStringLiteral("role"), QStringLiteral("user")},
                            {QStringLiteral("content"), QStringLiteral("Pinned context:\n") + context}};
    msgs << QJsonObject{{QStringLiteral("role"), QStringLiteral("user")},
                        {QStringLiteral("content"), QStringLiteral("Question: ") + question}};
    QJsonObject body;
    body.insert(QStringLiteral("model"), model);
    body.insert(QStringLiteral("stream"), true);
    body.insert(QStringLiteral("messages"), msgs);
    if (!effort.isEmpty())
        body.insert(QStringLiteral("reasoning_effort"), effort);
    return QJsonDocument(body).toJson(QJsonDocument::Compact);
}

OpenAiSseEvent parseOpenAiSseLine(const QByteArray &raw)
{
    OpenAiSseEvent ev;
    const QByteArray line = raw.trimmed();
    if (!line.startsWith("data:"))
        return ev;
    const QByteArray payload = line.mid(5).trimmed();
    if (payload.isEmpty())
        return ev;
    if (payload == "[DONE]") {
        ev.kind = OpenAiSseEvent::Done;
        return ev;
    }
    const QJsonObject o = QJsonDocument::fromJson(payload).object();
    if (o.isEmpty())
        return ev;
    if (o.contains(QLatin1String("error"))) {
        ev.kind = OpenAiSseEvent::Error;
        const QJsonValue err = o.value(QLatin1String("error"));
        ev.error = err.isObject() ? err.toObject().value(QLatin1String("message")).toString()
                                  : err.toString();
        return ev;
    }
    const QJsonArray choices = o.value(QLatin1String("choices")).toArray();
    if (choices.isEmpty())
        return ev;
    const QJsonObject delta = choices.first().toObject().value(QLatin1String("delta")).toObject();
    ev.thinking = delta.value(QLatin1String("reasoning_content")).toString();
    if (ev.thinking.isEmpty())
        ev.thinking = delta.value(QLatin1String("reasoning")).toString();
    ev.text = delta.value(QLatin1String("content")).toString();
    if (!ev.thinking.isEmpty() || !ev.text.isEmpty())
        ev.kind = OpenAiSseEvent::Delta;
    return ev;
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
    if (m_mode == Mode::Mock) {
        ensureFixtureFiles();
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
    m_boardTimer.start();
    refresh();
    pollBoard();
    m_ageTimer.setInterval(30000);
    connect(&m_ageTimer, &QTimer::timeout, this, &Backend::publish);
    m_ageTimer.start();
    if (warmHelpers) {
        // The default model's daemon is the one the first Enter will use; every other provider
        // channel starts lazily the first time the picker sends to it, then stays up.
        QTimer::singleShot(400, this, &Backend::warmChannel);
        QTimer::singleShot(900, this, &Backend::ensureTraceHelper);  // warm the agy trace helper
    }
}

Backend::~Backend()
{
    if (m_prefsSave.isActive())
        savePrefs();
    if (m_compatReply) {
        m_compatReply->disconnect(this);
        m_compatReply->abort();
        m_compatReply->deleteLater();
        m_compatReply.clear();
    }
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
    m_chatEffort = clampEffort(model, m_chatEffort);
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

void Backend::setPrefsPath(const QString &path)
{
    if (m_prefsSave.isActive()) {
        m_prefsSave.stop();
        savePrefs();  // flush pending changes to the old file first
    }
    m_prefsPath = path;
    m_firstLaunch = !path.isEmpty() && !QFileInfo::exists(path);
    loadPrefs();
    emit prefsChanged();
}

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
    const QString model = o.value(QLatin1String("defaultModel")).toString();
    if (specFor(model))
        m_chatModel = model;
    m_chatEffort = clampEffort(m_chatModel, o.value(QLatin1String("defaultEffort")).toString(m_chatEffort));
    m_modelSuggestions = o.value(QLatin1String("modelSuggestions")).toBool(false);
    m_openaiCompatUrl = o.value(QLatin1String("openaiCompatUrl")).toString().trimmed();
    const QString compatModel = o.value(QLatin1String("openaiCompatModel")).toString().trimmed();
    if (!compatModel.isEmpty())
        m_openaiCompatModel = compatModel;
    m_openaiCompatKey = o.value(QLatin1String("openaiCompatKey")).toString().trimmed();
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

void Backend::setOpenaiCompatUrl(const QString &url)
{
    const QString trimmed = url.trimmed();
    if (trimmed == m_openaiCompatUrl)
        return;
    m_openaiCompatUrl = trimmed;
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

bool Backend::savePrefs()
{
    m_prefsSave.stop();
    if (m_prefsPath.isEmpty())
        return false;
    const QFileInfo fi(m_prefsPath);
    if (!QFileInfo::exists(fi.absolutePath())) {
        if (!QDir().mkpath(fi.absolutePath()))
            return false;
        QFile::setPermissions(fi.absolutePath(),
                              QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner);
    }
    QJsonObject o;
    o.insert(QStringLiteral("uiScale"), m_uiScale);
    o.insert(QStringLiteral("navMode"), m_navMode);
    o.insert(QStringLiteral("defaultModel"), m_chatModel);
    o.insert(QStringLiteral("defaultEffort"), m_chatEffort);
    o.insert(QStringLiteral("modelSuggestions"), m_modelSuggestions);
    if (!m_openaiCompatUrl.isEmpty())
        o.insert(QStringLiteral("openaiCompatUrl"), m_openaiCompatUrl);
    if (!m_openaiCompatModel.isEmpty() && m_openaiCompatModel != QLatin1String("qwen3.8-flash-next"))
        o.insert(QStringLiteral("openaiCompatModel"), m_openaiCompatModel);
    if (!m_openaiCompatKey.isEmpty())
        o.insert(QStringLiteral("openaiCompatKey"), m_openaiCompatKey);
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
    m[QStringLiteral("isActiveInLapis")] = activeInLapis;
    // Never a blank island: with no metric to show (redacted infinity work, prose-only sessions),
    // the island says whether it is live and for how long.
    QString headline = p.ex.headline;
    if (headline.isEmpty() && !p.sessions.isEmpty()) {
        qint64 started = 0;
        for (const SessionInfo &s : p.sessions)
            if (s.started > started)  // the newest tab: how long the current push has run
                started = s.started;
        headline = statusMetric(status, started > 0 ? QDateTime::currentMSecsSinceEpoch() - started : 0);
    }
    m[QStringLiteral("headline")] = headline;
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
    m[QStringLiteral("artifacts")] = p.ex.artifacts;
    m[QStringLiteral("prose")] = p.ex.prose;
    m[QStringLiteral("thinking")] = p.ex.thinking;

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
    std::stable_sort(m_projects.begin(), m_projects.end(), [](const Project &a, const Project &b) {
        if (a.pending != b.pending)
            return a.pending;
        return a.mtime > b.mtime;
    });
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
    for (const Project &p : m_projects) {
        if (p.key != projectKey)
            continue;
        QString ctx = QStringLiteral("Project: %1 (%2).").arg(p.name, tildePath(expandHome(p.dir)));
        const QVariantMap a = p.ex.artifacts.value(artifactIndex).toMap();
        if (!a.isEmpty()) {
            ctx += QStringLiteral(" On screen: %1 \"%2\" (%3).")
                       .arg(a.value(QStringLiteral("kind")).toString(), a.value(QStringLiteral("title")).toString(),
                            a.value(QStringLiteral("metric")).toString());
            const QStringList labels = a.value(QStringLiteral("labels")).toStringList();
            const QStringList vt = a.value(QStringLiteral("valueText")).toStringList();
            for (int i = 0; i < labels.size() && i < 12; ++i)
                ctx += QStringLiteral(" %1=%2").arg(labels[i], vt.value(i));
        }
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
        return ctx;
    }
    return {};
}

void Backend::clearAnswers()
{
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
    m_quickQuestion = q;
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
    const QString prompt = contextFor(projectKey, artifactIndex) + QStringLiteral("\n\nQuestion: ") + q;
    m_quickClock.start();
    m_quickTicker.start();
    if (providerFor(m_chatModel) == QLatin1String("openai-compat")) {
        askOpenAiCompat(projectKey, artifactIndex, q);
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
    if (m_mode != Mode::Live || providerFor(model) == QLatin1String("openai-compat"))
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

void Backend::finishQuick()
{
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
        finishQuick();
        return;
    }
    while (base.endsWith(QLatin1Char('/')))
        base.chop(1);
    const QString endpoint = base.endsWith(QLatin1String("/chat/completions"))
                                 ? base
                                 : base + QStringLiteral("/chat/completions");
    QString model = m_openaiCompatModel;
    if (model.isEmpty())
        model = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_MODEL").trimmed();
    if (model.isEmpty())
        model = m_chatModel;
    QString key = m_openaiCompatKey;
    if (key.isEmpty())
        key = qEnvironmentVariable("LAPIS_OPENAI_COMPAT_KEY").trimmed();

    if (m_compatReply) {
        m_compatReply->disconnect(this);
        m_compatReply->abort();
        m_compatReply->deleteLater();
        m_compatReply.clear();
    }
    if (m_suggestReply) {
        m_suggestReply->disconnect(this);
        m_suggestReply->abort();
        m_suggestReply->deleteLater();
        m_suggestReply.clear();
    }
    m_compatBuf.clear();
    m_quickChannel = channelKeyFor(m_chatModel, m_chatEffort);

    static const QString kSystemPrompt = QStringLiteral(
        "You are Lapis Studio's fast-feedback engineering assistant. "
        "Answer concisely in plain English, grounded in the pinned context.");
    const QString ctx = contextFor(projectKey, artifactIndex);
    const QByteArray payload = buildOpenAiCompatRequest(model, kSystemPrompt, ctx, question, m_chatEffort);

    QNetworkRequest req{QUrl(endpoint)};
    req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    req.setRawHeader("Accept", "text/event-stream");
    if (!key.isEmpty())
        req.setRawHeader("Authorization", QByteArray("Bearer ") + key.toUtf8());

    QNetworkReply *reply = m_nam.post(req, payload);
    m_compatReply = reply;
    connect(reply, &QNetworkReply::readyRead, this, &Backend::onOpenAiCompatReadyRead);
    connect(reply, &QNetworkReply::finished, this, &Backend::onOpenAiCompatFinished);
}

void Backend::onOpenAiCompatReadyRead()
{
    if (!m_compatReply)
        return;
    m_compatBuf += m_compatReply->readAll();
    bool changed = false;
    bool done = false;
    qsizetype nl;
    while ((nl = m_compatBuf.indexOf('\n')) >= 0) {
        const QByteArray line = m_compatBuf.left(nl);
        m_compatBuf.remove(0, nl + 1);
        const OpenAiSseEvent ev = parseOpenAiSseLine(line);
        if (ev.kind == OpenAiSseEvent::Delta) {
            if (!ev.thinking.isEmpty()) {
                noteThinking(ev.thinking);
                changed = true;
            }
            if (!ev.text.isEmpty()) {
                appendAnswer(ev.text);
                changed = true;
            }
        } else if (ev.kind == OpenAiSseEvent::Error) {
            m_quickAnswer = QStringLiteral("openai-compat: ") + oneLine(ev.error, 12);
            done = true;
        } else if (ev.kind == OpenAiSseEvent::Done) {
            done = true;
        }
    }
    if (done)
        finishQuick();
    else if (changed)
        emit quickChanged();
}

void Backend::onOpenAiCompatFinished()
{
    QNetworkReply *reply = m_compatReply.data();
    m_compatReply.clear();
    if (!reply)
        return;
    if (reply->bytesAvailable() > 0) {
        m_compatBuf += reply->readAll();
    }
    if (!m_compatBuf.trimmed().isEmpty()) {
        const QList<QByteArray> rest = m_compatBuf.split('\n');
        m_compatBuf.clear();
        for (const QByteArray &line : rest) {
            const OpenAiSseEvent ev = parseOpenAiSseLine(line);
            if (ev.kind == OpenAiSseEvent::Delta) {
                if (!ev.thinking.isEmpty())
                    noteThinking(ev.thinking);
                if (!ev.text.isEmpty())
                    appendAnswer(ev.text);
            } else if (ev.kind == OpenAiSseEvent::Error) {
                m_quickAnswer = QStringLiteral("openai-compat: ") + oneLine(ev.error, 12);
            }
        }
    }
    if (reply->error() != QNetworkReply::NoError && reply->error() != QNetworkReply::OperationCanceledError
        && m_quickAnswer.isEmpty()) {
        m_quickAnswer = QStringLiteral("openai-compat: ") + oneLine(reply->errorString(), 12);
    }
    reply->deleteLater();
    if (m_quickBusy)
        finishQuick();
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
    m_quickThinking = tailWords(m_quickThinkRaw, 40);
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
    noteAnswer(oneLine(m_quickAnswerRaw, 40));
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
                        noteAnswer(oneLine(resp, 40));
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
                noteAnswer(oneLine(text, 40));
                changed = true;
            }
        } else if (type == QLatin1String("result")) {
            c->inFlight = std::max(0, c->inFlight - 1);
            if (current) {
                const QString r = o.value(QLatin1String("result")).toString();
                if (!r.isEmpty())
                    noteAnswer(oneLine(r, 40));
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
        const QByteArray payload = buildOpenAiCompatRequest(
            model, kSuggestSys, contextFor(projectKey, artifactIndex),
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
    Channel *c = ensureChannel(QStringLiteral("gpt-6-luna"), QStringLiteral("low"));
    if (!c || !c->proc)
        return;
    if (c->suggestReq < 0) {
        c->suggestReq = c->nextId++;
        c->proc->write(codexRpc(c->suggestReq, QStringLiteral("thread/start"),
                                codexThreadStartParams(QStringLiteral(
                                    "Suggest the single most useful next question the user could ask about what is on "
                                    "screen. Reply with only the question, under 9 words."))));
    }
    c->suggestQueue = {codexTurnParams(QString(), contextFor(projectKey, artifactIndex), QStringLiteral("gpt-6-luna"),
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
    m_traceAnswer = oneLine(m_traceRaw, 28);
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
    QDir dev(home() + QStringLiteral("/dev"));
    QFileInfoList dirs = dev.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Time);
    m_folders.clear();
    for (const QFileInfo &fi : dirs) {
        if (m_folders.size() >= 60)
            break;
        m_folders << tildePath(fi.absoluteFilePath());
    }
    m_defaultFolder = QStringLiteral("~/dev");
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
        const ProjectId pid = projectIdFor(s.directory, s.title, s.category);
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
    using Statuses = QList<QPair<QString, QString>>;
    auto *watcher = new QFutureWatcher<std::optional<Statuses>>(this);
    connect(watcher, &QFutureWatcher<std::optional<Statuses>>::finished, this, [this, watcher] {
        QElapsedTimer apply;
        apply.start();
        const double publishBefore = m_stall.publishMs;
        const std::optional<Statuses> r = watcher->result();
        watcher->deleteLater();
        m_boardBusy = false;
        if (!r)
            return;
        bool changed = false;
        for (const auto &[id, st] : *r)
            if (m_statusById.value(id) != st) {
                m_statusById.insert(id, st);
                changed = true;
            }
        if (changed)
            publish();
        const double ms = apply.nsecsElapsed() / 1e6;
        m_stall.boardMs += ms - (m_stall.publishMs - publishBefore);
        m_stall.boards += 1;
        noteStall(ms);
    });
    watcher->setFuture(QtConcurrent::run([program = lapisPeers(), cwd = home()]() -> std::optional<Statuses> {
        QProcess p;
        p.setWorkingDirectory(cwd);
        p.start(program, {QStringLiteral("board"), QStringLiteral("--json"), QStringLiteral("--snippet-chars"),
                          QStringLiteral("0")});
        if (!p.waitForFinished(15000)) {
            p.kill();
            p.waitForFinished(1000);
            return std::nullopt;
        }
        if (p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0)
            return std::nullopt;
        Statuses out;
        const QJsonObject b = QJsonDocument::fromJson(p.readAllStandardOutput()).object();
        for (const QJsonValue &c : b.value(QLatin1String("categories")).toArray())
            for (const QJsonValue &a : c.toObject().value(QLatin1String("agents")).toArray()) {
                const QJsonObject ao = a.toObject();
                out.append({ao.value(QLatin1String("id")).toString(), ao.value(QLatin1String("status")).toString()});
            }
        return out;
    }));
    const double ms = clock.nsecsElapsed() / 1e6;  // dispatch only
    m_stall.boardMs += ms;
    noteStall(ms);
}

}  // namespace studio
