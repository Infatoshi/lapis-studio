#include "llm/prefix.hpp"

#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>

namespace studio {

const QString kCompatSystemPrompt = QStringLiteral(
    "You are Lapis Studio's fast-feedback engineering assistant. "
    "Answer in plain English, grounded in the pinned context, in 1-3 crisp sentences unless asked for detail. "
    "To control the Studio UI itself, start your answer with [[ACTION:name]] or [[ACTION:name:arg]] "
    "(open_music, close_music, play_track:<song>, pause_music, resume_music, open_review, filter_needs_me, "
    "filter_all, reset_layout, open_settings, open_briefing, set_theme:blue|gold, show_diffs:1|0, "
    "spawn_agent:<task> to start a coding agent on real work).");

namespace {

QJsonObject message(const QString &role, const QString &content)
{
    return QJsonObject{{QStringLiteral("role"), role}, {QStringLiteral("content"), content}};
}

constexpr qint64 kStudioContextCapBytes = 4096;

}  // namespace

QString loadLocalStudioContext(const QString &path)
{
    const QString p = path.isEmpty() ? QDir::homePath() + QStringLiteral("/.lapis-next/studio_context.md") : path;
    QFile f(p);
    if (!f.open(QIODevice::ReadOnly))
        return {};
    QByteArray bytes = f.read(kStudioContextCapBytes);
    // Never split a UTF-8 sequence at the cap: drop a trailing incomplete multi-byte sequence.
    if (bytes.size() == kStudioContextCapBytes && !f.atEnd()) {
        qsizetype lead = bytes.size() - 1;
        while (lead > 0 && (quint8(bytes[lead]) & 0xC0) == 0x80)
            --lead;
        const quint8 b = quint8(bytes[lead]);
        const qsizetype need = b >= 0xF0 ? 4 : b >= 0xE0 ? 3 : b >= 0xC0 ? 2 : 1;
        if (bytes.size() - lead < need)
            bytes.truncate(lead);
    }
    return QString::fromUtf8(bytes).trimmed();
}

QString tier0SystemText(const QString &systemPrompt, const QString &model, const QString &extraSystemContext)
{
    QStringList parts;
    if (!systemPrompt.isEmpty())
        parts << systemPrompt;
    if (!model.isEmpty())
        parts << QStringLiteral("Your model ID is ") + model
                     + QStringLiteral(" (running locally via OpenAI-compatible vLLM). Transcripts in context come from "
                                      "other agent sessions (Claude, Codex, Antigravity); never claim to be Claude or "
                                      "any other model. Answer directly from context whenever sufficient to minimize "
                                      "latency; emit compound [[ACTION:...]] tags together in one turn when multiple "
                                      "UI actions or agent spawns are needed.");
    if (!extraSystemContext.isEmpty())
        parts << QStringLiteral("Local studio context:\n") + extraSystemContext;
    return parts.join(QStringLiteral("\n\n"));
}

QJsonArray tieredPrefixMessages(const QString &systemPrompt, const QString &trunk, const QVector<ChatTurn> &history)
{
    QJsonArray msgs;
    if (!systemPrompt.isEmpty())
        msgs << message(QStringLiteral("system"), systemPrompt);
    if (!trunk.isEmpty())
        msgs << message(QStringLiteral("user"), QStringLiteral("Pinned context:\n") + trunk);
    for (const ChatTurn &t : history) {
        if (t.content.isEmpty())
            continue;
        msgs << message(t.role.isEmpty() ? QStringLiteral("user") : t.role, t.content);
    }
    return msgs;
}

QString tieredLeafText(const TieredPrompt &p)
{
    QString leaf;
    auto section = [&leaf](const QString &s) {
        if (!leaf.isEmpty())
            leaf += QStringLiteral("\n\n");
        leaf += s;
    };
    if (!p.targetSlice.isEmpty())
        section(QStringLiteral("On screen now:\n") + p.targetSlice);
    if (!p.liveState.isEmpty())
        section(QStringLiteral("Studio live state:\n") + p.liveState);
    if (!p.recentActions.isEmpty())
        section(QStringLiteral("Recent user actions:\n- ") + p.recentActions.join(QStringLiteral("\n- ")));
    if (!p.phononTail.isEmpty())
        section(QStringLiteral("Live Phonon audio transcript:\n") + p.phononTail);
    if (!p.memory.isEmpty())
        section(QStringLiteral("Relevant past conversations:\n") + p.memory);
    section(QStringLiteral("Question: ") + p.question);
    return leaf;
}

QByteArray buildTieredChatRequest(const TieredPrompt &p)
{
    QJsonArray msgs = tieredPrefixMessages(tier0SystemText(p.systemPrompt, p.model, p.extraSystemContext), p.trunk,
                                           p.history);
    msgs << message(QStringLiteral("user"), tieredLeafText(p));
    QJsonObject body;
    body.insert(QStringLiteral("model"), p.model);
    body.insert(QStringLiteral("stream"), true);
    body.insert(QStringLiteral("messages"), msgs);
    if (!p.effort.isEmpty())
        body.insert(QStringLiteral("reasoning_effort"), p.effort);
    return QJsonDocument(body).toJson(QJsonDocument::Compact);
}

QByteArray buildTieredWarmRequest(const TieredPrompt &p)
{
    QJsonObject body;
    body.insert(QStringLiteral("model"), p.model);
    body.insert(QStringLiteral("stream"), false);
    body.insert(QStringLiteral("max_tokens"), 1);
    body.insert(QStringLiteral("messages"),
                tieredPrefixMessages(tier0SystemText(p.systemPrompt, p.model, p.extraSystemContext), p.trunk, p.history));
    return QJsonDocument(body).toJson(QJsonDocument::Compact);
}

QByteArray buildOpenAiCompatRequest(const QString &model, const QString &systemPrompt, const QString &context,
                                    const QString &question, const QString &effort)
{
    TieredPrompt p;
    p.model = model;
    p.effort = effort;
    p.systemPrompt = systemPrompt;
    p.trunk = context;
    p.question = question;
    return buildTieredChatRequest(p);
}

QByteArray buildOpenAiWarmRequest(const QString &model, const QString &systemPrompt, const QString &context)
{
    TieredPrompt p;
    p.model = model;
    p.systemPrompt = systemPrompt;
    p.trunk = context;
    return buildTieredWarmRequest(p);
}

ParsedModelOutput parseStudioModelOutput(const QString &raw)
{
    ParsedModelOutput out;
    if (raw.contains(QStringLiteral("[[IGNORE]]"))) {
        out.ignore = true;
        return out;
    }
    // [[ACTION:name]] or [[ACTION:name:arg]]; the arg may itself contain ':' but not ']'.
    static const QRegularExpression kAction(QStringLiteral(R"(\[\[ACTION:([^\]:]+)(?::([^\]]*))?\]\])"));
    QString text;
    qsizetype last = 0;
    auto it = kAction.globalMatch(raw);
    while (it.hasNext()) {
        const QRegularExpressionMatch m = it.next();
        const QString name = m.captured(1).trimmed();
        if (!name.isEmpty())
            out.actions.push_back(StudioAction{name, m.captured(2).trimmed()});
        text += raw.mid(last, m.capturedStart() - last);
        last = m.capturedEnd();
        // "a [[ACTION:x]] b" -> "a b": drop the space before a tag when whitespace or punctuation follows it.
        const QChar next = last < raw.size() ? raw.at(last) : QChar(u'\n');
        if (next.isSpace() || next.isPunct())
            while (text.endsWith(u' ') || text.endsWith(u'\t'))
                text.chop(1);
    }
    text += raw.mid(last);

    // Only tidy what tag removal can leave behind: trailing blanks and runs of empty lines.
    static const QRegularExpression kTrailing(QStringLiteral("[ \\t]+\\n"));
    static const QRegularExpression kBlankRuns(QStringLiteral("\\n{3,}"));
    text.replace(kTrailing, QStringLiteral("\n"));
    text.replace(kBlankRuns, QStringLiteral("\n\n"));
    out.cleanText = text.trimmed();
    return out;
}

}  // namespace studio
