#include "llm/prefix.hpp"

#include <QJsonDocument>
#include <QJsonObject>

namespace studio {

const QString kCompatSystemPrompt = QStringLiteral(
    "You are Lapis Studio's fast-feedback engineering assistant. "
    "Answer concisely in plain English, grounded in the pinned context.");

namespace {

QJsonObject message(const QString &role, const QString &content)
{
    return QJsonObject{{QStringLiteral("role"), role}, {QStringLiteral("content"), content}};
}

}  // namespace

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
    QJsonArray msgs = tieredPrefixMessages(p.systemPrompt, p.trunk, p.history);
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
    body.insert(QStringLiteral("messages"), tieredPrefixMessages(p.systemPrompt, p.trunk, p.history));
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

}  // namespace studio
