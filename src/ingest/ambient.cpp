#include "ingest/ambient.hpp"

#include "llm/prefix.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>

#include <algorithm>

namespace studio {

const QStringList kAmbientActionVocabulary = {
    QStringLiteral("focus_project:<name>"), QStringLiteral("open_stage:<name>"), QStringLiteral("open_ladder:<name>"),
    QStringLiteral("go_home"),              QStringLiteral("open_music"),        QStringLiteral("play_track:<song>"),
    QStringLiteral("pause_music"),          QStringLiteral("open_review"),       QStringLiteral("filter_needs_me"),
    QStringLiteral("reset_layout"),         QStringLiteral("spawn_agent:<task>"), QStringLiteral("set_volume:<pct>"),
    QStringLiteral("spotify_play"),         QStringLiteral("spotify_play:<song>"), QStringLiteral("spotify_pause"),
    QStringLiteral("spotify_next"),         QStringLiteral("spotify_prev"),      QStringLiteral("spotify_volume:<pct>"),
};

namespace {

QStringList turnLines(const QString &tail)
{
    QStringList lines;
    for (const QString &l : tail.split(u'\n')) {
        const QString t = l.trimmed();
        if (!t.isEmpty())
            lines << t;
    }
    return lines;
}

}  // namespace

QStringList extractNewUserUtterances(const QString &prevTail, const QString &nextTail)
{
    const QStringList prev = turnLines(prevTail);
    const QStringList next = turnLines(nextTail);

    // Largest k with prev[-k:] == next[:k].
    qsizetype overlap = 0;
    for (qsizetype k = std::min(prev.size(), next.size()); k > 0; --k) {
        bool match = true;
        for (qsizetype i = 0; match && i < k; ++i)
            match = prev[prev.size() - k + i] == next[i];
        if (match) {
            overlap = k;
            break;
        }
    }

    // Optional "m:ss" / "h:mm:ss" timestamp, then the speaker.
    static const QRegularExpression kYou(QStringLiteral(R"(^(?:\d+(?::\d+)+\s+)?You:\s*(.*)$)"));
    QStringList out;
    for (qsizetype i = overlap; i < next.size(); ++i) {
        const QRegularExpressionMatch m = kYou.match(next[i]);
        if (!m.hasMatch())
            continue;
        const QString text = m.captured(1).trimmed();
        if (!text.isEmpty())
            out << text;
    }
    return out;
}

QByteArray buildAmbientIntentRequest(const QString &model, const QString &systemPrompt,
                                     const QString &extraSystemContext, const QString &trunk,
                                     const QString &liveState, const QString &hoverTargetSlice,
                                     const QString &spokenText)
{
    QJsonArray msgs = tieredPrefixMessages(tier0SystemText(systemPrompt, model, extraSystemContext), trunk);

    QString leaf = QStringLiteral(
        "Ambient voice mode. The user said the line below out loud while working; it may be casual banter, "
        "talk with someone else, or noise rather than a request to you.\n"
        "- If it is not addressed to Lapis Studio, reply exactly [[IGNORE]] and nothing else.\n"
        "- If it asks for UI changes or work, emit every needed tag together in this one reply, in order, "
        "using only: ");
    for (qsizetype i = 0; i < kAmbientActionVocabulary.size(); ++i) {
        if (i)
            leaf += QStringLiteral(", ");
        leaf += QStringLiteral("[[ACTION:") + kAmbientActionVocabulary[i] + QStringLiteral("]]");
    }
    leaf += QStringLiteral(
        ".\n- If it asks about what is on screen, answer in 1-2 plain sentences (tags may come first).\n"
        "No preamble, no explanation of the tags.");
    if (!liveState.isEmpty())
        leaf += QStringLiteral("\n\nLive studio state:\n") + liveState;
    if (!hoverTargetSlice.isEmpty())
        leaf += QStringLiteral("\n\nOn screen now:\n") + hoverTargetSlice;
    leaf += QStringLiteral("\n\nSpoken: ") + spokenText;

    msgs << QJsonObject{{QStringLiteral("role"), QStringLiteral("user")}, {QStringLiteral("content"), leaf}};

    QJsonObject body;
    body.insert(QStringLiteral("model"), model);
    body.insert(QStringLiteral("stream"), true);
    body.insert(QStringLiteral("max_tokens"), 160);
    body.insert(QStringLiteral("chat_template_kwargs"), QJsonObject{{QStringLiteral("enable_thinking"), false}});
    body.insert(QStringLiteral("messages"), msgs);
    return QJsonDocument(body).toJson(QJsonDocument::Compact);
}

}  // namespace studio
