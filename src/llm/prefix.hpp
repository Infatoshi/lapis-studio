// APC-aligned prompt builder for OpenAI-compatible servers (vLLM Automatic Prefix Cache).
//
// Messages go out in four tiers, most stable first, so a change in a later tier never
// invalidates the cached KV of an earlier one:
//   Tier 0  static system prompt (kCompatSystemPrompt)
//   Tier 1  canonical workspace / session trunk ("Pinned context:"), stable, append-only
//   Tier 2  prior conversation turns (empty for an ephemeral hover ask)
//   Tier 3  the ephemeral leaf, strictly last: on-screen target slice, recent actions,
//           live Phonon tail and the question
// A warm request sends Tiers 0-2 with max_tokens 1; its messages are a byte-identical prefix
// of the ask that follows.
#pragma once

#include "core/types.hpp"

#include <QByteArray>
#include <QJsonArray>
#include <QString>
#include <QStringList>
#include <QVector>

namespace studio {

// The quick-ask system prompt shared by asks and warms (byte-identical prefix).
extern const QString kCompatSystemPrompt;

struct TieredPrompt {
    QString model;
    QString effort;               // reasoning_effort; empty = omit
    QString systemPrompt;         // Tier 0
    QString trunk;                // Tier 1
    QVector<ChatTurn> history;    // Tier 2
    // Tier 3 (volatile)
    QString targetSlice;
    QStringList recentActions;
    QString phononTail;
    QString question;
};

// Tiers 0 + 1 (+ 2 when history is non-empty), in order.
QJsonArray tieredPrefixMessages(const QString &systemPrompt, const QString &trunk,
                                const QVector<ChatTurn> &history = {});
// The Tier 3 leaf text. With only a question it is exactly "Question: <q>".
QString tieredLeafText(const TieredPrompt &p);
// Streamed chat request: Tiers 0-2 then the Tier 3 leaf as the final user message.
QByteArray buildTieredChatRequest(const TieredPrompt &p);
// Prefix warm: Tiers 0-2 only, max_tokens 1, stream false, no reasoning_effort.
QByteArray buildTieredWarmRequest(const TieredPrompt &p);

// Compatibility wrappers (Tier 2 empty, Tier 3 = question).
QByteArray buildOpenAiCompatRequest(const QString &model, const QString &systemPrompt,
                                    const QString &context, const QString &question,
                                    const QString &effort = QString());
QByteArray buildOpenAiWarmRequest(const QString &model, const QString &systemPrompt,
                                  const QString &context);

}  // namespace studio
