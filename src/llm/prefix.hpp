// APC-aligned prompt builder for OpenAI-compatible servers (vLLM Automatic Prefix Cache).
//
// Messages go out in four tiers, most stable first, so a change in a later tier never
// invalidates the cached KV of an earlier one:
//   Tier 0  static system prompt (kCompatSystemPrompt)
//   Tier 1  canonical workspace / session trunk ("Pinned context:"), stable, append-only
//   Tier 2  prior conversation turns (empty for an ephemeral hover ask)
//   Tier 3  the ephemeral leaf, strictly last: on-screen target slice, recent actions,
//           live Phonon tail, recalled past conversations and the question
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
    QString extraSystemContext;   // Tier 0 tail: optional local studio context (loadLocalStudioContext)
    QString trunk;                // Tier 1
    QVector<ChatTurn> history;    // Tier 2
    // Tier 3 (volatile)
    QString targetSlice;
    QStringList recentActions;
    QString phononTail;
    QString memory;               // cross-session recall ("Relevant past conversations:")
    QString question;
};

// Optional local studio context (default ~/.lapis-next/studio_context.md), capped at 4 KB.
// Returns empty when the file is missing or unreadable. Lives in Tier 0, so it stays APC-hot.
QString loadLocalStudioContext(const QString &path = {});

// The full Tier 0 system text: systemPrompt, then (when model is non-empty) the model identity
// anchor, then extraSystemContext. Deterministic, so warm, ask and ambient share it byte for byte.
QString tier0SystemText(const QString &systemPrompt, const QString &model, const QString &extraSystemContext);

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

// Model output post-processing: [[IGNORE]] and compound [[ACTION:name]] / [[ACTION:name:arg]] tags.
struct StudioAction {
    QString name;
    QString arg;
};

struct ParsedModelOutput {
    QVector<StudioAction> actions;  // every action tag, in order
    QString cleanText;              // raw with all tags stripped and whitespace tidied
    bool ignore = false;            // [[IGNORE]] seen: actions empty, cleanText empty
};

ParsedModelOutput parseStudioModelOutput(const QString &raw);

}  // namespace studio
