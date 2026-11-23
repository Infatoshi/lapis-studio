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
#include <QDateTime>
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
    QString liveState;            // "Studio live state:" (view, cards, highlight, overlays, music)
    QStringList recentActions;
    QString phononTail;
    QString memory;               // cross-session recall ("Relevant past conversations:")
    QString question;
    // Wall-clock line ("Current time: 6:16 pm"), Tier 3 only so Tiers 0/1 stay byte-stable and
    // APC-hot. Empty = omitted (deterministic tests); Backend::tieredPrompt fills it.
    QString currentTime;
    // A live screenshot (JPEG) for the leaf user message; empty on normal text turns. Only an
    // Inkling model's buildTieredChatRequest sends it (an OpenAI image_url content part); it is
    // never part of history, so Tiers 0-2 stay byte-identical and APC-hot.
    QByteArray imageJpeg;
};

// "Current time: 6:16 pm" from the local clock (12-hour, lowercase am/pm).
QString currentTimeLine(const QDateTime &now);

// OpenRouter: the same tiered messages, no chat_template_kwargs / reasoning_effort; instead
// reasoning.effort ("none" for fast / none) and provider routing (Wafer, InferenceNet, Nebius
// by latency, fallbacks allowed).
QByteArray buildOpenRouterChatRequest(const TieredPrompt &p);

// "fast" and "none": the request sends chat_template_kwargs.enable_thinking = false and no
// reasoning_effort (Qwen3.8 no_think). Any other non-empty effort rides as reasoning_effort.
bool isNoThinkEffort(const QString &effort);

// Inkling (any model id containing "inkling", case-insensitive): both builders above switch to
// chat_template_kwargs.reasoning_effort (0.0 fast/none/empty, 0.2 low, 0.5 medium, 0.8 high),
// temperature 0, repeat_penalty 1.0 and bounded max_tokens; OpenRouter pins PrimeIntellect FP4.
bool isInklingModel(const QString &model);
double inklingReasoningEffort(const QString &effort);

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
// Streamed chat request: Tiers 0-2 then the Tier 3 leaf as the final user message. With an
// Inkling model and a non-empty imageJpeg, that message's content is [image_url, text].
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
