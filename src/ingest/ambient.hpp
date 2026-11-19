// Ambient voice intent co-pilot: turns newly spoken "You:" lines from Phonon's live transcript
// into a small no-think intent request that shares the ask path's byte-identical Tier 0 + Tier 1
// prefix (so vLLM's Automatic Prefix Cache serves it warm). The model answers [[IGNORE]] for
// banter or noise, or compound [[ACTION:...]] tags and/or a 1-2 sentence answer; parse the reply
// with parseStudioModelOutput (llm/prefix.hpp).
#pragma once

#include <QByteArray>
#include <QString>
#include <QStringList>

namespace studio {

// The action vocabulary the ambient prompt offers, e.g. "focus_project:<name>", "go_home".
extern const QStringList kAmbientActionVocabulary;

// Diffs two successive Phonon tails (each the last N turn lines, "0:12 You: ..." / "0:15 Call: ...").
// The longest suffix of prevTail that is also a prefix of nextTail is treated as already seen; of
// the remaining lines, only "You:" turns are returned, as their spoken text (timestamp and speaker
// stripped, empty turns skipped). With no overlap every line of nextTail counts as new.
QStringList extractNewUserUtterances(const QString &prevTail, const QString &nextTail);

// Streamed request: Tier 0 (tier0SystemText(systemPrompt, model, extraSystemContext)) and Tier 1
// ("Pinned context:\n" + trunk) byte-identical to buildTieredChatRequest/buildTieredWarmRequest,
// then one leaf with the ambient instructions, live state, hover target slice and spoken text.
// No thinking (chat_template_kwargs.enable_thinking = false), max_tokens 160.
QByteArray buildAmbientIntentRequest(const QString &model, const QString &systemPrompt,
                                     const QString &extraSystemContext, const QString &trunk,
                                     const QString &liveState, const QString &hoverTargetSlice,
                                     const QString &spokenText);

}  // namespace studio
