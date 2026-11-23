// Jarvis (the Spotlight bar / CUA loop) zero-stall failover: an ordered list of endpoint legs,
// tried one after another on the same stream id while no text has reached the user yet.
//   1. the self-hosted jarvisEndpoint (prefs / LAPIS_JARVIS_ENDPOINT), unless its breaker is open
//   2. OpenRouter Inkling 2 Mini pinned to PrimeIntellect FP4
//   3. OpenRouter DeepSeek V4.1 Flash
// Pure data + decisions, so the order, timeouts and retry rule are unit-testable without a network.
#pragma once

#include "llm/prefix.hpp"

#include <QByteArray>
#include <QString>
#include <QVector>

namespace studio {

constexpr char kOpenRouterChatUrl[] = "https://openrouter.ai/api/v1/chat/completions";
constexpr char kJarvisDefaultModel[] = "Infatoshi/Inkling-2-Mini-heretic";
constexpr char kOpenRouterInklingModel[] = "thinkingmachines/inkling-2-mini";
constexpr char kOpenRouterDeepSeekModel[] = "deepseek/deepseek-v4.1-flash-1017";
constexpr int kJarvisBreakerMs = 15000;

struct JarvisLeg {
    QString name;      // "self", "inkling", "deepseek": logged as planner_turns.jsonl "leg"
    QString endpoint;  // full chat/completions URL
    QString apiKey;
    QString model;
    bool openRouter = false;  // true: buildOpenRouterChatRequest, else buildTieredChatRequest
    int connectTimeoutMs = 0;
    int firstByteTimeoutMs = 0;
};

struct JarvisLegConfig {
    QString jarvisEndpoint;  // empty: no leg 1
    QString jarvisModel;     // empty: kJarvisDefaultModel
    QString openrouterKey;   // empty: no legs 2 and 3
    bool selfBreakerOpen = false;
};

// "http://host:8000/v1" or ".../v1/chat/completions" -> the full chat/completions URL.
QString jarvisChatUrl(const QString &endpoint);

// The ordered legs for one turn; empty when nothing is configured.
QVector<JarvisLeg> planJarvisLegs(const JarvisLegConfig &cfg);

// A leg's failure before any text delta moves to the next leg when it is a transport error,
// a timeout (connect or first byte), HTTP 5xx, 408 or 429. Other 4xx and clean finishes do not.
bool jarvisLegRetryable(const QString &error);

// The request body for one leg: the same tiered prompt with the leg's model.
QByteArray buildJarvisLegRequest(const JarvisLeg &leg, TieredPrompt p);

// A simple time-based breaker for leg 1 (now in ms since epoch, or any monotonic clock).
struct JarvisBreaker {
    qint64 openUntilMs = 0;
    bool isOpen(qint64 nowMs) const { return nowMs < openUntilMs; }
    void trip(qint64 nowMs, int forMs = kJarvisBreakerMs) { openUntilMs = nowMs + forMs; }
};

}  // namespace studio
