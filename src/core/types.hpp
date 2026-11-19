// Plain value types shared by every Lapis Studio layer. Depends on QtCore only.
#pragma once

#include <QRectF>
#include <QString>
#include <QVariantMap>
#include <QVector>

namespace studio {

// What an ask or a warm is about: a session, an artifact inside it, or a file.
struct TargetRef {
    QString kind;        // "session" | "artifact" | "file" | ...
    QString sessionId;
    QString artifactId;
    QString path;
    qint64 rev = 0;      // transcript mtime / artifact revision the reference was taken at

    bool isNull() const { return kind.isEmpty() && sessionId.isEmpty() && artifactId.isEmpty() && path.isEmpty(); }
    // Stable identity across revisions: "kind|session|artifact|path".
    QString key() const;
    // key() plus "@rev": a different revision is a different prefix.
    QString revKey() const;
    friend bool operator==(const TargetRef &a, const TargetRef &b)
    {
        return a.kind == b.kind && a.sessionId == b.sessionId && a.artifactId == b.artifactId && a.path == b.path
            && a.rev == b.rev;
    }
    friend bool operator!=(const TargetRef &a, const TargetRef &b) { return !(a == b); }
};

// A slice of context cut from a target: the prompt text, the structured data it came from,
// where it sits on screen, and its token estimate (-1 = not counted).
struct ContextSlice {
    QString text;
    QVariantMap data;
    QRectF sceneRect;
    int tokens = -1;

    bool isEmpty() const { return text.isEmpty(); }
};

// One prior conversation turn (OpenAI chat roles: "user" | "assistant").
struct ChatTurn {
    QString role;
    QString content;
    friend bool operator==(const ChatTurn &a, const ChatTurn &b) { return a.role == b.role && a.content == b.content; }
};

// Where an ask goes: a one-shot hover ask (no history), a running conversation (history rides
// in Tier 2), or a full agent session.
enum class AskRoute { HoverAsk, Conversation, AgentSession };
QString askRouteName(AskRoute r);

// Rough token estimate for budget decisions (about 4 UTF-16 chars per token).
int estimateTokens(const QString &text);

}  // namespace studio
