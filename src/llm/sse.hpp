// OpenAI-compatible chat-completions SSE line parsing. Reentrant: safe on any thread.
#pragma once

#include <QByteArray>
#include <QString>

namespace studio {

struct OpenAiSseEvent {
    enum Kind { None, Delta, Done, Error };
    Kind kind = None;
    QString thinking;
    QString text;
    QString error;
};

// One "data: {...}" line: content and reasoning_content / reasoning deltas, [DONE], or an error.
OpenAiSseEvent parseOpenAiSseLine(const QByteArray &line);

}  // namespace studio
