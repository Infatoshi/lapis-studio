#include "llm/sse.hpp"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace studio {

OpenAiSseEvent parseOpenAiSseLine(const QByteArray &raw)
{
    OpenAiSseEvent ev;
    const QByteArray line = raw.trimmed();
    if (!line.startsWith("data:"))
        return ev;
    const QByteArray payload = line.mid(5).trimmed();
    if (payload.isEmpty())
        return ev;
    if (payload == "[DONE]") {
        ev.kind = OpenAiSseEvent::Done;
        return ev;
    }
    const QJsonObject o = QJsonDocument::fromJson(payload).object();
    if (o.isEmpty())
        return ev;
    if (o.contains(QLatin1String("error"))) {
        ev.kind = OpenAiSseEvent::Error;
        const QJsonValue err = o.value(QLatin1String("error"));
        ev.error = err.isObject() ? err.toObject().value(QLatin1String("message")).toString()
                                  : err.toString();
        return ev;
    }
    const QJsonArray choices = o.value(QLatin1String("choices")).toArray();
    if (choices.isEmpty())
        return ev;
    const QJsonObject delta = choices.first().toObject().value(QLatin1String("delta")).toObject();
    ev.thinking = delta.value(QLatin1String("reasoning_content")).toString();
    if (ev.thinking.isEmpty())
        ev.thinking = delta.value(QLatin1String("reasoning")).toString();
    ev.text = delta.value(QLatin1String("content")).toString();
    if (!ev.thinking.isEmpty() || !ev.text.isEmpty())
        ev.kind = OpenAiSseEvent::Delta;
    return ev;
}

}  // namespace studio
