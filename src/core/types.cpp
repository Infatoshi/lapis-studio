#include "core/types.hpp"

namespace studio {

QString TargetRef::key() const
{
    return kind + QLatin1Char('|') + sessionId + QLatin1Char('|') + artifactId + QLatin1Char('|') + path;
}

QString TargetRef::revKey() const
{
    return key() + QLatin1Char('@') + QString::number(rev);
}

QString askRouteName(AskRoute r)
{
    switch (r) {
    case AskRoute::HoverAsk:
        return QStringLiteral("hover");
    case AskRoute::Conversation:
        return QStringLiteral("conversation");
    case AskRoute::AgentSession:
        return QStringLiteral("agent");
    }
    return {};
}

int estimateTokens(const QString &text)
{
    return int((text.size() + 3) / 4);
}

}  // namespace studio
