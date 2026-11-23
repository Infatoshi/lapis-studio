#include "llm/jarvis_legs.hpp"

#include <QRegularExpression>

namespace studio {

QString jarvisChatUrl(const QString &endpoint)
{
    QString base = endpoint.trimmed();
    while (base.endsWith(QLatin1Char('/')))
        base.chop(1);
    if (base.isEmpty() || base.endsWith(QLatin1String("/chat/completions")))
        return base;
    return base + QStringLiteral("/chat/completions");
}

QVector<JarvisLeg> planJarvisLegs(const JarvisLegConfig &cfg)
{
    QVector<JarvisLeg> legs;
    const QString self = jarvisChatUrl(cfg.jarvisEndpoint);
    if (!self.isEmpty() && !cfg.selfBreakerOpen) {
        const QString model = cfg.jarvisModel.trimmed();
        legs << JarvisLeg{QStringLiteral("self"), self, QString(),
                          model.isEmpty() ? QString::fromLatin1(kJarvisDefaultModel) : model, false, 200, 1800};
    }
    if (!cfg.openrouterKey.isEmpty()) {
        legs << JarvisLeg{QStringLiteral("inkling"), QString::fromLatin1(kOpenRouterChatUrl), cfg.openrouterKey,
                          QString::fromLatin1(kOpenRouterInklingModel), true, 3000, 2500};
        legs << JarvisLeg{QStringLiteral("deepseek"), QString::fromLatin1(kOpenRouterChatUrl), cfg.openrouterKey,
                          QString::fromLatin1(kOpenRouterDeepSeekModel), true, 4000, 4000};
    }
    return legs;
}

bool jarvisLegRetryable(const QString &error)
{
    if (error.isEmpty())
        return false;
    static const QRegularExpression http(QStringLiteral("^HTTP (\\d+)"));
    const auto m = http.match(error);
    if (!m.hasMatch())
        return true;  // curl transport errors, connect / first-byte timeouts, SSE error events
    const int status = m.captured(1).toInt();
    return status == 0 || status >= 500 || status == 408 || status == 429;
}

QByteArray buildJarvisLegRequest(const JarvisLeg &leg, TieredPrompt p)
{
    p.model = leg.model;
    return leg.openRouter ? buildOpenRouterChatRequest(p) : buildTieredChatRequest(p);
}

}  // namespace studio
