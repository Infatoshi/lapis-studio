#include "targets/target_registry.hpp"

#include <QElapsedTimer>
#include <QStringList>

#include <algorithm>

namespace studio {

namespace {

QString clampText(const QString &s, int maxChars)
{
    if (maxChars <= 0)
        return {};
    if (s.size() <= maxChars)
        return s;
    if (maxChars == 1)
        return QStringLiteral("…");
    return s.left(maxChars - 1).trimmed() + QStringLiteral("…");
}

QString indent(const QString &s)
{
    QStringList lines = s.split(QLatin1Char('\n'));
    for (QString &l : lines)
        l.prepend(QStringLiteral("  "));
    return lines.join(QLatin1Char('\n'));
}

}  // namespace

QVariantMap SemanticTarget::toVariant() const
{
    if (!valid)
        return {};
    return {
        {QStringLiteral("kind"), ref.kind},
        {QStringLiteral("sessionId"), ref.sessionId},
        {QStringLiteral("artifactId"), ref.artifactId},
        {QStringLiteral("path"), ref.path},
        {QStringLiteral("label"), label},
        {QStringLiteral("contextText"), contextText},
        {QStringLiteral("x"), sceneRect.x()},
        {QStringLiteral("y"), sceneRect.y()},
        {QStringLiteral("w"), sceneRect.width()},
        {QStringLiteral("h"), sceneRect.height()},
        {QStringLiteral("key"), identity()},
    };
}

SemanticTarget SemanticTarget::fromVariant(const QVariantMap &m)
{
    SemanticTarget t;
    t.ref.kind = m.value(QStringLiteral("kind")).toString();
    t.ref.sessionId = m.value(QStringLiteral("sessionId")).toString();
    t.ref.artifactId = m.value(QStringLiteral("artifactId")).toString();
    t.ref.path = m.value(QStringLiteral("path")).toString();
    t.label = m.value(QStringLiteral("label")).toString();
    t.contextText = m.value(QStringLiteral("contextText")).toString();
    t.sceneRect = QRectF(m.value(QStringLiteral("x")).toReal(), m.value(QStringLiteral("y")).toReal(),
                         m.value(QStringLiteral("w")).toReal(), m.value(QStringLiteral("h")).toReal());
    t.valid = !t.ref.kind.isEmpty();
    return t;
}

QString formatTargetSlice(const QVector<SemanticTarget> &targets, int maxChars)
{
    QVector<SemanticTarget> live;
    for (const SemanticTarget &t : targets)
        if (t.valid)
            live << t;
    if (live.isEmpty() || maxChars <= 0)
        return {};
    const int n = int(live.size());
    QStringList heads;
    int headChars = 0;
    for (int i = 0; i < n; ++i) {
        const SemanticTarget &t = live[i];
        QString h = QStringLiteral("[%1] %2: %3").arg(i + 1).arg(t.ref.kind, t.label.isEmpty() ? t.ref.path : t.label);
        if (!t.ref.path.isEmpty() && !t.label.isEmpty())
            h += QStringLiteral(" (%1)").arg(t.ref.path);
        heads << h;
        headChars += int(h.size()) + 2;
    }
    const QString title = n == 1 ? QStringLiteral("Pointed-at target:") : QStringLiteral("Pointed-at targets (%1):").arg(n);
    // Every target gets an equal share of what the headers leave (2 spaces of indent per line).
    const int share = std::max(0, (maxChars - int(title.size()) - headChars) / n - 4);
    QStringList parts{title};
    for (int i = 0; i < n; ++i) {
        QString block = heads[i];
        const QString ctx = clampText(live[i].contextText.trimmed(), share);
        if (!ctx.isEmpty())
            block += QLatin1Char('\n') + indent(ctx);
        parts << block;
    }
    return clampText(parts.join(QLatin1Char('\n')), maxChars);
}

TargetRegistry::TargetRegistry(QObject *parent) : QObject(parent) {}

qint64 TargetRegistry::now() const
{
    if (m_now)
        return m_now();
    static QElapsedTimer clock = [] {
        QElapsedTimer t;
        t.start();
        return t;
    }();
    return clock.elapsed();
}

QVariantList TargetRegistry::pinnedTargets() const
{
    QVariantList out;
    out.reserve(m_pinned.size());
    for (const SemanticTarget &t : m_pinned)
        out << t.toVariant();
    return out;
}

void TargetRegistry::setScopeSessionId(const QString &id)
{
    if (id == m_scopeSession)
        return;
    m_scopeSession = id;
    emit scopeChanged();
}

void TargetRegistry::setScopeArtifactId(const QString &id)
{
    if (id == m_scopeArtifact)
        return;
    m_scopeArtifact = id;
    emit scopeChanged();
}

void TargetRegistry::prune(qint64 nowMs)
{
    // Drop entries older than the window, but keep the newest of them: it is what was active at
    // the window's start (a cell hovered 8 s ago and still under the pointer).
    const qint64 cutoff = nowMs - kHistoryMs;
    qsizetype firstInWindow = 0;
    while (firstInWindow < m_history.size() && m_history[firstInWindow].ms < cutoff)
        ++firstInWindow;
    if (firstInWindow > 1)
        m_history.remove(0, firstInWindow - 1);
    if (m_history.size() > kHistoryCap)
        m_history.remove(0, m_history.size() - kHistoryCap);
}

void TargetRegistry::record(const SemanticTarget &t)
{
    const qint64 ms = now();
    m_history.append(TimedTarget{ms, t});
    prune(ms);
}

void TargetRegistry::setHoverTarget(const QString &kind, const QString &sessionId, const QString &artifactId,
                                    const QString &path, const QString &label, const QString &contextText,
                                    qreal x, qreal y, qreal w, qreal h)
{
    if (kind.isEmpty())
        return;
    SemanticTarget t;
    t.ref.kind = kind;
    t.ref.sessionId = sessionId.isEmpty() ? m_scopeSession : sessionId;
    // A session target is the whole project: it never inherits the artifact on screen.
    t.ref.artifactId = artifactId.isEmpty() && kind != QLatin1String("session") ? m_scopeArtifact : artifactId;
    t.ref.path = path;
    t.label = label;
    t.contextText = contextText;
    t.sceneRect = QRectF(x, y, w, h);
    t.valid = true;

    const bool sameThing = m_active.valid && m_active.identity() == t.identity();
    if (sameThing && m_active.label == t.label && m_active.contextText == t.contextText
        && m_active.sceneRect == t.sceneRect)
        return;  // pointer moved inside the same target
    m_active = t;
    if (sameThing && !m_history.isEmpty() && m_history.last().target.identity() == t.identity())
        m_history.last().target = t;  // refreshed in place (scroll, live data): same history entry
    else
        record(t);
    emit activeTargetChanged();
}

void TargetRegistry::clearHoverTarget(const QString &kind)
{
    if (!m_active.valid)
        return;
    if (!kind.isEmpty() && m_active.ref.kind != kind)
        return;
    m_active = SemanticTarget{};
    record(m_active);
    emit activeTargetChanged();
}

QVariantMap TargetRegistry::targetAtMsAgo(int msAgo) const
{
    if (msAgo > kHistoryMs)
        return {};
    const qint64 at = now() - std::max(0, msAgo);
    for (qsizetype i = m_history.size() - 1; i >= 0; --i) {
        if (m_history[i].ms <= at)
            return m_history[i].target.toVariant();
    }
    return {};
}

bool TargetRegistry::pin(const SemanticTarget &t)
{
    if (!t.valid)
        return false;
    const QString id = t.identity();
    for (SemanticTarget &p : m_pinned) {
        if (p.identity() == id) {
            p = t;  // same thing pinned again: keep one chip, refresh its text
            return false;
        }
    }
    m_pinned.append(t);
    return true;
}

void TargetRegistry::pinActiveTarget()
{
    if (pin(m_active))
        emit pinnedChanged();
}

void TargetRegistry::pinTarget(const QVariantMap &target)
{
    if (pin(SemanticTarget::fromVariant(target)))
        emit pinnedChanged();
}

void TargetRegistry::unpinTarget(int index)
{
    if (index < 0 || index >= m_pinned.size())
        return;
    m_pinned.remove(index);
    emit pinnedChanged();
}

void TargetRegistry::clearPinnedTargets()
{
    if (m_pinned.isEmpty())
        return;
    m_pinned.clear();
    emit pinnedChanged();
}

bool TargetRegistry::isPinned(const QVariantMap &target) const
{
    const QString id = SemanticTarget::fromVariant(target).identity();
    return std::any_of(m_pinned.cbegin(), m_pinned.cend(), [&](const SemanticTarget &p) { return p.identity() == id; });
}

QString TargetRegistry::buildCombinedSlice(int maxChars) const
{
    if (!m_pinned.isEmpty())
        return formatTargetSlice(m_pinned, maxChars);
    return formatTargetSlice(QVector<SemanticTarget>{m_active}, maxChars);
}

QString TargetRegistry::buildSliceFor(const QVariantList &targets, int maxChars) const
{
    QVector<SemanticTarget> ts;
    for (const QVariant &v : targets)
        ts << SemanticTarget::fromVariant(v.toMap());
    return formatTargetSlice(ts, maxChars);
}

}  // namespace studio
