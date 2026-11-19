// What the pointer is on, what was under it a moment ago, and what the user pinned.
//
// Viewers push one semantic target per hovered thing (an island, a table cell, a diff line, a
// chart point, a conversation turn) in window coordinates. The registry keeps:
//   - the active target (the one under the pointer now),
//   - a 5 s hover history, so a voice question that ends after the pointer moved on can still
//     resolve "what is this?" to the target that was hovered when speech started,
//   - pinned targets (deduplicated by kind + session + artifact + path), so one ask can cover
//     several things at once,
// and formats any of them into a Tier 3 context slice. Depends on lapis_core and QtCore/QtGui.
#pragma once

#include "core/types.hpp"

#include <QObject>
#include <QRectF>
#include <QString>
#include <QVariantList>
#include <QVariantMap>
#include <QVector>

#include <functional>

namespace studio {

struct SemanticTarget {
    TargetRef ref;
    QString label;
    QString contextText;
    QRectF sceneRect;  // window coordinates
    bool valid = false;

    // {kind, sessionId, artifactId, path, label, contextText, x, y, w, h, key} ({} when invalid).
    QVariantMap toVariant() const;
    static SemanticTarget fromVariant(const QVariantMap &m);
    // Pin identity: kind|session|artifact|path (TargetRef::key()).
    QString identity() const { return ref.key(); }
};

struct TimedTarget {
    qint64 ms = 0;          // registry clock when this target became active
    SemanticTarget target;  // invalid = nothing hovered from then on
};

// The Tier 3 slice for a set of targets, clamped to maxChars ("" when the list is empty).
QString formatTargetSlice(const QVector<SemanticTarget> &targets, int maxChars = 4000);

class TargetRegistry : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantMap activeTarget READ activeTarget NOTIFY activeTargetChanged)
    Q_PROPERTY(bool hasActiveTarget READ hasActiveTarget NOTIFY activeTargetChanged)
    Q_PROPERTY(QVariantList pinnedTargets READ pinnedTargets NOTIFY pinnedChanged)
    Q_PROPERTY(int pinnedCount READ pinnedCount NOTIFY pinnedChanged)
    // Filled into a hover target whose sessionId / artifactId is empty (StudioApp binds these to
    // the project and artifact on screen, so viewers never need to know them).
    Q_PROPERTY(QString scopeSessionId READ scopeSessionId WRITE setScopeSessionId NOTIFY scopeChanged)
    Q_PROPERTY(QString scopeArtifactId READ scopeArtifactId WRITE setScopeArtifactId NOTIFY scopeChanged)

public:
    static constexpr qint64 kHistoryMs = 5000;
    static constexpr int kHistoryCap = 128;

    explicit TargetRegistry(QObject *parent = nullptr);

    // Tests: replace the monotonic clock (milliseconds).
    void setClock(std::function<qint64()> now) { m_now = std::move(now); }

    QVariantMap activeTarget() const { return m_active.toVariant(); }
    bool hasActiveTarget() const { return m_active.valid; }
    const SemanticTarget &active() const { return m_active; }
    QVariantList pinnedTargets() const;
    int pinnedCount() const { return int(m_pinned.size()); }
    const QVector<SemanticTarget> &pinned() const { return m_pinned; }
    const QVector<TimedTarget> &history() const { return m_history; }

    QString scopeSessionId() const { return m_scopeSession; }
    void setScopeSessionId(const QString &id);
    QString scopeArtifactId() const { return m_scopeArtifact; }
    void setScopeArtifactId(const QString &id);

    Q_INVOKABLE void setHoverTarget(const QString &kind, const QString &sessionId, const QString &artifactId,
                                    const QString &path, const QString &label, const QString &contextText,
                                    qreal x, qreal y, qreal w, qreal h);
    // Clears the active target; with a kind, only when the active target is of that kind.
    Q_INVOKABLE void clearHoverTarget(const QString &kind = {});
    // The target that was active msAgo milliseconds ago ({} when nothing was, or msAgo is
    // outside the 5 s window).
    Q_INVOKABLE QVariantMap targetAtMsAgo(int msAgo) const;

    Q_INVOKABLE void pinActiveTarget();
    Q_INVOKABLE void pinTarget(const QVariantMap &target);
    Q_INVOKABLE void unpinTarget(int index);
    Q_INVOKABLE void clearPinnedTargets();
    Q_INVOKABLE bool isPinned(const QVariantMap &target) const;

    // Pinned targets, or the active one when nothing is pinned, as a Tier 3 slice.
    Q_INVOKABLE QString buildCombinedSlice(int maxChars = 4000) const;
    // The same format for an explicit list of target maps (the omnibar's snapshot).
    Q_INVOKABLE QString buildSliceFor(const QVariantList &targets, int maxChars = 4000) const;

signals:
    void activeTargetChanged();
    void pinnedChanged();
    void scopeChanged();

private:
    qint64 now() const;
    void record(const SemanticTarget &t);
    void prune(qint64 nowMs);
    bool pin(const SemanticTarget &t);

    SemanticTarget m_active;
    QVector<TimedTarget> m_history;  // oldest first
    QVector<SemanticTarget> m_pinned;
    QString m_scopeSession, m_scopeArtifact;
    std::function<qint64()> m_now;
};

}  // namespace studio
