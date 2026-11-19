// test_targets: lapis_targets (TargetRegistry hover state, 5 s hover history, pins, slices).
// Runs on a fake clock, so the history tests are exact and instant.
#include "targets/target_registry.hpp"

#include <QCoreApplication>

#include <cstdio>

using namespace studio;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const char *name, const QString &detail = QString())
{
    if (ok) {
        ++g_pass;
        return;
    }
    ++g_fail;
    std::printf("FAIL %s%s%s\n", name, detail.isEmpty() ? "" : ": ", qPrintable(detail));
}

void hoverTests()
{
    qint64 t = 1000;
    TargetRegistry r;
    r.setClock([&t] { return t; });
    int changes = 0;
    QObject::connect(&r, &TargetRegistry::activeTargetChanged, [&changes] { ++changes; });

    check(!r.hasActiveTarget() && r.activeTarget().isEmpty(), "hover: starts empty");
    r.setHoverTarget("table_cell", "proj-a", "2", "row=3/col=1", "tok/s: 126", "Table: sweep", 10, 20, 100, 30);
    const QVariantMap a = r.activeTarget();
    check(r.hasActiveTarget() && a.value("kind") == "table_cell" && a.value("sessionId") == "proj-a"
              && a.value("artifactId") == "2" && a.value("path") == "row=3/col=1" && a.value("label") == "tok/s: 126"
              && a.value("x").toReal() == 10 && a.value("h").toReal() == 30 && changes == 1,
          "hover: setHoverTarget fills every field and notifies once");
    r.setHoverTarget("table_cell", "proj-a", "2", "row=3/col=1", "tok/s: 126", "Table: sweep", 10, 20, 100, 30);
    check(changes == 1 && r.history().size() == 1, "hover: same target again is a no-op (no signal, no history)");
    r.setHoverTarget("table_cell", "proj-a", "2", "row=3/col=1", "tok/s: 126", "Table: sweep", 10, 50, 100, 30);
    check(changes == 2 && r.history().size() == 1 && r.activeTarget().value("y").toReal() == 50,
          "hover: same target moved (scroll) updates in place, one history entry");

    r.clearHoverTarget("diff_line");
    check(r.hasActiveTarget(), "hover: clear with another kind keeps the target");
    r.clearHoverTarget("table_cell");
    check(!r.hasActiveTarget() && r.activeTarget().isEmpty() && changes == 3, "hover: clear with its kind clears");
    r.clearHoverTarget();
    check(changes == 3, "hover: clearing nothing is a no-op");

    // Scope fills the session / artifact a viewer leaves empty; a session target never takes the artifact.
    r.setScopeSessionId("proj-b");
    r.setScopeArtifactId("4");
    r.setHoverTarget("diff_line", "", "", "line=12", "fix.diff L12", "ctx", 0, 0, 10, 10);
    check(r.activeTarget().value("sessionId") == "proj-b" && r.activeTarget().value("artifactId") == "4",
          "hover: empty session / artifact come from the scope");
    r.setHoverTarget("session", "proj-c", "", "", "Kernel", "ctx", 0, 0, 10, 10);
    check(r.activeTarget().value("sessionId") == "proj-c" && r.activeTarget().value("artifactId").toString().isEmpty(),
          "hover: a session target never inherits the scoped artifact");
    r.setHoverTarget("", "x", "", "", "", "", 0, 0, 0, 0);
    check(r.activeTarget().value("kind") == "session", "hover: an empty kind is ignored");
}

void historyTests()
{
    qint64 t = 10000;
    TargetRegistry r;
    r.setClock([&t] { return t; });
    check(r.targetAtMsAgo(0).isEmpty(), "history: empty registry resolves nothing");

    r.setHoverTarget("chart_point", "p", "1", "series=tok/s/idx=0", "c1: 126", "", 0, 0, 1, 1);  // t=10000
    t = 11000;
    r.setHoverTarget("chart_point", "p", "1", "series=tok/s/idx=1", "c2: 200", "", 0, 0, 1, 1);  // t=11000
    t = 12500;
    r.clearHoverTarget();                                                                        // t=12500
    t = 13000;
    check(r.targetAtMsAgo(0).isEmpty(), "history: now = nothing hovered after the clear");
    check(r.targetAtMsAgo(1000).value("path") == "series=tok/s/idx=1", "history: 1 s ago = the second point",
          r.targetAtMsAgo(1000).value("path").toString());
    check(r.targetAtMsAgo(2500).value("path") == "series=tok/s/idx=0", "history: 2.5 s ago = the first point",
          r.targetAtMsAgo(2500).value("path").toString());
    check(r.targetAtMsAgo(3500).isEmpty(), "history: before the first hover = nothing");
    check(r.targetAtMsAgo(5001).isEmpty(), "history: beyond 5 s is outside the window");
    check(r.targetAtMsAgo(-50).isEmpty(), "history: negative msAgo reads as now");

    // Pruning: entries older than 5 s go, but the newest of them stays as the window's anchor.
    t = 30000;
    r.setHoverTarget("turn", "p", "", "turn=3", "user turn 3", "hi", 0, 0, 1, 1);  // t=30000
    t = 34000;
    r.setHoverTarget("turn", "p", "", "turn=4", "assistant turn 4", "yo", 0, 0, 1, 1);  // t=34000
    for (const TimedTarget &e : r.history())
        check(e.ms >= 30000 - TargetRegistry::kHistoryMs || &e == &r.history().first(),
              "history: only the anchor may be older than 5 s");
    check(r.history().size() == 3, "history: stale entries pruned, anchor + 2 kept", QString::number(r.history().size()));
    t = 38000;
    check(r.targetAtMsAgo(4900).value("path") == "turn=3", "history: a long hover is still known at the window start",
          r.targetAtMsAgo(4900).value("path").toString());

    // Cap at 128 entries even inside the window.
    for (int i = 0; i < 300; ++i) {
        t += 1;
        r.setHoverTarget("diff_line", "p", "1", QStringLiteral("line=%1").arg(i), "", "", 0, 0, 1, 1);
    }
    check(r.history().size() == TargetRegistry::kHistoryCap, "history: capped at 128 entries",
          QString::number(r.history().size()));
    check(r.targetAtMsAgo(0).value("path") == "line=299", "history: newest entry survives the cap");
}

void pinTests()
{
    TargetRegistry r;
    struct {
        int n = 0;
        int count() const { return n; }
    } spy;
    QObject::connect(&r, &TargetRegistry::pinnedChanged, [&spy] { ++spy.n; });
    r.pinActiveTarget();
    check(r.pinnedCount() == 0 && spy.count() == 0, "pins: pinning with nothing hovered does nothing");

    r.setHoverTarget("table_cell", "p", "2", "row=1/col=1", "tok/s: 126", "Row 1: c1 | 126", 0, 0, 1, 1);
    r.pinActiveTarget();
    r.setHoverTarget("diff_line", "p", "3", "line=7", "fix.diff L7", "+ return x;", 0, 0, 1, 1);
    r.pinActiveTarget();
    check(r.pinnedCount() == 2 && spy.count() == 2, "pins: two different targets pin as two chips");
    r.setHoverTarget("table_cell", "p", "2", "row=1/col=1", "tok/s: 127", "Row 1: c1 | 127", 5, 5, 1, 1);
    r.pinActiveTarget();
    check(r.pinnedCount() == 2 && spy.count() == 2
              && r.pinnedTargets().value(0).toMap().value("label") == "tok/s: 127",
          "pins: re-pinning the same kind+session+artifact+path dedups (and refreshes its text)");
    QVariantMap other = r.activeTarget();
    other["path"] = "row=2/col=1";
    r.pinTarget(other);
    check(r.pinnedCount() == 3 && r.isPinned(other), "pins: pinTarget(map) pins a different path");
    r.pinTarget(QVariantMap{{"label", "no kind"}});
    check(r.pinnedCount() == 3, "pins: a map without a kind is refused");

    r.unpinTarget(1);
    check(r.pinnedCount() == 2 && r.pinnedTargets().value(1).toMap().value("path") == "row=2/col=1",
          "pins: unpin removes exactly that chip");
    r.unpinTarget(9);
    r.unpinTarget(-1);
    check(r.pinnedCount() == 2, "pins: out-of-range unpin is ignored");
    r.clearPinnedTargets();
    check(r.pinnedCount() == 0 && r.pinnedTargets().isEmpty(), "pins: clear empties the list");
}

void sliceTests()
{
    TargetRegistry r;
    check(r.buildCombinedSlice().isEmpty(), "slice: nothing hovered or pinned = empty slice");
    r.setHoverTarget("table_cell", "p", "2", "row=3/col=2", "tok/s: 126", "Table: sweep\nRow 3: c1 | 126", 0, 0, 1, 1);
    const QString one = r.buildCombinedSlice();
    check(one.startsWith("Pointed-at target:\n[1] table_cell: tok/s: 126 (row=3/col=2)\n  Table: sweep\n  Row 3: c1 | 126"),
          "slice: the active target when nothing is pinned", one);

    r.pinActiveTarget();
    r.setHoverTarget("chart_point", "p", "1", "series=tok/s/idx=4", "c8: 410", "Chart: tok/s", 0, 0, 1, 1);
    r.pinActiveTarget();
    r.setHoverTarget("turn", "p", "", "turn=9", "user turn 9", "unrelated", 0, 0, 1, 1);
    const QString two = r.buildCombinedSlice();
    check(two.startsWith("Pointed-at targets (2):\n[1] table_cell") && two.contains("[2] chart_point: c8: 410")
              && !two.contains("unrelated"),
          "slice: pins win over the hovered target, in pin order", two);

    r.clearPinnedTargets();
    r.setHoverTarget("diff_line", "p", "1", "line=1", "big.diff L1", QString(20000, QLatin1Char('x')), 0, 0, 1, 1);
    const QString capped = r.buildCombinedSlice(500);
    check(capped.size() <= 500 && capped.endsWith(QStringLiteral("…")), "slice: clamped to maxChars",
          QString::number(capped.size()));

    QVariantList list{r.activeTarget(), QVariantMap{{"kind", "session"}, {"sessionId", "k"}, {"label", "Kernel"}}};
    const QString explicitSlice = r.buildSliceFor(list, 4000);
    check(explicitSlice.startsWith("Pointed-at targets (2):") && explicitSlice.contains("[2] session: Kernel"),
          "slice: buildSliceFor formats an explicit snapshot", explicitSlice.left(120));
}

}  // namespace

int main(int argc, char **argv)
{
    QCoreApplication app(argc, argv);
    hoverTests();
    historyTests();
    pinTests();
    sliceTests();
    std::printf("test_targets: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
