// --test: unit tests plus offscreen UI tests against the mock backend.
#include "backend.hpp"
#include "demo.hpp"
#include "fixtures.hpp"
#include "gamepad.hpp"
#include "mac/cua.hpp"
#include "mac/cua_spotlight.hpp"
#include "pdfrender.hpp"
#include "qml_host.hpp"
#include "transcript.hpp"

#include <QDir>
#include <QFileInfo>
#include <QFontInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>
#include <QDateTime>
#include <QElapsedTimer>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QThread>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QPointer>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlComponent>
#include <QQmlEngine>
#include <QQuickItem>
#include <QQuickWindow>

#include <algorithm>
#include <csignal>
#include <cstdio>
#include <limits>

namespace studio {

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool ok, const char *name, const QString &detail = QString())
{
    if (ok) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("FAIL  %s  %s\n", name, qPrintable(detail));
    }
}

bool noFiles(const QString &) { return false; }

// Defined with the omnibar helpers below; the Stage tests click the Code pill.
void mouseClick(QQuickWindow *w, QPointF p, Qt::MouseButton b, Qt::KeyboardModifiers mods);

QVariantMap firstOfKind(const QVariantList &arts, const QString &kind)
{
    for (const QVariant &v : arts)
        if (v.toMap().value(QStringLiteral("kind")).toString() == kind)
            return v.toMap();
    return {};
}

void unitTests()
{
    // Claude Code JSONL.
    {
        const QList<Turn> turns = parseTranscript(claudeSampleJsonl(), Harness::Claude);
        check(turns.size() == 2, "claude: 2 turns (meta, tool results, reminders skipped)", QString::number(turns.size()));
        check(turns.value(0).role == "user" && turns.value(0).text.contains("DCP=4"), "claude: user prompt");
        check(turns.value(1).thinking == "spills dominate", "claude: thinking block", turns.value(1).thinking);
        check(turns.value(1).patches.size() == 1 && turns.value(1).patches.value(0).contains("+float acc[128];"),
              "claude: Edit tool becomes a diff");
        const Extracted ex = extractArtifacts(turns, noFiles);
        check(ex.headline == "29.6 tok/s · 2.63x", "claude: headline metric", ex.headline);
        const QVariantMap chart = firstOfKind(ex.artifacts, "chart");
        check(chart.value("labels").toStringList() == QStringList({"base", "128-reg"}), "claude: chart labels");
        check(chart.value("best").toInt() == 1, "claude: chart winner");
        check(!firstOfKind(ex.artifacts, "diff").isEmpty(), "claude: diff artifact");
        check(ex.assumptions.contains("DCP=4"), "claude: assumption DCP=4");
        check(ex.goal == "profile fp4 decode with DCP=4", "claude: goal", ex.goal);
        check(ex.winner == "128-reg · 29.6 tok/s", "claude: winner", ex.winner);
    }
    // Codex JSONL.
    {
        const QList<Turn> turns = parseTranscript(codexSampleJsonl(), Harness::Codex);
        check(turns.size() == 2, "codex: 2 turns (environment context skipped)", QString::number(turns.size()));
        check(turns.value(0).text == "tune the decode tile", "codex: user text");
        check(turns.value(1).thinking.contains("compare tiles"), "codex: reasoning summary");
        check(turns.value(1).patches.value(0).contains("+++ b/src/tile.rs"), "codex: apply_patch diff");
        const Extracted ex = extractArtifacts(turns, noFiles);
        const QVariantMap table = firstOfKind(ex.artifacts, "table");
        check(table.value("header").toStringList() == QStringList({"tile", "ms"}), "codex: csv table");
        check(firstOfKind(ex.artifacts, "image").isEmpty(), "codex: missing image path ignored");
        check(!firstOfKind(ex.artifacts, "diff").isEmpty(), "codex: diff artifact");
    }
    // Antigravity JSONL.
    {
        const QList<Turn> turns = parseTranscript(agySampleJsonl(), Harness::Agy);
        check(turns.size() == 2, "agy: 2 turns", QString::number(turns.size()));
        check(turns.value(0).text == "why did 128-reg win?", "agy: USER_REQUEST unwrapped", turns.value(0).text);
        check(turns.value(1).text.startsWith("Registers hold"), "agy: planner response");
        check(extractArtifacts(turns, noFiles).headline == "1.8x", "agy: speedup headline");
    }
    // Cut first line (tail read) must not break parsing.
    {
        QByteArray raw = claudeSampleJsonl();
        raw = raw.mid(17);
        check(parseTranscript(raw, Harness::Claude).size() == 2, "parse: survives a cut first line");
    }
    // Helpers.
    check(clampWords("one two three four five six", 5) == "one two three four five", "clampWords");
    check(shortModel("claude-opus-5-5") == "opus-5.5", "shortModel opus", shortModel("claude-opus-5-5"));
    check(shortModel("claude-fable-5-2[1m]") == "fable-5.2", "shortModel fable");
    check(shortModel("gemini-4-carbon-medium") == "carbon-medium", "shortModel carbon");
    check(highlightCode("if (a < 2) return;").contains("&lt;"), "highlight escapes");
    check(countWords("<font color=\"#fff\">two&nbsp;words</font> → 2.63x") == 3, "countWords strips markup",
          QString::number(countWords("<font color=\"#fff\">two&nbsp;words</font> → 2.63x")));
    {
        const QVariantList lines = diffLines("--- a/x.cu\n+++ b/x.cu\n@@ -10,2 +10,2 @@\n ctx\n-old\n+new\n");
        QString kinds;
        for (const QVariant &l : lines)
            kinds += l.toMap().value("k").toString();
        check(kinds == "f@ -+", "diffLines kinds", kinds);
        check(lines.value(4).toMap().value("no").toString() == "11", "diffLines numbering");
    }
    {
        QVariantMap t = tableFromMarkdown({"| build | latency ms |", "|---|---|", "| a | 142 |", "| b | 38 |"});
        const QVariantMap c = chartFromTable(t);
        check(c.value("best").toInt() == 1 && c.value("lowerBetter").toBool(), "chart: latency lower is better");
        check(c.value("metric").toString() == "38 ms", "chart: metric with unit", c.value("metric").toString());
    }
    check(Backend::isRedactedCategory("infinity") && !Backend::isRedactedCategory("quick Qs"),
          "infinity sessions are redacted");

    // $HOME / /tmp sessions: the tab title names the island, never one "home" bucket.
    {
        const QString h = QDir::homePath();
        check(isScratchDir(h) && isScratchDir("~") && isScratchDir("/tmp") && isScratchDir("/private/tmp")
                  && !isScratchDir(h + "/dev/lapis"),
              "grouping: $HOME and /tmp are scratch dirs");
        struct Case { const char *title; const char *slug; };
        const Case cases[] = {
            {"Kneron H3 blind review", "kneron-h3"},
            {"Photos to send Alex", "alex-photos"},
            {"Anthropic Interview Practice", "anthropic-interview"},
            {"GPT 6.2 Astra KernelBench run", "gpt-6.2-astra"},
            {"netherite-v2", "netherite-v2"},
            {"Roblox game development setup", "roblox-game"},
            {"Roblox game on gpu-node", "roblox-game"},
            {"phonon", "phonon"},
        };
        for (const Case &c : cases)
            check(slugFromTitle(c.title) == c.slug, "grouping: title slug",
                  QStringLiteral("'%1' -> '%2' (want '%3')").arg(c.title, slugFromTitle(c.title), c.slug));
        const ProjectId kneron = projectIdFor(h, "Kneron H3 blind review", "apps");
        check(kneron.fromTitle && kneron.name == "kneron-h3" && kneron.key == "~kneron-h3", "grouping: $HOME key from title",
              kneron.key);
        const ProjectId tmp = projectIdFor("/tmp", "Photos to send Alex", "quick Qs");
        check(tmp.name == "alex-photos", "grouping: /tmp key from title", tmp.name);
        const ProjectId bare = projectIdFor(h, QFileInfo(h).fileName(), "quick Qs");
        check(bare.name == "quick-qs", "grouping: a tab named after $HOME falls back to its category", bare.name);
        const ProjectId lapis = projectIdFor(h + "/dev/lapis", "I don't understand this", "apps");
        check(!lapis.fromTitle && lapis.key == "lapis", "grouping: real folders still group by basename", lapis.key);
        check(projectIdFor("~/dev/infinity/autoresearch-competition", "Kimi K3", "infinity").key == "autoresearch-competition",
              "grouping: nested folder basename");
        // The nine $HOME tabs from the first-launch audit no longer share an island.
        const QStringList homeTitles = {
            QStringLiteral("Roblox game development setup"), QStringLiteral("netherite-v2"),
            QStringLiteral("Roblox game on gpu-node"), QStringLiteral("Anthropic Interview Practice"),
            QFileInfo(h).fileName(), QStringLiteral("GPT 6.2 Astra KernelBench run"),
            QStringLiteral("Infinity"), QStringLiteral("phonon"), QStringLiteral("Kneron H3 blind review")};
        QSet<QString> keys;
        for (const QString &t : homeTitles) {
            const QString k = projectIdFor(h, t, "quick Qs").key;
            check(k != "home" && !k.isEmpty() && k != "~", "grouping: no home mega-bucket", k);
            keys.insert(k);
        }
        check(keys.size() == 8, "grouping: 9 $HOME tabs -> 8 islands (two Roblox tabs share one)",
              QStringList(keys.values()).join(','));

        // Live sessions: $HOME tabs still titled after the folder are separate islands, not one
        // "quick-qs" bucket (the --probe-live 14e5ecb8 / de6cd690 / 933d3ac5 case).
        auto session = [&](const QString &id, const QString &title, const QString &dir, const QString &cat) {
            SessionInfo s;
            s.id = id;
            s.title = title;
            s.directory = dir;
            s.category = cat;
            s.harness = "agy";
            return s;
        };
        const QString homeName = QFileInfo(h).fileName();
        check(isGenericTabTitle(homeName) && isGenericTabTitle("") && isGenericTabTitle("tmp")
                  && !isGenericTabTitle("Kneron H3 blind review"),
              "grouping: generic tab titles");
        check(promptTitle("can you fix the fn key toggle latency?") == "Fix fn key toggle latency",
              "grouping: prompt title from the first request", promptTitle("can you fix the fn key toggle latency?"));
        check(promptTitle("Okay, you are replacing the Phonon session.").isEmpty()
                  && promptTitle("`https://x.com/a/status/1` How did they do it?").isEmpty(),
              "grouping: meta requests and questions name no tab");
        Extracted e1;
        e1.promptName = promptTitle("please sweep tile sizes for the fp4 gemm on sm120");
        Extracted e2;
        e2.jobName = "Benchmarking Qwen 3.8 decode";
        const ProjectId t1 = sessionProjectId(session("14e5ecb8-aaaa-bbbb-cccc-000000000001", homeName, h, "quick Qs"), e1);
        const ProjectId t2 = sessionProjectId(session("de6cd690-aaaa-bbbb-cccc-000000000002", homeName, h, "quick Qs"), e2);
        const ProjectId t3 = sessionProjectId(session("933d3ac5-aaaa-bbbb-cccc-000000000003", homeName, h, "quick Qs"), Extracted());
        const ProjectId t4 = sessionProjectId(session("4d4cfbff-aaaa-bbbb-cccc-000000000004", homeName, h, "quick Qs"), Extracted(),
                                              "Deleted 86G of reproducible training checkpoints on the box.");
        check(t1.key == "~tab:14e5ecb8" && t2.key == "~tab:de6cd690" && t3.key == "~tab:933d3ac5" && t1.fromTitle,
              "grouping: each generic $HOME tab keys by its short id", t1.key + " " + t2.key + " " + t3.key);
        check(t1.name == "Sweep tile sizes for FP4 GEMM" && t2.name == "Benchmarking Qwen 3.8 decode" && t3.name == "tab 933d3ac5"
                  && t4.name.startsWith("Deleted 86G"),
              "grouping: tab islands named from the first request, else job name, else board snippet",
              t1.name + " | " + t2.name + " | " + t3.name + " | " + t4.name);
        check(sessionProjectId(session("aaaaaaaa-1", "Kneron H3 blind review", h, "apps"), e1).key == "~kneron-h3"
                  && sessionProjectId(session("bbbbbbbb-1", homeName, h + "/dev/lapis", "apps"), e1).key == "lapis",
              "grouping: titled tabs and real folders keep their islands");

        // Through the live scan path: three generic tabs + a titled one + a remote session with no
        // local transcript. Categories stay; the remote island takes the board snippet.
        QTemporaryDir td;
        QHash<QString, ParsedCache> cache;
        QList<SessionInfo> sessions;
        const QStringList ids = {"14e5ecb8-0000-0000-0000-000000000001", "de6cd690-0000-0000-0000-000000000002",
                                 "933d3ac5-0000-0000-0000-000000000003"};
        const QStringList asks = {"fix the fn key toggle latency", "sweep tile sizes for the fp4 gemm",
                                  "write the release notes for lapis studio"};
        for (int i = 0; i < ids.size(); ++i) {
            SessionInfo s = session(ids[i], homeName, h, "quick Qs");
            s.transcript = td.filePath(QStringLiteral("t%1.jsonl").arg(i));
            QFile f(s.transcript);
            if (f.open(QIODevice::WriteOnly))
                f.close();
            Turn u;
            u.role = "user";
            u.text = asks[i];
            Turn a;
            a.role = "assistant";
            a.text = QStringLiteral("Done with step %1.").arg(i);
            ParsedCache c;
            c.ex = extractArtifacts({u, a});
            cache.insert(s.transcript, c);
            sessions << s;
        }
        sessions << session("1a0d1a39-0000-0000-0000-000000000009", "netherite-v2", "~/dev/netherite-v2", "minecraft");
        Backend b(Backend::Mode::Mock);
        b.testApplyScan(sessions, cache);
        QHash<QString, QVariantMap> byKey;
        for (const QVariant &v : b.projects())
            byKey.insert(v.toMap().value("key").toString(), v.toMap());
        check(byKey.size() == 4 && byKey.contains("~tab:14e5ecb8") && byKey.contains("~tab:de6cd690")
                  && byKey.contains("~tab:933d3ac5") && !byKey.contains("~quick-qs"),
              "grouping: three generic $HOME tabs -> three islands, no quick-qs bucket", QStringList(byKey.keys()).join(','));
        check(byKey.value("~tab:14e5ecb8").value("name").toString() == "Fix fn key toggle latency"
                  && byKey.value("~tab:933d3ac5").value("name").toString() == "Write release notes for lapis studio",
              "grouping: tab islands named from their first request",
              byKey.value("~tab:14e5ecb8").value("name").toString() + " | " + byKey.value("~tab:933d3ac5").value("name").toString());
        check(byKey.value("~tab:de6cd690").value("category").toString() == "quick Qs"
                  && byKey.value("netherite-v2").value("category").toString() == "minecraft",
              "grouping: categories stay on the project");
        check(byKey.value("~tab:14e5ecb8").value("turns").toList().size() == 2,
              "turns: projectMap exposes the conversation", QString::number(byKey.value("~tab:14e5ecb8").value("turns").toList().size()));
        const QVariantMap before = byKey.value("netherite-v2");
        check(before.value("lastLine").toString().isEmpty() && before.value("turns").toList().isEmpty(),
              "board: a remote session starts with no local text");
        const QByteArray board = R"({"categories":[{"name":"minecraft","agents":[
            {"id":"1a0d1a39-0000-0000-0000-000000000009","status":"working","infinityRedacted":false,
             "lastReplySnippet":"lane/wood didn't land: its branch conflicts with lane/choice. I'm merging master into it now."}]},
            {"name":"infinity","agents":[{"id":"ba7843e9-0000","status":"working","infinityRedacted":true,"lastReplySnippet":"secret"}]}]})";
        const QList<BoardRow> rows = parseBoardJson(board);
        check(rows.size() == 2 && rows[0].status == "working" && rows[0].snippet.startsWith("lane/wood") && rows[1].snippet.isEmpty(),
              "board: parse status + lastReplySnippet, withheld snippets dropped");
        b.applyBoard(rows);
        QVariantMap after;
        for (const QVariant &v : b.projects())
            if (v.toMap().value("key").toString() == "netherite-v2")
                after = v.toMap();
        const QVariantList rt = after.value("turns").toList();
        check(after.value("lastLine").toString().startsWith("lane/wood didn't land") && after.value("status").toString() == "working",
              "board: the snippet fills lastLine when the local transcript is empty", after.value("lastLine").toString());
        check(rt.size() == 1 && rt[0].toMap().value("text").toString().startsWith("lane/wood")
                  && rt[0].toMap().value("source").toString() == "board",
              "board: the snippet is the remote session's conversation");
        // A session with its own transcript keeps its own lines.
        b.applyBoard({{ids[0], "idle", "Something the board says"}});
        for (const QVariant &v : b.projects())
            if (v.toMap().value("key").toString() == "~tab:14e5ecb8")
                check(v.toMap().value("turns").toList().size() == 2 && !v.toMap().value("lastLine").toString().contains("board says"),
                      "board: a local transcript wins over the snippet");
    }
    // Never a blank island.
    check(elapsedLabel(30 * 1000) == "just now" && elapsedLabel(42 * 60000) == "42m"
              && elapsedLabel((3 * 60 + 12) * 60000LL) == "3h 12m" && elapsedLabel(2 * 3600000LL) == "2h"
              && elapsedLabel((5 * 24 + 3) * 3600000LL) == "5d 3h",
          "status: elapsed labels");
    // Island state: plain-English lines, never an uptime or a metric.
    {
        Extracted ex;
        ex.jobName = "Sweeping sm120 FP4 tile sizes";
        ex.headline = "29.6 tok/s · 2.63x";
        ex.prose = "**128-reg** won the decode sweep at 29.6 tok/s. Spills dropped to zero.\n\nTLDR: done";
        ex.recentTurns = {"User: sweep the tiles", "Assistant: 128-reg won the decode sweep at 29.6 tok/s."};
        const IslandState w = islandState(ex, "working");
        check(w.now == "Sweeping sm120 FP4 tile sizes" && w.last == "128-reg won the decode sweep at 29.6 tok/s" && !w.blocker,
              "island: job line + latest completed step", w.now + " / " + w.last);
        check(!w.now.contains("live ·") && !w.last.contains("3h"), "island: no uptime in the state lines");
        ex.prose = "Both arms are ready. Should I run the 65B sweep overnight?";
        ex.recentTurns.last() = "Assistant: Both arms are ready. Should I run the 65B sweep overnight?";
        const IslandState q = islandState(ex, "working");
        check(q.blocker && q.last == "Should I run the 65B sweep overnight?", "island: a trailing agent question is a gold blocker", q.last);
        ex.recentTurns << "User: yes";
        check(!islandState(ex, "working").blocker, "island: an answered question is not a blocker");
        ex.prose.clear();
        ex.recentTurns.clear();
        const IslandState n = islandState(ex, "needs-ok");
        check(n.blocker && n.last == "Waiting on your OK to continue", "island: needs-ok without a question still blocks", n.last);
        Extracted bare;
        bare.goal = "Profile fp4 decode with DCP=4. Then write it up.";
        bare.toolCalls = {"Bash: ./bench.sh --dcp 4"};
        const IslandState b = islandState(bare, "idle");
        check(b.now == "Profile fp4 decode with DCP=4" && b.last == "Last step: Bash: ./bench.sh --dcp 4",
              "island: goal line + last tool receipt when no write-up", b.now + " / " + b.last);
    }
    // Morning Briefing, local fallback + model merge.
    {
        auto proj = [](const QString &key, const QString &status, bool active, int fresh, bool blocker, const QString &last) {
            return QVariantMap{{"key", key}, {"name", key}, {"status", status}, {"active", active}, {"fresh", fresh},
                               {"blocker", blocker}, {"nowLine", "Tuning " + key}, {"lastLine", last},
                               {"headline", key == "a" ? "2.63x" : ""},
                               {"artifacts", key == "b" ? QVariantList{QVariantMap{{"title", "Spill Report"}}} : QVariantList{}}};
        };
        const QVariantList ps{proj("idle", "idle", false, 0, false, "Nothing new"),
                              proj("a", "working", true, 0, false, "128-reg won the sweep"),
                              proj("b", "idle", false, 3, false, "Spills dropped to zero"),
                              proj("c", "needs-ok", true, 0, true, "Should I ship it?")};
        QVariantList rows = localBriefing(ps);
        QStringList order;
        for (const QVariant &v : rows)
            order << v.toMap().value("key").toString();
        check(order == QStringList({"c", "b", "a"}), "briefing: only changed / active projects, blockers first", order.join(','));
        const QVariantMap c = rows.value(0).toMap(), b = rows.value(1).toMap(), a = rows.value(2).toMap();
        check(c.value("line2").toString() == "Decide: Should I ship it?" && c.value("blocker").toBool() && c.value("index").toInt() == 3,
              "briefing: a blocker's next action is the decision", c.value("line2").toString());
        check(b.value("line1").toString() == "Spills dropped to zero · Spill Report · 3 new turns"
                  && b.value("line2").toString() == "Review the spill report",
              "briefing: line 1 carries proof, line 2 one action", b.value("line1").toString() + " / " + b.value("line2").toString());
        check(a.value("line1").toString() == "128-reg won the sweep · 2.63x" && a.value("line2").toString() == "Check in on tuning a",
              "briefing: live project with its metric as proof", a.value("line1").toString() + " / " + a.value("line2").toString());
        const int n = mergeBriefingReply(rows, "- a | 128-reg hit 29.6 tok/s, 2.63x over base | Approve the 128-reg config\nnonsense line\nzz | x | y");
        check(n == 1 && rows.value(2).toMap().value("line2").toString() == "Approve the 128-reg config"
                  && rows.value(0).toMap().value("line1").toString() == c.value("line1").toString(),
              "briefing: model lines merge by key, the rest keep local lines", QString::number(n));
    }

    // Stripped helper command lines.
    {
        const QString sid = "0b0e5c1a-7d3e-4c1a-9f00-1234567890ab";
        const QStringList want = {"-p", "--setting-sources", "", "--strict-mcp-config", "--session-id", sid,
                                  "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
                                  "--include-partial-messages", "--model", "claude-sonnet-5-5", "--effort", "low",
                                  "--tools", "", "--no-session-persistence", "--append-system-prompt"};
        const QStringList got = quickHelperArgs(sid);
        check(got.mid(0, want.size()) == want && got.size() == want.size() + 1, "cli: claude helper args exact",
              got.join(' '));
        check(got.value(2).isEmpty() && got.value(17).isEmpty(), "cli: --setting-sources and --tools are empty strings");
        const QJsonObject hook = QJsonDocument::fromJson(cswapHookPayload(sid)).object();
        check(hook.value("hook_event_name").toString() == "SessionStart" && hook.value("session_id").toString() == sid
                  && hook.value("cwd").toString() == "/tmp" && hook.value("source").toString() == "startup"
                  && hook.size() == 4,
              "cli: cswap SessionStart payload", QString::fromUtf8(cswapHookPayload(sid)));
        const QJsonObject cl = QJsonDocument::fromJson(claudeUserLine("hi")).object();
        check(cl.value("type").toString() == "user"
                  && cl.value("message").toObject().value("content").toArray().at(0).toObject().value("text").toString() == "hi"
                  && claudeUserLine("hi").endsWith('\n'),
              "cli: claude stream-json user line");
        const QStringList agy = {"--input-format", "stream-json", "--output-format", "stream-json", "--model",
                                 "gemini-3.8-flash-low", "--sandbox", "--disable-slash-commands", "-p="};
        check(traceHelperArgs() == agy, "cli: warm agy args exact", traceHelperArgs().join(' '));
        const QJsonObject al = QJsonDocument::fromJson(agyUserLine("why?")).object();
        check(al.value("event").toString() == "user"
                  && al.value("message").toObject().value("content").toArray().at(0).toObject().value("text").toString() == "why?",
              "cli: agy stream-json user line uses \"event\"", QString::fromUtf8(agyUserLine("why?")));
        check(codexAppServerArgs().value(0) == "app-server" && codexAppServerArgs().contains("model=\"gpt-6-astra\""),
              "cli: codex app-server args", codexAppServerArgs().join(' '));
        const QJsonObject rpc = QJsonDocument::fromJson(codexRpc(3, "turn/start", {{"threadId", "t1"}})).object();
        check(rpc.value("id").toInt() == 3 && rpc.value("method").toString() == "turn/start"
                  && rpc.value("params").toObject().value("threadId").toString() == "t1",
              "cli: codex JSON-RPC request line");
        check(!QJsonDocument::fromJson(codexRpc(-1, "initialized", {})).object().contains("id"),
              "cli: codex notification has no id");
    }

    // Fixture pipeline: the demo session yields chart, diff, table, pdf in that order.
    ensureFixtureFiles();
    {
        const Extracted ex = extractArtifacts(kickoffTurns());
        QStringList kinds;
        for (const QVariant &a : ex.artifacts)
            kinds << a.toMap().value("kind").toString();
        check(kinds == QStringList({"chart", "diff", "table", "pdf"}), "fixture: artifact order", kinds.join(','));
        check(ex.headline == "29.6 tok/s · 2.63x", "fixture: headline", ex.headline);
        check(firstOfKind(ex.artifacts, "chart").value("metric").toString() == "2.63x", "fixture: chart metric");
        check(firstOfKind(ex.artifacts, "table").value("highlightLabel").toString() == "128-reg", "fixture: table winner");
        check(ex.assumptions == QStringList({"DCP=4", "MTP=3"}), "fixture: assumptions", ex.assumptions.join(','));
        for (const QVariant &a : ex.artifacts)
            check(countWords(a.toMap().value("title").toString()) <= 5, "fixture: titles <= 5 words",
                  a.toMap().value("title").toString());
    }

    // PDFKit page render.
    {
        const QString pdf = fixtureDir() + "/FP4_decode_spec.pdf";
        check(pdfPageCount(pdf) == 1, "pdfkit: page count", QString::number(pdfPageCount(pdf)));
        const QImage raw = renderPdfPage(pdf, 0, 800, false);
        check(raw.width() == 800 && raw.height() > 1100 && raw.height() < 1160, "pdfkit: page size",
              QStringLiteral("%1x%2").arg(raw.width()).arg(raw.height()));
        int dark = 0, gold = 0;
        for (int y = 0; y < raw.height(); y += 2)
            for (int x = 0; x < raw.width(); x += 2) {
                const QRgb p = raw.pixel(x, y);
                if (qGray(p) < 90) ++dark;
                if (qAbs(qRed(p) - 0xe0) < 12 && qAbs(qGreen(p) - 0x9f) < 12 && qAbs(qBlue(p) - 0x3e) < 12) ++gold;
            }
        check(dark > 800, "pdfkit: text rendered", QString::number(dark));
        check(gold > 500, "pdfkit: figure rendered", QString::number(gold));
        check(qGray(raw.pixel(5, 5)) > 250, "pdfkit: white paper");
        const QImage ink = renderPdfPage(pdf, 0, 800, true);
        const QRgb c = ink.pixel(5, 5);
        check(qRed(c) == 0x0d && qGreen(c) == 0x0c && qBlue(c) == 0x0b, "pdfkit: inked page sits on the card colour");
        check(forbiddenPixels(ink) == 0, "pdfkit: inked page has no blue/green");
        check(renderPdfPage(pdf, 3, 800, false).isNull(), "pdfkit: out-of-range page is null");
    }
    {
        const QVariantList peaks = audioPeaks(fixtureDir() + "/fn_key_click.wav", 160);
        double mx = 0;
        for (const QVariant &p : peaks)
            mx = std::max(mx, p.toDouble());
        check(peaks.size() == 160 && mx > 0.99, "audio: waveform peaks");
    }

    // Tool calls, touched files and the last turns, for the quick Q&A context.
    {
        const QList<Turn> turns = parseTranscript(claudeSampleJsonl(), Harness::Claude);
        check(turns.value(1).tools.size() == 2 && turns.value(1).tools.value(0) == "Edit /tmp/k.cu"
                  && turns.value(1).tools.value(1) == "Bash: nvcc -O3 /tmp/k.cu && ./a.out",
              "claude: tool calls summarised", turns.value(1).tools.join(" | "));
        const Extracted ex = extractArtifacts(turns, noFiles);
        check(ex.files == QStringList({"/tmp/k.cu"}), "claude: touched files", ex.files.join(","));
        check(ex.toolCalls.size() == 2, "claude: tool calls kept", ex.toolCalls.join(" | "));
        check(ex.recentTurns.size() == 2 && ex.recentTurns.value(0).startsWith("User: profile fp4")
                  && ex.recentTurns.value(1).startsWith("Assistant: Won at 29.6"),
              "claude: recent turns", ex.recentTurns.join(" | "));
    }
    {
        const QList<Turn> turns = parseTranscript(codexSampleJsonl(), Harness::Codex);
        check(turns.size() == 2, "codex: tool calls do not add turns", QString::number(turns.size()));
        check(turns.value(1).tools.contains("shell: cargo bench --bench tile"), "codex: function_call summarised",
              turns.value(1).tools.join(" | "));
        const Extracted ex = extractArtifacts(turns, noFiles);
        check(ex.files.contains("src/tile.rs"), "codex: apply_patch file touched", ex.files.join(","));
    }
    {
        // Ten long turns: only the last four survive, within 2,500 characters.
        QList<Turn> turns;
        for (int i = 0; i < 10; ++i)
            turns << Turn{i % 2 ? QStringLiteral("assistant") : QStringLiteral("user"),
                          QStringLiteral("turn%1 ").arg(i) + QString(1500, QLatin1Char('x')), {}, {}, {}, {}, {}, false};
        const Extracted ex = extractArtifacts(turns, noFiles);
        int chars = 0;
        for (const QString &s : ex.recentTurns)
            chars += int(s.size());
        check(ex.recentTurns.size() == 4 && ex.recentTurns.value(0).startsWith("User: turn6")
                  && ex.recentTurns.value(3).startsWith("Assistant: turn9") && chars <= 2500,
              "context: last 4 turns, <= 2,500 chars", QStringLiteral("%1 turns, %2 chars").arg(ex.recentTurns.size()).arg(chars));
    }
    {
        Backend b(Backend::Mode::Mock);
        const QString ctx = b.contextFor("cluster-serving", 0);
        check(ctx.contains("Last 4 turns:") && ctx.contains("User: serve the 65B")
                  && ctx.contains("User: compare against TP4") && ctx.contains("Assistant: DCP=4 KV split"),
              "context: multi-turn history", ctx);
        check(ctx.contains("Files touched: ~/dev/cluster-serving/serve.sh, ~/dev/cluster-serving/bench_c1.py"),
              "context: files touched", ctx);
        check(ctx.contains("Recent tool calls:") && ctx.contains("Bash: ./serve.sh --dcp 4"), "context: tool calls", ctx);
        check(ctx.contains("On screen: chart") && ctx.contains("Agent's latest write-up:"),
              "context: artifact and write-up kept", ctx);
        QString tblCtx;
        for (const QVariant &pv : b.projects()) {
            const QVariantMap pm = pv.toMap();
            const QVariantList arts = pm.value("artifacts").toList();
            for (int ai = 0; ai < arts.size(); ++ai) {
                if (arts[ai].toMap().value("kind") == "table") {
                    tblCtx = b.contextFor(pm.value("key").toString(), ai);
                    break;
                }
            }
            if (!tblCtx.isEmpty())
                break;
        }
        check(tblCtx.contains("Table:") && tblCtx.contains("|"),
              "context: on-screen table headers and rows included", tblCtx.left(180));
        check(ctx.contains("Artifacts the session produced"), "context: artifact list", ctx);
    }

    // UI prefs: uiScale clamping and the navMode keymap value set.
    {
        const QFileInfo realPrefs(Backend::defaultPrefsPath());
        const bool existedBefore = realPrefs.exists();
        const QDateTime mtimeBefore = existedBefore ? realPrefs.lastModified() : QDateTime();
        Backend b(Backend::Mode::Mock);
        check(b.prefsPath().isEmpty(), "prefs: mock backend has no prefs file");
        check(qFuzzyCompare(b.uiScale(), 1.0) && b.navMode() == "controller", "prefs: defaults 1.0x, Controller mode");
        check(Backend::navModes() == QStringList({"controller", "vim", "wasd", "mouse"})
                  && !Backend::navModes().contains("arrows"),
              "modes: exactly controller | vim | wasd | mouse, no arrows mode", Backend::navModes().join(','));
        check(!b.firstLaunch(), "onboarding: never shown by default in offscreen/mock modes");
        b.setUiScale(9.0);
        check(qFuzzyCompare(b.uiScale(), 1.60), "prefs: uiScale clamps to 1.60", QString::number(b.uiScale()));
        b.setUiScale(0.1);
        check(qFuzzyCompare(b.uiScale(), 0.75), "prefs: uiScale clamps to 0.75", QString::number(b.uiScale()));
        b.setUiScale(std::numeric_limits<double>::quiet_NaN());
        check(qFuzzyCompare(b.uiScale(), 1.0), "prefs: NaN uiScale resets to 1.0", QString::number(b.uiScale()));
        for (int i = 0; i < 30; ++i)
            b.zoomBy(1.1);
        check(qFuzzyCompare(b.uiScale(), 1.60), "prefs: zoomBy saturates at 1.60", QString::number(b.uiScale()));
        b.resetZoom();
        check(qFuzzyCompare(b.uiScale(), 1.0), "prefs: resetZoom", QString::number(b.uiScale()));
        b.setNavMode("emacs");
        check(b.navMode() == "controller", "prefs: unknown navMode ignored", b.navMode());
        b.cycleNavMode();
        const QString m1 = b.navMode();
        b.cycleNavMode();
        const QString m2 = b.navMode();
        b.cycleNavMode();
        const QString m3 = b.navMode();
        b.cycleNavMode();
        check(m1 == "vim" && m2 == "wasd" && m3 == "mouse" && b.navMode() == "controller",
              "prefs: Cmd+K cycles controller -> vim -> wasd -> mouse -> controller",
              m1 + "," + m2 + "," + m3 + "," + b.navMode());
        b.setNavMode("arrows");
        check(b.navMode() == "wasd", "modes: 'arrows' is gaming mode (wasd), never its own mode", b.navMode());
        b.setNavMode("vim");
        check(!b.savePrefs(), "prefs: mock without a path never writes");

        // Inline picker state machine: W/S (k/j) cycle the model across providers, A/D (h/l) the effort.
        check(b.chatModel() == "claude-sonnet-5-5" && b.chatEffort() == "low" && b.chatProvider() == "anthropic",
              "picker: default Sonnet 5.5 · low", b.chatModel() + " " + b.chatEffort());
        b.cycleEffort(1);
        b.cycleEffort(1);
        b.cycleEffort(1);
        check(b.chatEffort() == "xhigh", "picker: D/l x3 -> xhigh", b.chatEffort());
        b.cycleEffort(1);
        check(b.chatEffort() == "xhigh", "picker: effort stops at the model's top (no wrap)", b.chatEffort());
        // Catalog order: DeepSeek, Inkling, Qwen, Fable, Opus, Sonnet, Astra, Sol, Carbon, Neon, Hydrogen.
        b.cycleModel(-1);
        check(b.chatModel() == "claude-opus-5-5" && b.chatEffort() == "xhigh" && b.chatProvider() == "anthropic",
              "picker: W from Sonnet steps to Opus 5.5, xhigh kept", b.chatModel() + " " + b.chatEffort());
        b.cycleModel(-1);
        b.cycleModel(-1);
        check(b.chatModel() == "qwen3.8-flash-next" && b.chatEffort() == "fast" && b.chatProvider() == "openai-compat",
              "picker: W x2 -> Qwen3.8 Flash Next on no-think fast", b.chatModel() + " " + b.chatEffort());
        b.cycleModel(-1);
        check(b.chatModel() == "thinkingmachines/inkling-2-mini" && b.chatEffort() == "fast"
                  && b.chatProvider() == "openrouter" && b.chatEfforts() == QStringList({"fast", "low", "medium", "high"})
                  && modelDisplayLabel(b.chatModel()) == "Inkling 2 Mini · Fast",
              "picker: W -> Inkling 2 Mini (OpenRouter), fast default",
              b.chatModel() + " " + b.chatEffort() + " " + b.chatEfforts().join(','));
        b.cycleModel(-1);
        check(b.chatModel() == "deepseek/deepseek-v4.1-flash-1017" && b.chatEffort() == "fast"
                  && b.chatProvider() == "openrouter" && b.chatEfforts() == QStringList({"fast", "low", "medium", "high"}),
              "picker: W -> DeepSeek V4.1 Flash (OpenRouter), fast default",
              b.chatModel() + " " + b.chatEffort() + " " + b.chatEfforts().join(','));
        b.cycleModel(-1);  // wraps from the first model to the last
        check(b.chatModel() == "gemini-4-hydrogen" && b.chatEffort() == "xhigh" && b.chatProvider() == "google",
              "picker: W wraps to Gemini 4 Hydrogen; leaving fast restores xhigh", b.chatModel() + " " + b.chatEffort());
        b.cycleModel(-1);
        check(b.chatModel() == "gemini-4-neon" && b.chatEfforts().contains("xhigh"), "picker: Neon offers xhigh",
              b.chatEfforts().join(','));
        b.cycleModel(-1);
        check(b.chatModel() == "gemini-4-carbon" && b.chatEffort() == "high",
              "picker: W -> Gemini 4 Carbon, xhigh clamps to high", b.chatModel() + " " + b.chatEffort());
        b.cycleEffort(-1);
        b.cycleEffort(-1);
        b.cycleEffort(-1);
        check(b.chatEffort() == "low", "picker: A x3 -> low (stops at low)", b.chatEffort());
        b.cycleModel(-1);
        check(b.chatModel() == "gpt-6-sol" && b.chatProvider() == "openai", "picker: W crosses Google -> OpenAI", b.chatModel());
        b.cycleModel(1);
        b.cycleModel(1);
        check(b.chatModel() == "gemini-4-neon", "picker: S x2 back across to Neon", b.chatModel());
        QStringList ids;
        for (const ModelSpec &m : modelCatalog())
            ids << m.id;
        check(ids == QStringList({"deepseek/deepseek-v4.1-flash-1017", "thinkingmachines/inkling-2-mini", "qwen3.8-flash-next",
                                  "claude-fable-5-2", "claude-opus-5-5", "claude-sonnet-5-5", "gpt-6-astra", "gpt-6-sol",
                                  "gemini-4-carbon", "gemini-4-neon", "gemini-4-hydrogen"}),
              "picker: exactly the curated 11-model lineup, in order", ids.join(','));
        for (const char *gone : {"gpt-6-luna", "gemini-3.8-flash", "gemini-4-argon", "claude-haiku-5-5"}) {
            b.setChatModel(gone);
            check(b.chatModel() == "gemini-4-neon" && providerFor(gone).isEmpty(), "picker: retired model is not selectable", gone);
        }
        b.setChatEffort("fast");
        check(b.chatEffort() == "low", "picker: an effort the model lacks is ignored", b.chatEffort());
        check(clampEffort("gemini-4-carbon", "xhigh") == "high" && clampEffort("gpt-6-astra", "xhigh") == "xhigh",
              "picker: clampEffort per model");

        // Channel routing: one daemon per provider channel, reused across turns.
        check(channelKeyFor("gpt-6-astra", "low") == "openai" && channelKeyFor("gpt-6-sol", "xhigh") == "openai",
              "daemons: every GPT-6 model shares one codex app-server");
        check(channelKeyFor("claude-opus-5-5", "high") == "anthropic:claude-opus-5-5:high",
              "daemons: claude channel per model+effort", channelKeyFor("claude-opus-5-5", "high"));
        check(channelKeyFor("gemini-4-carbon", "medium") == "google:gemini-4-carbon-medium"
                  && agyModelId("gemini-4-hydrogen", "xhigh") == "gemini-4-hydrogen-xhigh",
              "daemons: agy channel names the model-effort id", channelKeyFor("gemini-4-carbon", "medium"));
        {
            const QStringList want = {"-p", "", "--input-format", "stream-json", "--output-format", "stream-json",
                                      "--disable-slash-commands", "--sandbox", "--model", "gemini-4-carbon-medium"};
            check(agyChatArgs("gemini-4-carbon", "medium") == want, "daemons: agy -p \"\" stream-json args exact",
                  agyChatArgs("gemini-4-carbon", "medium").join(' '));
            const QStringList c = quickHelperArgs("s", "claude-fable-5-2", "xhigh");
            check(c.value(c.indexOf("--model") + 1) == "claude-fable-5-2" && c.value(c.indexOf("--effort") + 1) == "xhigh"
                      && c.contains("--no-session-persistence") && c.contains("--strict-mcp-config")
                      && c.value(c.indexOf("--tools") + 1).isEmpty() && c.value(c.indexOf("--setting-sources") + 1).isEmpty(),
                  "daemons: claude args carry the picked model + effort", c.join(' '));
        }
        // Mock asks route to the selection without spawning anything.
        b.setChatModel("gpt-6-sol");
        b.setChatEffort("high");
        b.ask("cluster-serving", 0, "which arm?");
        check(b.lastAskModel() == "gpt-6-sol" && b.lastAskEffort() == "high" && b.lastAskChannel() == "openai"
                  && b.spawnCount() == 0 && b.channelCount() == 0,
              "daemons: Enter sends with the picked model + effort (mock, zero spawns)",
              b.lastAskModel() + " " + b.lastAskEffort() + " " + b.lastAskChannel());

        // Warm model suggestion (opt-in): off by default, lands for the exact artifact when on.
        b.requestSuggestion("cluster-serving", 0);
        b.advance(0.5);
        check(b.suggestion().isEmpty(), "predictor: model suggestions are opt-in (off: none)");
        b.setModelSuggestions(true);
        b.requestSuggestion("cluster-serving", 0);
        b.advance(0.2);
        check(b.suggestion().isEmpty(), "predictor: suggestion not there before 0.4 s");
        b.advance(0.3);
        check(!b.suggestion().isEmpty() && b.suggestionKey() == "cluster-serving:0", "predictor: warm suggestion lands",
              b.suggestion() + " @ " + b.suggestionKey());
    }

    // codex app-server JSON-RPC framing: one compact object per line, requests carry ids,
    // notifications do not, turn/start carries model + effort + summary.
    {
        const QByteArray line = codexRpc(7, "turn/start", codexTurnParams("th1", "hi", "gpt-6-sol", "high"));
        check(line.endsWith('\n') && line.count('\n') == 1, "codex: one JSON object per line");
        const QJsonObject o = QJsonDocument::fromJson(line).object();
        const QJsonObject p = o.value("params").toObject();
        check(o.value("id").toInt() == 7 && o.value("method").toString() == "turn/start" && p.value("threadId").toString() == "th1"
                  && p.value("model").toString() == "gpt-6-sol" && p.value("effort").toString() == "high"
                  && p.value("summary").toString() == "concise"
                  && p.value("input").toArray().at(0).toObject().value("text").toString() == "hi",
              "codex: turn/start params (model, effort, summary)", QString::fromUtf8(line));
        const QJsonObject ts = codexThreadStartParams("be brief");
        check(ts.value("approvalPolicy").toString() == "never" && ts.value("sandbox").toString() == "read-only"
                  && ts.value("ephemeral").toBool() && ts.value("developerInstructions").toString() == "be brief",
              "codex: thread/start is read-only, ephemeral, never asks");
        CodexEvent ev = parseCodexLine(R"({"id":2,"result":{"thread":{"id":"th-9"}}})");
        check(ev.kind == CodexEvent::Response && ev.id == 2 && ev.threadId == "th-9", "codex: thread/start response -> thread id");
        ev = parseCodexLine(R"({"method":"item/agentMessage/delta","params":{"threadId":"th-9","turnId":"u1","itemId":"i1","delta":"12"}})");
        check(ev.kind == CodexEvent::AgentDelta && ev.text == "12" && ev.threadId == "th-9", "codex: agentMessage delta");
        ev = parseCodexLine(R"({"method":"item/reasoning/summaryTextDelta","params":{"threadId":"th-9","delta":"thinking"}})");
        check(ev.kind == CodexEvent::ReasoningDelta && ev.text == "thinking", "codex: reasoning summary delta");
        ev = parseCodexLine(R"({"method":"turn/completed","params":{"threadId":"th-9","turn":{"id":"u1"}}})");
        check(ev.kind == CodexEvent::TurnCompleted && ev.threadId == "th-9", "codex: turn/completed");
        ev = parseCodexLine(R"({"id":5,"error":{"code":-32600,"message":"bad model"}})");
        check(ev.kind == CodexEvent::Error && ev.id == 5 && ev.text == "bad model", "codex: JSON-RPC error");
        check(parseCodexLine("{\"method\":\"item/agentMessage/de").kind == CodexEvent::None, "codex: cut line ignored");

        // A whole reasoning turn as codex app-server streams it: item/started reasoning, summary
        // deltas, item/started agentMessage, text deltas, turn/completed. Reasoning and text
        // accumulate separately and in order.
        const QList<QByteArray> turn = {
            R"({"method":"turn/started","params":{"threadId":"th-9","turn":{"id":"u2"}}})",
            R"({"method":"item/started","params":{"threadId":"th-9","item":{"type":"reasoning","id":"r1"}}})",
            R"({"method":"item/reasoning/summaryTextDelta","params":{"threadId":"th-9","delta":"**Comparing arms**"}})",
            R"({"method":"item/reasoning/summaryTextDelta","params":{"threadId":"th-9","delta":" by spills"}})",
            R"({"method":"item/reasoning/textDelta","params":{"threadId":"th-9","delta":" and regs"}})",
            R"({"method":"item/started","params":{"threadId":"th-9","item":{"type":"agentMessage","id":"m1"}}})",
            R"({"method":"item/agentMessage/delta","params":{"threadId":"th-9","delta":"128-reg"}})",
            R"({"method":"item/agentMessage/delta","params":{"threadId":"th-9","delta":" wins."}})",
            R"({"method":"turn/completed","params":{"threadId":"th-9","turn":{"id":"u2","status":"completed"}}})",
        };
        QString reasoning, text;
        QStringList started;
        int completed = 0, other = 0;
        for (const QByteArray &l : turn) {
            const CodexEvent e = parseCodexLine(l);
            switch (e.kind) {
            case CodexEvent::ReasoningDelta: reasoning += e.text; break;
            case CodexEvent::AgentDelta: text += e.text; break;
            case CodexEvent::ItemStarted: started << e.text; break;
            case CodexEvent::TurnCompleted: completed += 1; break;
            default: other += 1; break;
            }
        }
        check(reasoning == "**Comparing arms** by spills and regs" && text == "128-reg wins." && completed == 1
                  && started == QStringList({"reasoning", "agentMessage"}) && other == 1,
              "codex: reasoning + text deltas of one turn parse in order", reasoning + " | " + text + " | " + started.join(','));
        // Multi-turn reuse: the second turn/start rides the same thread id with a new request id.
        const QJsonObject t1 = QJsonDocument::fromJson(codexRpc(3, "turn/start", codexTurnParams("th-9", "a", "gpt-6-astra", "low"))).object();
        const QJsonObject t2 = QJsonDocument::fromJson(codexRpc(4, "turn/start", codexTurnParams("th-9", "b", "gpt-6-sol", "low"))).object();
        check(t1.value("params").toObject().value("threadId") == t2.value("params").toObject().value("threadId")
                  && t1.value("id").toInt() != t2.value("id").toInt()
                  && t2.value("params").toObject().value("model").toString() == "gpt-6-sol",
              "codex: turn 2 reuses the thread, switches model per turn, new request id");
        for (const char *m : {"gpt-6-astra", "gpt-6-sol"})
            check(channelKeyFor(m, "low") == "openai" && providerFor(m) == "openai",
                  "codex: astra / sol both ride the one app-server", m);
        check(channelKeyFor("deepseek/deepseek-v4.1-flash-1017", "fast") == "openrouter:deepseek/deepseek-v4.1-flash-1017",
              "openrouter: DeepSeek gets its own HTTP channel key, no daemon",
              channelKeyFor("deepseek/deepseek-v4.1-flash-1017", "fast"));
        check(codexAppServerArgs().value(0) == "app-server" && codexAppServerArgs().contains("mcp_servers={}"),
              "codex: app-server args (no MCP servers)", codexAppServerArgs().join(' '));
    }

    // agy / claude stream progress: the live phase and thinking text before the answer.
    {
        auto obj = [](const char *s) { return QJsonDocument::fromJson(QByteArray(s)).object(); };
        // Real gemini-4-hydrogen lines (captured 2026-11-17): no thinking text at all, only steps.
        StreamProgress p = agyProgress(obj(R"({"event":"step_update","step_update":{"step_index":0,"state":"DONE","step_type":"user_input"}})"));
        check(p.phase == "thinking" && p.thinking.isEmpty(), "agy: prompt accepted -> phase thinking", p.phase);
        p = agyProgress(obj(R"({"event":"step_update","step_update":{"step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"run_command"}})"));
        check(p.phase == "running run_command", "agy: active tool step -> running <tool>", p.phase);
        p = agyProgress(obj(R"({"event":"step_update","step_update":{"step_index":2,"state":"DONE","step_type":"tool","tool_name":"run_command"}})"));
        check(p.phase == "thinking", "agy: tool done -> back to thinking", p.phase);
        p = agyProgress(obj(R"({"event":"step_update","step_update":{"step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"No."}})"));
        check(p.text == "No." && p.phase == "writing", "agy: text_delta -> writing", p.text);
        p = agyProgress(obj(R"({"event":"step_update","step_update":{"state":"DONE","step_type":"agent_response","text_delta":"\n","usage":{"output_tokens":424,"thinking_tokens":418}}})"));
        check(p.thinkingTokens == 418, "agy: step usage carries thinking_tokens", QString::number(p.thinkingTokens));
        p = agyProgress(obj(R"({"event":"result","result":{"status":"SUCCESS","response":"No.\n","usage":{"thinking_tokens":418}}})"));
        check(p.thinkingTokens == 418, "agy: result usage carries thinking_tokens");
        p = agyProgress(obj(R"({"event":"step_update","step_update":{"state":"ACTIVE","step_type":"agent_response","thinking_delta":"weighing 7*11*13"}})"));
        check(p.thinking == "weighing 7*11*13", "agy: step_update.thinking_delta streams as thinking", p.thinking);
        // Claude stream-json partial messages.
        p = claudeProgress(obj(R"({"type":"stream_event","event":{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}})"));
        check(p.phase == "thinking", "claude: thinking block start -> phase thinking", p.phase);
        p = claudeProgress(obj(R"({"type":"stream_event","event":{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"spills first"}}})"));
        check(p.thinking == "spills first" && p.text.isEmpty(), "claude: thinking_delta -> thinking text", p.thinking);
        p = claudeProgress(obj(R"({"type":"stream_event","event":{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Blue"}}})"));
        check(p.text == "Blue" && p.thinking.isEmpty(), "claude: text_delta -> answer text", p.text);
        check(tailWords("one two three four five", 3) == QStringLiteral("… three four five")
                  && tailWords("a b", 3) == "a b",
              "thinking: the live line keeps the newest words", tailWords("one two three four five", 3));
        // Raw tool-call markup never reaches the quick line or the trace.
        auto clean = [](const char *s) { return stripToolNoise(QString::fromUtf8(s)).simplified(); };
        const QString a1 = clean(
            "Checking the kernel.\n<tool_call>{\"name\":\"bash\",\"arguments\":{\"cmd\":\"nvidia-smi\"}}</tool_call>\nIt is 126 tok/s.");
        check(a1 == "Checking the kernel. It is 126 tok/s.", "noise: <tool_call> block stripped from the answer", a1);
        const QString a2 = clean("{\"name\": \"read_file\", \"arguments\": {\"path\": \"x\"}}\nThe diff swaps the tile to 128.");
        check(a2 == "The diff swaps the tile to 128.", "noise: JSON tool-call line dropped", a2);
        const QString a3 = clean(
            "Short answer.<|tool_calls_section_begin|><|tool_call_begin|>functions.run:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>");
        check(a3 == "Short answer.", "noise: special tool tokens stripped", a3);
        const QString a4 = clean("Looking.\n<function_calls>\n<invoke name=\"bash\">\n<parameter name=\"cmd\">ls");
        check(a4 == "Looking.", "noise: an unclosed block mid-stream is cut to the end", a4);
        const QString a5 = QStringLiteral("Use a {tile} of 128 [see run 3] <- faster");
        check(stripToolNoise(a5) == a5, "noise: plain prose with braces and brackets is untouched", stripToolNoise(a5));
        const QString t1 = tailWords(stripToolNoise(QStringLiteral("weighing KV layout\n[tool_call: run_command]\nthen the answer")), 40);
        check(t1 == "weighing KV layout then the answer", "noise: [tool_call: ...] lines leave the thinking line", t1);
    }

    // Honest speed classes in the catalog and picker.
    {
        const QHash<QString, QString> want = {
            {"claude-sonnet-5-5", "Sonnet 5.5 · Fast"}, {"deepseek/deepseek-v4.1-flash-1017", "DeepSeek V4.1 Flash · Fast"},
            {"gemini-4-hydrogen", "Gemini 4 Hydrogen · Deep Think"}, {"gemini-4-carbon", "Gemini 4 Carbon · Teach"},
            {"gpt-6-astra", "GPT 6.2 Astra · Frontier"}, {"gpt-6-sol", "GPT 6.2 Sol · Balanced"},
            {"qwen3.8-flash-next", "Qwen3.8 Flash Next · Fast"}, {"gemini-4-neon", "Gemini 4 Neon · Balanced"}};
        for (auto it = want.cbegin(); it != want.cend(); ++it)
            check(modelDisplayLabel(it.key()) == it.value(), "catalog: model speed class label", modelDisplayLabel(it.key()));
        bool allTagged = true;
        for (const ModelSpec &m : modelCatalog())
            allTagged = allTagged && !m.speed.isEmpty();
        check(allTagged, "catalog: every model has a speed class");
    }

    // Hyperparameters / configs and key terms from a session's turns.
    {
        const QList<Turn> turns = kickoffTurns();
        const Extracted ex = extractArtifacts(turns, noFiles);
        QHash<QString, QVariantMap> byName;
        for (const QVariant &v : ex.params)
            byName.insert(v.toMap().value("name").toString(), v.toMap());
        const QVariantMap tile = byName.value("tile"), regs = byName.value("regs");
        check(tile.value("runs").toStringList() == QStringList({"64x128", "128x256", "128x128"}),
              "params: tile values across runs from the sweep table", tile.value("runs").toStringList().join(','));
        check(regs.value("value").toString() == "128" && regs.value("runs").toStringList().contains("168"),
              "params: regs latest value 128, runs include 168", regs.value("value").toString() + " / " + regs.value("runs").toStringList().join(','));
        check(byName.value("DCP").value("value").toString() == "4" && byName.value("MTP").value("value").toString() == "3",
              "params: DCP=4 and MTP=3 from prose", byName.value("DCP").value("value").toString());
        check(byName.value("spills").value("value").toString() == "0" && tile.value("value").toString() == "128x128",
              "params: the winning row sets the value (spills 0, tile 128x128), not the newest prose mention",
              byName.value("spills").value("value").toString() + " / " + tile.value("value").toString());
        check(!tile.value("detail").toString().isEmpty() && tile.value("kind").toString() == "param",
              "params: each param carries a plain-English detail");
        int terms = 0;
        for (const QVariant &v : ex.params)
            terms += v.toMap().value("kind").toString() == "term";
        check(ex.params.size() <= 9 && terms >= 1 && terms <= 3, "params: sparse (<= 6 params + <= 3 terms)",
              QString::number(ex.params.size()));
        // Training run: lr, steps, LoRA rank; sm_120 normalised.
        Turn u;
        u.role = "user";
        u.text = "train a voice-clone lora on my recordings, lora_rank 16, lr=2e-4 on sm120";
        Turn a;
        a.role = "assistant";
        a.text = "Started. step 200 loss 2.31";
        a.tools = {"Bash: python train.py --steps=1200 --batch-size 8"};
        Turn a2;
        a2.role = "assistant";
        a2.text = "step 400 loss 1.92, lr=1e-4 after warmup";
        const QVariantList ps = extractParams({u, a, a2}, {});
        QHash<QString, QVariantMap> m;
        for (const QVariant &v : ps)
            m.insert(v.toMap().value("name").toString(), v.toMap());
        check(m.value("lr").value("value").toString() == "1e-4" && m.value("lr").value("runs").toStringList() == QStringList({"2e-4", "1e-4"}),
              "params: lr latest 1e-4, runs 2e-4 -> 1e-4", m.value("lr").value("runs").toStringList().join(','));
        check(m.value("steps").value("value").toString() == "400" && m.value("rank").value("value").toString() == "16"
                  && m.value("batch").value("value").toString() == "8" && m.value("arch").value("value").toString() == "sm_120",
              "params: steps 400, LoRA rank 16, batch 8, arch sm_120",
              m.value("steps").value("value").toString() + " " + m.value("rank").value("value").toString() + " "
                  + m.value("batch").value("value").toString() + " " + m.value("arch").value("value").toString());
        // Plain-English job names.
        const QString train = synthesizeJobName({u, a, a2}, "");
        check(train == "Training voice-clone LoRA step 400", "jobname: training run with progress", train);
        Turn s;
        s.role = "user";
        s.text = "sweep tile sizes for the fp4 gemm on sm120";
        Turn sa;
        sa.role = "assistant";
        sa.tools = {"Bash: for t in 64 128 256; do ./bench --tile $t; done"};
        const QString sweep = synthesizeJobName({s, sa}, "");
        const int sw = int(sweep.split(' ').size());
        check(sweep.startsWith("Sweeping ") && sweep.contains("FP4") && sweep.contains("tile") && sw >= 4 && sw <= 7,
              "jobname: sweep reads 'Sweeping ... FP4 ...' in 4-7 words", sweep);
        Turn h;
        h.role = "user";
        h.text = "run the h3 blind review tree benchmark";
        const QString h3 = synthesizeJobName({h}, "");
        const int hw = int(h3.split(' ').size());
        check(h3.contains("H3") && h3.contains("benchmark") && hw >= 4 && hw <= 7, "jobname: H3 blind review in 4-7 words", h3);
        // Real-transcript failure modes seen in --probe-live: a URL goal, a meta request about the
        // chat itself, and a tab title that already names the job.
        {
            Turn m;
            m.role = "user";
            m.text = "Okay, you are replacing the Phonon session.";
            Turn f;
            f.role = "user";
            f.text = "fix the fn key toggle latency";
            const QString meta = synthesizeJobName({m, f}, m.text);
            check(meta == "Fixing fn key toggle latency", "jobname: meta requests about the chat are skipped", meta);
            Turn u2;
            u2.role = "user";
            u2.text = "`https://x.com/someone/status/2122784279604383914` How did they do it?";
            const QString url = synthesizeJobName({u2}, u2.text);
            check(!url.contains("http") && !url.contains("x.com"), "jobname: URLs never leak into a name", url);
            const QString titled = synthesizeJobName({}, "", "GPT 6.2 Astra KernelBench run");
            check(titled.contains("GPT 6.2 Astra KernelBench"), "jobname: a phrase-like tab title is the subject", titled);
            Turn v;
            v.role = "assistant";
            v.tools = {"Bash: cmake --build build -j 2"};
            check(synthesizeJobName({v}, "").isEmpty(), "jobname: a verb with no subject gives no name (never just 'Building')",
                  synthesizeJobName({v}, ""));
            // Word salad seen in --probe-live ("Rendering dont even ask fable", "Fixing mean it's
            // unreachable fact"): chat is skipped, a real request keeps its word order.
            auto user = [](const QString &text) {
                Turn t;
                t.role = "user";
                t.text = text;
                return t;
            };
            Turn r;
            r.role = "assistant";
            r.tools = {"Bash: ffmpeg -i in.mov out.mp4"};
            check(synthesizeJobName({user("dont even ask fable, just do it yourself"), r}, "").isEmpty(),
                  "jobname: chat ('dont even ask ...') never becomes a name", synthesizeJobName({user("dont even ask fable"), r}, ""));
            check(synthesizeJobName({user("I mean it's unreachable in fact")}, "").isEmpty(),
                  "jobname: 'I mean it's ...' never becomes a name");
            const QString slack = synthesizeJobName({user("can you write a slack to Alex about her own blog")}, "");
            check(slack == "Writing slack to Alex about blog", "jobname: a request keeps its word order", slack);
            const QString fact = synthesizeJobName(
                {user("What do you mean it's unreachable? Fix the fact that it's unreachable. There's gotta be a way.")}, "");
            check(!fact.contains("unreachable") || fact.contains("fact that"), "jobname: a relative clause never glues on ('Fixing fact unreachable')", fact);
            const QString ps4 = promptTitle("Add support to my PS4 controller with the most obvious keybinds and setups for things.");
            check(ps4 == "Add support to PS4 controller", "jobname: a cut name ends on a whole phrase", ps4);
            const QString later = synthesizeJobName({user("I mean it's unreachable in fact"), user("ok, render the demo video at 240 Hz")}, "");
            check(later == "Rendering demo video at 240 Hz", "jobname: the first request that names a job wins", later);
        }
        // The readable conversation: last 16 turns, tool noise stripped, <= 800 chars each.
        {
            QList<Turn> conv;
            for (int i = 0; i < 20; ++i) {
                Turn t;
                t.role = i % 2 ? "assistant" : "user";
                t.text = QStringLiteral("turn %1 ").arg(i) + QString(i == 19 ? 1200 : 20, QLatin1Char('x'));
                if (i == 17)
                    t.text = "<tool_call>{\"name\": \"Bash\", \"arguments\": {}}</tool_call>\nRebuilt cleanly, **0 warnings**.";
                conv << t;
                if (i == 10) {
                    Turn tool;  // tool-only step: nothing to read
                    tool.role = "assistant";
                    tool.tools = {"Bash: ls"};
                    conv << tool;
                }
            }
            const Extracted ce = extractArtifacts(conv, noFiles);
            const QVariantList tt = ce.threadTurns;
            bool noNoise = true;
            for (const QVariant &v : tt)
                noNoise = noNoise && !v.toMap().value("text").toString().contains("tool_call")
                    && !v.toMap().value("text").toString().contains("\"arguments\"");
            check(tt.size() == 16 && tt.first().toMap().value("text").toString().startsWith("turn 4 ")
                      && tt.last().toMap().value("role").toString() == "assistant",
                  "turns: the last 16 readable turns, oldest first", QString::number(tt.size()));
            check(noNoise && tt.value(13).toMap().value("text").toString() == "Rebuilt cleanly, 0 warnings.",
                  "turns: tool-call markup and bold markers stripped", tt.value(13).toMap().value("text").toString());
            check(tt.last().toMap().value("text").toString().size() <= 801
                      && tt.last().toMap().value("text").toString().endsWith(QChar(0x2026)),
                  "turns: a long turn is cut at 800 chars", QString::number(tt.last().toMap().value("text").toString().size()));
        }
        bool allShort = true;
        QStringList names;
        for (const FixtureProject &f : fixtureProjects()) {
            const QString n = extractArtifacts(f.turns, noFiles).jobName;
            names << n;
            const int c = int(n.split(' ').size());
            allShort = allShort && c >= 3 && c <= 7;
        }
        check(allShort, "jobname: every fixture name is 3-7 words", names.join(" | "));
        std::printf("info  fixture job names: %s\n", qPrintable(names.join(" | ")));
        check(isCrypticName("cluster-serving") && isCrypticName("a1b2c3d4") && isCrypticName("q3_v2") && !isCrypticName("studio")
                  && !isCrypticName("Phonon fn key toggle"),
              "jobname: cryptic folder detection");
    }

    // Fresh-turn counting across rescans, and the +N badge bookkeeping.
    {
        check(newTurnsSince({1, 2, 3}, {1, 2, 3, 4, 5}) == 2, "fresh: two appended turns", QString::number(newTurnsSince({1, 2, 3}, {1, 2, 3, 4, 5})));
        check(newTurnsSince({1, 2, 3}, {1, 2, 9}) == 0, "fresh: a grown last turn is not new");
        check(newTurnsSince({1, 2, 3}, {1, 2, 9, 4}) == 1, "fresh: grown last turn + one new");
        check(newTurnsSince({}, {1, 2}) == 0 && newTurnsSince({7}, {1, 2}) == 0, "fresh: first parse or no overlap counts 0");
        Backend b(Backend::Mode::Mock);
        auto fresh = [&](const QString &key) {
            for (const QVariant &v : b.projects())
                if (v.toMap().value("key").toString() == key)
                    return v.toMap().value("fresh").toInt();
            return -1;
        };
        Turn t;
        t.role = "assistant";
        t.text = "Rerun with tile 128x64: 4.4x over eager.";
        const int idxBefore = b.projectIndex("kernel-search");
        b.mockAppendTurn("kernel-search", t);
        b.mockAppendTurn("kernel-search", t);
        check(b.projectIndex("kernel-search") == idxBefore,
              "order: background turns never shuffle existing card positions",
              QStringLiteral("%1 -> %2").arg(idxBefore).arg(b.projectIndex("kernel-search")));
        check(fresh("kernel-search") == 2, "fresh: background turns raise +2 on an unfocused island", QString::number(fresh("kernel-search")));
        b.setFocusedProject("kernel-search");
        check(fresh("kernel-search") == 0, "fresh: opening the project clears its badge");
        b.mockAppendTurn("kernel-search", t);
        check(fresh("kernel-search") == 0, "fresh: turns on the focused project never raise a badge");
        bool hasTile = false;
        for (const QVariant &v : b.projects())
            if (v.toMap().value("key").toString() == "kernel-search")
                for (const QVariant &p : v.toMap().value("params").toList())
                    hasTile = hasTile || p.toMap().value("name").toString() == "tile";
        check(hasTile, "fresh: an appended turn re-synthesizes params (tile 128x64 appears)");
        QString servingJob, phononJob;
        for (const QVariant &v : b.projects()) {
            const QVariantMap p = v.toMap();
            if (p.value("key") == "cluster-serving")
                servingJob = p.value("jobName").toString();
            if (p.value("key") == "phonon")
                phononJob = p.value("jobName").toString();
        }
        check(!servingJob.isEmpty() && phononJob.isEmpty(),
              "jobname: shown for a cryptic long-running folder, not for 'phonon' (short, idle, plain)",
              servingJob + " / " + phononJob);
    }

    // First-launch onboarding: shown only when the interactive prefs file does not exist yet.
    {
        QTemporaryDir tmp;
        const QString path = tmp.path() + "/studio_prefs.json";
        {
            Backend w(Backend::Mode::Mock);
            w.setPrefsPath(path);
            check(w.firstLaunch() && w.navMode() == "controller",
                  "onboarding: missing prefs file -> first launch, Controller preselected");
            w.completeOnboarding("wasd");
            check(!w.firstLaunch() && w.navMode() == "wasd" && QFileInfo::exists(path),
                  "onboarding: confirming writes the prefs file at once");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(!r.firstLaunch() && r.navMode() == "wasd", "onboarding: second launch skips it, mode kept", r.navMode());
        }
        {
            QFile f(path);
            if (f.open(QIODevice::WriteOnly))
                f.write("{\"navMode\": \"arrows\", \"defaultModel\": \"gemini-4-hydrogen\", \"defaultEffort\": \"xhigh\"}");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.navMode() == "wasd" && r.chatModel() == "gemini-4-hydrogen" && r.chatEffort() == "xhigh",
                  "prefs: old 'arrows' file -> wasd; default model + effort load",
                  r.navMode() + " " + r.chatModel() + " " + r.chatEffort());
            r.setChatModel("gemini-4-carbon");
            check(r.chatEffort() == "high", "prefs: switching to Carbon clamps xhigh -> high", r.chatEffort());
            check(r.savePrefs(), "prefs: save model + effort");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.chatModel() == "gemini-4-carbon" && r.chatEffort() == "high", "prefs: default model + effort round trip",
                  r.chatModel() + " " + r.chatEffort());
        }
    }

    // Accent theme persists in studio_prefs.json: gold (default) -> lapis -> gold.
    {
        QTemporaryDir tmp;
        const QString path = tmp.path() + "/studio_prefs.json";
        auto fileTheme = [&path] {
            QFile f(path);
            return f.open(QIODevice::ReadOnly) ? QJsonDocument::fromJson(f.readAll()).object().value("theme").toString()
                                               : QStringLiteral("<missing>");
        };
        {
            Backend w(Backend::Mode::Mock);
            check(w.theme() == "gold", "theme: default is gold", w.theme());
            w.setPrefsPath(path);
            w.setTheme("teal");
            check(w.theme() == "gold", "theme: unknown id ignored", w.theme());
            w.setTheme("lapis");
            check(w.theme() == "lapis" && w.savePrefs() && fileTheme() == "lapis", "theme: lapis written to prefs",
                  w.theme() + " / " + fileTheme());
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.theme() == "lapis", "theme: lapis survives relaunch", r.theme());
            r.setTheme("gold");
            check(r.savePrefs() && fileTheme() == "gold", "theme: back to gold written", fileTheme());
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.theme() == "gold", "theme: gold -> lapis -> gold round trip", r.theme());
        }
        {
            QFile f(path);
            if (f.open(QIODevice::WriteOnly))
                f.write("{\"theme\": \"neon\"}");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.theme() == "gold", "theme: bad file value falls back to gold", r.theme());
        }
    }

    // lapis-viz and other media printed by tools become video / image artifacts.
    {
        const QByteArray claude =
            R"({"type":"user","message":{"role":"user","content":"render the decode sweep"}})" "\n"
            R"({"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"tu1","name":"Bash","input":{"command":"lapis-viz sweep.csv --mp4"}}]}})" "\n"
            R"({"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu1","content":"wrote /tmp/viz/decode_sweep.mp4 (2.1 s)\nposter /tmp/viz/decode_sweep.png"}]}})" "\n"
            R"({"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","id":"tu2","name":"Bash","input":{"command":"ls"}}]}})" "\n"
            R"({"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"tu2","content":[{"type":"text","text":"/tmp/shots/frame_0.png"}]}]}})" "\n"
            R"({"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Rendered the sweep."}]}})" "\n";
        const QList<Turn> turns = parseTranscript(claude, Harness::Claude);
        check(turns.size() == 2 && turns.value(1).viz && turns.value(1).toolOutput.contains("/tmp/viz/decode_sweep.mp4"),
              "viz: tool_result paths attach to the assistant turn", QString::number(turns.size()));
        const Extracted ex = extractArtifacts(turns, [](const QString &p) { return p.startsWith("/tmp/"); });
        const QVariantMap vid = firstOfKind(ex.artifacts, "video");
        check(vid.value("path").toString() == "/tmp/viz/decode_sweep.mp4" && vid.value("source").toString() == "lapis-viz"
                  && vid.value("metric").toString() == "viz · mp4",
              "viz: lapis-viz .mp4 -> video artifact", vid.value("path").toString() + " " + vid.value("metric").toString());
        int images = 0;
        for (const QVariant &a : ex.artifacts)
            images += a.toMap().value("kind").toString() == "image" ? 1 : 0;
        check(images == 2, "viz: .png from lapis-viz and from any tool output -> image artifacts", QString::number(images));
        check(extractArtifacts(turns, noFiles).artifacts.isEmpty(), "viz: missing files are not artifacts");

        const QByteArray codex =
            R"({"type":"response_item","payload":{"type":"function_call","name":"shell","call_id":"c1","arguments":"{\"command\":[\"bash\",\"-lc\",\"lapis-viz bench.json\"]}"}})" "\n"
            R"({"type":"response_item","payload":{"type":"function_call_output","call_id":"c1","output":"saved ~/viz/bench.png"}})" "\n";
        const QList<Turn> ct = parseTranscript(codex, Harness::Codex);
        const Extracted cx = extractArtifacts(ct, [](const QString &p) { return p.endsWith("bench.png"); });
        const QVariantMap img = firstOfKind(cx.artifacts, "image");
        check(img.value("path").toString() == QDir::homePath() + "/viz/bench.png" && img.value("source").toString() == "lapis-viz",
              "viz: codex function_call_output .png -> image artifact", img.value("path").toString());
        check(mediaKindForExt("MP4") == "video" && mediaKindForExt("webm") == "video" && mediaKindForExt("png") == "image",
              "viz: media kinds");
        check(mediaPathsIn("a /x/y.mp4 b ~/z.png c /no.txt") == QStringList({"/x/y.mp4", "~/z.png"}), "viz: media path scan");
    }

    {
        const QFileInfo realPrefs(Backend::defaultPrefsPath());
        const bool existedBefore = realPrefs.exists();
        const QDateTime mtimeBefore = existedBefore ? realPrefs.lastModified() : QDateTime();
        // Round trip through a temp file: atomic write, 0600, read back by a fresh backend.
        QTemporaryDir tmp;
        const QString path = tmp.path() + "/sub/studio_prefs.json";
        {
            Backend w(Backend::Mode::Mock);
            w.setPrefsPath(path);
            w.setUiScale(1.15);
            w.setNavMode("vim");
            check(w.savePrefs(), "prefs: save");
        }
        const QFileInfo fi(path);
        check(fi.exists() && fi.permissions() == (QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ReadUser | QFileDevice::WriteUser),
              "prefs: file is 0600", QString::number(int(fi.permissions()), 16));
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(qFuzzyCompare(r.uiScale(), 1.15) && r.navMode() == "vim", "prefs: round trip 1.15x vim",
                  QStringLiteral("%1 %2").arg(r.uiScale()).arg(r.navMode()));
            r.setNavMode("wasd");  // debounced; the destructor flushes it
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.navMode() == "wasd", "prefs: pending change flushed on exit", r.navMode());
        }
        {
            QFile f(path);
            if (f.open(QIODevice::WriteOnly))
                f.write("{\"uiScale\": 7, \"navMode\": \"qwerty\"}");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(qFuzzyCompare(r.uiScale(), 1.60) && r.navMode() == "controller",
                  "prefs: bad file values clamped/ignored",
                  QStringLiteral("%1 %2").arg(r.uiScale()).arg(r.navMode()));
        }
        const QFileInfo after(Backend::defaultPrefsPath());
        check(after.exists() == existedBefore && (!existedBefore || after.lastModified() == mtimeBefore),
              "safety: --test never touches ~/.lapis-next/studio_prefs.json");
    }
}

QQuickItem *firstItem(QQuickWindow *w, const char *name)
{
    const auto items = findItems(w->contentItem(), QString::fromLatin1(name));
    return items.isEmpty() ? nullptr : items.first();
}

// Constellation focus chrome ("Open Stage R1" badge, "✕ Expand" chip) against the map: each
// pill may sit on the one card it belongs to (the card it overlaps most) but must not touch a
// visible cluster label or any other card.
int mapChromeCollisions(QQuickWindow *win, QStringList *hits)
{
    auto sceneRect = [](QQuickItem *it) { return it->mapRectToScene(QRectF(0, 0, it->width(), it->height())); };
    QList<QRectF> cards, labels;
    for (QQuickItem *it : findItems(win->contentItem(), "island"))
        if (it->isVisible() && it->opacity() > 0.5)
            cards << sceneRect(it);
    for (QQuickItem *t : findItems(win->contentItem(), "clusterLabel"))
        if (t->isVisible() && t->opacity() > 0.5)
            labels << sceneRect(t).adjusted(0, 0, 0, 2);
    int n = 0;
    for (const char *name : {"tabBadge", "focusCursorHint"}) {
        QQuickItem *pill = firstItem(win, name);
        if (!pill || !pill->isVisible() || !pill->parentItem() || pill->parentItem()->opacity() < 0.5)
            continue;
        const QRectF pr = sceneRect(pill);
        int owner = -1;
        qreal best = 0;
        for (int i = 0; i < cards.size(); ++i) {
            const QRectF x = cards[i].intersected(pr);
            if (x.width() * x.height() > best) { best = x.width() * x.height(); owner = i; }
        }
        for (int i = 0; i < cards.size(); ++i)
            if (i != owner && cards[i].intersects(pr)) {
                ++n;
                if (hits)
                    *hits << QStringLiteral("%1 touches card at %2,%3").arg(name).arg(cards[i].x()).arg(cards[i].y());
            }
        for (const QRectF &l : labels)
            if (l.intersects(pr)) {
                ++n;
                if (hits)
                    *hits << QStringLiteral("%1 touches cluster label at %2,%3").arg(name).arg(l.x()).arg(l.y());
            }
    }
    return n;
}

void scanBanned(QQuickWindow *win, const char *where)
{
    QStringList texts;
    visibleWords(win, &texts);
    const QString all = texts.join('\n');
    for (const char *bad : {"TOUCHES PROD", "SPENDS MONEY", "$100", "Slack", "rerun", " or keep "})
        check(!all.contains(QLatin1String(bad), Qt::CaseInsensitive), "ui: no policy badges or option prompts",
              QStringLiteral("%1: %2").arg(where, bad));
    QStringList pairs;
    check(textOverlaps(win, &pairs) == 0, "ui: no overlapping text", QStringLiteral("%1: %2").arg(where, pairs.join("; ")));
}

void uiTests()
{
    check(QGuiApplication::platformName() == "offscreen", "ui: platform is offscreen", QGuiApplication::platformName());
    Backend backend(Backend::Mode::Mock);
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    check(win != nullptr, "ui: QML loads");
    if (!win)
        return;
    QObject *app = win->findChild<QObject *>("studioApp");
    check(app != nullptr, "ui: StudioApp present");
    if (!app)
        return;
    check(win->rendererInterface()->graphicsApi() == QSGRendererInterface::Software, "ui: software scene graph");
    settle(450);

    // Constellation.
    QImage img = win->grabWindow();
    check(img.size() == QSize(1920, 1080), "ui: grab is 1920x1080");
    check(forbiddenPixels(img) == 0, "ui: constellation has zero blue/green pixels", QString::number(forbiddenPixels(img)));
    check(qRed(img.pixel(3, 3)) == 0 && qGreen(img.pixel(3, 3)) == 0 && qBlue(img.pixel(3, 3)) == 0,
          "ui: background is #000000");
    check(findItems(win->contentItem(), "island").size() == 5, "ui: five fixture islands");
    {
        const int w = visibleWords(win);
        // Each island now carries two plain-English state lines (job + latest step, <= 9 words
        // each) instead of a sparkbar and a metric, so the budget is per island: <= 17 words.
        check(w < 90, "ui: constellation stays sparse", QString::number(w));
    }
    scanBanned(win, "constellation");

    // Keyboard focus ring: a 120 ms OutExpo glide on x/y only. Its size never changes, it lands
    // within ~150 ms, never overshoots the target, and stays inside the window.
    if (QQuickItem *ring = firstItem(win, "focusRing")) {
        const auto isl = findItems(win->contentItem(), "island");
        const QSizeF size0(ring->width(), ring->height());
        check(qAbs(size0.width() - (isl.value(0)->width() + 18)) < 0.01 && qAbs(size0.height() - (isl.value(0)->height() + 18)) < 0.01,
              "spring: ring is exactly islandW + 18 by islandH + 18",
              QStringLiteral("%1x%2").arg(size0.width()).arg(size0.height()));
        const double x0 = ring->x();
        const QRectF i1(isl.value(1)->x(), isl.value(1)->y(), isl.value(1)->width(), isl.value(1)->height());
        const double tx = i1.x() - 9;
        QElapsedTimer clock;
        sendKey(win, Qt::Key_Right);
        clock.start();
        bool sizeConstant = true, overshoot = false, inBounds = true;
        qint64 landedMs = -1;
        double lastX = x0;
        bool monotonic = true;
        while (clock.elapsed() < 400) {
            settle(4);
            const double x = ring->x();
            sizeConstant = sizeConstant && ring->width() == size0.width() && ring->height() == size0.height();
            overshoot = overshoot || (tx > x0 ? x > tx + 0.5 : x < tx - 0.5);
            monotonic = monotonic && (tx > x0 ? x >= lastX - 0.01 : x <= lastX + 0.01);
            inBounds = inBounds && x >= -1 && ring->y() >= -1 && x + ring->width() <= win->width() + 1
                && ring->y() + ring->height() <= win->height() + 1;
            lastX = x;
            if (landedMs < 0 && qAbs(x - tx) < 1.0)
                landedMs = clock.elapsed();
        }
        check(sizeConstant, "spring: ring width/height never animate during a hop");
        check(!overshoot && monotonic, "spring: hop is monotonic with zero overshoot (no wobble)",
              QStringLiteral("x0 %1 target %2 last %3").arg(x0).arg(tx).arg(lastX));
        check(landedMs >= 0 && landedMs <= 200, "spring: hop lands within 1 px in <= 200 ms (120 ms OutExpo)",
              QString::number(landedMs) + " ms");
        check(inBounds, "spring: ring stays inside the window while it moves");
        std::printf("info  focus ring hop landed within 1 px after %lld ms (wall clock, offscreen)\n",
                    static_cast<long long>(landedMs));
        const QRectF r(ring->x(), ring->y(), ring->width(), ring->height());
        check(qAbs(r.center().x() - i1.center().x()) < 2 && qAbs(r.center().y() - i1.center().y()) < 2,
              "ui: focus ring springs to island 2",
              QStringLiteral("ring %1,%2 island %3,%4").arg(r.center().x()).arg(r.center().y()).arg(i1.center().x()).arg(i1.center().y()));
        // Lift spring (22 / 0.92): the selected island's body grows to 1.03 without ringing.
        if (QQuickItem *body = findItems(isl.value(1), "islandBody").value(0)) {
            double peak = 0;
            for (int i = 0; i < 40; ++i) {
                settle(10);
                peak = std::max(peak, body->scale());
            }
            check(qAbs(body->scale() - 1.03) < 0.002 && peak < 1.03 + 0.004, "spring: island lift settles at 1.03, overshoot < 0.004",
                  QStringLiteral("now %1 peak %2").arg(body->scale()).arg(peak));
        } else {
            check(false, "spring: island body exists");
        }
        // Rapid hops (key repeat at 40 Hz): the ring retargets mid-flight and ends on the last island.
        for (int i = 0; i < 6; ++i) {
            sendKey(win, i % 2 ? Qt::Key_Left : Qt::Key_Right);
            settle(25);
        }
        settle(250);
        const int sel = app->property("selectedIsland").toInt();
        const QRectF is(isl.value(sel)->x(), isl.value(sel)->y(), isl.value(sel)->width(), isl.value(sel)->height());
        check(qAbs(ring->x() - (is.x() - 9)) < 1 && qAbs(ring->y() - (is.y() - 9)) < 1,
              "spring: rapid hops settle exactly on the selected island", QString::number(sel));
        // Active blinking dot on islands; no border waves, no uptime, no sparkbar.
        {
            settle(120);
            check(findItems(win->contentItem(), "islandWave").isEmpty(), "dot: border waves removed");
            const auto halos = findItems(win->contentItem(), "islandHalo");
            check(!halos.isEmpty(), "dot: active islands show the blinking pulse dot", QString::number(halos.size()));
            QStringList lines;
            for (QQuickItem *t : findItems(win->contentItem(), "islandLast"))
                if (t->isVisible())
                    lines << t->property("text").toString();
            bool noUptime = true;
            for (QQuickItem *t : findItems(win->contentItem(), "island")) {
                for (QQuickItem *m : findItems(t, "islandMetric"))
                    noUptime = noUptime && !m->isVisible();
            }
            check(!lines.isEmpty() && noUptime, "island: latest-step lines shown, metrics hidden unless pinned", lines.join(" | "));
            std::printf("info  island state lines: %s\n", qPrintable(lines.join(" | ")));
            const QString k0 = backend.projects().value(0).toMap().value("key").toString();
            backend.toggleMetricPin(k0);
            settle(30);
            QQuickItem *m0 = nullptr;
            for (QQuickItem *it : findItems(win->contentItem(), "island"))
                if (it->property("proj").toMap().value("key").toString() == k0)
                    m0 = findItems(it, "islandMetric").value(0);
            const bool hasMetric = !backend.projects().value(0).toMap().value("headline").toString().isEmpty();
            check(!hasMetric || (m0 && m0->isVisible()), "island: a pinned metric shows", hasMetric ? "metric" : "no metric");
            backend.toggleMetricPin(k0);
            settle(30);
        }
        // +N fresh badge: a background turn on an island that is not open.
        {
            Turn t;
            t.role = "assistant";
            t.text = "Still tuning the launch config.";  // no metric: the fixture's artifacts stay put
            backend.mockAppendTurn("kernel-search", t);
            settle(50);
            QString badge;
            for (QQuickItem *b : findItems(win->contentItem(), "freshBadgeText"))
                if (b->isVisible())
                    badge = b->property("text").toString();
            check(badge == "+1", "fresh: the island shows a +1 badge for a background turn", badge);
            backend.setFocusedProject("kernel-search");
            backend.setFocusedProject("");
            settle(30);
            int shown = 0;
            for (QQuickItem *b : findItems(win->contentItem(), "freshBadge"))
                shown += b->isVisible();
            check(shown == 0, "fresh: opening the island clears the badge");
        }
        {
            QStringList jobs;
            for (QQuickItem *j : findItems(win->contentItem(), "islandJob"))
                if (j->isVisible())
                    jobs << j->property("text").toString();
            check(!jobs.isEmpty(), "jobname: islands show plain-English job names", jobs.join(" | "));
            std::printf("info  island job names: %s\n", qPrintable(jobs.join(" | ")));
        }
        while (app->property("selectedIsland").toInt() != 0) {
            sendKey(win, Qt::Key_Left);
            settle(20);
        }
        settle(300);
    } else {
        check(false, "ui: focus ring exists");
    }

    // No title or "⌘N new" hint on the Constellation; Cmd+N still works (tested below).
    {
        QStringList texts;
        visibleWords(win, &texts);
        const QString all = texts.join('\n');
        check(!all.contains("lapis studio", Qt::CaseInsensitive) && !all.contains(QStringLiteral("⌘N")),
              "ui: constellation has no 'lapis studio' title or '⌘N  new' hint", all);
        QQuickItem *pill = firstItem(win, "navPill");
        check(pill && pill->isVisible() && pill->x() > 1700 && pill->y() > 1030, "ui: nav pill sits bottom-right",
              pill ? QStringLiteral("%1,%2").arg(pill->x()).arg(pill->y()) : "none");
    }

    // Input modes: controller (default) | vim | wasd | mouse. Arrow keys are a silent fallback, not a mode.
    {
        auto sel = [&] { return app->property("selectedIsland").toInt(); };
        auto letter = [&](int key, const char *t) { sendKey(win, key, Qt::NoModifier, QString::fromLatin1(t)); };
        check(backend.navMode() == "controller" && sel() == 0, "nav: starts in Controller mode on island 1");
        // Cmd+K opens the floating ask card now; Cmd+Shift+K cycles the keymap.
        sendKey(win, Qt::Key_K, Qt::ControlModifier | Qt::ShiftModifier);
        check(backend.navMode() == "vim" && sel() == 0, "nav: Cmd+Shift+K -> vim (Neovim)", backend.navMode());
        letter(Qt::Key_D, "d");
        check(sel() == 0, "nav: vim mode ignores wasd letters", QString::number(sel()));
        letter(Qt::Key_L, "l");
        const int afterL = sel();
        letter(Qt::Key_J, "j");
        const int afterJ = sel();
        letter(Qt::Key_K, "k");
        const int afterK = sel();
        letter(Qt::Key_H, "h");
        check(afterL == 1 && afterJ == 4 && afterK == 1 && sel() == 0, "nav: vim h/j/k/l move between islands",
              QStringLiteral("%1 %2 %3 %4").arg(afterL).arg(afterJ).arg(afterK).arg(sel()));
        sendKey(win, Qt::Key_G, Qt::ShiftModifier, "G");
        const int afterBigG = sel();
        letter(Qt::Key_G, "g");
        check(afterBigG == 4 && sel() == 0, "nav: vim G / g jump to the last / first island",
              QStringLiteral("%1 %2").arg(afterBigG).arg(sel()));
        sendKey(win, Qt::Key_Right);
        check(sel() == 1, "nav: arrow keys still work in vim mode");
        sendKey(win, Qt::Key_Left);
        sendKey(win, Qt::Key_K, Qt::ControlModifier | Qt::ShiftModifier);
        check(backend.navMode() == "wasd", "nav: Cmd+Shift+K -> wasd (Gaming)", backend.navMode());
        letter(Qt::Key_L, "l");
        check(sel() == 0, "nav: wasd mode ignores vim keys");
        letter(Qt::Key_D, "d");
        const int afterD = sel();
        letter(Qt::Key_S, "s");
        const int afterS = sel();
        letter(Qt::Key_W, "w");
        const int afterW = sel();
        letter(Qt::Key_A, "a");
        check(afterD == 1 && afterS == 4 && afterW == 1 && sel() == 0, "nav: wasd w/a/s/d move between islands",
              QStringLiteral("%1 %2 %3 %4").arg(afterD).arg(afterS).arg(afterW).arg(sel()));
        letter(Qt::Key_E, "e");
        settle(350);
        const int lensAfterE = app->property("lens").toInt();
        letter(Qt::Key_Q, "q");
        settle(350);
        check(lensAfterE > 0 && app->property("lens").toInt() == 0, "nav: wasd e interacts (zoom in), q goes back",
              QStringLiteral("%1 -> %2").arg(lensAfterE).arg(app->property("lens").toInt()));
        sendKey(win, Qt::Key_Down);
        check(sel() == 3, "nav: arrow keys still work in wasd mode", QString::number(sel()));
        sendKey(win, Qt::Key_Up);
        sendKey(win, Qt::Key_K, Qt::ControlModifier | Qt::ShiftModifier);
        check(backend.navMode() == "mouse", "nav: Cmd+Shift+K -> mouse", backend.navMode());
        letter(Qt::Key_L, "l");
        letter(Qt::Key_D, "d");
        check(sel() == 0, "nav: mouse mode has no letter navigation");
        sendKey(win, Qt::Key_K, Qt::ControlModifier | Qt::ShiftModifier);
        check(backend.navMode() == "controller" && sel() == 0, "nav: Cmd+Shift+K wraps back to controller");
        settle(700);
    }

    // First-launch onboarding modal (explicit trigger: offscreen never shows it on its own).
    {
        QQuickItem *modal = firstItem(win, "onboardingModal");
        check(modal && !modal->isVisible(), "onboarding: hidden in --test without a trigger");
        backend.showOnboarding();
        settle(300);
        check(modal && modal->isVisible() && modal->property("choice").toInt() == 0,
              "onboarding: shows with Controller Mode (Recommended) pre-selected");
        const auto cards = findItems(win->contentItem(), "onboardingCard");
        QStringList texts;
        visibleWords(win, &texts);
        const QString all = texts.join('\n');
        check(cards.size() == 4 && all.contains("Controller Mode") && all.contains("Neovim Mode")
                  && all.contains("Gaming Mode") && all.contains("Mouse Mode") && all.contains("Recommended")
                  && !all.contains("Arrow", Qt::CaseInsensitive),
              "onboarding: four cards (Controller / Neovim / Gaming / Mouse), no arrows mode",
              QString::number(cards.size()));
        const QImage shot = win->grabWindow();
        check(forbiddenPixels(shot) == 0 && qRed(shot.pixel(4, 4)) == 0 && qGreen(shot.pixel(4, 4)) == 0,
              "onboarding: #000000 + gold, no blue/green");
        QStringList pairs;
        check(textOverlaps(win, &pairs) == 0, "onboarding: no overlapping text", pairs.join("; "));
        sendKey(win, Qt::Key_J, Qt::NoModifier, "j");
        const int afterJ = modal->property("choice").toInt();
        sendKey(win, Qt::Key_S, Qt::NoModifier, "s");
        const int afterS = modal->property("choice").toInt();
        sendKey(win, Qt::Key_K, Qt::NoModifier, "k");
        check(afterJ == 1 && afterS == 2 && modal->property("choice").toInt() == 1, "onboarding: j/s/k switch cards",
              QStringLiteral("%1 %2 %3").arg(afterJ).arg(afterS).arg(modal->property("choice").toInt()));
        check(app->property("selectedIsland").toInt() == 0, "onboarding: keys never reach the islands underneath");
        sendKey(win, Qt::Key_W, Qt::NoModifier, "w");
        sendKey(win, Qt::Key_Return);
        settle(300);
        check(!backend.firstLaunch() && backend.navMode() == "controller" && !modal->isVisible(),
              "onboarding: Enter confirms Controller and closes", backend.navMode());
    }

    // Cmd+, Command Settings: Mode, UI Scale, Default Model & Effort, keybinds.
    {
        sendKey(win, Qt::Key_Comma, Qt::ControlModifier);
        settle(250);
        QQuickItem *ov = firstItem(win, "settingsOverlay");
        check(ov && ov->isVisible() && ov->property("open").toBool(), "settings: Cmd+, opens the overlay");
        QStringList texts;
        visibleWords(win, &texts);
        const QString all = texts.join('\n');
        // Rows run most-used first; the Model row shows every model in the lineup at once.
        const QStringList captions = {"Model", "Default Effort", "Controller", "Ambient Voice", "Navigation Mode",
                                      "Theme", "UI Scale", "Code Diffs", "Model Suggestions"};
        int lastAt = -1;
        bool ordered = true;
        for (const QString &c : captions) {
            const int at = int(texts.indexOf(c));
            ordered = ordered && at > lastAt;
            lastAt = at;
        }
        check(ordered, "settings: rows ranked Model, Effort, Controller, Ambient, Nav, Theme, Scale, Diffs, Suggestions",
              texts.mid(0, 40).join(" | "));
        QStringList missing;
        for (const char *m : {"DeepSeek V4.1 Flash", "Inkling 2 Mini", "Qwen3.8 Flash Next", "Fable 5.2", "Opus 5.5",
                              "Sonnet 5.5", "GPT 6.2 Astra", "GPT 6.2 Sol", "Gemini 4 Carbon", "Gemini 4 Neon",
                              "Gemini 4 Hydrogen"})
            if (!texts.contains(QString::fromUtf8(m)))
                missing << m;
        check(missing.isEmpty() && findItems(win->contentItem(), "settingsModelChip").size() == 11
                  && !all.contains("Luna") && !all.contains("Gemini 3"),
              "settings: Model grid shows all 11 models at once, no retired ones", missing.join(","));
        check(all.contains("Neovim") && all.contains("Gaming") && all.contains("Mouse")
                  && !all.contains("arrows", Qt::CaseInsensitive),
              "settings: Navigation Mode shows Controller / Neovim / Gaming / Mouse only");
        check(all.contains("1.00x") && all.contains("0.75x – 1.60x") && all.contains("wheel"),
              "settings: UI scale with range and wheel hint");
        check(all.contains("confirm") && all.contains("settings"),
              "settings: Controller keybind cheat-sheet");
        QStringList pairs;
        check(textOverlaps(win, &pairs) == 0, "settings: no overlapping text", pairs.join("; "));
        check(forbiddenPixels(win->grabWindow()) == 0, "settings: no blue/green pixels");
        if (QQuickItem *card = firstItem(win, "settingsCard")) {
            const QRectF r = card->mapRectToScene(QRectF(0, 0, card->width(), card->height()));
            check(r.top() >= 0 && r.bottom() <= win->height(), "settings: the card fits the window",
                  QStringLiteral("%1..%2 of %3").arg(r.top()).arg(r.bottom()).arg(win->height()));
        }
        sendKey(win, Qt::Key_L, Qt::NoModifier, "l");  // row 0: Model -> next in the lineup
        check(backend.chatModel() == "gpt-6-astra", "settings: l on Model -> GPT 6.2 Astra", backend.chatModel());
        sendKey(win, Qt::Key_H, Qt::NoModifier, "h");
        sendKey(win, Qt::Key_J, Qt::NoModifier, "j");  // row 1: effort
        sendKey(win, Qt::Key_L, Qt::NoModifier, "l");
        check(backend.chatEffort() == "medium", "settings: l on Default Effort -> medium", backend.chatEffort());
        sendKey(win, Qt::Key_H, Qt::NoModifier, "h");
        for (int i = 0; i < 3; ++i)
            sendKey(win, Qt::Key_J, Qt::NoModifier, "j");  // row 4: navigation mode
        sendKey(win, Qt::Key_L, Qt::NoModifier, "l");
        settle(50);
        check(backend.navMode() == "vim", "settings: l on Navigation Mode switches to Neovim", backend.navMode());
        texts.clear();
        visibleWords(win, &texts);
        check(texts.join('\n').contains("h j k l"), "settings: cheat-sheet follows the mode");
        sendKey(win, Qt::Key_H, Qt::NoModifier, "h");
        sendKey(win, Qt::Key_J, Qt::NoModifier, "j");
        sendKey(win, Qt::Key_J, Qt::NoModifier, "j");  // row 6: UI scale
        sendKey(win, Qt::Key_L, Qt::NoModifier, "l");
        check(qFuzzyCompare(backend.uiScale(), 1.1), "settings: l on UI Scale zooms 10%", QString::number(backend.uiScale()));
        sendKey(win, Qt::Key_H, Qt::NoModifier, "h");
        check(app->property("selectedIsland").toInt() == 0 && app->property("lens").toInt() == 0,
              "settings: keys stay in the overlay");
        sendKey(win, Qt::Key_Escape);
        settle(250);
        check(ov && !ov->isVisible() && backend.navMode() == "controller" && backend.chatModel() == "claude-sonnet-5-5"
                  && qFuzzyCompare(backend.uiScale(), 1.0),
              "settings: Esc closes; values as set", backend.navMode() + " " + backend.chatModel());
        settle(1400);  // scale pill fades
    }

    // Settings Theme row: l -> Lapis Blue (blue accents appear), h -> Gold (zero blue again).
    // Every forbiddenPixels check in this suite assumes gold, so this block must end on gold.
    {
        sendKey(win, Qt::Key_Comma, Qt::ControlModifier);
        settle(250);
        QQuickItem *ov = firstItem(win, "settingsOverlay");
        for (int i = 0; i < 5; ++i)
            sendKey(win, Qt::Key_J, Qt::NoModifier, "j");
        QStringList texts;
        visibleWords(win, &texts);
        check(ov && ov->property("row").toInt() == 5 && texts.join('\n').contains("Lapis Blue"),
              "theme: Settings shows a Theme row (Gold / Lapis Blue) at row 5");
        check(forbiddenPixels(win->grabWindow()) == 0, "theme: gold mode shows no blue, not even in the Theme row");
        sendKey(win, Qt::Key_Right);
        settle(250);
        const QImage lapisShot = win->grabWindow();
        int accent = 0;
        for (int y = 0; y < lapisShot.height(); ++y)
            for (int x = 0; x < lapisShot.width(); ++x) {
                const QRgb c = lapisShot.pixel(x, y);
                accent += std::abs(qRed(c) - 0x4f) < 10 && std::abs(qGreen(c) - 0x8c) < 10 && std::abs(qBlue(c) - 0xff) < 10;
            }
        check(backend.theme() == "lapis" && accent > 20, "theme: → switches to Lapis Blue; #4f8cff accent on screen",
              QStringLiteral("%1 px").arg(accent));
        check(qRed(lapisShot.pixel(4, 4)) == 0 && qGreen(lapisShot.pixel(4, 4)) == 0 && qBlue(lapisShot.pixel(4, 4)) == 0,
              "theme: lapis keeps the #000000 OLED background");
        sendKey(win, Qt::Key_Left);
        settle(250);
        check(backend.theme() == "gold" && forbiddenPixels(win->grabWindow()) == 0,
              "theme: ← back to Gold; zero blue/green pixels again", backend.theme());
        sendKey(win, Qt::Key_Escape);
        settle(250);
    }

    // Settings Jarvis Bar row: last (row 9), so no older row index moved; ← / → pick the mode.
    {
        sendKey(win, Qt::Key_Comma, Qt::ControlModifier);
        settle(250);
        QQuickItem *ov = firstItem(win, "settingsOverlay");
        for (int i = 0; i < 9; ++i)
            sendKey(win, Qt::Key_J, Qt::NoModifier, "j");
        QStringList texts;
        visibleWords(win, &texts);
        const QString all = texts.join('\n');
        check(ov && ov->property("row").toInt() == 9 && ov->property("jarvisRow").toInt() == 9
                  && ov->property("rowCount").toInt() == 10 && all.contains("Jarvis Bar") && all.contains("Unified Overlay")
                  && all.contains("Split Studio / Desktop"),
              "jarvis settings: a Jarvis Bar row (Unified Overlay / Split Studio / Desktop) at the end");
        const QString before = backend.jarvisBarMode();
        sendKey(win, Qt::Key_Left);
        check(backend.jarvisBarMode() == "overlay", "jarvis settings: ← picks Unified Overlay", backend.jarvisBarMode());
        sendKey(win, Qt::Key_Right);
        check(backend.jarvisBarMode() == "split", "jarvis settings: → picks Split Studio / Desktop", backend.jarvisBarMode());
        check(forbiddenPixels(win->grabWindow()) == 0, "jarvis settings: the row has zero blue/green pixels");
        backend.setJarvisBarMode(before);
        sendKey(win, Qt::Key_Escape);
        settle(250);
    }

    // Morning Briefing: B on the Constellation; changed / active projects only, two lines each.
    {
        QObject *mb = win->findChild<QObject *>("morningBriefing");
        check(mb && !mb->property("open").toBool(), "briefing: closed by default (no auto-trigger offscreen)");
        sendKey(win, Qt::Key_B, Qt::NoModifier, "b");
        settle(250);
        const auto rows = findItems(win->contentItem(), "briefingRow");
        int expected = 0;
        for (const QVariant &v : backend.projects()) {
            const QVariantMap p = v.toMap();
            expected += p.value("active").toBool() || p.value("fresh").toInt() > 0 || p.value("status").toString() == "needs-ok"
                        || p.value("blocker").toBool();
        }
        check(mb && mb->property("open").toBool() && backend.briefingSource() == "local" && !backend.briefingBusy(),
              "briefing: B opens it; mock uses the local latest-turn summary", backend.briefingSource());
        check(!rows.isEmpty() && rows.size() == std::min(expected, 8), "briefing: one row per changed / active project",
              QStringLiteral("%1 rows, %2 expected").arg(rows.size()).arg(expected));
        bool twoLines = true;
        QStringList shown;
        for (QQuickItem *r : rows) {
            QQuickItem *l1 = findItems(r, "briefingLine1").value(0), *l2 = findItems(r, "briefingLine2").value(0);
            const QString a = l1 ? stripMarkup(l1->property("text").toString()) : QString();
            const QString b = l2 ? l2->property("text").toString() : QString();
            twoLines = twoLines && a.size() > 6 && b.size() > 4 && l1->property("lineCount").toInt() == 1
                       && l2->property("lineCount").toInt() == 1;
            shown << a + " // " + b;
        }
        check(twoLines, "briefing: each row is exactly two non-empty lines", shown.join(" | "));
        std::printf("info  briefing: %s\n", qPrintable(shown.join(" | ")));
        check(forbiddenPixels(win->grabWindow()) == 0, "briefing: zero blue/green pixels");
        sendKey(win, Qt::Key_Down);
        settle(30);
        check(mb && mb->property("row").toInt() == (rows.size() > 1 ? 1 : 0) && app->property("lens").toInt() == 0,
              "briefing: Down moves the row, keys stay in the overlay");
        sendKey(win, Qt::Key_Escape);
        settle(250);
        check(mb && !mb->property("open").toBool() && app->property("lens").toInt() == 0, "briefing: Esc closes it");
        settle(300);
    }

    // Tab predictor on the Constellation: gold ring + "Open Stage ⇥ Tab" on the predicted island.
    {
        settle(700);  // idle: the ring breathes in
        QQuickItem *ring = firstItem(win, "predictorRingShape");
        QQuickItem *badge = firstItem(win, "tabBadgeLabel");
        QQuickItem *badgeBox = firstItem(win, "tabBadge");
        const QVariantMap p = app->property("predictedAction").toMap();
        const int target = p.value("index").toInt();
        const auto isl = findItems(win->contentItem(), "island");
        check(p.value("actionKind").toString() == "open" && p.value("targetId").toString().startsWith("island:"),
              "predictor: Constellation predicts an island", p.value("targetId").toString());
        check(ring && firstItem(win, "predictorRing")->opacity() > 0.9 && badge && badge->isVisible(),
              "predictor: gold ring / badge visible when idle");
        // On the Constellation the badge is an opaque tab tucked onto the island's top border
        // (it covers the ring line under it), so it never climbs into a cluster label or the
        // card above.
        if (badgeBox && isl.value(target)) {
            const QRectF br = badgeBox->mapRectToScene(QRectF(0, 0, badgeBox->width(), badgeBox->height()));
            QQuickItem *it = isl.value(target);
            const QRectF ir = it->mapRectToScene(QRectF(0, 0, it->width(), it->height()));
            check(qAbs(br.center().y() - ir.top()) < 1.5 && qAbs(br.center().x() - ir.center().x()) < 1.5,
                  "predictor: tabBadge tucked on the predicted island's top border",
                  QStringLiteral("badge cy %1 island top %2").arg(br.center().y()).arg(ir.top()));
        }
        QStringList chrome;
        check(mapChromeCollisions(win, &chrome) == 0, "predictor: badge / hint clear every cluster label and other card",
              chrome.join("; "));
        if (ring && isl.value(target)) {
            const QRectF rr = ring->mapRectToScene(QRectF(0, 0, ring->width(), ring->height()));
            const QRectF ir = isl.value(target)->mapRectToScene(QRectF(0, 0, isl.value(target)->width(), isl.value(target)->height()));
            check(qAbs(rr.center().x() - ir.center().x()) < 3 && qAbs(rr.center().y() - ir.center().y()) < 3 && rr.contains(ir),
                  "predictor: ring wraps the predicted island",
                  QStringLiteral("ring %1,%2 island %3,%4").arg(rr.center().x()).arg(rr.center().y()).arg(ir.center().x()).arg(ir.center().y()));
        }
        check(badge && badge->property("text").toString() == p.value("label").toString()
                  && (p.value("label").toString() == "Open Stage" || p.value("label").toString() == "Open Ladder"),
              "predictor: badge reads the action + ⇥ Tab", badge ? badge->property("text").toString() : "none");
        {
            const QImage shot = win->grabWindow();
            int gold = 0;
            const QRectF rr = ring ? ring->mapRectToScene(QRectF(0, 0, ring->width(), ring->height())) : QRectF();
            for (int x = int(rr.left()) + 30; x < int(rr.right()) - 30; ++x) {
                const QRgb c = shot.pixel(x, int(rr.top()) + 1);
                gold += qRed(c) > 150 && qGreen(c) > 90 && qGreen(c) < 190 && qBlue(c) < 110 ? 1 : 0;
            }
            check(gold > 20, "predictor: ring pixels are warm gold (#e09f3e)", QString::number(gold));
            QStringList pairs;
            check(textOverlaps(win, &pairs) == 0, "predictor: badge overlaps no text", pairs.join("; "));
        }
        // It breathes only while idle; a key press hides it until the user pauses again.
        const qreal ph0 = firstItem(win, "predictorRing")->property("phase").toReal();
        settle(200);
        const qreal ph1 = firstItem(win, "predictorRing")->property("phase").toReal();
        check(qAbs(ph1 - ph0) > 0.01, "predictor: ring pulses while idle", QStringLiteral("%1 -> %2").arg(ph0).arg(ph1));
        sendKey(win, Qt::Key_L, Qt::NoModifier, "l");
        sendKey(win, Qt::Key_H, Qt::NoModifier, "h");
        settle(300);
        check(firstItem(win, "predictorRing")->opacity() < 0.5, "predictor: ring rests while keys are pressed");
        settle(700);
        sendKey(win, Qt::Key_Tab);
        settle(450);
        check(app->property("lens").toInt() > 0 && backend.projectIndex(app->property("projectKey").toString()) == target
                  && app->property("predictorAccepts").toInt() == 1,
              "predictor: Tab opens the predicted island", app->property("projectKey").toString());
        sendKey(win, Qt::Key_Escape);
        settle(450);
        const QVariantMap p2 = app->property("predictedAction").toMap();
        check(p2.value("index").toInt() != target, "predictor: a visited island is not predicted again",
              p2.value("targetId").toString());
    }

    // Controller hints + the element-aware focus cursor on the Constellation.
    {
        settle(700);
        QQuickItem *key = firstItem(win, "tabBadgeKey");
        check(backend.navMode() == "controller" && key && key->property("text").toString() == "R1",
              "hints: Controller mode badge shows only the bound button (R1), no Tab or ⇥",
              key ? key->property("text").toString() : "none");
        backend.setNavMode("vim");
        settle(50);
        check(key && key->property("text").toString() == "⇥ Tab", "hints: keyboard mode badge reads ⇥ Tab",
              key ? key->property("text").toString() : "none");
        backend.setNavMode("controller");
        settle(700);
        QQuickItem *ring = firstItem(win, "focusCursorRing");
        QQuickItem *hint = firstItem(win, "focusCursorHintText");
        const auto isl = findItems(win->contentItem(), "island");
        auto islandScene = [&](int i) {
            QQuickItem *it = isl.value(i);
            return it ? it->parentItem()->mapRectToScene(QRectF(it->x(), it->y(), it->width(), it->height())) : QRectF();
        };
        auto ringScene = [&] { return ring->mapRectToScene(QRectF(0, 0, ring->width(), ring->height())); };
        const int sel = app->property("selectedIsland").toInt();
        check(ring && ring->isVisible() && firstItem(win, "focusCursor")->opacity() > 0.9, "cursor: visible on the Constellation");
        if (ring && isl.value(sel)) {
            const QRectF rr = ringScene(), ir = islandScene(sel);
            check(qAbs(rr.center().x() - ir.center().x()) < 1.5 && qAbs(rr.center().y() - ir.center().y()) < 1.5 && rr.contains(ir),
                  "cursor: ring wraps the selected island",
                  QStringLiteral("ring %1,%2 island %3,%4").arg(rr.center().x()).arg(rr.center().y()).arg(ir.center().x()).arg(ir.center().y()));
            QObject *pen = ring->property("border").value<QObject *>();
            check(pen && pen->property("color").value<QColor>() == QColor("#e09f3e"), "cursor: ring border is #e09f3e");
            check(rr.width() - ir.width() > 20, "cursor: ring opens up around the element once it lands",
                  QString::number(rr.width() - ir.width()));
        }
        check(hint && hint->isVisible() && hint->property("text").toString() == "✕ Expand",
              "cursor: contextual hint reads ✕ Expand on an island", hint ? hint->property("text").toString() : "none");
        QQuickItem *fr = firstItem(win, "focusRing");
        check(fr && fr->opacity() < 0.01, "cursor: the old island ring yields to the cursor (one ring, not two)");
        {
            QStringList pairs;
            check(textOverlaps(win, &pairs) == 0, "cursor: hint chip overlaps no text", pairs.join("; "));
        }
        // Spring: critically damped at a fixed 240 Hz step, so a hop never overshoots and lands fast.
        if (ring && isl.size() > sel + 1) {
            const double x0 = ringScene().center().x();
            const double tx = islandScene(sel + 1).center().x();
            sendKey(win, Qt::Key_Right);
            QElapsedTimer clock;
            clock.start();
            bool overshoot = false;
            qint64 landed = -1;
            while (clock.elapsed() < 600) {
                settle(4);
                const double x = ringScene().center().x();
                overshoot = overshoot || (tx > x0 ? x > tx + 0.5 : x < tx - 0.5);
                if (landed < 0 && qAbs(x - tx) < 1.0)
                    landed = clock.elapsed();
            }
            check(!overshoot, "cursor: spring hop has zero overshoot", QStringLiteral("x0 %1 target %2").arg(x0).arg(tx));
            check(landed >= 0 && landed <= 400, "cursor: spring hop lands within 1 px in <= 400 ms", QString::number(landed) + " ms");
            std::printf("info  focus cursor hop landed within 1 px after %lld ms (wall clock, offscreen)\n",
                        static_cast<long long>(landed));
            sendKey(win, Qt::Key_Left);
            settle(500);
        }
        // Keyboard modes speak keys: ⇥ Tab, ⏎ / E to expand. Mouse mode has no cursor.
        backend.setNavMode("vim");
        settle(50);
        check(key && key->property("text").toString() == "⇥ Tab", "hints: vim mode badge reads ⇥ Tab",
              key ? key->property("text").toString() : "none");
        check(hint && hint->property("text").toString() == "⏎ Expand", "cursor: vim hint reads ⏎ Expand",
              hint ? hint->property("text").toString() : "none");
        backend.setNavMode("wasd");
        settle(50);
        check(hint && hint->property("text").toString() == "E Expand", "cursor: wasd hint reads E Expand",
              hint ? hint->property("text").toString() : "none");
        backend.setNavMode("mouse");
        settle(300);
        check(firstItem(win, "focusCursor")->opacity() < 0.05 && fr && fr->opacity() > 0.9,
              "cursor: mouse mode hides the cursor, the plain island ring returns");
        backend.setNavMode("controller");
        settle(300);
    }

    // UI zoom: plain wheel on the Constellation, Cmd +/-/0, the 1.2 s pill, islands and text grow.
    {
        QQuickItem *isl0 = findItems(win->contentItem(), "island").value(0);
        const qreal w0 = isl0 ? isl0->width() : 0;
        const QPointF centre(960, 540);
        QWheelEvent wheel(centre, win->mapToGlobal(centre), QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier,
                          Qt::NoScrollPhase, false);
        QCoreApplication::sendEvent(win, &wheel);
        check(backend.uiScale() > 1.1 && backend.uiScale() < 1.2, "zoom: plain wheel on the Constellation zooms in",
              QString::number(backend.uiScale()));
        settle(100);
        QQuickItem *pill = firstItem(win, "scalePill");
        check(pill && pill->isVisible() && firstItem(win, "scalePillText")
                  && firstItem(win, "scalePillText")->property("text").toString() == QString::number(backend.uiScale(), 'f', 2) + "x",
              "zoom: scale pill shows the new scale",
              firstItem(win, "scalePillText") ? firstItem(win, "scalePillText")->property("text").toString() : "none");
        check(isl0 && isl0->width() > w0 + 1, "zoom: islands grow with the scale",
              QStringLiteral("%1 -> %2").arg(w0).arg(isl0 ? isl0->width() : 0));
        sendKey(win, Qt::Key_0, Qt::ControlModifier);
        check(qFuzzyCompare(backend.uiScale(), 1.0), "zoom: Cmd+0 resets", QString::number(backend.uiScale()));
        sendKey(win, Qt::Key_Equal, Qt::ControlModifier);
        check(qFuzzyCompare(backend.uiScale(), 1.1), "zoom: Cmd+= zooms in 10%", QString::number(backend.uiScale()));
        sendKey(win, Qt::Key_Minus, Qt::ControlModifier);
        sendKey(win, Qt::Key_Minus, Qt::ControlModifier);
        check(backend.uiScale() < 0.92 && backend.uiScale() > 0.88, "zoom: Cmd+- zooms out", QString::number(backend.uiScale()));
        for (const double s : {Backend::kMaxScale, Backend::kMinScale}) {
            backend.setUiScale(s);
            settle(120);
            QStringList pairs;
            check(textOverlaps(win, &pairs) == 0, "zoom: no overlapping text on the Constellation at the scale limits",
                  QStringLiteral("%1x: %2").arg(s).arg(pairs.join("; ")));
        }
        sendKey(win, Qt::Key_0, Qt::ControlModifier);
        settle(1700);
        check(pill && !pill->isVisible(), "zoom: scale pill fades after 1.2 s");
        check(isl0 && qAbs(isl0->width() - w0) < 0.5, "zoom: islands back to size at 1.0x");
    }

    // Zoom into project 1 -> Stage by default, as a shared-element zoom out of island 1.
    QQuickItem *morph = firstItem(win, "zoomMorph");
    QQuickItem *island1 = findItems(win->contentItem(), "island").value(0);
    check(morph && !morph->isVisible(), "ui: zoom layer hidden on the Constellation");
    sendKey(win, Qt::Key_1);
    {
        const QRectF ir = app->property("islandRect").toRectF();
        check(island1 && qAbs(ir.x() - island1->x()) < 0.5 && qAbs(ir.y() - island1->y()) < 0.5
                  && qAbs(ir.width() - island1->width()) < 0.5 && qAbs(ir.height() - island1->height()) < 0.5,
              "ui: zoom starts from the clicked island's geometry",
              QStringLiteral("%1,%2 %3x%4").arg(ir.x()).arg(ir.y()).arg(ir.width()).arg(ir.height()));
        settle(70);
        const qreal w = morph ? morph->width() : 0;
        check(w > ir.width() + 1 && w < 1919, "ui: mid-zoom the island card is growing", QString::number(w));
        if (island1 && island1->parentItem()) {
            const auto isl = findItems(win->contentItem(), "island");
            check(isl.value(1)->scale() < 0.999 && isl.value(1)->opacity() < 0.999, "ui: siblings recede mid-zoom",
                  QStringLiteral("scale %1 opacity %2").arg(isl.value(1)->scale()).arg(isl.value(1)->opacity()));
        }
    }
    settle(400);
    check(morph && qAbs(morph->width() - 1920) < 0.5 && qAbs(morph->height() - 1080) < 0.5 && morph->x() == 0,
          "ui: zoom lands full-bleed");
    {
        const auto isl = findItems(win->contentItem(), "island");
        check(qAbs(isl.value(1)->scale() - 0.94) < 0.002, "ui: siblings settle at scale 0.94",
              QString::number(isl.value(1)->scale()));
    }
    check(app->property("lens").toInt() == 2, "ui: digit zooms into Stage", app->property("lens").toString());
    img = win->grabWindow();
    check(forbiddenPixels(img) == 0, "ui: stage has zero blue/green pixels", QString::number(forbiddenPixels(img)));
    {
        QStringList texts;
        const int w = visibleWords(win, &texts);
        // 35 before the hyperparameter pills and the plain-English job name joined the header.
        check(w < 40, "ui: default Stage shows < 40 words", QStringLiteral("%1: %2").arg(w).arg(texts.join(" | ")));
        std::printf("info  default Stage visible words: %d\n", w);
    }
    // Hyperparameter / term pills under the header; Cmd+P, l, Enter scopes a quick question.
    {
        const auto pills = findItems(win->contentItem(), "paramPill");
        QStringList labels;
        for (QQuickItem *p : pills)
            if (QQuickItem *t = findItems(p, "paramPillText").value(0))
                labels << t->property("text").toString();
        check(pills.size() >= 2, "pills: the Stage shows hyperparameter pills", labels.join(" | "));
        QQuickItem *header = firstItem(win, "headerJob");
        check(header && header->isVisible() && header->property("text").toString().size() > 6,
              "jobname: the Stage header shows the plain-English job name", header ? header->property("text").toString() : "none");
        QQuickItem *card = firstItem(win, "paramCard");
        check(card && !card->isVisible(), "pills: no card until a pill is picked");
        sendKey(win, Qt::Key_P, Qt::ControlModifier);
        settle(60);
        QQuickItem *detail = firstItem(win, "paramDetail");
        check(card && card->isVisible() && detail && detail->property("text").toString().size() > 20,
              "pills: Cmd+P opens the first pill's card with a plain-English line",
              detail ? detail->property("text").toString() : "none");
        const QString first = firstItem(win, "paramCardName") ? firstItem(win, "paramCardName")->property("text").toString() : "";
        sendKey(win, Qt::Key_Right);
        settle(60);
        const QString second = firstItem(win, "paramCardName") ? firstItem(win, "paramCardName")->property("text").toString() : "";
        check(!first.isEmpty() && !second.isEmpty() && first != second, "pills: l moves to the next pill's card",
              first + " -> " + second);
        // The card floats: the artifact title does not move while it is open.
        QQuickItem *title = firstItem(win, "stageTitle");
        const double titleY = title ? title->mapToScene(QPointF(0, 0)).y() : -1;
        sendKey(win, Qt::Key_Return);
        settle(60);
        QQuickItem *chip = firstItem(win, "scopeChip");
        check(chip && chip->isVisible() && firstItem(win, "scopeChipText")->property("text").toString().contains(second),
              "pills: Enter scopes the composer to the parameter", chip ? firstItem(win, "scopeChipText")->property("text").toString() : "none");
        check(!card->isVisible() && title && qAbs(title->mapToScene(QPointF(0, 0)).y() - titleY) < 0.5,
              "pills: card closes, artifact never moved");
        typeText(win, "why?");
        sendKey(win, Qt::Key_Return);
        settle(30);
        check(backend.quickQuestion().startsWith("About " + second) && backend.quickQuestion().endsWith(": why?"),
              "pills: the quick question is scoped to the parameter", backend.quickQuestion());
        check(!chip->isVisible(), "pills: the scope is spent after one question");
        for (int i = 0; i < 8; ++i)
            backend.advance(0.25);
        backend.clearAnswers();
        QMetaObject::invokeMethod(app, "takeFocus");
        settle(100);
        check(app->property("lens").toInt() == 2, "pills: still on the Stage afterwards");
    }
    // Regression: with a term card down, the focus cursor must never draw the big Stage ring
    // (or its halo) through the card. Both ways in: a click-style open and Cmd+P.
    {
        QObject *bar = firstItem(win, "termZoomBar");
        QQuickItem *card = firstItem(win, "paramCard");
        QQuickItem *ring = firstItem(win, "focusCursorRing");
        auto ringScene = [&] {
            // The ring plus its landing halo (4 px) plus the 2 px stroke: everything it draws.
            return ring->mapRectToScene(QRectF(0, 0, ring->width(), ring->height())).adjusted(-6, -6, 6, 6);
        };
        auto cardScene = [&] { return card->mapRectToScene(QRectF(0, 0, card->width(), card->height())); };
        auto pillScene = [&](int i) {
            QQuickItem *p = findItems(win->contentItem(), "paramPill").value(i);
            return p ? p->mapRectToScene(QRectF(0, 0, p->width(), p->height())) : QRectF();
        };
        if (bar && card && ring) {
            settle(500);
            const QRectF before = ringScene();
            bar->setProperty("expanded", 0);  // what a click on the first pill does
            settle(600);
            const QRectF rr = ringScene(), cr = cardScene();
            check(card->isVisible() && !rr.intersects(cr), "cursor: clicked term card is clear of the focus ring",
                  QStringLiteral("ring %1,%2 %3x%4 card %5,%6 %7x%8").arg(rr.x()).arg(rr.y()).arg(rr.width()).arg(rr.height())
                      .arg(cr.x()).arg(cr.y()).arg(cr.width()).arg(cr.height()));
            check(rr.contains(pillScene(0)) && rr.height() < before.height() / 2,
                  "cursor: wraps the open pill, not the artifact behind the card");
            QColor bg = card->property("color").value<QColor>();
            check(bg == QColor("#0b0a08") && bg.alpha() == 255 && card->z() >= 60,
                  "pills: card is opaque #0b0a08 at z 60", bg.name() + " z " + QString::number(card->z()));
            bar->setProperty("expanded", -1);
            settle(300);
            sendKey(win, Qt::Key_P, Qt::ControlModifier);
            sendKey(win, Qt::Key_Right);
            settle(600);
            const QRectF rk = ringScene(), ck = cardScene();
            check(card->isVisible() && !rk.intersects(ck), "cursor: Cmd+P term card is clear of the focus ring");
            sendKey(win, Qt::Key_Escape);
            settle(300);
            QMetaObject::invokeMethod(app, "takeFocus");
            settle(300);
            check(!card->isVisible() && app->property("lens").toInt() == 2, "pills: card closed, still on the Stage");
        } else {
            check(false, "cursor: term bar, card and ring exist");
        }
    }
    if (auto *title = firstItem(win, "stageTitle")) {
        const QFont f = title->property("font").value<QFont>();
        check(QFontInfo(f).family() == "Avenir Next", "ui: title renders in Avenir Next", QFontInfo(f).family());
        check(countWords(title->property("text").toString()) <= 5, "ui: stage title <= 5 words");
    } else {
        check(false, "ui: stage title exists");
    }
    {
        QFont mono(QStringLiteral("Menlo"));
        check(QFontInfo(mono).family() == "Menlo", "ui: Menlo resolves", QFontInfo(mono).family());
    }
    check(firstItem(win, "chartView") != nullptr, "ui: first artifact is a chart");
    {
        // On the Stage the cursor wraps the artifact and offers ✕ Prompt; the composer speaks pad.
        settle(500);
        QQuickItem *hint = firstItem(win, "focusCursorHintText");
        check(hint && hint->isVisible() && hint->property("text").toString() == "✕ Prompt",
              "cursor: Stage hint reads ✕ Prompt", hint ? hint->property("text").toString() : "none");
        QQuickItem *ring = firstItem(win, "focusCursorRing");
        QQuickItem *chart = firstItem(win, "chartView");
        if (ring && chart) {
            const QRectF rr = ring->mapRectToScene(QRectF(0, 0, ring->width(), ring->height()));
            const QRectF cr = chart->mapRectToScene(QRectF(0, 0, chart->width(), chart->height()));
            check(rr.contains(cr.adjusted(2, 2, -2, -2)), "cursor: ring wraps the Stage artifact",
                  QStringLiteral("ring %1x%2 chart %3x%4").arg(rr.width()).arg(rr.height()).arg(cr.width()).arg(cr.height()));
        }
        QQuickItem *hints = firstItem(win, "composerHints");
        const QString ht = hints ? hints->property("text").toString() : QString();
        check(ht.contains("L2 Voice") && !ht.contains("○ Back") && !ht.contains("⏎"),
              "hints: composer shows controller glyphs in Controller mode", ht);
        QQuickItem *composer = firstItem(win, "composer");
        QMetaObject::invokeMethod(composer, "focusInput", Q_ARG(QVariant, QVariant(QStringLiteral("why"))));
        settle(50);
        const QString hf = hints ? hints->property("text").toString() : QString();
        check(hf.contains("R2 Send") && hf.contains("✕ Select") && hf.contains("○ Back") && !hf.contains("⏎"),
              "hints: typing shows R2 Send, ✕ Select, ○ Back", hf);
        // The cursor follows focus into the composer (no chip there: the bar carries its own hints).
        QQuickItem *bar = firstItem(win, "composerBar");
        if (ring && bar) {
            settle(450);
            const QRectF rr = ring->mapRectToScene(QRectF(0, 0, ring->width(), ring->height()));
            const QRectF br = bar->mapRectToScene(QRectF(0, 0, bar->width(), bar->height()));
            check(rr.contains(br) && qAbs(rr.center().y() - br.center().y()) < 1.5, "cursor: wraps the composer while typing");
        }
        check(hint && !firstItem(win, "focusCursorHint")->isVisible(), "cursor: no chip over the composer");
        if (QQuickItem *in = firstItem(win, "composerInput"))
            in->setProperty("text", QString());
        QMetaObject::invokeMethod(app, "takeFocus");
        settle(450);
        backend.setNavMode("vim");
        settle(50);
        const QString hv = hints ? hints->property("text").toString() : QString();
        check(hv.contains("⏎ ask") && !hv.contains("R2"), "hints: composer shows keys in vim mode", hv);
        backend.setNavMode("controller");
        settle(50);
    }
    scanBanned(win, "stage");

    // Stage text scales with the UI zoom.
    if (QQuickItem *title = firstItem(win, "stageTitle")) {
        const int p0 = title->property("font").value<QFont>().pixelSize();
        sendKey(win, Qt::Key_Equal, Qt::ControlModifier);
        sendKey(win, Qt::Key_Equal, Qt::ControlModifier);
        settle(60);
        const int p1 = title->property("font").value<QFont>().pixelSize();
        check(p1 == qRound(p0 * backend.uiScale()) && p1 > p0, "zoom: Cmd+= scales Stage text",
              QStringLiteral("%1 -> %2 at %3x").arg(p0).arg(p1).arg(backend.uiScale()));
        sendKey(win, Qt::Key_0, Qt::ControlModifier);
        settle(60);
        check(title->property("font").value<QFont>().pixelSize() == p0, "zoom: Cmd+0 restores Stage text");
        const QPointF centre(960, 540);
        QWheelEvent plain(centre, win->mapToGlobal(centre), QPoint(), QPoint(0, 120), Qt::NoButton, Qt::NoModifier,
                          Qt::NoScrollPhase, false);
        QCoreApplication::sendEvent(win, &plain);
        check(qFuzzyCompare(backend.uiScale(), 1.0), "zoom: plain wheel on the Stage does not zoom",
              QString::number(backend.uiScale()));
        QWheelEvent cmd(centre, win->mapToGlobal(centre), QPoint(), QPoint(0, 120), Qt::NoButton, Qt::ControlModifier,
                        Qt::NoScrollPhase, false);
        QCoreApplication::sendEvent(win, &cmd);
        check(backend.uiScale() > 1.1, "zoom: Cmd+wheel zooms on the Stage", QString::number(backend.uiScale()));
        sendKey(win, Qt::Key_0, Qt::ControlModifier);
        settle(60);
    }

    // Space unfolds prose, Space folds it.
    QQuickItem *prose = firstItem(win, "prosePanel");
    sendKey(win, Qt::Key_Space, Qt::NoModifier, " ");
    settle(300);
    check(prose && prose->isVisible() && prose->opacity() > 0.95, "ui: Space unfolds prose");
    sendKey(win, Qt::Key_Space, Qt::NoModifier, " ");
    settle(300);
    check(prose && !prose->isVisible(), "ui: Space folds prose");

    // Ladder lens.
    sendKey(win, Qt::Key_2, Qt::ControlModifier);
    settle(350);
    check(app->property("lens").toInt() == 1, "ui: Cmd+2 shows Ladder");
    const auto rungs = findItems(win->contentItem(), "rung");
    check(rungs.size() == 4, "ui: four rungs", QString::number(rungs.size()));
    for (QQuickItem *r : rungs) {
        int words = 0;
        for (QQuickItem *t : findItems(r, "rungText"))
            if (t->isVisible())
                words += countWords(t->property("text").toString());
        check(words < 8, "ui: rung headline < 8 words", QString::number(words));
    }
    check(forbiddenPixels(win->grabWindow()) == 0, "ui: ladder has zero blue/green pixels");

    // Esc back to Constellation, then Cmd+N kicks off a new session (mock: no process).
    sendKey(win, Qt::Key_Escape);
    settle(300);
    check(app->property("lens").toInt() == 0, "ui: Esc returns to Constellation");
    sendKey(win, Qt::Key_N, Qt::ControlModifier);
    settle(250);
    QQuickItem *dialog = firstItem(win, "newSessionDialog");
    check(dialog && dialog->property("open").toBool(), "ui: Cmd+N opens the dialog");
    typeText(win, "profile fp4 kv-cache decode on sm120");
    check(firstItem(win, "goalInput") && firstItem(win, "goalInput")->property("text").toString()
                                             == "profile fp4 kv-cache decode on sm120",
          "ui: goal typed into dialog");
    scanBanned(win, "dialog");
    sendKey(win, Qt::Key_Return);
    settle(300);
    check(app->property("lens").toInt() == 1, "ui: kickoff auto-shows Ladder");
    check(app->property("projectKey").toString() == "fp4-decode", "ui: new project selected",
          app->property("projectKey").toString());
    for (int i = 0; i < 14; ++i)
        backend.advance(0.5);
    settle(300);
    check(backend.projectIndex("fp4-decode") >= 0, "ui: new island exists");
    {
        const QVariantMap p = backend.projects().value(backend.projectIndex("fp4-decode")).toMap();
        check(p.value("ladder").toMap().value("winner").toString() == "128-reg · 29.6 tok/s", "ui: winner rung filled");
        check(p.value("headline").toString() == "29.6 tok/s · 2.63x", "ui: island headline");
    }
    check(forbiddenPixels(win->grabWindow()) == 0, "ui: filled ladder has zero blue/green pixels");

    // Stage across chart -> table -> pdf: visual first, the session's diff stays out of the deck.
    sendKey(win, Qt::Key_3, Qt::ControlModifier);
    settle(350);
    {
        QStringList kinds;
        for (const QVariant &v : app->property("artifacts").toList())
            kinds << v.toMap().value("kind").toString();
        const QVariantMap pm = backend.projects().value(backend.projectIndex("fp4-decode")).toMap();
        check(!backend.showCodeDiffs() && kinds == QStringList({"chart", "table", "pdf"}) && pm.value("diffCount").toInt() == 1,
              "visual-first: showCodeDiffs defaults off; the default deck is chart, table, pdf (diff hidden)", kinds.join(','));
        QQuickItem *pill = firstItem(win, "codePill");
        check(pill && pill->isVisible(), "visual-first: a Code pill sits in the Stage header when the project has diffs");
        check(!firstItem(win, "diffView") || !firstItem(win, "diffView")->isVisible(), "visual-first: no diff on the default slide");
    }
    const char *viewers[3] = {"chartView", "tableView", "pdfView"};
    for (int i = 0; i < 3; ++i) {
        if (i > 0) {
            sendKey(win, Qt::Key_Right);
            // Mid-slide: the outgoing pane leaves left while the incoming one arrives from the right.
            bool sawSlide = false;
            QString seen;
            for (int k = 0; k < 30 && !sawSlide; ++k) {
                settle(6);
                int shown = 0;
                qreal outX = 0, inX = 0;
                for (QQuickItem *pn : findItems(win->contentItem(), "stagePane"))
                    if (pn->isVisible()) {
                        ++shown;
                        (pn->property("current").toBool() ? inX : outX) = pn->x();
                    }
                seen += QStringLiteral(" %1:%2/%3").arg(shown).arg(outX, 0, 'f', 0).arg(inX, 0, 'f', 0);
                sawSlide = shown == 2 && outX < -1 && inX > 1;
            }
            check(sawSlide, "ui: Right slides the stage horizontally (old exits left, new enters right)", seen);
            settle(320);
            int shown = 0;
            for (QQuickItem *pn : findItems(win->contentItem(), "stagePane"))
                if (pn->isVisible()) {
                    ++shown;
                    check(qAbs(pn->x()) < 0.5 && qAbs(pn->scale() - 1) < 0.001, "ui: slide settles at x=0, scale 1");
                }
            check(shown == 1, "ui: one stage pane after the slide", QString::number(shown));
        }
        QQuickItem *v = firstItem(win, viewers[i]);
        check(v && v->isVisible(), "ui: stage viewer shown", QString::fromLatin1(viewers[i]));
        const QImage shot = win->grabWindow();
        check(forbiddenPixels(shot) == 0, "ui: viewer has zero blue/green pixels",
              QStringLiteral("%1: %2").arg(viewers[i]).arg(forbiddenPixels(shot)));
        if (i == 1) {
            sendKey(win, Qt::Key_Down);
            sendKey(win, Qt::Key_Down);
            settle(300);
        }
    }
    if (QQuickItem *pdfImg = firstItem(win, "pdfImage"))
        check(pdfImg->property("status").toInt() == 1, "ui: PDF page image ready", pdfImg->property("status").toString());
    else
        check(false, "ui: PDF image item exists");

    // Keymaps on the Stage: h/l (vim) and a/d (wasd) step artifacts; j/k, w/s scroll, never type.
    {
        auto idx = [&] { return app->property("artifactIndex").toInt(); };
        auto letter = [&](int key, const char *t) { sendKey(win, key, Qt::NoModifier, QString::fromLatin1(t)); };
        const int n = int(app->property("artifacts").toList().size());
        const int i0 = idx();
        auto step = [&](int d) { return (i0 + d + n) % n; };
        QQuickItem *in = firstItem(win, "composerInput");
        check(n >= 3, "nav: Stage project has several artifacts", QString::number(n));
        backend.setNavMode("mouse");
        letter(Qt::Key_L, "l");
        check(idx() == i0 && in && in->property("text").toString() == "l", "nav: mouse mode leaves l to the composer",
              QStringLiteral("%1 '%2'").arg(idx()).arg(in ? in->property("text").toString() : "?"));
        sendKey(win, Qt::Key_Escape);  // clears the composer; focus stays in it
        QMetaObject::invokeMethod(app, "takeFocus");
        backend.setNavMode("vim");
        sendKey(win, Qt::Key_G, Qt::ShiftModifier, "G");
        const int afterBigG = idx();
        letter(Qt::Key_G, "g");
        const int afterG = idx();
        check(afterBigG == n - 1 && afterG == 0 && in && in->property("text").toString().isEmpty(),
              "nav: vim G / g jump to the last / first artifact", QStringLiteral("%1 %2").arg(afterBigG).arg(afterG));
        for (int s = 0; s < 40 && idx() != i0; ++s)
            sendKey(win, Qt::Key_Right);
        settle(300);
        letter(Qt::Key_H, "h");
        const int afterH = idx();
        letter(Qt::Key_L, "l");
        const int afterL = idx();
        letter(Qt::Key_J, "j");
        letter(Qt::Key_K, "k");
        check(afterH == step(-1) && afterL == i0 && idx() == i0, "nav: vim h/l step artifacts on the Stage",
              QStringLiteral("%1 -> %2 -> %3").arg(i0).arg(afterH).arg(afterL));
        check(in && in->property("text").toString().isEmpty() && !in->hasActiveFocus(),
              "nav: vim j/k scroll instead of typing", in ? in->property("text").toString() : "none");
        backend.setNavMode("wasd");
        letter(Qt::Key_A, "a");
        const int afterA = idx();
        letter(Qt::Key_D, "d");
        letter(Qt::Key_S, "s");
        letter(Qt::Key_W, "w");
        check(afterA == step(-1) && idx() == i0 && in && in->property("text").toString().isEmpty(),
              "nav: wasd a/d step artifacts, w/s scroll", QStringLiteral("%1 -> %2 -> %3").arg(i0).arg(afterA).arg(idx()));
        sendKey(win, Qt::Key_Left);
        check(idx() == step(-1), "nav: Left still steps in wasd mode");
        sendKey(win, Qt::Key_Right);
        backend.setNavMode("vim");
        settle(400);
        check(app->property("lens").toInt() == 2 && idx() == i0, "nav: back on the same Stage artifact");
    }

    // The Code pill: clicking it turns diffs on and lands on the first one (after the visuals).
    {
        auto idx = [&] { return app->property("artifactIndex").toInt(); };
        QQuickItem *pill = firstItem(win, "codePill");
        if (pill)
            mouseClick(win, pill->mapToScene(QPointF(pill->width() / 2, pill->height() / 2)), Qt::LeftButton, {});
        settle(450);
        QStringList kinds;
        for (const QVariant &v : app->property("artifacts").toList())
            kinds << v.toMap().value("kind").toString();
        check(backend.showCodeDiffs() && kinds == QStringList({"chart", "table", "pdf", "diff"}),
              "visual-first: the Code pill appends the diff after the visuals", kinds.join(','));
        check(idx() == 3 && firstItem(win, "diffView") && firstItem(win, "diffView")->isVisible(),
              "visual-first: the Code pill lands on the diff", QString::number(idx()));
        check(backend.contextSlice("fp4-decode", 3).contains("diff"), "visual-first: asks index the deck the user sees",
              backend.contextSlice("fp4-decode", 3).left(80));
    }

    // Tab predictor on the Stage (diffs on): rings the next artifact's pager dot ("View Diff"
    // after a chart) and Tab steps there; once everything is seen it suggests a prompt instead.
    {
        auto idx = [&] { return app->property("artifactIndex").toInt(); };
        QMetaObject::invokeMethod(app, "takeFocus");
        while (idx() != 0)
            sendKey(win, Qt::Key_Left);
        app->setProperty("seenArtifacts", QVariantList{0});  // as if this Stage was just opened
        settle(900);
        const QVariantList arts = app->property("artifacts").toList();
        const QVariantMap p = app->property("predictedAction").toMap();
        check(arts.value(0).toMap().value("kind").toString() == "chart" && p.value("label").toString() == "View Diff"
                  && p.value("targetId").toString() == "pager:3",
              "predictor: a chart predicts its diff", p.value("label").toString() + " " + p.value("targetId").toString());
        QQuickItem *ring = firstItem(win, "predictorRingShape");
        check(ring && ring->isVisible() && qAbs(ring->width() - ring->height()) < 0.5 && ring->property("radius").toReal() > ring->width() / 2 - 0.5,
              "predictor: a round gold circle around the pager dot");
        QStringList pairs;
        check(textOverlaps(win, &pairs) == 0, "predictor: Stage badge overlaps no text", pairs.join("; "));
        sendKey(win, Qt::Key_Tab);
        settle(400);
        check(idx() == 3 && firstItem(win, "diffView") && firstItem(win, "diffView")->isVisible(),
              "predictor: Tab executes View Diff", QString::number(idx()));
        for (int i = 0; i < 6 && app->property("predictedAction").toMap().value("actionKind").toString() == "artifact"; ++i) {
            sendKey(win, Qt::Key_Tab);
            settle(350);
        }
        const QVariantMap pp = app->property("predictedAction").toMap();
        check(pp.value("actionKind").toString() == "prompt" && pp.value("targetId").toString() == "composer"
                  && !pp.value("promptSuggestion").toString().isEmpty(),
              "predictor: all artifacts seen -> a prompt for the composer", pp.value("promptSuggestion").toString());
        QQuickItem *ghost = firstItem(win, "composerGhost");
        check(ghost && ghost->isVisible() && ghost->property("text").toString() == pp.value("promptSuggestion").toString(),
              "predictor: the suggestion shows as ghost text in the composer",
              ghost ? ghost->property("text").toString() : "none");
        sendKey(win, Qt::Key_Tab);
        QQuickItem *in = firstItem(win, "composerInput");
        check(in && in->hasActiveFocus() && in->property("text").toString() == pp.value("promptSuggestion").toString(),
              "predictor: Tab accepts the suggested prompt", in ? in->property("text").toString() : "none");
        sendKey(win, Qt::Key_Escape);  // clear it
        QMetaObject::invokeMethod(app, "takeFocus");
        while (idx() != 0)
            sendKey(win, Qt::Key_Left);
        settle(400);
    }

    // Inline Model & Effort picker in the composer.
    {
        QQuickItem *in = firstItem(win, "composerInput");
        QQuickItem *strip = firstItem(win, "modelStrip");
        QQuickItem *composer = firstItem(win, "composer");
        auto letter = [&](int key, const char *t) { sendKey(win, key, Qt::NoModifier, QString::fromLatin1(t)); };
        check(firstItem(win, "modelChipLabel") && firstItem(win, "modelChipLabel")->property("text").toString() == "Sonnet 5.5 · low",
              "picker: composer chip shows the model + effort",
              firstItem(win, "modelChipLabel") ? firstItem(win, "modelChipLabel")->property("text").toString() : "none");
        sendKey(win, Qt::Key_M, Qt::ControlModifier | Qt::ShiftModifier);
        settle(60);
        check(strip && strip->isVisible() && in && in->hasActiveFocus(), "picker: Cmd+Shift+M opens the strip and focuses the composer");
        QStringList pairs;
        check(textOverlaps(win, &pairs) == 0, "picker: no overlapping text", pairs.join("; "));
        letter(Qt::Key_S, "s");
        const QString afterS = backend.chatModel();
        letter(Qt::Key_D, "d");
        letter(Qt::Key_D, "d");
        const QString afterDD = backend.chatEffort();
        letter(Qt::Key_W, "w");
        const QString afterW = backend.chatModel();
        letter(Qt::Key_J, "j");
        const QString afterJ = backend.chatModel();
        letter(Qt::Key_H, "h");
        const QString afterH = backend.chatEffort();
        sendKey(win, Qt::Key_Down);
        sendKey(win, Qt::Key_Right);
        // Lineup order: ... Fable, Opus, Sonnet, Astra, Sol, ...
        check(afterS == "gpt-6-astra" && afterDD == "high" && afterW == "claude-sonnet-5-5" && afterJ == "gpt-6-astra"
                  && afterH == "medium" && backend.chatModel() == "gpt-6-sol" && backend.chatEffort() == "high",
              "picker: S/W/j/Down cycle models, D/h/Right cycle efforts",
              QStringList({afterS, afterDD, afterW, afterJ, afterH, backend.chatModel(), backend.chatEffort()}).join(' '));
        check(in->property("text").toString().isEmpty(), "picker: picker keys never type into the composer");
        if (QQuickItem *prov = firstItem(win, "pickerProvider"))
            check(prov->property("text").toString() == "OpenAI", "picker: strip names the provider", prov->property("text").toString());
        sendKey(win, Qt::Key_Escape);
        check(strip && !strip->isVisible() && in->hasActiveFocus() && app->property("lens").toInt() == 2,
              "picker: Esc closes the strip without leaving the composer");
        sendKey(win, Qt::Key_K, Qt::NoModifier, "k");  // vim: k in an empty composer reopens it
        check(strip->isVisible(), "picker: k in an empty composer opens the strip (vim)");
        sendKey(win, Qt::Key_Escape);
        sendKey(win, Qt::Key_Up);
        check(strip->isVisible(), "picker: Up in an empty composer opens the strip");
        letter(Qt::Key_A, "a");
        sendKey(win, Qt::Key_Escape);
        typeText(win, "which arm?");
        sendKey(win, Qt::Key_M, Qt::ControlModifier | Qt::ShiftModifier);
        letter(Qt::Key_D, "d");
        sendKey(win, Qt::Key_Return);
        check(backend.lastAskModel() == "gpt-6-sol" && backend.lastAskEffort() == "high" && backend.lastAskChannel() == "openai"
                  && !strip->isVisible() && in->property("text").toString().isEmpty(),
              "picker: Enter sends with the selected model + effort and closes the strip",
              backend.lastAskModel() + " " + backend.lastAskEffort());
        check(composer && composer->property("pickerOpen").toBool() == false, "picker: closed after send");
        for (int i = 0; i < 6; ++i)
            backend.advance(0.3);
        backend.setChatModel("claude-sonnet-5-5");
        backend.setChatEffort("low");
        backend.clearAnswers();
        sendKey(win, Qt::Key_Escape);  // leave the composer: Esc with an empty field zooms out
        settle(450);
        sendKey(win, Qt::Key_1);
        settle(450);
        check(app->property("lens").toInt() == 2, "picker: back on the Stage");
    }

    // Enter quick answer and Opt+Enter collapsing trace (mock replies, no processes).
    typeText(win, "best arm?");
    {
        QQuickItem *in = firstItem(win, "composerInput");
        check(in && in->property("text").toString() == "best arm?", "ui: typing on Stage fills the composer",
              QStringLiteral("text='%1' focus=%2").arg(in ? in->property("text").toString() : "?",
                                                      win->activeFocusItem() ? win->activeFocusItem()->objectName() + "/" + win->activeFocusItem()->metaObject()->className() : "none"));
    }
    sendKey(win, Qt::Key_Return);
    {
        // The first frame after Enter already says something (never a blank, hung-looking line).
        settle(30);
        QQuickItem *qa = firstItem(win, "quickAnswer");
        check(qa && !qa->property("text").toString().isEmpty() && backend.quickPhase() == "sending",
              "thinking: the answer line shows a phase before any stream event", qa ? qa->property("text").toString() : "?");
    }
    backend.advance(0.2);
    check(backend.quickBusy() && backend.quickAnswer().isEmpty(), "ui: quick answer waits for its first token");
    {
        settle(30);
        QQuickItem *qa = firstItem(win, "quickAnswer");
        const QString shown = qa ? qa->property("text").toString() : QString();
        check(!backend.quickThinking().isEmpty() && backend.quickPhase() == "thinking"
                  && shown == backend.quickThinking() && qa->property("color").value<QColor>() == QColor("#6e665e"),
              "thinking: thinking tokens stream live, dim #6e665e, before the answer", shown);
        QQuickItem *qs = firstItem(win, "quickSeconds");
        check(qs && qs->property("text").toString() == "0.2s", "thinking: live elapsed seconds tick while busy",
              qs ? qs->property("text").toString() : "?");
    }
    backend.advance(0.6);
    {
        settle(30);
        QQuickItem *qa = firstItem(win, "quickAnswer");
        check(qa && qa->property("text").toString() == backend.quickAnswer() && !backend.quickAnswer().isEmpty(),
              "thinking: the first answer token replaces the thinking line", qa ? qa->property("text").toString() : "?");
    }
    {
        const QString part = backend.quickAnswer();
        // The mock alternates two canned replies; the picker test above already used one.
        const QString a = "128-reg: dequant stays in registers, zero spills.";
        const QString full = a.startsWith(part) ? a : QStringLiteral("Baseline spilled 212 registers per thread.");
        check(backend.quickBusy() && !part.isEmpty() && part.size() < full.size() && full.startsWith(part)
                  && (part.endsWith(' ') || full.at(part.size()) == ' '),
              "ui: quick answer streams token by token", part);
    }
    for (int i = 0; i < 4; ++i)
        backend.advance(0.3);
    check(!backend.quickBusy() && !backend.quickAnswer().isEmpty(), "ui: quick answer arrives");
    {
        settle(30);
        QQuickItem *qs = firstItem(win, "quickSeconds");
        check(qs && qs->property("text").toString() == QStringLiteral("✦ 1.3s"),
              "thinking: a thought answer ends with ✦ and its total time", qs ? qs->property("text").toString() : "?");
    }
    typeText(win, "why did 128-reg win?");
    sendKey(win, Qt::Key_Return, Qt::AltModifier);
    backend.advance(1.0);
    check(backend.traceBusy() && !backend.traceThinking().isEmpty(), "ui: trace streams while thinking");
    for (int i = 0; i < 8; ++i)
        backend.advance(0.25);
    settle(350);
    check(!backend.traceBusy() && !backend.traceAnswer().isEmpty(), "ui: trace collapses to an answer");
    if (QQuickItem *h = firstItem(win, "traceHeader"))
        check(h->property("text").toString() == "✦ Thought 2.4s", "ui: collapsed header", h->property("text").toString());
    check(forbiddenPixels(win->grabWindow()) == 0, "ui: answers have zero blue/green pixels");
    scanBanned(win, "answers");
    sendKey(win, Qt::Key_Return, Qt::ControlModifier);  // empty steer is a no-op
    typeText(win, "keep 128-reg");
    sendKey(win, Qt::Key_Return, Qt::ControlModifier);
    check(backend.toast().startsWith("→"), "ui: Cmd+Enter steers the session (mock)", backend.toast());

    sendKey(win, Qt::Key_Escape);
    settle(60);
    {
        const QRectF ir = app->property("islandRect").toRectF();
        const qreal w = morph ? morph->width() : 0;
        check(w < 1919 && w > ir.width(), "ui: Esc shrinks the card back toward its island", QString::number(w));
    }
    settle(400);
    check(app->property("lens").toInt() == 0, "ui: Esc from composer zooms out");
    check(morph && !morph->isVisible(), "ui: zoom layer gone after Esc");
    {
        const int idx = backend.projectIndex(app->property("projectKey").toString());
        QQuickItem *home = findItems(win->contentItem(), "island").value(idx);
        check(home && home->opacity() > 0.99 && qAbs(home->scale() - 1) < 0.001, "ui: Esc lands back on the island",
              home ? QString::number(home->opacity()) : "none");
    }

    // Safety: the mock never spawns a process and never reads a transcript.
    check(backend.spawnCount() == 0, "safety: zero processes spawned in --test", QString::number(backend.spawnCount()));
    check(backend.transcriptsRead().isEmpty(), "safety: zero transcripts read in --test");
}


// --- PS4 DualShock 4 ------------------------------------------------------------------------------

// Records every signal a Gamepad emits, with the clock value of the poll that produced it.
struct PadLog {
    struct Ev { QString name; int a = 0, b = 0; double v = 0; qint64 t = 0; };
    QList<Ev> evs;
    qint64 now = 0;
    void attach(Gamepad *g)
    {
        QObject::connect(g, &Gamepad::nav, g, [this](int dx, int dy) { evs.append({"nav", dx, dy, 0, now}); });
        QObject::connect(g, &Gamepad::confirm, g, [this] { evs.append({"confirm", 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::back, g, [this] { evs.append({"back", 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::acceptPrediction, g, [this] { evs.append({"tab", 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::secondary, g, [this] { evs.append({"secondary", 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::step, g, [this](int d) { evs.append({"step", d, 0, 0, now}); });
        QObject::connect(g, &Gamepad::zoomStep, g, [this](double d) { evs.append({"zoom", 0, 0, d, now}); });
        QObject::connect(g, &Gamepad::zoomReset, g, [this] { evs.append({"zoomReset", 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::toggleSettings, g, [this] { evs.append({"settings", 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::toggleNewSession, g, [this] { evs.append({"newSession", 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::toggleBriefing, g, [this] { evs.append({"briefing", 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::scrollDelta, g, [this](double d) { evs.append({"scroll", 0, 0, d, now}); });
        QObject::connect(g, &Gamepad::touchpadPan, g, [this](double dx, double dy) { evs.append({"pan", int(dx), int(dy), 0, now}); });
        QObject::connect(g, &Gamepad::horizontalInspect, g, [this](int dir) { evs.append({"rinspect", dir, 0, 0, now}); });
        QObject::connect(g, &Gamepad::voiceHold, g, [this](bool on) { evs.append({"voice", on ? 1 : 0, 0, 0, now}); });
        QObject::connect(g, &Gamepad::sendPressed, g, [this] { evs.append({"send", 0, 0, 0, now}); });
    }
    void feed(Gamepad *g, const GamepadSnapshot &s, qint64 t)
    {
        now = t;
        g->processSnapshot(s, t);
    }
    // Hold s from t0 to t1 (inclusive), polling every dt ms.
    void hold(Gamepad *g, const GamepadSnapshot &s, qint64 t0, qint64 t1, qint64 dt = 8)
    {
        for (qint64 t = t0; t <= t1; t += dt)
            feed(g, s, t);
    }
    int count(const QString &n) const
    {
        int c = 0;
        for (const Ev &e : evs)
            c += e.name == n;
        return c;
    }
    QString names() const
    {
        QStringList l;
        for (const Ev &e : evs)
            l << e.name;
        return l.join(',');
    }
};

void gamepadUnitTests()
{
    using S = GamepadSnapshot;
    // Left stick: 0.35 deadzone, immediate step, 320 ms delay then 110 ms repeat.
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        S s;
        s.lx = 0.34;
        log.hold(&g, s, 0, 400);
        check(log.evs.isEmpty(), "pad: left stick inside the 0.35 deadzone is silent", log.names());
        s.lx = 0.36;
        log.feed(&g, s, 1000);
        check(log.evs.size() == 1 && log.evs[0].name == "nav" && log.evs[0].a == 1 && log.evs[0].b == 0,
              "pad: crossing 0.35 steps right immediately", log.names());
        log.evs.clear();
        s.lx = 0.9;
        log.hold(&g, s, 1010, 2000, 10);  // a 10 ms poll grid lands on the nominal times
        QStringList times;
        for (const auto &e : log.evs)
            times << QString::number(e.t - 1000);
        check(times.join(',') == "320,430,540,650,760,870,980", "pad: stick repeats after 320 ms, then every 110 ms",
              times.join(','));
        // Release hysteresis: 0.30 (between 0.25 and 0.35) keeps the hold, no new step; 0.2 lets go.
    }
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        S s;
        s.lx = 0.5;
        log.feed(&g, s, 3000);
        s.lx = 0.30;
        log.hold(&g, s, 3008, 3200);
        check(log.count("nav") == 1, "pad: stick at 0.30 after engaging holds without chatter", log.names());
        s.lx = 0.2;
        log.feed(&g, s, 3208);
        s.lx = 0.4;
        log.feed(&g, s, 3216);
        check(log.count("nav") == 2, "pad: dropping under 0.25 re-arms the next step", log.names());
        // Dominant axis, no diagonals; stick up is screen up.
        log.evs.clear();
        log.feed(&g, S(), 3300);
        s = S();
        s.lx = 0.5;
        s.ly = 0.8;
        log.feed(&g, s, 4000);
        check(log.evs.size() == 1 && log.evs[0].a == 0 && log.evs[0].b == -1, "pad: stick up-left (mostly up) steps up");
        s = S();
        s.ly = -0.9;
        log.feed(&g, s, 4008);
        check(log.evs.size() == 2 && log.evs[1].a == 0 && log.evs[1].b == 1, "pad: changing direction steps down at once",
              log.names());
    }
    // D-pad: discrete step on press, same 320 / 110 ms repeat, wins over the stick.
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        S s;
        s.dpadY = -1;
        s.lx = 0.9;
        log.hold(&g, s, 0, 545);
        QStringList seen;
        for (const auto &e : log.evs)
            seen << QStringLiteral("%1:%2,%3").arg(e.t).arg(e.a).arg(e.b);
        check(seen.join(' ') == "0:0,1 320:0,1 432:0,1 544:0,1", "pad: d-pad down repeats 320 / 110 and beats the stick",
              seen.join(' '));
        log.evs.clear();
        log.feed(&g, S(), 600);
        s = S();
        s.dpadX = -1;
        log.feed(&g, s, 608);
        check(log.evs.size() == 1 && log.evs[0].a == -1, "pad: d-pad left steps left on press");
        // A stalled poll fires once, never a burst.
        log.evs.clear();
        log.feed(&g, s, 2000);
        log.feed(&g, s, 2008);
        check(log.count("nav") == 1, "pad: a 1.4 s stall fires one repeat, not a backlog", log.names());
    }
    // Buttons: edge-triggered, one action per press however long it is held.
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        qint64 t = 0;
        auto press = [&](auto set) {
            S s;
            set(s);
            log.hold(&g, s, t, t + 200);
            log.feed(&g, S(), t + 208);
            t += 300;
        };
        press([](S &s) { s.a = true; });
        press([](S &s) { s.b = true; });
        press([](S &s) { s.y = true; });
        press([](S &s) { s.x = true; });
        press([](S &s) { s.l3 = true; });
        press([](S &s) { s.r3 = true; });
        press([](S &s) { s.l1 = true; });
        press([](S &s) { s.r1 = true; });
        press([](S &s) { s.menu = true; });
        press([](S &s) { s.options = true; });
        press([](S &s) { s.touchpad = true; });
        check(log.names() == "confirm,back,step,secondary,zoomReset,zoom,step,tab,settings,newSession,newSession",
              "pad: ✕ ○ △ □ L3 R3 L1 R1 ☰ share touchpad map once per press", log.names());
        check(log.evs.value(2).a == 1 && log.evs.value(6).a == -1, "pad: △ steps +1 (Next View), L1 steps -1 (Previous View)");
        check(log.evs.value(7).name == "tab", "pad: R1 is Tab / Full Send");
        check(g.glyphFor("accept_prediction") == "R1" && g.glyphFor("step_next") == "△" && g.glyphFor("step_prev") == "L1"
                  && g.glyphFor("confirm") == "✕" && g.glyphFor("none").isEmpty(),
              "pad: glyphFor names the bound button (R1 Tab, △ next, L1 previous)", g.glyphFor("accept_prediction"));
        check(g.actionLabel("accept_prediction") == "Tab / Full Send" && g.actionLabel("step_prev") == "Previous View"
                  && g.actionLabel("step_next") == "Next View",
              "pad: plain-English action labels", g.actionLabel("step_prev"));
        log.evs.clear();
        S s;
        s.options = true;
        log.feed(&g, s, t);
        s.touchpad = true;
        log.feed(&g, s, t + 8);
        check(log.count("newSession") == 1, "pad: share + touchpad together toggle the dialog once", log.names());
    }
    // L3 + R3 together: one confirmProposalRequested, never zoom_reset / zoom_in / r3Pressed.
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        int confirms = 0, r3s = 0;
        QObject::connect(&g, &Gamepad::confirmProposalRequested, [&confirms] { ++confirms; });
        QObject::connect(&g, &Gamepad::r3Pressed, [&r3s] { ++r3s; });
        S both;
        both.l3 = both.r3 = true;
        log.hold(&g, both, 0, 400);  // same-poll press, held well past the 80 ms window
        log.feed(&g, S(), 408);
        check(confirms == 1 && r3s == 0 && log.evs.isEmpty(), "pad: L3+R3 same poll emits one confirm and no single actions",
              log.names());
        // Staggered within the window: L3 first, R3 48 ms later.
        S l3;
        l3.l3 = true;
        log.hold(&g, l3, 1000, 1040);
        log.hold(&g, both, 1048, 1300);
        S r3only;
        r3only.r3 = true;
        log.hold(&g, r3only, 1308, 1400);  // L3 lets go first: still the chord, no R3 zoom
        log.feed(&g, S(), 1408);
        check(confirms == 2 && r3s == 0 && log.evs.isEmpty(), "pad: L3 then R3 within 80 ms is the chord", log.names());
        // Outside the window: L3 fires alone after 80 ms, R3 later fires alone too.
        log.hold(&g, l3, 2000, 2120);
        log.hold(&g, both, 2128, 2200);
        log.feed(&g, S(), 2208);
        check(confirms == 2 && r3s == 1 && log.names() == "zoomReset,zoom",
              "pad: L3 held past 80 ms then R3 are two single clicks", log.names());
        log.evs.clear();
        // A quick single tap still fires on release, inside the window.
        log.feed(&g, r3only, 3000);
        log.feed(&g, S(), 3016);
        check(confirms == 2 && r3s == 2 && log.names() == "zoom" && log.evs.value(0).t == 3016,
              "pad: a 16 ms R3 tap fires on release", log.names());
    }
    // Triggers: > 0.55 presses, < 0.45 re-arms. L2 = Phonon hold/release, R2 = Send.
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        S s;
        s.r2 = 0.5;
        log.feed(&g, s, 0);
        check(log.evs.isEmpty(), "pad: R2 at 0.50 does not fire");
        s.r2 = 0.6;
        log.hold(&g, s, 8, 400);
        s.r2 = 0.5;
        log.feed(&g, s, 408);
        s.r2 = 0.7;
        log.feed(&g, s, 416);
        check(log.count("send") == 1, "pad: R2 fires send once, hysteresis holds at 0.50", log.names());
        s.r2 = 0.4;
        log.feed(&g, s, 424);
        s.r2 = 0.6;
        log.feed(&g, s, 432);
        s = S();
        s.l2 = 1.0;
        log.feed(&g, s, 440);
        check(g.voiceActive(), "pad: L2 hold activates Phonon voiceActive");
        s.l2 = 0.0;
        log.feed(&g, s, 448);
        check(!g.voiceActive() && log.count("send") == 2 && log.count("voice") == 2
                  && log.evs[2].a == 1 && log.evs[3].a == 0,
              "pad: R2 re-arms under 0.45; L2 fires voiceHold(true) on press and false on release",
              log.names());
    }
    // Touchpad split click + 2D pan + Right Stick horizontal inspect.
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        S s;
        s.touchpad = true;
        s.touchpadX = -0.6;
        log.feed(&g, s, 0);
        log.feed(&g, S(), 16);
        s.touchpad = true;
        s.touchpadX = 0.5;
        log.feed(&g, s, 32);
        log.feed(&g, S(), 48);
        check(log.count("briefing") == 1 && log.count("newSession") == 1,
              "pad: touchpad left half -> Morning Briefing, right half -> New Session", log.names());
        log.evs.clear();
        S p0; p0.touchpadX = 0.1; p0.touchpadY = 0.2;
        log.feed(&g, p0, 64);
        S p1; p1.touchpadX = 0.3; p1.touchpadY = 0.1;
        log.feed(&g, p1, 72);
        check(log.count("pan") == 1 && log.evs[0].a > 0 && log.evs[0].b > 0,
              "pad: touchpad drag emits 2D pan deltas", log.names());
        log.evs.clear();
        S r; r.rx = 0.8;
        log.feed(&g, r, 100);
        check(log.count("rinspect") == 1 && log.evs[0].a == 1,
              "pad: right stick horizontal triggers horizontalInspect", log.names());
    }
    // Remapping + capture mode + prefs persistence.
    {
        Gamepad g;
        QString captured;
        QObject::connect(&g, &Gamepad::rawInputCaptured, &g, [&](const QString &id) { captured = id; });
        g.setCaptureMode(true);
        S s;
        s.y = true;
        g.processSnapshot(s, 0);
        check(captured == "y" && !g.captureMode(), "pad: captureMode captures raw input and disarms", captured);

        QTemporaryDir tmp;
        const QString path = tmp.path() + "/pad_prefs.json";
        {
            Backend b(Backend::Mode::Mock);
            b.setPrefsPath(path);
            b.setGamepadBinding(QStringLiteral("l2"), QStringLiteral("zoom_out"));
            b.setGamepadBinding(QStringLiteral("x"), QStringLiteral("voice_hold"));
            check(b.gamepad()->actionFor(QStringLiteral("l2")) == "zoom_out"
                      && b.gamepad()->actionFor(QStringLiteral("x")) == "voice_hold",
                  "pad: setGamepadBinding updates live map");
            check(b.savePrefs(), "pad: savePrefs writes custom gamepadBindings");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.gamepad()->actionFor(QStringLiteral("l2")) == "zoom_out"
                      && r.gamepad()->actionFor(QStringLiteral("x")) == "voice_hold",
                  "pad: custom gamepadBindings round-trip from studio_prefs.json");
            r.resetGamepadBindings();
            check(r.gamepad()->actionFor(QStringLiteral("l2")) == "voice_hold"
                      && r.gamepad()->actionFor(QStringLiteral("r2")) == "send",
                  "pad: resetGamepadBindings restores defaults");
        }
    }
    // Right stick: 0.20 deadzone, one delta per 8 ms tick, quadratic speed, stick up scrolls up.
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        S s;
        s.ry = -0.19;
        log.hold(&g, s, 0, 200, 4);
        check(log.evs.isEmpty(), "pad: right stick inside 0.20 is silent");
        s.ry = -1.0;
        log.hold(&g, s, 1000, 1096, 4);  // polled at 250 Hz: still one delta per 8 ms
        double sum = 0;
        bool even = true;
        for (const auto &e : log.evs) {
            sum += e.v;
            even = even && qAbs(e.v - 16.0) < 1e-9;
        }
        check(log.count("scroll") == 13 && even, "pad: full right stick down: 13 ticks of 16 px over 96 ms",
              QStringLiteral("%1 ticks, sum %2").arg(log.count("scroll")).arg(sum));
        check(qAbs(sum - 208) < 1e-6, "pad: right stick scrolls 2000 px/s at full deflection", QString::number(sum));
        log.evs.clear();
        s.ry = 0.6;  // half way past the deadzone: a quarter of the speed, upward
        log.feed(&g, s, 1104);
        log.feed(&g, s, 1112);
        check(log.evs.size() == 2 && qAbs(log.evs[1].v + 4.0) < 1e-9, "pad: half deflection up scrolls -4 px per tick",
              log.evs.isEmpty() ? QString() : QString::number(log.evs.last().v));
    }
    // Disconnect resets the debouncer: a button still held on reconnect never fires.
    {
        Gamepad g;
        PadLog log;
        log.attach(&g);
        int changed = 0;
        QObject::connect(&g, &Gamepad::connectedChanged, &g, [&] { ++changed; });
        g.setConnected(true, QStringLiteral("DualShock 4"));
        check(g.connected() && g.controllerName() == "DualShock 4" && changed == 1, "pad: connect sets name");
        g.setConnected(false, QString());
        check(!g.connected() && g.controllerName().isEmpty() && changed == 2, "pad: disconnect clears name");
        check(!g.enabled(), "pad: a fresh Gamepad is disabled (no GameController)");
    }
}

void gamepadUiTests()
{
    using S = GamepadSnapshot;
    Backend backend(Backend::Mode::Mock);
    Gamepad *pad = backend.gamepad();
    check(pad && !pad->enabled(), "pad: the --test backend's gamepad stays disabled");
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    check(win != nullptr, "pad ui: QML loads");
    if (!win || !pad)
        return;
    QObject *app = win->findChild<QObject *>("studioApp");
    QObject *settings = win->findChild<QObject *>("settingsOverlay");
    QObject *dialog = win->findChild<QObject *>("newSessionDialog");
    QObject *onboarding = win->findChild<QObject *>("onboardingModal");
    QObject *composer = win->findChild<QObject *>("composer");
    QObject *stage = win->findChild<QObject *>("stage");
    if (!app || !settings || !dialog || !onboarding || !composer || !stage) {
        check(false, "pad ui: overlays present");
        return;
    }
    settle(300);
    qint64 t = 0;
    auto tap = [&](auto set) {
        S s;
        set(s);
        pad->processSnapshot(s, t);
        pad->processSnapshot(S(), t + 16);
        t += 40;
        settle(30);
    };
    auto prop = [](QObject *o, const char *n) { return o->property(n); };

    // Status pill: hidden without a pad, "DualShock 4" with one.
    QQuickItem *pill = firstItem(win, "padPill");
    check(pill && !pill->isVisible(), "pad ui: no controller, no pill");
    pad->setConnected(true, QStringLiteral("DualShock 4"));
    settle(30);
    QQuickItem *pillText = firstItem(win, "padPillText");
    check(pill && pill->isVisible() && pillText && pillText->property("text").toString() == "DualShock 4",
          "pad ui: connected pad shows a dim DualShock 4 pill");
    scanBanned(win, "pad pill");

    // Onboarding: stick down picks Gaming, ✕ confirms.
    backend.showOnboarding();
    settle(250);
    tap([](S &s) { s.dpadY = -1; });
    check(prop(onboarding, "choice").toInt() == 1, "pad ui: onboarding d-pad down picks the second card");
    tap([](S &s) { s.b = true; });
    check(prop(onboarding, "open").toBool(), "pad ui: ○ cannot skip onboarding");
    tap([](S &s) { s.a = true; });
    check(!prop(onboarding, "open").toBool() && backend.navMode() == "vim", "pad ui: ✕ confirms onboarding (vim)",
          backend.navMode());
    backend.setNavMode(QStringLiteral("controller"));
    settle(250);

    // Constellation: L1 / △ and the stick move the island ring, R1 accepts the prediction.
    app->setProperty("selectedIsland", 0);
    tap([](S &s) { s.y = true; });
    check(prop(app, "selectedIsland").toInt() == 1, "pad ui: △ selects the next island (Next View)");
    tap([](S &s) { s.l1 = true; });
    check(prop(app, "selectedIsland").toInt() == 0, "pad ui: L1 selects the previous island");
    {
        // The left stick drives the free map cursor (analog), not island hops: gliding it onto
        // the next card selects that card. The d-pad still hops, and the cursor follows.
        QObject *map = win->findChild<QObject *>("constellation");
        check(pad->analogCursor(), "pad ui: on the Constellation the pad is in analog cursor mode");
        settle(250);  // the cursor glides onto island 1's neighbour, the selection
        S s;
        s.lx = 1.0;
        bool picked = false;
        for (int k = 0; k < 250 && !picked; ++k) {
            pad->processSnapshot(s, t);
            t += 8;
            settle(8);
            picked = prop(app, "selectedIsland").toInt() == 1;
        }
        pad->processSnapshot(S(), t);
        t += 40;
        settle(30);
        check(picked && map && map->property("stickX").toDouble() == 0,
              "pad ui: left stick glides the free cursor onto the next island (and stops on release)");
        tap([](S &s) { s.dpadX = -1; });
        check(prop(app, "selectedIsland").toInt() == 0, "pad ui: d-pad left still hops back");
        settle(300);  // the cursor follows the hop
    }
    {
        const int before = prop(app, "predictorAccepts").toInt();
        const bool predicted = !prop(app, "predictedAction").isNull();
        tap([](S &s) { s.r1 = true; });
        settle(400);
        check(predicted && prop(app, "predictorAccepts").toInt() == before + 1 && prop(app, "lens").toInt() > 0,
              "pad ui: R1 accepts the Tab prediction on the constellation", QString::number(prop(app, "lens").toInt()));
        tap([](S &s) { s.b = true; });
        settle(400);
        check(prop(app, "lens").toInt() == 0, "pad ui: ○ zooms back out to the constellation");
    }
    {
        QObject *mb = win->findChild<QObject *>("morningBriefing");
        tap([](S &s) { s.r2 = 1.0; });
        settle(200);
        check(mb && prop(mb, "open").toBool(), "pad ui: R2 on the constellation opens the Morning Briefing");
        tap([](S &s) { s.b = true; });
        settle(200);
        check(mb && !prop(mb, "open").toBool() && prop(app, "lens").toInt() == 0, "pad ui: ○ closes the briefing");
    }
    // □ on the constellation opens Settings on the Default Model row; ☰ toggles Settings.
    tap([](S &s) { s.x = true; });
    check(prop(settings, "open").toBool() && prop(settings, "row").toInt() == prop(settings, "modelRow").toInt()
              && prop(settings, "modelRow").toInt() == 0,
          "pad ui: □ on constellation opens settings on the Model row (top)");
    tap([](S &s) { s.menu = true; });
    check(!prop(settings, "open").toBool(), "pad ui: ☰ closes settings");
    tap([](S &s) { s.menu = true; });
    check(prop(settings, "open").toBool() && prop(settings, "row").toInt() == 0, "pad ui: ☰ opens settings");
    {
        settle(200);
        QQuickItem *hdr = firstItem(win, "settingsPadHeader");
        QQuickItem *binds = firstItem(win, "settingsPadBinds");
        check(hdr && hdr->isVisible() && hdr->property("text").toString().contains("DualShock 4") && binds && binds->isVisible(),
              "pad ui: settings shows the PS4 Controller keybinds");
        scanBanned(win, "settings with pad");
        if (QQuickItem *card = firstItem(win, "settingsCard")) {
            const QRectF r = card->mapRectToScene(QRectF(0, 0, card->width(), card->height()));
            check(r.top() >= 0 && r.bottom() <= win->height(), "pad ui: settings card with the PS4 section fits the window",
                  QStringLiteral("%1..%2 of %3").arg(r.top()).arg(r.bottom()).arg(win->height()));
        }
        check(forbiddenPixels(win->grabWindow()) == 0, "pad ui: settings pad section has zero blue/green pixels");
    }
    tap([](S &s) { s.dpadY = -1; });
    check(prop(settings, "row").toInt() == 1, "pad ui: d-pad down moves the settings row");
    {
        const QString before = backend.chatEffort();
        tap([](S &s) { s.dpadX = 1; });
        check(backend.chatEffort() != before, "pad ui: d-pad right changes the Default Effort row",
              before + " -> " + backend.chatEffort());
        backend.setChatEffort(before);
    }
    tap([](S &s) { s.b = true; });
    check(!prop(settings, "open").toBool(), "pad ui: ○ closes settings");

    // R3 / L3: uiScale +0.10 and reset to 1.00x anywhere; L2 triggers Phonon voiceActive.
    tap([](S &s) { s.r3 = true; });
    check(qAbs(backend.uiScale() - 1.10) < 1e-6, "pad ui: R3 zooms in to 1.10x", QString::number(backend.uiScale()));
    tap([](S &s) { s.l3 = true; });
    check(qAbs(backend.uiScale() - 1.00) < 1e-6, "pad ui: L3 resets zoom to 1.00x", QString::number(backend.uiScale()));
    pad->processSnapshot([]() { S s; s.l2 = 1.0; return s; }(), t += 20);
    check(pad->voiceActive(), "pad ui: L2 hold sets voiceActive");
    pad->processSnapshot(S(), t += 20);
    check(!pad->voiceActive(), "pad ui: L2 release clears voiceActive");
    settle(30);
    // L2 opened the ask card (Phonon types into it) beside the island the cursor is on.
    QObject *omni = win->findChild<QObject *>("floatingOmnibar");
    {
        const QVariantMap chip = omni ? prop(omni, "chips").toList().value(0).toMap() : QVariantMap();
        check(omni && prop(omni, "open").toBool() && chip.value("kind") == "island",
              "pad ui: L2 opens the ask card with the highlighted island as its target", chip.value("label").toString());
    }
    tap([](S &s) { s.b = true; });
    check(omni && !prop(omni, "open").toBool(), "pad ui: ○ closes the ask card");

    // Share / touchpad: the new-session dialog; stick changes folder / harness; ○ closes.
    tap([](S &s) { s.touchpad = true; });
    check(prop(dialog, "open").toBool(), "pad ui: touchpad opens the new-session dialog");
    tap([](S &s) { s.dpadX = 1; });
    check(prop(dialog, "harnessIdx").toInt() == 1, "pad ui: d-pad right picks the next harness");
    tap([](S &s) { s.y = true; });
    check(prop(dialog, "modelIdx").toInt() == 1 || backend.harnessModels().value("codex").toList().size() < 2,
          "pad ui: △ picks the next model in the dialog");
    tap([](S &s) { s.b = true; });
    settle(150);
    check(!prop(dialog, "open").toBool(), "pad ui: ○ closes the dialog");
    tap([](S &s) { s.options = true; });
    check(prop(dialog, "open").toBool(), "pad ui: share opens the dialog");
    tap([](S &s) { s.options = true; });
    settle(150);
    check(!prop(dialog, "open").toBool(), "pad ui: share again closes it");

    // ✕ on an island zooms in, ○ zooms back out.
    app->setProperty("selectedIsland", 0);
    tap([](S &s) { s.a = true; });
    settle(500);
    check(prop(app, "lens").toInt() > 0, "pad ui: ✕ zooms into the selected island");
    tap([](S &s) { s.b = true; });
    settle(450);
    check(prop(app, "lens").toInt() == 0, "pad ui: ○ zooms back out");

    // A kickoff session (mock) yields chart, diff, table, pdf: ✕ on its Ladder enters the Stage.
    // The pad walks every viewer, the diff included, so code diffs are on for this run.
    backend.setShowCodeDiffs(true);
    backend.startSession(QStringLiteral("~/dev"), QStringLiteral("claude"),
                         backend.harnessModels().value("claude").toList().value(0).toString(),
                         QStringLiteral("profile fp4 kv-cache decode on sm120"));
    settle(300);
    check(prop(app, "lens").toInt() == 1, "pad ui: kickoff shows the Ladder");
    for (int i = 0; i < 14; ++i)
        backend.advance(0.5);
    settle(300);
    tap([](S &s) { s.a = true; });
    settle(500);
    check(prop(app, "lens").toInt() == 2, "pad ui: ✕ on the Ladder zooms into the Stage");
    const int arts = prop(stage, "artifactCount").toInt();
    check(arts >= 4, "pad ui: the kickoff Stage has chart, diff, table, pdf", QString::number(arts));
    tap([](S &s) { s.y = true; });
    check(arts < 2 || prop(app, "artifactIndex").toInt() == 1, "pad ui: △ slides to the next artifact");
    tap([](S &s) { s.l1 = true; });
    check(prop(app, "artifactIndex").toInt() == 0, "pad ui: L1 slides back");
    tap([](S &s) { s.dpadX = 1; });
    check(arts < 2 || prop(app, "artifactIndex").toInt() == 1, "pad ui: d-pad right slides too");
    tap([](S &s) { s.dpadX = -1; });
    settle(400);

    // Right stick on every scrollable viewer kind that the fixture has.
    auto stickScroll = [&](double ry, int ms) {
        S s;
        s.ry = ry;
        for (int i = 0; i <= ms; i += 8) {
            pad->processSnapshot(s, t + i);
            settle(1);
        }
        pad->processSnapshot(S(), t + ms + 8);
        t += ms + 40;
        settle(30);
    };
    QSet<QString> scrolled;
    QSet<QString> tried;
    const struct { const char *kind; const char *list; } viewers[] = {{"diff", "diffList"}, {"table", "csvList"}, {"pdf", "pdfFlick"}};
    for (int i = 0; i < arts; ++i) {
        app->setProperty("artifactIndex", i);
        settle(450);
        const QString kind = prop(stage, "kind").toString();
        for (const auto &v : viewers) {
            if (kind != QLatin1String(v.kind) || tried.contains(kind))
                continue;
            tried << kind;
            QQuickItem *list = nullptr;
            for (QQuickItem *it : findItems(win->contentItem(), QString::fromLatin1(v.list)))
                if (it->isVisible() && it->property("contentHeight").toDouble() > it->height() + 40)
                    list = it;
            if (!list) {
                std::printf("info  pad scroll: %s viewer too short to scroll, skipped\n", v.kind);
                continue;
            }
            list->setProperty("contentY", 0.0);
            settle(250);
            // 200 ms of full stick is ~26 ticks of 16 px (~416 px), clamped to the viewer's range.
            const double maxY = list->property("contentHeight").toDouble() - list->height();
            stickScroll(-1.0, 200);
            const double y1 = list->property("contentY").toDouble();
            stickScroll(1.0, 100);
            const double y2 = list->property("contentY").toDouble();
            const double want = std::min(maxY, 380.0);
            check(y1 >= want - 1 && y1 < 470 && y2 < y1, QStringLiteral("pad ui: right stick scrolls the %1 smoothly").arg(kind).toUtf8().constData(),
                  QStringLiteral("down %1 px (range %3), then up to %2").arg(y1).arg(y2).arg(maxY));
            if (y1 > 0 && y2 < y1)
                scrolled << kind;
        }
    }
    std::printf("info  pad scroll: viewers scrolled by the right stick: %s\n", qPrintable(QStringList(scrolled.values()).join(',')));
    check(!scrolled.isEmpty(), "pad ui: right stick scrolled at least one Stage viewer", QStringList(tried.values()).join(','));
    app->setProperty("artifactIndex", 0);
    settle(450);

    tap([](S &s) { s.x = true; });
    settle(400);
    check(prop(app, "proseOpen").toBool(), "pad ui: □ opens the write-up in the Stage");
    if (QQuickItem *flick = firstItem(win, "proseFlick")) {
        if (flick->property("contentHeight").toDouble() > flick->height() + 40) {
            stickScroll(-1.0, 120);
            check(flick->property("contentY").toDouble() > 100, "pad ui: right stick scrolls the write-up",
                  QString::number(flick->property("contentY").toDouble()));
        } else {
            std::printf("info  pad scroll: write-up too short to scroll, skipped\n");
        }
    }
    tap([](S &s) { s.b = true; });
    settle(300);
    check(!prop(app, "proseOpen").toBool() && prop(app, "lens").toInt() == 2, "pad ui: ○ closes the write-up first");

    // ✕ in the Stage (Prompt) opens the ask card beside the artifact, its chip attached; ○ closes
    // it. The docked composer stays collapsed until summoned; □ there toggles the model picker.
    check(!prop(app, "composerShown").toBool(), "pad ui: the docked composer is collapsed on the Stage");
    tap([](S &s) { s.a = true; });
    {
        const QVariantMap chip = omni ? prop(omni, "chips").toList().value(0).toMap() : QVariantMap();
        QQuickItem *card = firstItem(win, "omnibarCard");
        const QRectF cr = card ? QRectF(card->mapToScene(QPointF(0, 0)), card->size()) : QRectF();
        check(omni && prop(omni, "open").toBool() && chip.value("kind") == "artifact" && cr.width() > 0
                  && QRectF(0, 0, 1920, 1080).contains(cr),
              "pad ui: ✕ in the Stage opens the ask card at the highlighted artifact", chip.value("label").toString());
    }
    tap([](S &s) { s.b = true; });
    check(omni && !prop(omni, "open").toBool() && prop(app, "lens").toInt() == 2, "pad ui: ○ closes the ask card, Stage stays");
    QMetaObject::invokeMethod(composer, "focusInput", Q_ARG(QVariant, QVariant(QString())));
    settle(60);
    check(prop(composer, "inputFocused").toBool() && prop(app, "composerShown").toBool(),
          "pad ui: the composer slides in when summoned");
    tap([](S &s) { s.x = true; });
    check(prop(composer, "pickerOpen").toBool(), "pad ui: □ in the composer opens the model & effort picker");
    {
        const QString m0 = backend.chatModel();
        tap([](S &s) { s.dpadY = -1; });
        check(backend.chatModel() != m0, "pad ui: d-pad down cycles the model in the picker", backend.chatModel());
        backend.setChatModel(m0);
    }
    tap([](S &s) { s.b = true; });
    check(!prop(composer, "pickerOpen").toBool() && prop(composer, "inputFocused").toBool(), "pad ui: ○ closes the picker, composer keeps focus");
    tap([](S &s) { s.r1 = true; });
    check(!prop(composer, "inputText").toString().isEmpty() || prop(app, "promptSuggestion").toString().isEmpty(),
          "pad ui: R1 in the composer accepts the suggested prompt", prop(composer, "inputText").toString());
    tap([](S &s) { s.b = true; });
    tap([](S &s) { s.b = true; });
    check(!prop(composer, "inputFocused").toBool(), "pad ui: ○ clears then leaves the composer");
    tap([](S &s) { s.b = true; });
    settle(450);
    check(prop(app, "lens").toInt() == 0, "pad ui: ○ from the Stage zooms out");

    check(Gamepad::platformCalls() == 0, "safety: zero GameController.framework calls in --test",
          QString::number(Gamepad::platformCalls()));
    check(backend.spawnCount() == 0, "safety: pad tests spawned zero processes");
}

void openAiCompatUnitTests()
{
    const QByteArray body = buildOpenAiCompatRequest(
        QStringLiteral("qwen3.8-flash-next"), QStringLiteral("SYS"), QStringLiteral("CTX"),
        QStringLiteral("What changed?"), QStringLiteral("low"));
    const QJsonObject req = QJsonDocument::fromJson(body).object();
    const QJsonArray msgs = req.value(QLatin1String("messages")).toArray();
    check(req.value(QLatin1String("model")).toString() == "qwen3.8-flash-next"
              && req.value(QLatin1String("stream")).toBool() && msgs.size() == 3
              && msgs[0].toObject().value(QLatin1String("role")).toString() == "system"
              && msgs[1].toObject().value(QLatin1String("content")).toString().startsWith("Pinned context:")
              && msgs[2].toObject().value(QLatin1String("content")).toString() == "Question: What changed?",
          "openai-compat: request keeps system + pinned context first for prefix cache");

    const OpenAiSseEvent d1 = parseOpenAiSseLine(
        R"(data: {"choices":[{"delta":{"reasoning_content":"checking KV","content":""}}]})");
    check(d1.kind == OpenAiSseEvent::Delta && d1.thinking == "checking KV" && d1.text.isEmpty(),
          "openai-compat: SSE reasoning_content delta parsed");

    const OpenAiSseEvent d2 = parseOpenAiSseLine(
        R"(data: {"choices":[{"delta":{"content":"126 tok/s"}}]})");
    check(d2.kind == OpenAiSseEvent::Delta && d2.text == "126 tok/s" && d2.thinking.isEmpty(),
          "openai-compat: SSE content delta parsed");

    const OpenAiSseEvent done = parseOpenAiSseLine("data: [DONE]");
    check(done.kind == OpenAiSseEvent::Done, "openai-compat: SSE [DONE] parsed");

    const OpenAiSseEvent err = parseOpenAiSseLine(R"(data: {"error":{"message":"model overloaded"}})");
    check(err.kind == OpenAiSseEvent::Error && err.error == "model overloaded",
          "openai-compat: SSE error parsed");

    check(channelKeyFor(QStringLiteral("qwen3.8-flash-next"), QStringLiteral("low"))
              == "openai-compat:qwen3.8-flash-next",
          "openai-compat: channel key stable");

    Backend b(Backend::Mode::Mock);
    for (int i = 0; i < 15; ++i)
        b.recordAction(QStringLiteral("act %1").arg(i));
    check(b.recentActions().size() == 12 && b.recentActions().first() == "act 3"
              && b.recentActions().last() == "act 14",
          "openai-compat: recordAction keeps last 12 actions for intent inference");

    // Phonon live transcript tail: missing returns empty; fresh symlink returns last N non-empty lines.
    {
        QTemporaryDir td;
        check(readPhononTail(td.path() + "/missing.txt").isEmpty(),
              "phonon tail: missing file returns empty");
        const QString realPath = td.path() + "/transcript.txt";
        QFile f(realPath);
        if (f.open(QIODevice::WriteOnly)) {
            f.write("line 1\n\nline 2\nline 3\nline 4\nline 5\nline 6\n  \nline 7\n");
            f.close();
        }
        const QString linkPath = td.path() + "/latest.txt";
        QFile::link(realPath, linkPath);
        const QString tail = readPhononTail(linkPath, 5, 60000);
        check(tail == QStringLiteral("line 3\nline 4\nline 5\nline 6\nline 7"),
              "phonon tail: fresh symlink returns last 5 non-empty lines");
    }

    // KV pre-warm: one token, not streamed, and the same system + pinned-context prefix as an ask.
    {
        const QString model = QStringLiteral("qwen3.8-flash-next");
        const QString ctx = QStringLiteral("Project: lapis (~/dev/lapis).\nLast 2 turns:\nUser: why\nAssistant: KV");
        const QJsonObject warm = QJsonDocument::fromJson(buildOpenAiWarmRequest(model, kCompatSystemPrompt, ctx)).object();
        const QJsonObject ask = QJsonDocument::fromJson(
            buildOpenAiCompatRequest(model, kCompatSystemPrompt, ctx, QStringLiteral("what changed?"), QStringLiteral("low"))).object();
        const QJsonArray wm = warm.value("messages").toArray();
        const QJsonArray am = ask.value("messages").toArray();
        check(warm.value("max_tokens").toInt() == 1 && warm.contains("stream") && !warm.value("stream").toBool(true)
                  && warm.value("model").toString() == model,
              "kv warm: max_tokens 1, stream false, same model");
        check(wm.size() == 2 && am.size() == 3 && wm[0] == am[0] && wm[1] == am[1]
                  && wm[1].toObject().value("content").toString() == "Pinned context:\n" + ctx,
              "kv warm: messages are a byte-identical prefix of the ask's");
        check(!warm.contains("reasoning_effort"), "kv warm: no effort field (nothing is generated)");

        Backend m(Backend::Mode::Mock);
        m.setOpenaiCompatUrl(QStringLiteral("http://127.0.0.1:9/v1"));
        const QVariantList ps = m.projects();
        if (!ps.isEmpty())
            m.setFocusedProject(ps.first().toMap().value("key").toString());
        m.warmProjectPrefix(QStringLiteral("anything"));
        check(m.warmRequests() == 0 && m.spawnCount() == 0, "kv warm: mock mode never sends a warm request",
              QString::number(m.warmRequests()));
    }
}

// Backend on top of lapis_llm: the context trunk / slice split, askStream refusing in mock mode,
// and a two-turn conversation through an injected LlmRouter against a local SSE server.
void llmRouterBackendTests()
{
    Backend b(Backend::Mode::Mock);
    const QVariantList ps = b.projects();
    QString key;
    for (const QVariant &v : ps) {
        const QString k = v.toMap().value("key").toString();
        if (!b.contextSlice(k, 0).isEmpty()) {
            key = k;
            break;
        }
    }
    check(!key.isEmpty(), "llm ctx: a mock project has an on-screen slice");
    const QString full = b.contextFor(key, 0);
    const QString trunk = b.contextTrunk(key);
    const QString slice = b.contextSlice(key, 0);
    check(slice.startsWith("On screen:") && !trunk.contains("On screen:"),
          "llm ctx: slice holds the on-screen artifact, the trunk never does", slice.left(80));
    check(QString(full).replace(QLatin1Char(' ') + slice, QString()) == trunk,
          "llm ctx: contextFor = trunk with the slice spliced in (unchanged output)");
    check(b.contextTrunk(key) == trunk && b.contextSlice(key, 1) != slice,
          "llm ctx: switching artifacts changes only the slice, never the Tier 1 trunk");

    // Mock mode with no router: refuse with an error, touch nothing.
    QStringList finished;
    QObject::connect(&b, &Backend::streamFinished, [&finished](const QString &id, const QString &err) {
        finished << id + "|" + err;
    });
    b.askStream(QStringLiteral("s1"), key, 0, QStringLiteral("why?"));
    check(finished.size() == 1 && finished[0].startsWith("s1|no LLM router") && b.spawnCount() == 0,
          "llm stream: mock mode without a router finishes with an error, no network", finished.join(';'));
    b.askStream(QStringLiteral("__quick__"), key, 0, QStringLiteral("x"));
    check(finished.size() == 1, "llm stream: the internal quick stream id is refused");

    // Injected router + local SSE server: a two-turn conversation keeps Tier 2.
    QTcpServer server;
    QList<QByteArray> bodies;
    int replyNo = 0;
    QObject::connect(&server, &QTcpServer::newConnection, [&] {
        while (QTcpSocket *s = server.nextPendingConnection()) {
            auto buf = std::make_shared<QByteArray>();
            QObject::connect(s, &QTcpSocket::readyRead, s, [&, s, buf] {
                *buf += s->readAll();
                const qsizetype he = buf->indexOf("\r\n\r\n");
                if (he < 0)
                    return;
                const QByteArray head = buf->left(he).toLower();
                const qsizetype cl = head.indexOf("content-length:");
                const qsizetype eol = head.indexOf("\r\n", cl);
                const qsizetype len = cl < 0 ? 0 : head.mid(cl + 15, eol < 0 ? -1 : eol - cl - 15).trimmed().toLongLong();
                if (buf->size() < he + 4 + len)
                    return;
                bodies << buf->mid(he + 4, len);
                const QByteArray answer = ++replyNo == 1 ? "126 tok/s at c1." : "MTP adds 15 tok/s.";
                const QByteArray body = "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"checking\"}}]}\n\n"
                                        "data: {\"choices\":[{\"delta\":{\"content\":\"" + answer + "\"}}]}\n\n"
                                        "data: [DONE]\n\n";
                s->write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: "
                         + QByteArray::number(body.size()) + "\r\n\r\n" + body);
                s->disconnectFromHost();
            });
            QObject::connect(s, &QTcpSocket::disconnected, s, &QObject::deleteLater);
        }
    });
    check(server.listen(QHostAddress::LocalHost, 0), "llm stream: local SSE server listens");
    LlmRouter router;
    b.setLlmRouter(&router);
    b.setOpenaiCompatUrl(QStringLiteral("http://127.0.0.1:%1/v1").arg(server.serverPort()));
    QString text, thinking;
    QObject::connect(&b, &Backend::streamDelta, [&](const QString &id, const QString &t, const QString &th) {
        if (id == "conv") {
            text += t;
            thinking += th;
        }
    });
    auto waitFinished = [&](int n) {
        QElapsedTimer t;
        t.start();
        while (finished.size() < n && t.elapsed() < 5000) {
            QCoreApplication::processEvents(QEventLoop::AllEvents, 5);
            QThread::msleep(1);
        }
        return finished.size() >= n;
    };
    b.askStream(QStringLiteral("conv"), key, 0, QStringLiteral("what is the tok/s?"));
    check(waitFinished(2) && finished.value(1) == "conv|" && text == "126 tok/s at c1." && thinking == "checking",
          "llm stream: askStream streams deltas through the injected router", finished.join(';') + " / " + text);
    const QVector<ChatTurn> h = b.streamHistory(QStringLiteral("conv"));
    check(h.size() == 2 && h[0].content == "what is the tok/s?" && h[1].role == "assistant"
              && h[1].content == "126 tok/s at c1.",
          "llm stream: a clean finish appends the exchange to Tier 2");
    b.askStream(QStringLiteral("conv"), key, 0, QStringLiteral("and with MTP?"));
    check(waitFinished(3) && finished.value(2) == "conv|", "llm stream: second turn finished", finished.join(';'));
    const QJsonArray m1 = QJsonDocument::fromJson(bodies.value(0)).object().value("messages").toArray();
    const QJsonArray m2 = QJsonDocument::fromJson(bodies.value(1)).object().value("messages").toArray();
    check(m1.size() == 3 && m2.size() == 5 && m1[0] == m2[0] && m1[1] == m2[1]
              && m2[1].toObject().value("content").toString() == "Pinned context:\n" + trunk,
          "llm stream: both turns send byte-identical Tier 0 + Tier 1 (the trunk)",
          QStringLiteral("%1/%2").arg(m1.size()).arg(m2.size()));
    check(m2[2].toObject().value("content") == "what is the tok/s?" && m2[3].toObject().value("content") == "126 tok/s at c1."
              && m2[4].toObject().value("content").toString().endsWith("Question: and with MTP?")
              && m2[4].toObject().value("content").toString().contains("On screen now:\n" + slice),
          "llm stream: Tier 2 history precedes the Tier 3 leaf (slice + question last)",
          m2.size() > 4 ? m2[4].toObject().value("content").toString().left(160) : QString());
    b.clearStreamHistory(QStringLiteral("conv"));
    check(b.streamHistory(QStringLiteral("conv")).isEmpty() && b.spawnCount() == 0,
          "llm stream: history clears; no processes spawned");
    b.setLlmRouter(nullptr);
}

// --- Floating omnibar / hover targets ---------------------------------------------------------

// A local OpenAI-compatible SSE endpoint: every POST gets one reasoning delta and then `answer`,
// unless `hold` is set, in which case the request stays open with no reply (a slow model).
struct FakeSse {
    QTcpServer server;
    QList<QByteArray> bodies;
    QString answer = QStringLiteral("Row 2 leads.");
    bool hold = false;

    static QByteArray chunk(const char *field, const QString &v)
    {
        const QJsonObject delta{{QString::fromLatin1(field), v}};
        const QJsonObject o{{QStringLiteral("choices"), QJsonArray{QJsonObject{{QStringLiteral("delta"), delta}}}}};
        return "data: " + QJsonDocument(o).toJson(QJsonDocument::Compact) + "\n\n";
    }
    bool start()
    {
        QObject::connect(&server, &QTcpServer::newConnection, &server, [this] {
            while (QTcpSocket *s = server.nextPendingConnection()) {
                auto buf = std::make_shared<QByteArray>();
                auto done = std::make_shared<bool>(false);
                QObject::connect(s, &QTcpSocket::readyRead, s, [this, s, buf, done] {
                    *buf += s->readAll();
                    const qsizetype he = buf->indexOf("\r\n\r\n");
                    if (*done || he < 0)
                        return;
                    const QByteArray head = buf->left(he).toLower();
                    const qsizetype cl = head.indexOf("content-length:");
                    const qsizetype eol = head.indexOf("\r\n", cl);
                    const qsizetype len =
                        cl < 0 ? 0 : head.mid(cl + 15, eol < 0 ? -1 : eol - cl - 15).trimmed().toLongLong();
                    if (buf->size() < he + 4 + len)
                        return;
                    *done = true;
                    bodies << buf->mid(he + 4, len);
                    if (hold)
                        return;
                    const QByteArray body = chunk("reasoning_content", QStringLiteral("comparing the pinned cells"))
                                            + chunk("content", answer) + "data: [DONE]\n\n";
                    s->write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n"
                             "Content-Length: " + QByteArray::number(body.size()) + "\r\n\r\n" + body);
                    s->disconnectFromHost();
                });
                QObject::connect(s, &QTcpSocket::disconnected, s, &QObject::deleteLater);
            }
        });
        return server.listen(QHostAddress::LocalHost, 0);
    }
    QString url() const { return QStringLiteral("http://127.0.0.1:%1/v1").arg(server.serverPort()); }
};

bool waitUntil(const std::function<bool()> &ok, int ms)
{
    QElapsedTimer t;
    t.start();
    while (!ok() && t.elapsed() < ms) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 5);
        QThread::msleep(1);
    }
    return ok();
}

void mouseMove(QQuickWindow *w, QPointF p, Qt::MouseButtons held = Qt::NoButton)
{
    QMouseEvent ev(QEvent::MouseMove, p, w->mapToGlobal(p), Qt::NoButton, held, Qt::NoModifier);
    QCoreApplication::sendEvent(w, &ev);
}

void mouseButton(QQuickWindow *w, QEvent::Type type, QPointF p, Qt::MouseButton b, Qt::KeyboardModifiers mods = {})
{
    QMouseEvent ev(type, p, w->mapToGlobal(p), b, type == QEvent::MouseButtonPress ? Qt::MouseButtons(b) : Qt::NoButton,
                   mods);
    QCoreApplication::sendEvent(w, &ev);
}

void mouseClick(QQuickWindow *w, QPointF p, Qt::MouseButton b, Qt::KeyboardModifiers mods = {})
{
    mouseButton(w, QEvent::MouseButtonPress, p, b, mods);
    mouseButton(w, QEvent::MouseButtonRelease, p, b, mods);
}

// Left-drag from `from` by `delta` in small steps (past the drag threshold).
void mouseDrag(QQuickWindow *w, QPointF from, QPointF delta)
{
    mouseMove(w, from);
    mouseButton(w, QEvent::MouseButtonPress, from, Qt::LeftButton);
    const int steps = 24;
    for (int i = 1; i <= steps; ++i) {
        mouseMove(w, from + delta * (qreal(i) / steps), Qt::LeftButton);
        QCoreApplication::processEvents();
    }
    // Hold still before letting go: a drop, not a flick (Constellation cards fly on release speed).
    waitUntil([] { return false; }, 90);
    mouseButton(w, QEvent::MouseButtonRelease, from + delta, Qt::LeftButton);
}

QQuickItem *visibleItem(QQuickWindow *w, const char *name)
{
    for (QQuickItem *it : findItems(w->contentItem(), QString::fromLatin1(name)))
        if (it->isVisible() && it->width() > 0 && it->height() > 0)
            return it;
    return nullptr;
}

int visibleCount(QQuickWindow *w, const char *name)
{
    int n = 0;
    for (QQuickItem *it : findItems(w->contentItem(), QString::fromLatin1(name)))
        n += it->isVisible() ? 1 : 0;
    return n;
}

// Opens the first mock project with an artifact of KIND on the Stage, showing that artifact.
bool showArtifactKind(QObject *app, Backend &b, const QString &kind, QString *projectKey = nullptr)
{
    const QVariantList ps = b.projects();
    for (int i = 0; i < ps.size(); ++i) {
        const QVariantList arts = ps[i].toMap().value(QStringLiteral("artifacts")).toList();
        for (int a = 0; a < arts.size(); ++a) {
            if (arts[a].toMap().value(QStringLiteral("kind")).toString() != kind)
                continue;
            QMetaObject::invokeMethod(app, "enterProject", Q_ARG(QVariant, QVariant(i)));
            app->setProperty("artifactIndex", a);
            settle(700);
            if (projectKey)
                *projectKey = ps[i].toMap().value(QStringLiteral("key")).toString();
            return true;
        }
    }
    return false;
}

// Window point at the centre of table row `row`, column `col` of the visible CsvTable.
QPointF tableCellPoint(QQuickWindow *win, int row, int col)
{
    QQuickItem *table = visibleItem(win, "tableView");
    QQuickItem *list = visibleItem(win, "csvList");
    if (!table || !list)
        return {-1, -1};
    QQuickItem *rowItem = nullptr;
    QMetaObject::invokeMethod(list, "itemAtIndex", Q_RETURN_ARG(QQuickItem *, rowItem), Q_ARG(int, row));
    if (!rowItem)
        return {-1, -1};
    const QVariantList colW = table->property("colW").toList();
    qreal x = 0;
    for (int c = 0; c < col && c < colW.size(); ++c)
        x += colW[c].toReal();
    const qreal w = colW.value(col).toReal();
    return rowItem->mapToScene(QPointF(x + std::min<qreal>(w / 2, 40), rowItem->height() / 2));
}

// Every slot id and on-screen glyph a PadIcon accepts.
QStringList padIconProbeButtons()
{
    return {"a", "b", "x", "y", "l1", "r1", "l2", "r2", "l3", "r3", "lstick", "rstick", "menu", "options",
            "dpad", "dpad_up", "dpad_down", "dpad_left", "dpad_right", "dpad_ud", "dpad_lr",
            "✕", "○", "□", "△", "☰", "Share", "PAD", "↕", "◄►"};
}

// A black sheet over everything: one row of PadIcons idle, one active (objectName padIconProbe).
// *owner gets the created object (delete it when done).
QQuickItem *makePadIconSheet(QQmlEngine &engine, QQuickItem *parent, int size, QObject **owner)
{
    QQmlComponent comp(&engine);
    comp.setData(QByteArrayLiteral(
                     "import QtQuick\nimport Studio\n"
                     "Rectangle { property var buttons: []; property int iconSize: 28; color: \"#000000\"; z: 100000\n"
                     "  width: col.implicitWidth + 24; height: col.implicitHeight + 24\n"
                     "  Column { id: col; x: 12; y: 12; spacing: iconSize * 0.6\n"
                     "    Repeater { model: 2\n"
                     "      Row { id: row; required property int index; spacing: iconSize * 0.45\n"
                     "        Repeater { model: buttons\n"
                     "          PadIcon { required property string modelData; objectName: \"padIconProbe\"\n"
                     "                    anchors.verticalCenter: parent.verticalCenter\n"
                     "                    button: modelData; size: iconSize; active: row.index === 1 } } } } } }\n"),
                 QUrl(QStringLiteral("qrc:/padiconprobe.qml")));
    QObject *obj = comp.createWithInitialProperties({{"buttons", padIconProbeButtons()}, {"iconSize", size}});
    if (!obj)
        std::fprintf(stderr, "pad icon sheet: %s\n", qPrintable(comp.errorString()));
    auto *sheet = qobject_cast<QQuickItem *>(obj);
    if (sheet)
        sheet->setParentItem(parent);
    if (owner)
        *owner = obj;
    return sheet;
}

void omnibarUiTests()
{
    Backend backend(Backend::Mode::Mock);
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    check(win != nullptr, "omni: QML loads");
    if (!win)
        return;
    QObject *app = win->findChild<QObject *>("studioApp");
    QQuickItem *omni = firstItem(win, "floatingOmnibar");
    QQuickItem *card = firstItem(win, "omnibarCard");
    TargetRegistry *reg = backend.targets();
    check(app && omni && card && backend.property("targets").value<QObject *>() == reg,
          "omni: FloatingOmnibar present; backend.targets is the registry");
    if (!app || !omni || !card)
        return;
    settle(450);
    auto isOpen = [&] { return omni->property("open").toBool(); };
    auto focusName = [&] {
        QQuickItem *f = win->activeFocusItem();
        return f ? f->objectName() : QString();
    };
    auto cardRect = [&] { return QRectF(card->x(), card->y(), card->width(), card->height()); };
    auto rectText = [](const QRectF &r) {
        return QStringLiteral("%1,%2 %3x%4").arg(r.x()).arg(r.y()).arg(r.width()).arg(r.height());
    };
    const QString key0 = backend.projects().value(0).toMap().value("key").toString();
    check(!isOpen() && !card->isVisible(), "omni: closed at start");

    // Cmd+K opens with focus in the prompt; Esc closes and hands focus back.
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(30);
    check(isOpen() && card->isVisible() && focusName() == "omnibarInput", "omni: Cmd+K opens, prompt focused",
          focusName());
    sendKey(win, Qt::Key_Escape);
    settle(30);
    check(!isOpen() && !card->isVisible() && focusName() != "omnibarInput", "omni: Esc closes and returns focus");
    check(backend.navMode() == "controller", "omni: Cmd+K no longer cycles the keymap", backend.navMode());

    // "/" with no text field focused opens it too.
    sendKey(win, Qt::Key_Slash, Qt::NoModifier, "/");
    settle(30);
    check(isOpen() && focusName() == "omnibarInput", "omni: / opens the card");
    sendKey(win, Qt::Key_Escape);

    // A real pointer over an island reports a session target; leaving clears it.
    {
        QQuickItem *isl = findItems(win->contentItem(), "island").value(0);
        const QPointF c = isl ? isl->mapToScene(QPointF(isl->width() / 2, isl->height() / 2)) : QPointF();
        mouseMove(win, c);
        settle(40);
        const QVariantMap t = reg->activeTarget();
        check(t.value("kind") == "session" && t.value("sessionId") == key0 && !t.value("label").toString().isEmpty()
                  && QRectF(t.value("x").toReal(), t.value("y").toReal(), t.value("w").toReal(), t.value("h").toReal())
                         .contains(c),
              "hover: island reports kind=session, its key, name and window rect",
              t.value("kind").toString() + " " + t.value("sessionId").toString());
        // Shift+Click on the hovered island opens the card under it (and does not enter it).
        mouseClick(win, c, Qt::LeftButton, Qt::ShiftModifier);
        settle(40);
        check(isOpen() && app->property("lens").toInt() == 0
                  && omni->property("chips").toList().value(0).toMap().value("kind") == "session",
              "omni: Shift+Click on a hovered island opens the card about it, no zoom",
              QString::number(app->property("lens").toInt()));
        sendKey(win, Qt::Key_Escape);
        mouseMove(win, QPointF(3, 3));
        settle(40);
        check(!reg->hasActiveTarget(), "hover: leaving the island clears the target");
    }

    // Middle-click anywhere opens it at the pointer.
    mouseClick(win, QPointF(3, 300), Qt::MiddleButton);
    settle(30);
    check(isOpen() && std::abs(card->x() - 16) < 1 && std::abs(card->y() - 310) < 1,
          "omni: middle-click opens the card at the pointer (clamped to the 16 px margin)", rectText(cardRect()));
    sendKey(win, Qt::Key_Escape);

    // Anchoring: right beside the target (x + w + 14, its top); a corner target keeps the card
    // inside the window.
    reg->setHoverTarget("table_cell", key0, "0", "row=2/col=3", "tok/s: 126", "Table: sweep\n> Row 2: c1 | 126",
                        300, 200, 110, 30);
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(30);
    check(isOpen() && card->x() == 424 && card->y() == 200, "omni: card anchors right beside the hovered target",
          rectText(cardRect()));
    sendKey(win, Qt::Key_Escape);
    reg->setHoverTarget("table_cell", key0, "0", "row=9/col=3", "tok/s: 410", "corner", 1860, 1050, 50, 20);
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(30);
    {
        const QRectF r = cardRect();
        check(r.left() >= 16 && r.top() >= 16 && r.right() <= 1920 - 16 + 0.5 && r.bottom() <= 1080 - 16 + 0.5,
              "omni: card is clamped inside the window bounds", rectText(r));
    }
    sendKey(win, Qt::Key_Escape);

    // Dragging the header moves the card.
    reg->setHoverTarget("table_cell", key0, "0", "row=2/col=3", "tok/s: 126", "Table: sweep\n> Row 2: c1 | 126",
                        300, 200, 110, 30);
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(30);
    {
        QQuickItem *header = firstItem(win, "omnibarHeader");
        const QPointF from = header ? header->mapToScene(QPointF(header->width() / 2, header->height() / 2)) : QPointF();
        const QRectF before = cardRect();
        mouseDrag(win, from, QPointF(240, 120));
        settle(30);
        const QRectF after = cardRect();
        check(header && std::abs(after.x() - before.x() - 240) < 2 && std::abs(after.y() - before.y() - 120) < 2,
              "omni: dragging the header moves the card", rectText(before) + " -> " + rectText(after));
        QMetaObject::invokeMethod(app, "takeFocus");
        sendKey(win, Qt::Key_Escape);  // focus left the prompt: the app's Esc closes it
        settle(30);
        check(!isOpen(), "omni: Esc from the app closes the card too");
    }

    // Multi-target pins: Tab in the prompt pins the hovered target; a repeat is deduplicated.
    reg->setHoverTarget("table_cell", key0, "0", "row=2/col=3", "tok/s: 126", "Table: sweep\n> Row 2: c1 | 126",
                        300, 200, 110, 30);
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(20);
    sendKey(win, Qt::Key_Tab);
    reg->setHoverTarget("diff_line", key0, "1", "line=7", "fix.diff L7", "Diff: fix\n> 7 + return x;", 300, 600, 400, 30);
    sendKey(win, Qt::Key_Tab);
    sendKey(win, Qt::Key_Tab);
    settle(40);
    check(reg->pinnedCount() == 2 && visibleCount(win, "omnibarChip") == 2 && isOpen() && focusName() == "omnibarInput",
          "omni: Tab pins two targets as two chips (repeat deduplicated), prompt keeps focus",
          QStringLiteral("%1 pins, %2 chips").arg(reg->pinnedCount()).arg(visibleCount(win, "omnibarChip")));

    // Dispatch: Enter sends askStreamWithSlice on omni:<card> with both pins as the Tier 3 slice.
    FakeSse sse;
    check(sse.start(), "omni: local SSE server listens");
    LlmRouter router;
    backend.setLlmRouter(&router);
    backend.setOpenaiCompatUrl(sse.url());
    QStringList finished;
    const auto conn = QObject::connect(&backend, &Backend::streamFinished,
                                       [&finished](const QString &id, const QString &e) { finished << id + "|" + e; });
    typeText(win, "which is faster");
    sendKey(win, Qt::Key_Return);
    const QString sid = omni->property("streamId").toString();
    check(sid.startsWith("omni:") && omni->property("busy").toBool(), "omni: Enter starts a stream on omni:<card>", sid);
    waitUntil([&] { return !omni->property("busy").toBool(); }, 5000);
    check(finished.contains(sid + "|") && omni->property("answer").toString() == sse.answer
              && omni->property("thinking").toString() == "comparing the pinned cells",
          "omni: the answer and thinking stream into the card", finished.join(';'));
    {
        const QJsonArray msgs =
            QJsonDocument::fromJson(sse.bodies.value(0)).object().value("messages").toArray();
        const QString leaf = msgs.isEmpty() ? QString() : msgs.last().toObject().value("content").toString();
        check(leaf.contains("Pointed-at targets (2):") && leaf.contains("tok/s: 126") && leaf.contains("fix.diff L7")
                  && leaf.endsWith("Question: which is faster"),
              "omni: Tier 3 carries both pinned targets, then the question", leaf.left(200));
        check(msgs.size() == 3 && msgs[1].toObject().value("content").toString() == "Pinned context:\n" + backend.contextTrunk(key0),
              "omni: Tier 0 + Tier 1 are the warmed prefix, unchanged (cache-hot)", QString::number(msgs.size()));
        check(omni->property("lastProject").toString() == key0 && omni->property("lastArtifact").toInt() == 0,
              "omni: the first chip names the project and artifact of the ask");
    }
    {
        const QImage shot = win->grabWindow();
        check(forbiddenPixels(shot) == 0, "omni: open card with an answer has zero blue/green pixels",
              QString::number(forbiddenPixels(shot)));
    }

    // Pin card: the card detaches into a persistent answer card; a second card streams beside it.
    QMetaObject::invokeMethod(omni, "pinCard");
    settle(30);
    check(!isOpen() && omni->property("pinnedCardCount").toInt() == 1 && visibleCount(win, "pinnedAskCard") == 1,
          "omni: Pin card detaches it into a persistent floating card");
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(20);
    sse.answer = QStringLiteral("Because MTP adds 15 tok/s.");
    typeText(win, "and why");
    sendKey(win, Qt::Key_Return);
    const QString sid2 = omni->property("streamId").toString();
    waitUntil([&] { return !omni->property("busy").toBool(); }, 5000);
    QMetaObject::invokeMethod(omni, "pinCard");
    settle(30);
    check(sid2 != sid && sid2.startsWith("omni:") && omni->property("pinnedCardCount").toInt() == 2
              && visibleCount(win, "pinnedAskCard") == 2,
          "omni: a second card has its own stream id and pins beside the first", sid + " / " + sid2);

    // Cancel: Esc on a card whose request is still open cancels its stream; no late finish arrives.
    sse.hold = true;
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(20);
    typeText(win, "slow one");
    sendKey(win, Qt::Key_Return);
    const QString sid3 = omni->property("streamId").toString();
    waitUntil([&] { return sse.bodies.size() >= 3; }, 5000);
    check(router.activeStreams() == 1 && omni->property("busy").toBool(), "omni: the slow ask is in flight",
          QString::number(router.activeStreams()));
    sendKey(win, Qt::Key_Escape);
    settle(30);
    check(!isOpen() && !omni->property("busy").toBool() && router.activeStreams() == 0,
          "omni: Esc closes the card and cancels its stream", QString::number(router.activeStreams()));
    settle(200);
    check(!finished.join(';').contains(sid3), "omni: no finish event arrives for the cancelled card", finished.join(';'));

    // Closing a pinned card removes it.
    QMetaObject::invokeMethod(omni, "closePinned", Q_ARG(QVariant, QVariant(0)));
    QMetaObject::invokeMethod(omni, "closePinned", Q_ARG(QVariant, QVariant(0)));
    settle(20);
    check(omni->property("pinnedCardCount").toInt() == 0 && visibleCount(win, "pinnedAskCard") == 0,
          "omni: closing pinned cards removes them");
    QObject::disconnect(conn);

    // Harness tags never show: anywhere in the reply, mid-stream, and between a CUA turn and its
    // follow-up (which reads as its own paragraph). The QML and C++ strippers agree.
    {
        auto qmlShown = [&](const QString &raw, bool streaming) {
            QVariant out;
            QMetaObject::invokeMethod(omni, "displayAnswer", Q_RETURN_ARG(QVariant, out), Q_ARG(QVariant, raw),
                                      Q_ARG(QVariant, streaming));
            return out.toString();
        };
        const QString turn1 = QStringLiteral("Typing x.com into Chrome's address bar. [[ACTION:cua_type:Chrome address bar|x.com\n]]");
        const QString turn2 = QStringLiteral("x.com is loaded in Chrome.");
        const QString want = QStringLiteral("Typing x.com into Chrome's address bar.\n\nx.com is loaded in Chrome.");
        const struct { QString raw; bool streaming; QString want; } cases[] = {
            {turn1 + turn2, false, want},                                  // the screenshot: no separator
            {turn1 + cuaFollowupBreak(turn1) + turn2, false, want},        // with the backend's break
            {QStringLiteral("Opening it. [[ACTION:cua_cl"), true, QStringLiteral("Opening it.")},
            {QStringLiteral("Opening it. [["), true, QStringLiteral("Opening it.")},
            {QStringLiteral("Opening it. ["), true, QStringLiteral("Opening it.")},
            {QStringLiteral("[[ACTION:open_settings]] Done."), false, QStringLiteral("Done.")},
            {QStringLiteral("Click [[ACTION:cua_click:Send]] now."), false, QStringLiteral("Click now.")},
            {QStringLiteral("Sure. [[ACTION:cua_type:To: field|a|b]]"), false, QStringLiteral("Sure.")},
            {QStringLiteral("a [[ACTION:x]] [[ACTION:y]] b"), false, QStringLiteral("a b")},
            {QStringLiteral("[[IGNORE]]"), false, QString()},
        };
        int qmlBad = 0, cppBad = 0;
        QStringList misses;
        for (const auto &c : cases) {
            const QString q = qmlShown(c.raw, c.streaming), p = answerDisplayText(c.raw, c.streaming);
            qmlBad += q != c.want;
            cppBad += p != c.want;
            if (q != c.want || p != c.want)
                misses << QStringLiteral("[%1] qml=[%2] cpp=[%3]").arg(c.raw, q, p);
        }
        check(qmlBad == 0, "omni: displayAnswer strips every [[ACTION:..]] tag (mid-text, partial) with paragraph breaks",
              misses.join("; "));
        check(cppBad == 0, "spotlight: answerDisplayText strips every tag the same way", misses.join("; "));
        check(cuaFollowupBreak(turn1) == "\n\n" && cuaFollowupBreak(QStringLiteral("done.\n")) == "\n"
                  && cuaFollowupBreak(QStringLiteral("done.\n\n")).isEmpty() && cuaFollowupBreak(QStringLiteral("  ")).isEmpty(),
              "cua: a follow-up turn opens a fresh paragraph only when one is needed");

        // On screen: the answer text of an open card carries no tag and two paragraphs.
        sendKey(win, Qt::Key_K, Qt::ControlModifier);
        settle(20);
        omni->setProperty("question", QStringLiteral("open x.com"));
        omni->setProperty("answer", turn1 + turn2);
        settle(40);
        QQuickItem *ans = firstItem(win, "omnibarAnswer");
        const QString html = ans ? ans->property("text").toString() : QString();
        check(ans && ans->isVisible() && !html.contains("ACTION") && html.contains("address bar.")
                  && html.contains("x.com is loaded"),
              "omni: the rendered answer shows no harness tag", html.left(240));
        // A raw [[ACTION still streaming in never flashes.
        omni->setProperty("busy", true);
        omni->setProperty("answer", QStringLiteral("Typing x.com. [[ACTION:cua_type:Chrome addr"));
        settle(20);
        check(ans && !ans->property("text").toString().contains("[["), "omni: a partial tag never flashes while streaming");
        omni->setProperty("busy", false);

        // The header is quiet: no model slug, one close control.
        const QString slug = backend.openaiCompatModel();
        QStringList texts;
        visibleWords(win, &texts);
        check((slug.isEmpty() || !texts.join(' ').contains(slug)) && visibleCount(win, "omnibarClose") == 1,
              "omni: the header names no model slug and has exactly one close",
              QStringLiteral("%1 closes, slug '%2'").arg(visibleCount(win, "omnibarClose")).arg(slug));
        sendKey(win, Qt::Key_Escape);
        settle(20);
    }

    // L2 (focusForVoice) opens a centred command bar even over a target; the chip still names it.
    reg->clearPinnedTargets();
    reg->setHoverTarget("table_cell", key0, "0", "row=2/col=3", "tok/s: 126", "Table: sweep\n> Row 2: c1 | 126",
                        300, 200, 110, 30);
    QMetaObject::invokeMethod(omni, "focusForVoice");
    settle(30);
    check(isOpen() && std::abs(card->x() - (1920 - card->width()) / 2) < 1 && std::abs(card->y() - 173) < 1
              && omni->property("chips").toList().value(0).toMap().value("kind") == "table_cell",
          "omni: L2 opens the bar centred upper third, the hovered target as its chip", rectText(cardRect()));
    sendKey(win, Qt::Key_Escape);
    settle(20);

    // Every controller button as a vector PadIcon (idle and active): each draws ink, all on palette.
    {
        const QStringList buttons = padIconProbeButtons();
        QObject *obj = nullptr;
        QQuickItem *sheet = makePadIconSheet(engine, win->contentItem(), 28, &obj);
        settle(120);
        const QImage shot = win->grabWindow();
        const QList<QQuickItem *> icons = findItems(win->contentItem(), QStringLiteral("padIconProbe"));
        int blank = 0, unknown = 0;
        QStringList blankNames;
        for (QQuickItem *ic : icons) {
            if (ic->property("kind").toString().isEmpty())
                ++unknown;
            const QRect r = ic->mapRectToScene(QRectF(0, 0, ic->width(), ic->height())).toAlignedRect();
            int lit = 0;
            for (int y = r.top(); y <= r.bottom() && y < shot.height(); ++y)
                for (int x = r.left(); x <= r.right() && x < shot.width(); ++x)
                    lit += qGray(shot.pixel(x, y)) > 110;
            if (lit < 6) {
                ++blank;
                blankNames << ic->property("button").toString();
            }
        }
        const QImage crop = shot.copy(sheet ? sheet->mapRectToScene(QRectF(0, 0, sheet->width(), sheet->height())).toAlignedRect()
                                            : QRect());
        check(sheet && icons.size() == 2 * buttons.size() && unknown == 0 && blank == 0,
              "pad icons: every controller button and glyph draws a vector icon (idle + active)",
              QStringLiteral("%1 icons, %2 unknown, blank: %3").arg(icons.size()).arg(unknown).arg(blankNames.join(',')));
        check(forbiddenPixels(crop) == 0, "pad icons: zero blue/green pixels in the gold theme",
              QString::number(forbiddenPixels(crop)));
        delete obj;
        settle(20);
    }
    backend.setLlmRouter(nullptr);
    reg->clearPinnedTargets();
    reg->clearHoverTarget();

    // Viewer hover hooks, driven by real pointer moves over the mock Stage and Ladder.
    {
        QString key;
        const bool shown = showArtifactKind(app, backend, "table", &key);
        const QPointF p = tableCellPoint(win, 1, 1);
        mouseMove(win, p);
        settle(40);
        const QVariantMap t = reg->activeTarget();
        check(shown && t.value("kind") == "table_cell" && t.value("path") == "row=2/col=2"
                  && t.value("sessionId") == key && t.value("contextText").toString().contains("> Row 2:")
                  && t.value("contextText").toString().contains("Columns: "),
              "hover: table cell reports row/col, header: value, and adjacent rows",
              t.value("path").toString() + " | " + t.value("label").toString());
        mouseMove(win, QPointF(3, 3));
        settle(30);
        check(!reg->hasActiveTarget(), "hover: leaving the table clears the cell");
    }
    {
        // No mock island carries a diff: a DiffView in its own offscreen window over a known diff.
        QQuickWindow dwin;
        dwin.resize(1000, 600);
        QQmlComponent comp(&engine);
        comp.setData("import QtQuick\nimport Studio\nDiffView { width: 1000; height: 600 }", QUrl());
        auto *dv = qobject_cast<QQuickItem *>(comp.create());
        auto ln = [](const char *k, const char *no, const char *html) {
            return QVariantMap{{"k", QString::fromLatin1(k)}, {"no", QString::fromLatin1(no)}, {"html", QString::fromLatin1(html)}};
        };
        const QVariantList lines{ln("f", "", "src/attn.cpp"), ln(" ", "40", "int tile = 64;"),
                                 ln("-", "41", "return 0;"), ln("+", "41", "return tile &amp; mask;"),
                                 ln(" ", "42", "}")};
        if (dv) {
            dv->setParentItem(dwin.contentItem());
            dv->setProperty("artifact", QVariantMap{{"title", "attn tile fix"}, {"lines", lines}});
            dv->setProperty("targets", QVariant::fromValue(static_cast<QObject *>(reg)));
            dwin.show();
            settle(250);
            QQuickItem *row = nullptr;
            QQuickItem *list = findItems(dv, "diffList").value(0);
            if (list)
                QMetaObject::invokeMethod(list, "itemAtIndex", Q_RETURN_ARG(QQuickItem *, row), Q_ARG(int, 3));
            const QPointF p = row ? row->mapToScene(QPointF(300, row->height() / 2)) : QPointF();
            mouseMove(&dwin, p);
            settle(40);
        }
        const QVariantMap t = reg->activeTarget();
        const QString ctx = t.value("contextText").toString();
        check(dv && t.value("kind") == "diff_line" && t.value("path") == "line=4" && t.value("label") == "attn tile fix L41"
                  && ctx.contains("> 41 + return tile & mask;") && ctx.contains("  40   int tile = 64;"),
              "hover: diff line reports line=N, '<title> L<N>' and the surrounding lines (markup stripped)",
              t.value("path").toString() + " | " + t.value("label").toString() + " | " + ctx);
        if (dv) {
            mouseMove(&dwin, QPointF(-50, -50));
            settle(30);
            delete dv;
        }
        reg->clearHoverTarget();
    }
    {
        const bool shown = showArtifactKind(app, backend, "chart");
        QQuickItem *cv = visibleItem(win, "chartView");
        const QPointF p = cv ? cv->mapToScene(QPointF(cv->width() * 0.5, cv->height() * 0.8)) : QPointF();
        mouseMove(win, p);
        settle(40);
        const QVariantMap t = reg->activeTarget();
        check(shown && t.value("kind") == "chart_point" && t.value("path").toString().contains("/idx=")
                  && t.value("contextText").toString().contains("mean "),
              "hover: chart bar reports series/idx, label: value and series stats",
              t.value("path").toString() + " | " + t.value("label").toString());
        mouseMove(win, QPointF(3, 3));
        settle(30);
    }
    {
        QMetaObject::invokeMethod(app, "goLens", Q_ARG(QVariant, QVariant(1)));
        settle(500);
        QQuickItem *thread = visibleItem(win, "threadView");
        const int count = thread ? thread->property("count").toInt() : 0;
        QQuickItem *last = nullptr;
        if (thread && count > 0)
            QMetaObject::invokeMethod(thread, "itemAtIndex", Q_RETURN_ARG(QQuickItem *, last), Q_ARG(int, count - 1));
        const QPointF p = last ? last->mapToScene(QPointF(40, std::min<qreal>(last->height() / 2, 30))) : QPointF();
        mouseMove(win, p);
        settle(40);
        const QVariantMap t = reg->activeTarget();
        check(last && t.value("kind") == "turn" && t.value("path") == QStringLiteral("turn=%1").arg(count)
                  && t.value("label").toString().endsWith(QStringLiteral("turn %1").arg(count)),
              "hover: ladder turn reports turn=N and '<role> turn <N>'",
              QString::number(count) + " " + t.value("path").toString() + " | " + t.value("label").toString());
        mouseMove(win, QPointF(3, 3));
        settle(30);
        QMetaObject::invokeMethod(app, "goLens", Q_ARG(QVariant, QVariant(0)));
        settle(400);
        // The turn goes with its lens; in controller mode the cursor's island is the target now.
        check(!reg->hasActiveTarget() || reg->activeTarget().value("kind") == "island",
              "hover: changing lens clears the old lens's hover target", reg->activeTarget().value("kind").toString());
    }
}

Turn mkTurn(const QString &role, const QString &text)
{
    Turn t;
    t.role = role;
    t.text = text;
    return t;
}

void formFactorTests()
{
    // Qualitative / architectural write-up with incidental numbers: no forced chart, a summary leads.
    {
        const QList<Turn> turns = {
            mkTurn("user", "should the scheduler own retries or the worker?"),
            mkTurn("assistant",
                   "## Retry ownership\n"
                   "I recommend the scheduler owns retries, because it already tracks job state.\n\n"
                   "- The worker stays stateless, so restarts take about 2 s instead of a re-sync.\n"
                   "- Backoff lives in one place (src/sched.cpp), not in 3 worker copies.\n"
                   "- **Decision:** keep idempotency keys on the worker side.\n\n"
                   "| option | owner |\n|---|---|\n| A | scheduler |\n")};
        const Extracted ex = extractArtifacts(turns, noFiles);
        const QVariantMap s = ex.artifacts.value(0).toMap();
        check(firstOfKind(ex.artifacts, "chart").isEmpty() && firstOfKind(ex.artifacts, "line").isEmpty(),
              "form: qualitative turn emits no chart from incidental numbers");
        check(s.value("kind").toString() == "summary" && s.value("metric").toString().isEmpty()
                  && s.value("title").toString() == "Retry ownership",
              "form: qualitative turn leads with a summary artifact",
              s.value("kind").toString() + " / " + s.value("title").toString());
        const QStringList pts = s.value("points").toStringList();
        check(pts.size() == 3 && pts.value(0).startsWith("The worker stays stateless") && pts.value(2).startsWith("Decision: keep")
                  && s.value("text").toString() == pts.join('\n'),
              "form: summary text is the distilled key points (bullets, markdown stripped)", pts.join(" | "));
    }
    // Plain prose, no bullets: the opening sentences become the points.
    {
        const QStringList pts = distillKeyPoints(
            "The cache layer is the bottleneck here. Moving it behind the queue removes the lock. "
            "We should ship that first and measure after.\n```cpp\nint x = 1;\n```");
        check(pts.size() == 3 && pts.value(0) == "The cache layer is the bottleneck here." && !pts.join(' ').contains("int x"),
              "form: prose distills to its sentences, code dropped", pts.join(" | "));
    }
    // A one-row table is not a sweep, and neither is a table whose arms all measure the same.
    {
        const QVariantMap one = tableFromMarkdown({"| arm | ms |", "|---|---|", "| base | 38 |"});
        check(chartFromTable(one).isEmpty(), "form: a 1-bar table never becomes a chart");
        const QVariantMap flat = tableFromMarkdown({"| arm | ms |", "|---|---|", "| a | 38 |", "| b | 38 |"});
        check(chartFromTable(flat).isEmpty(), "form: identical values are filler, not a comparison");
    }
    // A genuine >= 2-arm sweep still charts, and leads (no summary pushed in front of it).
    {
        const QList<Turn> turns = {
            mkTurn("user", "sweep the decode tile"),
            mkTurn("assistant",
                   "The 128-reg tile wins clearly over the baseline across every decode run we tried today.\n\n"
                   "| tile | tok/s |\n|---|---|\n| base | 11.2 |\n| 128-reg | 29.6 |\n")};
        const Extracted ex = extractArtifacts(turns, noFiles);
        check(ex.artifacts.value(0).toMap().value("kind").toString() == "chart"
                  && firstOfKind(ex.artifacts, "summary").isEmpty(),
              "form: a 2-arm sweep keeps its chart and gets no summary",
              ex.artifacts.value(0).toMap().value("kind").toString());
    }
    // Newest turn qualitative, older turn a sweep: the summary leads, the chart is still there.
    {
        const QList<Turn> turns = {
            mkTurn("user", "sweep the decode tile"),
            mkTurn("assistant", "| tile | tok/s |\n|---|---|\n| base | 11.2 |\n| 128-reg | 29.6 |\n"),
            mkTurn("user", "what next?"),
            mkTurn("assistant", "Next we should fold the scale lookup into the dequant loop, then re-run the sweep on the 65B model.")};
        const Extracted ex = extractArtifacts(turns, noFiles);
        check(ex.artifacts.value(0).toMap().value("kind").toString() == "summary" && !firstOfKind(ex.artifacts, "chart").isEmpty(),
              "form: a qualitative newest turn leads with a summary, the older chart stays");
    }

    // Human-interest filter prompt: direction, pinned params and current focus all reach the model.
    {
        Backend b(Backend::Mode::Mock);
        b.recordAction(QStringLiteral("opened pill BLOCK_M=128"));
        b.recordAction(QStringLiteral("scrolled diff src/tile.cu"));
        const QVariantList params = {QVariantMap{{"name", "BLOCK_M"}, {"value", "128"}, {"kind", "param"}},
                                     QVariantMap{{"name", "DCP"}, {"value", "4"}, {"kind", "param"}},
                                     QVariantMap{{"name", "spills"}, {"kind", "term"}}};
        const QByteArray body = buildFilterPrompt(QStringLiteral("qwen3.8-flash-next"), QStringLiteral("profile fp4 decode with DCP=4"),
                                                  params, b.recentActions(), QStringLiteral("Ran the build. 128-reg removed all spills."));
        const QJsonObject req = QJsonDocument::fromJson(body).object();
        const QJsonArray msgs = req.value("messages").toArray();
        const QString sys = msgs.at(0).toObject().value("content").toString();
        const QString ctx = msgs.at(1).toObject().value("content").toString();
        check(req.value("model").toString() == "qwen3.8-flash-next" && req.value("stream").toBool(),
              "filter: request targets qwen3.8-flash-next, streamed");
        check(ctx.contains("Human direction (goal): profile fp4 decode with DCP=4"), "filter: prompt carries the human goal", ctx.left(200));
        check(ctx.contains("BLOCK_M=128") && ctx.contains("DCP=4") && ctx.contains("spills"),
              "filter: prompt carries pinned params and terms", ctx);
        check(ctx.contains("- opened pill BLOCK_M=128") && ctx.contains("- scrolled diff src/tile.cu"),
              "filter: prompt carries the human's recent actions (m_recentActions)", ctx);
        check(ctx.contains("128-reg removed all spills"), "filter: prompt carries the latest turn");
        check(sys.contains("boilerplate") && sys.contains("FORM: chart") && sys.contains("summary"),
              "filter: system asks to drop boilerplate and pick chart vs summary");

        const HumanFilterReply r = parseFilterReply(
            "**STATE:** 128-reg removed every spill at DCP=4\nFORM: summary\nPOINTS:\n- Spills went to zero\n* **BLOCK_M=128** is the keeper\n");
        check(r.stateLine == "128-reg removed every spill at DCP=4" && r.form == "summary"
                  && r.summary == "Spills went to zero\nBLOCK_M=128 is the keeper",
              "filter: reply parsed into state line, form and points", r.stateLine + " / " + r.form + " / " + r.summary);
        // Mock mode never hits the network, even with a URL configured.
        b.setOpenaiCompatUrl(QStringLiteral("http://127.0.0.1:9/v1"));
        b.filterForHuman(QStringLiteral("anything"));
        check(b.filterRequests() == 0 && b.spawnCount() == 0, "filter: mock mode sends nothing");
    }

    // Stage renders kind "summary" as centered text, and the right stick scrolls it.
    {
        QQmlEngine engine;
        QQmlComponent comp(&engine);
        comp.loadFromModule("Studio", "Stage");
        QStringList pts;
        for (int i = 0; i < 14; ++i)
            pts << QStringLiteral("Point %1: the scheduler owns retries so the worker stays stateless and restarts fast.").arg(i);
        const QVariantMap art{{"kind", "summary"}, {"title", "Retry ownership"}, {"metric", ""},
                              {"text", pts.join('\n')}, {"points", pts}};
        std::unique_ptr<QObject> obj(comp.createWithInitialProperties(
            {{"width", 1600}, {"height", 700}, {"artifact", art}, {"artifacts", QVariantList{art}}, {"artifactCount", 1}}));
        check(obj != nullptr, "summary ui: Stage loads", comp.errorString());
        auto *stage = qobject_cast<QQuickItem *>(obj.get());
        if (stage) {
            QCoreApplication::processEvents();
            const QList<QQuickItem *> views = findItems(stage, "summaryView");
            QQuickItem *view = views.value(0);
            const QList<QQuickItem *> texts = view ? findItems(view, "summaryText") : QList<QQuickItem *>();
            check(view && stage->property("kind").toString() == "summary", "summary ui: kind summary loads the summary view");
            check(texts.size() == 14 && texts.value(0)->property("text").toString().startsWith("Point 0")
                      && texts.value(0)->property("wrapMode").toInt() == 4 /* Text.Wrap */,
                  "summary ui: every key point is a wrapped text line", QString::number(texts.size()));
            QQuickItem *col = texts.isEmpty() ? nullptr : texts.value(0)->parentItem()->parentItem();
            if (view && col) {
                const double left = col->x();
                const double right = view->width() - col->x() - col->width();
                check(col->width() <= 780.5 && qAbs(left - right) < 1.5, "summary ui: text column <= 780 px and centered",
                      QStringLiteral("w %1 l %2 r %3").arg(col->width()).arg(left).arg(right));
                const QFont f = texts.value(0)->property("font").value<QFont>();
                check(f.family() == QStringLiteral("Avenir Next") && texts.value(0)->property("lineHeight").toDouble() > 1.3,
                      "summary ui: Theme.sans with generous line height", f.family());
                const double y0 = view->property("contentY").toDouble();
                QMetaObject::invokeMethod(stage, "scrollPixels", Q_ARG(QVariant, QVariant(120.0)));
                const double y1 = view->property("contentY").toDouble();
                QMetaObject::invokeMethod(stage, "scrollBy", Q_ARG(QVariant, QVariant(-1)));
                const double y2 = view->property("contentY").toDouble();
                check(y0 == 0 && qAbs(y1 - 120) < 0.5 && y2 == 0, "summary ui: scrollPixels / scrollBy scroll the summary",
                      QStringLiteral("%1 -> %2 -> %3").arg(y0).arg(y1).arg(y2));
            }
            const QVariantMap pred = stage->property("predictedAction").toMap();
            check(pred.value("actionKind").toString() == "prompt" && !pred.value("promptSuggestion").toString().isEmpty(),
                  "summary ui: Tab predictor has a prompt for a summary");
        }
    }
    // Ladder conversation: real turns, "You" in the accent, scrollable by scrollPixels / scrollBy.
    {
        QQmlEngine engine;
        QQmlComponent comp(&engine);
        comp.loadFromModule("Studio", "Ladder");
        QVariantList turns;
        for (int i = 0; i < 16; ++i)
            turns << QVariantMap{{"role", i % 2 ? "assistant" : "user"},
                                 {"text", QStringLiteral("Turn %1: the scheduler owns retries so the worker stays stateless, "
                                                         "restarts fast and never double-applies a step.").arg(i)}};
        const QVariantMap ladder{{"goal", "Fix fn key toggle latency"}, {"assumptions", QVariantList()},
                                 {"arms", QVariantList()}, {"armsLabel", ""}, {"winner", ""}};
        const QVariantMap project{{"key", "~tab:14e5ecb8"}, {"name", "Fix fn key toggle latency"}, {"ladder", ladder},
                                  {"artifacts", QVariantList()}, {"turns", turns}};
        std::unique_ptr<QObject> obj(comp.createWithInitialProperties({{"width", 1600}, {"height", 900}, {"project", project}}));
        check(obj != nullptr, "ladder ui: Ladder loads", comp.errorString());
        auto *lad = qobject_cast<QQuickItem *>(obj.get());
        if (lad) {
            for (int i = 0; i < 5; ++i)
                QCoreApplication::processEvents();
            QQuickItem *view = findItems(lad, "threadView").value(0);
            const QList<QQuickItem *> roles = view ? findItems(view, "turnRole") : QList<QQuickItem *>();
            const QList<QQuickItem *> texts = view ? findItems(view, "turnText") : QList<QQuickItem *>();
            check(view && view->isVisible() && !roles.isEmpty() && roles.size() == texts.size(),
                  "ladder ui: the conversation renders its turns", QString::number(texts.size()));
            bool labels = !roles.isEmpty();
            for (QQuickItem *r : roles) {
                const QString t = r->property("text").toString();
                const QString c = r->property("color").value<QColor>().name();
                labels = labels && ((t == "You" && c == "#e09f3e") || (t == "Assistant" && c != "#e09f3e"));
            }
            check(labels, "ladder ui: 'You' labels in gold, 'Assistant' otherwise");
            bool wrapped = !texts.isEmpty();
            for (QQuickItem *t : texts)
                wrapped = wrapped && t->property("wrapMode").toInt() == 4 /* Text.Wrap */
                    && t->property("text").toString().startsWith("Turn ");
            check(wrapped, "ladder ui: turn text is wrapped plain text");
            if (view) {
                const double origin = view->property("originY").toDouble();
                const double y0 = view->property("contentY").toDouble();
                QMetaObject::invokeMethod(lad, "scrollPixels", Q_ARG(QVariant, QVariant(-120.0)));
                const double y1 = view->property("contentY").toDouble();
                QMetaObject::invokeMethod(lad, "scrollBy", Q_ARG(QVariant, QVariant(1)));
                const double y2 = view->property("contentY").toDouble();
                QMetaObject::invokeMethod(lad, "scrollPixels", Q_ARG(QVariant, QVariant(-1e6)));
                const double y3 = view->property("contentY").toDouble();
                check(y0 > origin + 100 && qAbs(y1 - (y0 - 120)) < 0.5 && qAbs(y2 - y0) < 0.5 && qAbs(y3 - origin) < 0.5,
                      "ladder ui: opens at the newest turn; scrollPixels / scrollBy scroll the conversation",
                      QStringLiteral("origin %1: %2 -> %3 -> %4 -> %5").arg(origin).arg(y0).arg(y1).arg(y2).arg(y3));
            }
            QQuickItem *convo = findItems(lad, "conversation").value(0);
            const QList<QQuickItem *> rungs = findItems(lad, "rung");
            check(convo && !rungs.isEmpty() && convo->y() >= rungs.last()->mapToItem(lad, QPointF(0, rungs.last()->height())).y() - 0.5,
                  "ladder ui: the conversation sits below the four rungs");
        }
    }
}

// --- Phase 3: spatial cards, visual-first Stage, Needs Me, session review ----------------------

QString firstSessionId(const Backend &b, const QString &key)
{
    const QVariantMap p = b.projects().value(b.projectIndex(key)).toMap();
    return p.value("sessions").toList().value(0).toMap().value("id").toString();
}

// One island waiting on the human (phonon goes needs-ok over the board) and two user steers on
// lapis, so the Needs Me pill counts something and the review has rules to offer.
void seedPhase3Scene(Backend &b)
{
    b.applyBoard({BoardRow{firstSessionId(b, "phonon"), QStringLiteral("needs-ok"), QString()}});
    Turn steer;
    steer.role = QStringLiteral("user");
    steer.text = QStringLiteral("No, don't lead with the diff. Show the frame-time chart first.");
    b.mockAppendTurn("lapis", steer);
    Turn reply;
    reply.role = QStringLiteral("assistant");
    reply.text = QStringLiteral("Moved the chart to the front; the diff now sits behind the Code pill.");
    b.mockAppendTurn("lapis", reply);
    steer.text = QStringLiteral("Actually, keep the cards calm: one line each, never a terminal log.");
    b.mockAppendTurn("lapis", steer);
}

int needsMeCount(const Backend &b)
{
    int n = 0;
    for (const QVariant &v : b.projects())
        n += v.toMap().value("needsMe").toBool() ? 1 : 0;
    return n;
}

void phase3UnitTests()
{
    // Visual-first deck: non-diffs keep their order and lead; diffs follow only on request.
    {
        auto art = [](const char *k) { return QVariant(QVariantMap{{"kind", QString::fromLatin1(k)}, {"title", QString::fromLatin1(k)}}); };
        const QVariantList arts{art("diff"), art("chart"), art("summary"), art("diff"), art("table"), art("image"), art("pdf")};
        auto kinds = [](const QVariantList &l) {
            QStringList out;
            for (const QVariant &v : l)
                out << v.toMap().value("kind").toString();
            return out.join(',');
        };
        check(kinds(stageArtifacts(arts, false)) == "chart,summary,table,image,pdf", "visual-first: diffs dropped by default",
              kinds(stageArtifacts(arts, false)));
        check(kinds(stageArtifacts(arts, true)) == "chart,summary,table,image,pdf,diff,diff",
              "visual-first: Code on appends diffs after every visual", kinds(stageArtifacts(arts, true)));
        Backend b(Backend::Mode::Mock);
        check(!b.showCodeDiffs(), "visual-first: showCodeDiffs defaults to false");
        bool anyDiff = false;
        for (const QVariant &v : b.projects())
            for (const QVariant &a : v.toMap().value("artifacts").toList())
                anyDiff = anyDiff || a.toMap().value("kind").toString() == "diff";
        check(!anyDiff, "visual-first: no mock project publishes a diff on its default deck");
    }

    // Needs Me: needs-ok / waiting / gate statuses or a blocker question.
    {
        check(projectNeedsMe("needs-ok", false) && projectNeedsMe("waiting", false) && projectNeedsMe("gate", false)
                  && projectNeedsMe("idle", true) && !projectNeedsMe("working", false) && !projectNeedsMe("finished", false),
              "needs me: needs-ok / waiting / gate / blocker, not working or finished");
        Backend b(Backend::Mode::Mock);
        const int before = needsMeCount(b);
        b.applyBoard({BoardRow{firstSessionId(b, "phonon"), QStringLiteral("needs-ok"), QString()}});
        const QVariantMap ph = b.projects().value(b.projectIndex("phonon")).toMap();
        check(ph.value("needsMe").toBool() && needsMeCount(b) == before + 1, "needs me: a needs-ok board status flags the island",
              QStringLiteral("%1 -> %2").arg(before).arg(needsMeCount(b)));
    }

    // Card positions and showCodeDiffs persist in studio_prefs.json.
    {
        QTemporaryDir tmp;
        const QString path = tmp.path() + "/studio_prefs.json";
        {
            Backend w(Backend::Mode::Mock);
            w.setPrefsPath(path);
            w.setCardPosition("phonon", 0.31, 0.62);
            w.setCardPosition("lapis", 1.7, -0.2);  // clamped into 0..1
            w.setShowCodeDiffs(true);
            check(w.savePrefs(), "cards: prefs save");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            const QVariantMap ph = r.cardPositions().value("phonon").toMap();
            const QVariantMap la = r.cardPositions().value("lapis").toMap();
            check(qAbs(ph.value("x").toDouble() - 0.31) < 1e-9 && qAbs(ph.value("y").toDouble() - 0.62) < 1e-9,
                  "cards: a dragged position survives a restart", QStringLiteral("%1,%2").arg(ph.value("x").toDouble()).arg(ph.value("y").toDouble()));
            check(la.value("x").toDouble() == 1.0 && la.value("y").toDouble() == 0.0, "cards: positions clamp to 0..1");
            check(r.showCodeDiffs(), "visual-first: showCodeDiffs round-trips through studio_prefs.json");
            QFile f(path);
            const bool opened = f.open(QIODevice::ReadOnly);
            const QJsonObject o = QJsonDocument::fromJson(opened ? f.readAll() : QByteArray()).object();
            check(o.value("cardPositions").toObject().value("phonon").toObject().value("x").toDouble() == 0.31,
                  "cards: stored as cardPositions {key: {x, y}}");
            r.resetCardPositions();
            check(r.cardPositions().isEmpty() && r.savePrefs(), "cards: reset clears every position");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.cardPositions().isEmpty(), "cards: the reset persists");
        }
    }

    // Taste rules: JSONL round trip, newest rating wins, and they reach the filter prompt.
    {
        const QFileInfo realReviews(Backend::defaultReviewsPath());
        const qint64 realBefore = realReviews.exists() ? realReviews.lastModified().toMSecsSinceEpoch() : -1;
        {
            Backend m(Backend::Mode::Mock);
            check(m.reviewsPath().isEmpty() && !m.saveTasteRule("Lead with the chart.", 1) && m.tasteRules() == QStringList{"Lead with the chart."},
                  "taste: mock with no reviews path keeps the rule in memory, writes nothing");
        }
        const QFileInfo realAfter(Backend::defaultReviewsPath());
        check((realAfter.exists() ? realAfter.lastModified().toMSecsSinceEpoch() : -1) == realBefore,
              "taste: tests never touch ~/.lapis-next/session_reviews.jsonl");

        QTemporaryDir tmp;
        const QString path = tmp.path() + "/sub/session_reviews.jsonl";
        {
            Backend w(Backend::Mode::Mock);
            w.setReviewsPath(path);
            check(w.saveTasteRule("Lead with the chart, never the diff.", 1), "taste: saveTasteRule writes a line");
            check(w.saveTasteRule("Show terminal logs on cards.", 1) && w.saveTasteRule("show terminal logs on cards.", -1),
                  "taste: a later rating <= 0 retires a rule");
            check(w.tasteRules() == QStringList{"Lead with the chart, never the diff."}, "taste: in-memory rules follow ratings",
                  w.tasteRules().join(" | "));
        }
        QFile f(path);
        QStringList lines;
        if (f.open(QIODevice::ReadOnly))
            lines = QString::fromUtf8(f.readAll()).split('\n', Qt::SkipEmptyParts);
        bool wellFormed = lines.size() == 3;
        for (const QString &l : lines) {
            const QJsonObject o = QJsonDocument::fromJson(l.toUtf8()).object();
            wellFormed = wellFormed && !o.value("rule").toString().isEmpty() && o.value("rating").isDouble()
                         && QDateTime::fromString(o.value("ts").toString(), Qt::ISODate).isValid();
        }
        check(wellFormed, "taste: three JSONL lines of {ts, rule, rating}", lines.join(" // "));
        check((QFileInfo(path).permissions() & (QFileDevice::ReadGroup | QFileDevice::ReadOther)) == 0,
              "taste: session_reviews.jsonl is owner-only");
        Backend r(Backend::Mode::Mock);
        r.setReviewsPath(path);
        check(r.tasteRules() == QStringList{"Lead with the chart, never the diff."}, "taste: rules load on startup",
              r.tasteRules().join(" | "));
        const QByteArray prompt = buildFilterPrompt("qwen3.8-flash-next", "goal", {}, {}, "turn", r.tasteRules());
        check(prompt.contains("standing taste rules") && prompt.contains("Lead with the chart, never the diff."),
              "taste: saved rules ride in the fast-model filter prompt");
        check(!buildFilterPrompt("m", "goal", {}, {}, "turn").contains("taste rules"), "taste: no rules, no section");
    }

    // Reflection: parsing, the local fallback, and one reflection:1 stream through LlmRouter.
    {
        const SessionReflection p = parseReflectionReply(
            "**INTENT:** Wanted a calm canvas; got it once diffs moved behind a pill.\n"
            "CORRECTIONS: twice steered away from diffs and logs\n"
            "RULES:\n- Lead with results, never code.\n2. Keep cards to one line.\n- Ask before restarting the app.\n- A fourth one.");
        check(p.intent.startsWith("Wanted a calm canvas") && p.corrections.startsWith("twice steered") && p.rules.size() == 3
                  && p.rules.value(1) == "Keep cards to one line.",
              "reflection: parses INTENT / CORRECTIONS / RULES (<= 3)", p.rules.join(" | "));
        check(isSteerTurn("No, don't lead with the diff.") && isSteerTurn("Actually, use the other tile")
                  && !isSteerTurn("Profile the decode kernel on sm120"),
              "reflection: steers are corrections, not fresh requests");

        Backend b(Backend::Mode::Mock);
        seedPhase3Scene(b);
        b.runSessionReflection();
        const QVariantMap r = b.reflection();
        const QStringList rules = r.value("rules").toStringList();
        check(!b.reflectionBusy() && b.reflectionSource() == "local" && !r.value("intent").toString().isEmpty(),
              "reflection: offline -> clean local lines at once", r.value("intent").toString());
        check(rules.contains("Don't lead with the diff.") && rules.contains("Keep the cards calm: one line each, never a terminal log."),
              "reflection: local rules come from the human's steers", rules.join(" | "));
        check(r.value("corrections").toString().contains("lead with the diff"), "reflection: steers are quoted",
              r.value("corrections").toString());
        check(b.spawnCount() == 0, "reflection: no processes spawned");

        FakeSse sse;
        LlmRouter router;
        if (sse.start()) {
            b.setLlmRouter(&router);
            b.setOpenaiCompatUrl(sse.url());
            sse.answer = QStringLiteral("INTENT: Set out to calm the canvas; matched once diffs hid behind Code.\n"
                                        "CORRECTIONS: Steered away from diffs first and from logs on cards.\n"
                                        "RULES:\n- Lead with results; code only on request.\n- One calm line per card.");
            b.runSessionReflection();
            check(b.reflectionBusy() && b.reflectionSource() == "local", "reflection: local lines while the model streams");
            waitUntil([&] { return !b.reflectionBusy(); }, 5000);
            const QStringList mr = b.reflection().value("rules").toStringList();
            check(b.reflectionSource() == "qwen3.8-flash-next" && mr == QStringList{"Lead with results; code only on request.", "One calm line per card."},
                  "reflection: one reflection:1 stream to qwen3.8-flash-next rewrites the card",
                  b.reflectionSource() + " " + mr.join(" | "));
            const QByteArray body = sse.bodies.value(sse.bodies.size() - 1);
            check(body.contains("qwen3.8-flash-next") && body.contains("Reflect on this session") && body.contains("lead with the diff"),
                  "reflection: the request carries the session's turns");
            b.setLlmRouter(nullptr);
        } else {
            check(false, "reflection: fake SSE endpoint listens");
        }
    }
}

void phase3UiTests()
{
    Backend backend(Backend::Mode::Mock);
    QTemporaryDir tmp;
    backend.setPrefsPath(tmp.path() + "/studio_prefs.json");
    backend.completeOnboarding("controller");  // a fresh prefs file would raise the first-launch modal
    backend.setReviewsPath(tmp.path() + "/session_reviews.jsonl");
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    check(win != nullptr, "p3 ui: QML loads");
    if (!win)
        return;
    QObject *app = win->findChild<QObject *>("studioApp");
    if (!app)
        return;
    settle(450);
    seedPhase3Scene(backend);
    settle(200);
    auto islands = [&] {
        QList<QQuickItem *> out;
        for (QQuickItem *it : findItems(win->contentItem(), "island"))
            if (it->isVisible() && it->property("draggable").toBool())  // not the zoom morph's copy
                out << it;
        return out;
    };
    auto pillText = [&](const char *name) {
        QQuickItem *p = firstItem(win, name);
        const auto t = p ? findItems(p, "barPillText") : QList<QQuickItem *>();
        return t.isEmpty() ? QString() : t.first()->property("text").toString();
    };

    // Needs Me: one bell with a count badge (no All / Needs Me text pills), Cmd+2 filters to it,
    // Cmd+1 brings every island back.
    Q_UNUSED(pillText);
    auto bellCount = [&] {
        QQuickItem *t = firstItem(win, "needsBellCount");
        QQuickItem *b = firstItem(win, "needsBellBadge");
        return t && b && b->isVisible() ? t->property("text").toString() : QString();
    };
    const int needs = needsMeCount(backend);
    check(needs >= 1 && firstItem(win, "needsBell") && firstItem(win, "needsBell")->isVisible()
              && bellCount() == QString::number(needs) && !firstItem(win, "filterAll") && !firstItem(win, "filterNeedsMe"),
          "needs me: the top bar is one bell with an N badge (no All / Needs Me pills)", bellCount());
    {
        // Addressing a card decrements the badge live; at zero it is a bare bell.
        backend.applyBoard({BoardRow{firstSessionId(backend, "phonon"), QStringLiteral("idle"), QString()}});
        settle(60);
        const int after = needsMeCount(backend);
        check(after == needs - 1 && (after > 0 ? bellCount() == QString::number(after) : bellCount().isEmpty()),
              "needs me: the bell count drops live as a card is addressed", bellCount() + " vs " + QString::number(after));
        backend.applyBoard({BoardRow{firstSessionId(backend, "phonon"), QStringLiteral("needs-ok"), QString()}});
        settle(60);
        check(bellCount() == QString::number(needs), "needs me: and climbs back when it waits again", bellCount());
    }
    const int all = int(islands().size());
    sendKey(win, Qt::Key_2, Qt::ControlModifier);
    settle(300);
    check(app->property("lens").toInt() == 0 && app->property("islandFilter").toString() == "needs" && islands().size() == needs,
          "needs me: Cmd+2 on the Constellation shows only islands that need you",
          QStringLiteral("%1 of %2").arg(islands().size()).arg(all));
    bool allNeed = true;
    for (QQuickItem *it : islands())
        allNeed = allNeed && it->property("proj").toMap().value("needsMe").toBool();
    check(allNeed, "needs me: every island left on screen needs the human");
    {
        const int sel = app->property("selectedIsland").toInt();
        check(backend.projects().value(sel).toMap().value("needsMe").toBool(), "needs me: the selection moves onto a shown island");
        const QImage shot = win->grabWindow();
        QStringList pairs;
        check(forbiddenPixels(shot) == 0 && textOverlaps(win, &pairs) == 0, "needs me: filtered view clean (palette, overlaps)",
              pairs.join("; "));
    }
    sendKey(win, Qt::Key_1, Qt::ControlModifier);
    settle(300);
    check(app->property("islandFilter").toString() == "all" && islands().size() == all, "needs me: Cmd+1 shows All again",
          QString::number(islands().size()));
    if (QQuickItem *p = firstItem(win, "needsBell")) {
        mouseClick(win, p->mapToScene(QPointF(p->width() / 2, p->height() / 2)), Qt::LeftButton, {});
        settle(200);
        check(app->property("islandFilter").toString() == "needs", "needs me: clicking the bell filters too");
        sendKey(win, Qt::Key_1, Qt::ControlModifier);
        settle(250);
    }

    // Freeform drag: transform-only while moving, committed once on release, persisted. The last
    // card (bottom row) moves down-right into open space, so it lands exactly as dropped (a drop
    // onto another card would be nudged clear of it).
    QList<QQuickItem *> isl = islands();
    QQuickItem *card = nullptr;
    QQuickItem *left = nullptr;
    for (QQuickItem *it : isl) {
        if (it->property("index").toInt() == 4)
            card = it;
        if (it->property("index").toInt() == 3)
            left = it;
    }
    check(card != nullptr && left != nullptr, "cards: an island to drag");
    if (!card || !left)
        return;
    const QString key = card->property("proj").toMap().value("key").toString();
    const QPointF xy0(card->x(), card->y());
    const QPointF from = card->mapToScene(QPointF(card->width() / 2, card->height() / 2));
    const QPointF delta(60, 190);
    // Mid-drag: the card rides its Translate while x / y (the layout) stay put.
    mouseMove(win, from);
    mouseButton(win, QEvent::MouseButtonPress, from, Qt::LeftButton);
    for (int i = 1; i <= 12; ++i) {
        mouseMove(win, from + delta * (i / 24.0), Qt::LeftButton);
        QCoreApplication::processEvents();
    }
    check(qAbs(card->x() - xy0.x()) < 0.01 && qAbs(card->property("dragDx").toReal() - delta.x() / 2) < 0.5
              && !firstItem(win, "focusRing")->isVisible(),
          "cards: mid-drag moves by transform only (no relayout, ring hidden)",
          QStringLiteral("x %1 dragDx %2").arg(card->x()).arg(card->property("dragDx").toReal()));
    for (int i = 13; i <= 24; ++i) {
        mouseMove(win, from + delta * (i / 24.0), Qt::LeftButton);
        QCoreApplication::processEvents();
    }
    settle(90);  // hold still, then let go: a drop, not a flick
    mouseButton(win, QEvent::MouseButtonRelease, from + delta, Qt::LeftButton);
    settle(120);
    check(app->property("lens").toInt() == 0, "cards: a drag never opens the island");
    check(backend.cardPositions().contains(key), "cards: release commits the position", key);
    check(qAbs(card->x() - (xy0.x() + delta.x())) < 1 && qAbs(card->y() - (xy0.y() + delta.y())) < 1
              && card->property("dragDx").toReal() == 0,
          "cards: the island rests where it was dropped",
          QStringLiteral("%1,%2 -> %3,%4").arg(xy0.x()).arg(xy0.y()).arg(card->x()).arg(card->y()));
    QQuickItem *reset = firstItem(win, "resetLayoutPill");
    check(reset && reset->isVisible(), "cards: a Reset layout pill appears once a card moved");
    {
        const QImage shot = win->grabWindow();
        QStringList pairs;
        check(forbiddenPixels(shot) == 0 && textOverlaps(win, &pairs) == 0, "cards: dragged layout clean (palette, overlaps)",
              pairs.join("; "));
    }
    check(backend.savePrefs(), "cards: prefs flush");
    {
        Backend r(Backend::Mode::Mock);
        r.setPrefsPath(backend.prefsPath());
        check(r.cardPositions().value(key) == backend.cardPositions().value(key), "cards: the dragged position reloads");
    }
    // Background updates never shuffle cards: a fresh turn elsewhere leaves this card put.
    {
        Turn t;
        t.role = QStringLiteral("assistant");
        t.text = QStringLiteral("Loss settles at 2.39.");
        backend.mockAppendTurn("yoctogpt", t);
        settle(120);
        check(qAbs(card->x() - (xy0.x() + delta.x())) < 1, "cards: background updates leave the dragged card put");
    }
    // Cmd+0 snaps back to the auto-layout.
    sendKey(win, Qt::Key_0, Qt::ControlModifier);
    settle(200);
    check(backend.cardPositions().isEmpty() && qAbs(card->x() - xy0.x()) < 0.5 && qAbs(card->y() - xy0.y()) < 0.5
              && !(reset && reset->isVisible()),
          "cards: Cmd+0 snaps back to auto-layout", QStringLiteral("%1,%2").arg(card->x()).arg(card->y()));

    // Session review: Cmd+Shift+R, local lines at once, Enter saves the selected rule.
    QQuickItem *review = firstItem(win, "sessionReview");
    sendKey(win, Qt::Key_R, Qt::ControlModifier | Qt::ShiftModifier);
    settle(300);
    check(review && review->property("open").toBool() && firstItem(win, "reviewCard")->isVisible(),
          "review: Cmd+Shift+R opens the session review");
    check(!firstItem(win, "reviewIntent")->property("text").toString().isEmpty() && visibleCount(win, "reviewRule") >= 1,
          "review: intent and rules shown", QString::number(visibleCount(win, "reviewRule")));
    {
        const QImage shot = win->grabWindow();
        // Overlaps inside the card only: the Constellation shows faintly through the 0.88 dim.
        QStringList pairs;
        QList<QPair<QRectF, QString>> boxes;
        std::function<void(QQuickItem *)> walk = [&](QQuickItem *it) {
            if (!it->isVisible() || it->opacity() < 0.01)
                return;
            if (it->inherits("QQuickText") && !it->property("text").toString().trimmed().isEmpty()) {
                const QRectF r(it->mapToScene(QPointF(0, 0)),
                               QSizeF(std::min(it->width(), it->property("contentWidth").toReal()), it->property("contentHeight").toReal()));
                for (const auto &b : std::as_const(boxes))
                    if (b.first.intersects(r))
                        pairs << b.second + " x " + it->property("text").toString().left(24);
                boxes << qMakePair(r, it->property("text").toString().left(24));
            }
            for (QQuickItem *c : it->childItems())
                walk(c);
        };
        if (QQuickItem *rc = firstItem(win, "reviewCard"))
            walk(rc);
        check(forbiddenPixels(shot) == 0 && pairs.isEmpty() && boxes.size() >= 5, "review: card clean (palette, overlaps)",
              pairs.join("; "));
        QStringList texts;
        visibleWords(win, &texts);
        const QString joined = texts.join('\n');
        check(!joined.contains("+++") && !joined.contains("@@") && !joined.contains("$ "), "review: no diffs or terminal logs");
    }
    const QString rule0 = review ? review->property("rules").toStringList().value(0) : QString();
    sendKey(win, Qt::Key_Return);
    settle(100);
    QQuickItem *saveText = firstItem(win, "saveRuleText");
    check(backend.tasteRules().contains(rule0) && QFileInfo::exists(backend.reviewsPath()) && saveText
              && saveText->property("text").toString() == "Saved",
          "review: Enter saves the rule to session_reviews.jsonl", rule0);
    sendKey(win, Qt::Key_Escape);
    settle(250);
    check(review && !review->property("open").toBool(), "review: Esc closes it");
    if (QQuickItem *p = firstItem(win, "reviewPill")) {
        mouseClick(win, p->mapToScene(QPointF(p->width() / 2, p->height() / 2)), Qt::LeftButton, {});
        settle(250);
        check(review->property("open").toBool(), "review: the Review pill opens it too");
        sendKey(win, Qt::Key_R, Qt::ControlModifier | Qt::ShiftModifier);
        settle(250);
        check(!review->property("open").toBool(), "review: Cmd+Shift+R toggles it closed");
    }
}

}  // namespace

// --snap-review OUT.png: the session review over the mock Constellation, plus OUT-constellation.png
// (one card dragged, Needs Me pill counting) and OUT-stage.png (default visual-first slide).
int snapReview(const QString &outPath)
{
    Backend backend(Backend::Mode::Mock);
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    if (!win) {
        std::fprintf(stderr, "snap-review: QML failed to load\n");
        return 2;
    }
    QObject *app = win->findChild<QObject *>("studioApp");
    if (!app)
        return 3;
    QDir().mkpath(QFileInfo(outPath).absolutePath());
    auto sibling = [&](const char *tag) {
        QString p = outPath;
        p.insert(outPath.endsWith(QLatin1String(".png"), Qt::CaseInsensitive) ? outPath.size() - 4 : outPath.size(),
                 QLatin1Char('-') + QLatin1String(tag));
        return p;
    };
    settle(450);
    seedPhase3Scene(backend);
    settle(300);

    // Constellation: drag the island that needs the human into the open top-right corner, as a
    // user arranging work around what is waiting on them.
    for (QQuickItem *it : findItems(win->contentItem(), "island"))
        if (it->isVisible() && it->property("draggable").toBool() && it->property("proj").toMap().value("key") == "phonon") {
            const QPointF from = it->mapToScene(QPointF(it->width() / 2, it->height() / 2));
            mouseDrag(win, from, QPointF(1580, 250) - from);
            break;
        }
    mouseMove(win, QPointF(4, 1076));  // no hover lift in the frame
    settle(900);
    const QImage cons = win->grabWindow();
    const bool consSaved = cons.save(sibling("constellation"));

    // The review card over the Constellation.
    sendKey(win, Qt::Key_R, Qt::ControlModifier | Qt::ShiftModifier);
    settle(500);
    const QImage img = win->grabWindow();
    const bool saved = img.save(outPath);
    sendKey(win, Qt::Key_Escape);
    settle(300);

    // Stage: the default visual-first slide of a session with a diff (the mock kickoff yields
    // chart, diff, table, pdf); the diff stays behind the Code pill.
    const QString stageKey = backend.startSession(QStringLiteral("~/dev/fp4-decode"), QStringLiteral("claude"),
                                                  backend.harnessModels().value("claude").toList().value(0).toString(),
                                                  QStringLiteral("profile fp4 kv-cache decode on sm120"));
    settle(300);
    for (int i = 0; i < 14; ++i)
        backend.advance(0.5);
    settle(300);
    bool stageSaved = false;
    int stagePixels = 0;
    if (backend.projectIndex(stageKey) >= 0) {
        QMetaObject::invokeMethod(app, "enterProject", Q_ARG(QVariant, QVariant(backend.projectIndex(stageKey))));
        mouseMove(win, QPointF(4, 1076));
        settle(900);
        const QImage stage = win->grabWindow();
        stageSaved = stage.save(sibling("stage"));
        stagePixels = forbiddenPixels(stage);
    }
    std::printf("snap-review: %s %s (%d blue/green px)\n", saved ? "saved" : "FAILED", qPrintable(outPath), forbiddenPixels(img));
    std::printf("snap-review: %s %s (%d blue/green px), needs me %d, %lld custom card positions\n",
                consSaved ? "saved" : "FAILED", qPrintable(sibling("constellation")), forbiddenPixels(cons),
                needsMeCount(backend), static_cast<long long>(backend.cardPositions().size()));
    std::printf("snap-review: %s %s (%d blue/green px), project %s, deck %lld slides, %d diff hidden\n",
                stageSaved ? "saved" : "FAILED", qPrintable(sibling("stage")), stagePixels, qPrintable(stageKey),
                static_cast<long long>(app->property("artifacts").toList().size()),
                backend.projects().value(backend.projectIndex(stageKey)).toMap().value("diffCount").toInt());
    const QVariantMap r = backend.reflection();
    std::printf("intent: %s\nsteers: %s\nrules: %s\n", qPrintable(r.value("intent").toString()),
                qPrintable(r.value("corrections").toString()), qPrintable(r.value("rules").toStringList().join(" | ")));
    return saved && consSaved ? 0 : 1;
}

int snapOmnibar(const QString &outPath)
{
    Backend backend(Backend::Mode::Mock);
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    if (!win) {
        std::fprintf(stderr, "snap-omnibar: QML failed to load\n");
        return 2;
    }
    QObject *app = win->findChild<QObject *>("studioApp");
    QQuickItem *omni = firstItem(win, "floatingOmnibar");
    QQuickItem *card = firstItem(win, "omnibarCard");
    if (!app || !omni || !card)
        return 3;
    settle(450);
    backend.setNavMode("mouse");  // Tab pins the hovered target
    if (!showArtifactKind(app, backend, "table")) {
        std::fprintf(stderr, "snap-omnibar: no mock table artifact\n");
        return 4;
    }
    FakeSse sse;
    if (!sse.start())
        return 5;
    LlmRouter router;
    backend.setLlmRouter(&router);
    backend.setOpenaiCompatUrl(sse.url());
    auto askAndWait = [&](const QString &q) {
        typeText(win, q);
        sendKey(win, Qt::Key_Return);
        waitUntil([&] { return !omni->property("busy").toBool(); }, 5000);
    };

    // First card: one cell, answered, pinned, dragged to the right edge. Before it opens, the
    // mouse-mode hover pill sits under the hovered cell: saved beside OUT as OUT-pill.png.
    mouseMove(win, tableCellPoint(win, 0, 1));
    settle(60);
    QString pillPath = outPath;
    pillPath.insert(outPath.endsWith(QLatin1String(".png"), Qt::CaseInsensitive) ? outPath.size() - 4 : outPath.size(),
                    QStringLiteral("-pill"));
    win->grabWindow().save(pillPath);
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(30);
    sse.answer = QStringLiteral("This row is the baseline: no tiling change yet, so every other row is measured against it.");
    askAndWait(QStringLiteral("what is this row"));
    QMetaObject::invokeMethod(omni, "pinCard");
    settle(30);
    if (QQuickItem *pc = firstItem(win, "pinnedAskCard")) {
        const QPointF from = pc->mapToScene(QPointF(pc->width() / 2, 22));
        mouseDrag(win, from, QPointF(1920 - 16 - pc->width() - pc->x(), 140 - pc->y()));
        settle(30);
    }

    // Second card: Tab pins two cells (mouse mode), the card opens under the hovered one.
    mouseMove(win, tableCellPoint(win, 1, 1));
    settle(40);
    sendKey(win, Qt::Key_Tab);
    mouseMove(win, tableCellPoint(win, 2, 1));
    settle(40);
    sendKey(win, Qt::Key_Tab);
    settle(40);
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(30);
    // Markdown on purpose: inline code, bold and a fenced block must render, not show raw backticks.
    sse.answer = QStringLiteral("Row 3 is **faster**: the wider `TILE_N` keeps both warps fed, so it holds 410 tok/s "
                                "where row 2 stalls on shared-memory bank conflicts at 126.\n\n"
                                "```cpp\n"
                                "constexpr int TILE_N = 128;  // was 64\n"
                                "__shared__ float smem[TILE_N + 1];\n"
                                "```");
    askAndWait(QStringLiteral("which of these is faster and why"));
    settle(300);

    const QImage img = win->grabWindow();
    const bool saved = img.save(outPath);
    const QRectF r(card->x(), card->y(), card->width(), card->height());
    std::printf("snap-omnibar: %s %dx%d, card %.0f,%.0f %.0fx%.0f, %d pinned targets, %d pinned cards, "
                "%d blue/green pixels\nanswer: %s\n",
                saved ? "saved" : "FAILED to save", img.width(), img.height(), r.x(), r.y(), r.width(), r.height(),
                backend.targets()->pinnedCount(), omni->property("pinnedCardCount").toInt(), forbiddenPixels(img),
                qPrintable(omni->property("answer").toString()));

    // OUT-pad.png: controller mode, L2's centred command bar with a closed-loop CUA reply (the
    // tag between the two turns must not show). OUT-padicons.png: every PadIcon, idle and active.
    auto sibling = [&](const char *tag) {
        QString p = outPath;
        p.insert(outPath.endsWith(QLatin1String(".png"), Qt::CaseInsensitive) ? outPath.size() - 4 : outPath.size(),
                 QLatin1Char('-') + QLatin1String(tag));
        return p;
    };
    sendKey(win, Qt::Key_Escape);
    backend.targets()->clearPinnedTargets();
    backend.setNavMode("controller");
    settle(60);
    QMetaObject::invokeMethod(omni, "focusForVoice");
    settle(40);
    omni->setProperty("question", QStringLiteral("open x.com in chrome"));
    omni->setProperty("answer", QStringLiteral("Typing x.com into Chrome's address bar. "
                                               "[[ACTION:cua_type:Chrome address bar|x.com\n]]\n\n"
                                               "x.com is loaded in Chrome. Your home timeline is up; "
                                               "the first post is from 4 minutes ago."));
    settle(300);
    const QImage pad = win->grabWindow();
    const bool padSaved = pad.save(sibling("pad"));
    QObject *sheetObj = nullptr;
    QQuickItem *sheet = makePadIconSheet(engine, win->contentItem(), 44, &sheetObj);
    settle(150);
    const QImage icons = sheet ? win->grabWindow().copy(sheet->mapRectToScene(QRectF(0, 0, sheet->width(), sheet->height()))
                                                            .toAlignedRect())
                               : QImage();
    const bool iconsSaved = !icons.isNull() && icons.save(sibling("padicons"));
    delete sheetObj;
    std::printf("snap-omnibar: %s %s (%d blue/green px); %s %s (%d blue/green px)\n", padSaved ? "saved" : "FAILED",
                qPrintable(sibling("pad")), forbiddenPixels(pad), iconsSaved ? "saved" : "FAILED",
                qPrintable(sibling("padicons")), icons.isNull() ? -1 : forbiddenPixels(icons));
    backend.setLlmRouter(nullptr);
    return saved && padSaved && iconsSaved ? 0 : 1;
}

// Music overlay (slowmix) on the mock backend: a dry run, never an external process.
void slowmixUiTests()
{
    Backend backend(Backend::Mode::Mock);
    QTemporaryDir tmp;
    backend.setPrefsPath(tmp.path() + "/studio_prefs.json");
    backend.completeOnboarding("controller");
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    check(win != nullptr, "music ui: QML loads");
    if (!win)
        return;
    QObject *app = win->findChild<QObject *>("studioApp");
    QQuickItem *overlay = firstItem(win, "slowmixOverlay");
    SlowmixController *player = backend.slowmixController();
    if (!app || !overlay)
        return;
    settle(450);
    auto visibleItems = [&](const char *name) {
        QList<QQuickItem *> out;
        for (QQuickItem *it : findItems(overlay, name))
            if (it->isVisible())
                out << it;
        return out;
    };
    auto pillText = [&] {
        QQuickItem *p = firstItem(win, "musicPill");
        const auto t = p ? findItems(p, "barPillText") : QList<QQuickItem *>();
        return t.isEmpty() ? QString() : t.first()->property("text").toString();
    };
    auto chip = [&](const char *name, const QString &label) -> QQuickItem * {
        for (QQuickItem *it : visibleItems(name))
            if (it->property("label").toString() == label)
                return it;
        return nullptr;
    };
    auto click = [&](QQuickItem *it) {
        if (it)
            mouseClick(win, it->mapToScene(QPointF(it->width() / 2, it->height() / 2)), Qt::LeftButton);
        settle(60);
    };

    check(pillText() == QStringLiteral("♪ L1+R1") && !overlay->property("open").toBool(),
          "music ui: idle top-bar pill names the L1+R1 chord in controller mode, overlay closed", pillText());
    backend.setNavMode(QStringLiteral("vim"));
    settle(30);
    check(pillText() == QStringLiteral("♪"), "music ui: idle pill is a bare ♪ on the keyboard", pillText());
    backend.setNavMode(QStringLiteral("controller"));
    settle(30);
    sendKey(win, Qt::Key_M, Qt::ControlModifier);
    settle(300);
    check(overlay->property("open").toBool() && overlay->isVisible() && overlay->hasActiveFocus(),
          "music ui: Cmd+M opens the music overlay");

    QStringList names;
    for (QQuickItem *t : visibleItems("slowmixTrackName"))
        names << t->property("text").toString();
    check(names == QStringList({"night drive", "afterglow extended", "tidal lock", "glass harbor", "low orbit"}),
          "music ui: curated rotation by 30-day minutes (no takes, runner-ups, baked exports, stale or unplayed)",
          names.join(", "));
    QStringList pairs;
    check(textOverlaps(win, &pairs) == 0, "music ui: no overlapping text", pairs.join("; "));
    const QList<QQuickItem *> speeds = visibleItems("slowmixSpeed");
    const QList<QQuickItem *> reverbs = visibleItems("slowmixReverb");
    const QList<QQuickItem *> basses = visibleItems("slowmixBass");
    check(speeds.size() == 6 && reverbs.size() == 3 && basses.size() == 3 && visibleItems("slowmixPlay").size() == 1
              && visibleItems("slowmixLoop").size() == 1,
          "music ui: only Play, Loop, 6 speeds, 3 reverbs, 3 bass presets",
          QStringLiteral("%1 %2 %3").arg(speeds.size()).arg(reverbs.size()).arg(basses.size()));

    // Enter plays the selected (top) track at its learned favorite preset.
    sendKey(win, Qt::Key_Return);
    settle(100);
    check(player->playing() && player->currentTrack() == "night drive" && player->speed() == 0.85 && player->reverb() == "hall",
          "music ui: Enter plays the top track with its favorite 0.85x hall",
          player->currentTrack() + " " + QString::number(player->speed()) + " " + player->reverb());
    check(chip("slowmixSpeed", "0.85x") && chip("slowmixSpeed", "0.85x")->property("on").toBool()
              && chip("slowmixReverb", "Hall") && chip("slowmixReverb", "Hall")->property("on").toBool(),
          "music ui: the active speed / reverb chips light up");
    sendKey(win, Qt::Key_Down);
    sendKey(win, Qt::Key_Return);
    settle(60);
    check(player->currentTrack() == "afterglow extended" && player->speed() == 1.25 && player->reverb() == "hall",
          "music ui: Down + Enter switches track and applies its favorite 1.25x",
          player->currentTrack() + " " + QString::number(player->speed()));
    sendKey(win, Qt::Key_Right);
    check(player->speed() == 1.6, "music ui: Right steps speed to the next chip", QString::number(player->speed()));
    click(chip("slowmixSpeed", "0.90x"));
    click(chip("slowmixReverb", "Heavy"));
    click(chip("slowmixBass", "Warm"));
    check(qFuzzyCompare(player->speed(), 0.90) && player->reverb() == "heavy" && player->bass() == 2.0,
          "music ui: chip clicks set speed / reverb / bass",
          QString::number(player->speed()) + " " + player->reverb() + " " + QString::number(player->bass()));
    click(firstItem(win, "slowmixLoop"));
    check(!player->looping(), "music ui: Loop chip toggles looping");
    sendKey(win, Qt::Key_Space);
    check(!player->playing(), "music ui: Space pauses");
    sendKey(win, Qt::Key_Space);
    check(player->playing(), "music ui: Space resumes");

    // Volume: five chips, 80% lit by default, a click sets the daemon's linear gain.
    check(visibleItems("slowmixVolume").size() == 5 && chip("slowmixVolume", "80%")
              && chip("slowmixVolume", "80%")->property("on").toBool(),
          "music ui: VOLUME row shows 20..100% with 80% on by default",
          QString::number(visibleItems("slowmixVolume").size()));
    click(chip("slowmixVolume", "40%"));
    check(std::abs(player->volume() - 0.4) < 1e-9 && chip("slowmixVolume", "40%")->property("on").toBool(),
          "music ui: the 40% chip sets slowmix volume 0.40", QString::number(player->volume()));
    player->setVolume(0.8);

    // Tab flips to Spotify (header tabs), where the slowmix rows hide and Spotify's controls show.
    {
        SpotifyController *sp = backend.spotifyController();
        sendKey(win, Qt::Key_Tab);
        settle(60);
        QQuickItem *spView = firstItem(win, "spotifyView");
        check(overlay->property("tab").toString() == "spotify" && spView && spView->isVisible()
                  && visibleItems("slowmixVolume").isEmpty() && visibleItems("spotifyVolume").size() == 5,
              "music ui: Tab switches to the Spotify tab (5 volume chips, slowmix rows hidden)",
              overlay->property("tab").toString());
        check(sp->polling(), "music ui: the Spotify tab turns polling on while visible");
        click(firstItem(win, "spotifyPlay"));
        check(sp->playing() && !player->playing(),
              "music ui: Spotify Play pauses slowmix (never both)", QString::number(player->playing()));
        click(firstItem(win, "spotifyNext"));
        click(firstItem(win, "spotifyPrev"));
        click(chip("spotifyVolume", "60%"));
        const QStringList log = sp->commandLog();
        check(log.mid(log.size() - 4)
                      == QStringList{"api PUT /v1/me/player/play", "api POST /v1/me/player/next", "api POST /v1/me/player/previous",
                                     "api PUT /v1/me/player/volume?volume_percent=60"}
                  && sp->volume() == 60,
              "music ui: Play / Next / Prev / 60% map to Spotify Web API calls", log.join(" | "));
        if (QQuickItem *in = firstItem(win, "spotifySearch")) {
            in->forceActiveFocus();
            in->setProperty("text", QStringLiteral("midnight city"));
            sendKey(win, Qt::Key_Return);
            settle(60);
        }
        QQuickItem *searchBox = firstItem(win, "spotifySearch");
        check(sp->commandLog().contains("api GET /v1/search?type=track&limit=1&q=midnight%20city") && sp->track() == "midnight city"
                  && overlay->hasActiveFocus()
                  && searchBox && !searchBox->hasActiveFocus(),
              "music ui: the search box runs search-and-play and hands focus back",
              QStringLiteral("overlayFocus=%1 searchFocus=%2 track=%3 focusItem=%4")
                  .arg(overlay->hasActiveFocus())
                  .arg(searchBox && searchBox->hasActiveFocus())
                  .arg(sp->track())
                  .arg(win->activeFocusItem() ? win->activeFocusItem()->objectName() : QString("none")));
        sendKey(win, Qt::Key_Tab);
        settle(60);
        check(overlay->property("tab").toString() == "slowmix" && !sp->polling(),
              "music ui: Tab returns to Slowmix and stops Spotify polling");
        sendKey(win, Qt::Key_Space);
        check(player->playing() && !sp->playing(), "music ui: resuming slowmix pauses Spotify");
        check(sp->spawnCount() == 0, "music ui: mock mode never spawns spotify_player");
    }

    click(firstItem(win, "slowmixAll"));
    settle(100);
    check(visibleItems("slowmixTrack").size() == 7 && player->showArchive(),
          "music ui: All adds stale and never-played songs",
          QString::number(visibleItems("slowmixTrack").size()));
    click(firstItem(win, "slowmixAll"));
    settle(100);

    sendKey(win, Qt::Key_Escape);
    settle(300);
    check(!overlay->property("open").toBool() && player->playing() && player->currentTrack() == "afterglow extended",
          "music ui: Esc closes the overlay without stopping playback",
          QStringLiteral("open=%1 playing=%2 track=%3 focus=%4")
              .arg(overlay->property("open").toBool())
              .arg(player->playing())
              .arg(player->currentTrack())
              .arg(win->activeFocusItem() ? win->activeFocusItem()->objectName() + "/" + win->activeFocusItem()->metaObject()->className() : QString("none")));
    check(pillText() == QStringLiteral("♪ afterglow extended"), "music ui: top-bar pill names the playing track", pillText());
    sendKey(win, Qt::Key_M, Qt::ControlModifier);
    settle(200);
    const bool reopened = overlay->property("open").toBool();
    sendKey(win, Qt::Key_M, Qt::ControlModifier);
    settle(200);
    check(reopened && !overlay->property("open").toBool(), "music ui: Cmd+M toggles the overlay");
    // From a focused composer too: Cmd+M is music there, Cmd+Shift+M stays the model strip.
    QMetaObject::invokeMethod(app, "enterProject", Q_ARG(QVariant, QVariant(0)));
    settle(400);
    if (QQuickItem *in = firstItem(win, "composerInput")) {
        in->forceActiveFocus();
        settle(60);
        const bool composerFocused = in->hasActiveFocus();
        sendKey(win, Qt::Key_M, Qt::ControlModifier);
        settle(250);
        QQuickItem *strip = firstItem(win, "modelStrip");
        check(composerFocused && overlay->property("open").toBool() && !(strip && strip->isVisible()),
              "music ui: Cmd+M from the composer opens music, not the model strip");
        QQuickItem *hdr = firstItem(win, "headerMusicPill");
        check(hdr && hdr->isVisible(), "music ui: zoomed lenses show the ♪ pill in the header");
        sendKey(win, Qt::Key_Escape);
        settle(250);
    } else {
        check(false, "music ui: composer input found");
    }
    check(player->spawnCount() == 0 && !player->running() && !player->spawnAllowed(),
          "music ui: mock mode never spawns slowmix");
}

int snapSlowmix(const QString &outPath)
{
    Backend backend(Backend::Mode::Mock);
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    if (!win) {
        std::fprintf(stderr, "snap-slowmix: QML failed to load\n");
        return 2;
    }
    QQuickItem *overlay = firstItem(win, "slowmixOverlay");
    if (!overlay)
        return 3;
    QDir().mkpath(QFileInfo(outPath).absolutePath());
    QString barPath = outPath;
    barPath.insert(outPath.endsWith(QLatin1String(".png"), Qt::CaseInsensitive) ? outPath.size() - 4 : outPath.size(),
                   QLatin1String("-bar"));
    settle(450);
    // Playing the top of the rotation (a dry run: no audio, no process).
    SlowmixController *player = backend.slowmixController();
    player->refresh();
    if (!player->tracks().isEmpty())
        player->selectTrack(player->tracks().first().toMap().value("path").toString());
    mouseMove(win, QPointF(4, 1076));
    settle(500);
    const QImage bar = win->grabWindow();
    const bool barSaved = bar.save(barPath);
    sendKey(win, Qt::Key_M, Qt::ControlModifier);
    settle(500);
    const QImage img = win->grabWindow();
    const bool saved = img.save(outPath);
    std::printf("snap-slowmix: %s %s (%d blue/green px), %lld tracks, playing \"%s\" %.2fx %s\n",
                saved ? "saved" : "FAILED", qPrintable(outPath), forbiddenPixels(img),
                static_cast<long long>(player->tracks().size()), qPrintable(player->currentTrack()), player->speed(),
                qPrintable(player->reverb()));
    std::printf("snap-slowmix: %s %s (%d blue/green px), spawned %d\n", barSaved ? "saved" : "FAILED",
                qPrintable(barPath), forbiddenPixels(bar), player->spawnCount());
    // The Spotify tab, in a dry run (no process): a search-and-play stands in for real playback.
    QString spPath = outPath;
    spPath.insert(outPath.endsWith(QLatin1String(".png"), Qt::CaseInsensitive) ? outPath.size() - 4 : outPath.size(),
                  QLatin1String("-spotify"));
    SpotifyController *sp = backend.spotifyController();
    sp->searchAndPlay(QStringLiteral("midnight city"));
    sp->setVolume(60);
    sendKey(win, Qt::Key_Tab);
    settle(500);
    const QImage spImg = win->grabWindow();
    const bool spSaved = spImg.save(spPath);
    std::printf("snap-slowmix: %s %s (%d blue/green px), spotify spawned %d\n", spSaved ? "saved" : "FAILED",
                qPrintable(spPath), forbiddenPixels(spImg), sp->spawnCount());
    return saved && barSaved && spSaved ? 0 : 1;
}

// --- Cross-session memory in Tier 3 -------------------------------------------------------------

void memoryTierTests()
{
    Backend b(Backend::Mode::Mock);
    check(b.memoryIndex()->size() == 0 && !b.memoryIndex()->refreshing(),
          "memory tier: the mock backend's index stays empty (no transcript scan in --test)");
    QString key;
    for (const QVariant &v : b.projects()) {
        const QString k = v.toMap().value("key").toString();
        if (!b.contextTrunk(k).isEmpty() && !b.contextSlice(k, 0).isEmpty()) {
            key = k;
            break;
        }
    }
    check(!key.isEmpty(), "memory tier: a mock project with a trunk and a slice");
    FakeSse sse;
    check(sse.start(), "memory tier: fake SSE endpoint listens");
    LlmRouter router;
    b.setLlmRouter(&router);
    b.setOpenaiCompatUrl(sse.url());
    QStringList finished;
    QObject::connect(&b, &Backend::streamFinished, [&finished](const QString &id, const QString &) { finished << id; });
    const QString q = QStringLiteral("why did cublas lose to the fp4 kernel?");

    b.askStreamWithSlice(QStringLiteral("mem0"), key, 0, QStringLiteral("hovered: row 2"), q);
    check(waitUntil([&] { return finished.contains("mem0"); }, 5000) && sse.bodies.size() == 1, "memory tier: baseline ask sent");

    const qint64 now = QDateTime::currentMSecsSinceEpoch();
    const QString active = b.activeSessionId(key);
    check(!active.isEmpty(), "memory tier: the project names its active (Tier 1) session");
    MemoryTurn past{QStringLiteral("claude"), QStringLiteral("0badc0de-0000-0000-0000-000000000001"), QStringLiteral("gemm-lab"),
                    now - 2 * 86400000, 0, QStringLiteral("cuBLAS vs our sm120 FP4 kernel at 128x256 tiles?"),
                    QStringLiteral("cuBLAS chose a 64x128 tile and left the tensor cores half fed.")};
    MemoryTurn own = past;
    own.sessionId = active;
    own.userText = QStringLiteral("cuBLAS fp4 kernel lose question from the active session itself");
    b.memoryIndex()->setTurns({past, own});
    b.askStreamWithSlice(QStringLiteral("mem1"), key, 0, QStringLiteral("hovered: row 2"), q);
    b.askStream(QStringLiteral("mem2"), key, 0, q);
    check(waitUntil([&] { return finished.contains("mem1") && finished.contains("mem2"); }, 5000) && sse.bodies.size() == 3,
          "memory tier: asks with recall sent", QString::number(sse.bodies.size()));

    const QJsonArray m0 = QJsonDocument::fromJson(sse.bodies.value(0)).object().value("messages").toArray();
    // mem1 and mem2 run concurrently: tell them apart by the hovered slice, not arrival order.
    const bool firstIsHover = sse.bodies.value(1).contains("hovered: row 2");
    const QJsonArray m1 = QJsonDocument::fromJson(sse.bodies.value(firstIsHover ? 1 : 2)).object().value("messages").toArray();
    const QJsonArray m2 = QJsonDocument::fromJson(sse.bodies.value(firstIsHover ? 2 : 1)).object().value("messages").toArray();
    // The warmed prefix: exactly what warmProjectPrefix() prefills for this project.
    TieredPrompt warm;
    warm.model = b.tieredPrompt(key, QString(), q).model;
    warm.systemPrompt = kCompatSystemPrompt;
    warm.trunk = b.contextTrunk(key);
    const QJsonArray w = QJsonDocument::fromJson(buildTieredWarmRequest(warm)).object().value("messages").toArray();
    auto bytes = [](const QJsonValue &v) { return QJsonDocument(v.toObject()).toJson(QJsonDocument::Compact); };
    // mem0's exchange is the shared Tier 2 of mem1 and mem2 (2 messages between Tier 1 and the leaf).
    check(w.size() == 2 && m0.size() == 3 && m1.size() == 5 && m2.size() == 5 && bytes(m1[0]) == bytes(w[0])
              && bytes(m1[1]) == bytes(w[1]) && bytes(m2[0]) == bytes(w[0]) && bytes(m2[1]) == bytes(w[1])
              && bytes(m0[0]) == bytes(m1[0]) && bytes(m0[1]) == bytes(m1[1]),
          "memory tier: Tier 0 + Tier 1 stay byte-identical to the warmed prefix with recall on",
          QStringLiteral("%1/%2/%3/%4").arg(w.size()).arg(m0.size()).arg(m1.size()).arg(m2.size()));
    const QString leaf0 = m0.last().toObject().value("content").toString();
    const QString leaf1 = m1.last().toObject().value("content").toString();
    const QString leaf2 = m2.last().toObject().value("content").toString();
    const QString recall = QStringLiteral("Relevant past conversations:\n[claude 0badc0de] gemm-lab, 2d ago\nUser: cuBLAS vs our sm120");
    check(!leaf0.contains("Relevant past conversations") && leaf1.contains(recall) && leaf2.contains(recall)
              && leaf1.startsWith("On screen now:\nhovered: row 2") && leaf1.endsWith("Question: " + q),
          "memory tier: recall lands in the Tier 3 leaf only (slice first, question last)", leaf1.left(300));
    check(!leaf1.contains("from the active session itself") && !bytes(m1[1]).contains("Relevant past"),
          "memory tier: the active session's own turns are excluded; Tier 1 never carries recall");
    const qsizetype at = leaf1.indexOf("Relevant past conversations:\n");
    const qsizetype qAt = leaf1.lastIndexOf("\n\nQuestion: ");
    check(at >= 0 && qAt > at && qAt - at - 29 <= 3200, "memory tier: the recall block is <= 3,200 chars",
          QString::number(qAt - at - 29));
    const TieredPrompt quick = b.tieredPrompt(key, b.contextSlice(key, 0), q);
    check(quick.memory.contains("[claude 0badc0de]") && quick.trunk == b.contextTrunk(key),
          "memory tier: the quick-ask prompt (askOpenAiCompat) carries the same recall in Tier 3");
    b.setLlmRouter(nullptr);
    check(b.spawnCount() == 0, "memory tier: zero processes spawned");
}

// --- Fresh-engine QML hot reload ----------------------------------------------------------------

void hotReloadTests()
{
    Backend backend(Backend::Mode::Mock);
    QTemporaryDir td;
    const QString dir = td.path();
    // The compiled-in module's sources, copied to disk: the same files `--hot-reload` reads from qml/.
    const QDir src(QStringLiteral(":/qt/qml/Studio/qml"));
    int copied = 0;
    for (const QString &f : src.entryList({QStringLiteral("*.qml")}, QDir::Files))
        copied += QFile::copy(src.filePath(f), dir + "/" + f) ? 1 : 0;
    for (const QString &f : QDir(dir).entryList({QStringLiteral("*.qml")}, QDir::Files))
        QFile::setPermissions(dir + "/" + f, QFileDevice::ReadOwner | QFileDevice::WriteOwner);
    check(copied >= 20 && QmlHost::ensureQmldir(dir) && QFileInfo::exists(dir + "/qmldir"),
          "reload: qml sources + qmldir on disk", QString::number(copied));
    QFile orig(dir + "/StudioApp.qml");
    check(orig.open(QIODevice::ReadOnly), "reload: StudioApp.qml readable");
    const QByteArray original = orig.readAll();
    orig.close();
    auto writeApp = [&](const QByteArray &data) {
        QFile f(dir + "/StudioApp.qml");
        return f.open(QIODevice::WriteOnly | QIODevice::Truncate) && f.write(data) == data.size();
    };
    auto withMarker = [&](const char *v) {
        QByteArray d = original;
        d.replace("    property int lens: 0\n", QByteArray("    property int lens: 0\n    property string hotMarker: \"") + v + "\"\n");
        return d;
    };

    QQuickWindow win;
    win.setColor(Qt::black);
    win.resize(1920, 1080);
    win.show();
    QmlHost host(&win, &backend);
    host.setQmlDir(dir);
    check(host.load() && host.root() && host.root()->objectName() == "studioApp" && host.generation() == 1,
          "reload: StudioApp loads from the filesystem into a bare QQuickWindow", host.lastErrors().join("; "));
    if (!host.root())
        return;
    QPointer<QQuickItem> root1 = host.root();
    QPointer<QQmlEngine> engine1 = host.engine();
    settle(50);
    check(qAbs(root1->width() - 1920) < 0.5 && qAbs(root1->height() - 1080) < 0.5 && root1->parentItem() == win.contentItem(),
          "reload: root fills the window's contentItem");
    check(findItems(win.contentItem(), "island").size() == 5, "reload: the Theme singleton and child types resolve from disk");

    QString key;
    for (const QVariant &v : backend.projects())
        if (v.toMap().value("artifacts").toList().size() > 1) {
            key = v.toMap().value("key").toString();
            break;
        }
    root1->setProperty("projectKey", key);
    root1->setProperty("artifactIndex", 1);
    root1->setProperty("lens", 2);
    root1->setProperty("islandFilter", "needs");
    settle(300);

    // 1. Broken QML: errors logged, the live root and engine untouched.
    check(writeApp(original + "\n}}} not qml {{{\n"), "reload: broken StudioApp.qml written");
    check(!host.reload() && host.failures() == 1 && !host.lastErrors().isEmpty(), "reload: broken QML is rejected with errors",
          host.lastErrors().value(0));
    settle(50);
    check(root1 && host.root() == root1 && engine1 && host.engine() == engine1 && root1->isVisible()
              && root1->parentItem() == win.contentItem() && host.generation() == 1,
          "reload: after broken QML the old root and engine stay live (no blank window)");
    check(root1 && root1->property("lens").toInt() == 2, "reload: the live UI keeps its state after a failed reload");

    // 2. A valid edit: fresh engine, new root swapped in, state restored, old root + engine freed.
    check(writeApp(withMarker("v2")), "reload: edited StudioApp.qml written");
    check(host.reload() && host.generation() == 2, "reload: valid edit swaps in a new root", host.lastErrors().join("; "));
    QQuickItem *root2 = host.root();
    check(root2 && root2 != root1.data() && host.engine() != engine1.data() && root2->property("hotMarker").toString() == "v2",
          "reload: the new root comes from a fresh QQmlEngine and shows the edit");
    check(root2 && root2->property("backend").value<QObject *>() == &backend,
          "reload: the same C++ Backend pointer is bound into the new root");
    check(root2 && root2->property("projectKey").toString() == key && root2->property("artifactIndex").toInt() == 1
              && root2->property("lens").toInt() == 2 && root2->property("islandFilter").toString() == "needs",
          "reload: lens, project, slide and Needs Me filter carry over",
          root2 ? QStringLiteral("%1/%2/%3/%4").arg(root2->property("projectKey").toString())
                      .arg(root2->property("artifactIndex").toInt()).arg(root2->property("lens").toInt())
                      .arg(root2->property("islandFilter").toString())
                : QString());
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    settle(50);
    check(root1.isNull() && engine1.isNull(), "reload: the old root and old engine are destroyed");
    check(win.isVisible() && root2 && qAbs(root2->width() - 1920) < 0.5, "reload: same window, still shown, new root fills it");
    win.resize(1600, 1000);
    settle(20);
    check(root2 && qAbs(root2->width() - 1600) < 0.5 && qAbs(root2->height() - 1000) < 0.5, "reload: new root tracks window resizes");
    win.resize(1920, 1080);

    // 3. The live triggers: a saved file (watcher, 150 ms debounce), Cmd+Shift+L, SIGUSR1.
    host.setHotReload(true);
    check(writeApp(withMarker("v3")), "reload: v3 written");
    check(waitUntil([&] { return host.generation() == 3; }, 4000) && host.root()->property("hotMarker").toString() == "v3",
          "reload: a file save reloads through the watcher", QString::number(host.generation()));
    sendKey(&win, Qt::Key_L, Qt::ControlModifier | Qt::ShiftModifier);
    check(waitUntil([&] { return host.generation() == 4; }, 2000), "reload: Cmd+Shift+L reloads", QString::number(host.generation()));
    ::raise(SIGUSR1);
    check(waitUntil([&] { return host.generation() == 5; }, 2000), "reload: SIGUSR1 reloads", QString::number(host.generation()));
    host.setHotReload(false);
    QCoreApplication::sendPostedEvents(nullptr, QEvent::DeferredDelete);
    check(forbiddenPixels(win.grabWindow()) == 0 && backend.spawnCount() == 0,
          "reload: reloaded UI renders on-palette; zero processes spawned");
}

// Spec gates: natural-language Studio UI actions, the L1+R1 Music chord and the DualShock inside
// the Music card, shared cross-model history, live screen state, and prefs that tests never touch.
void studioControlTests()
{
    // --- Parser (pure) -------------------------------------------------------------------------
    struct Case {
        const char *text;
        const char *action;
        const char *arg;
    };
    const Case cases[] = {
        {"Can you open the Slowmix tab inside of this?", "open_music", ""},
        {"open music", "open_music", ""},
        {"hey lapis, could you please pull up the music player", "open_music", ""},
        {"close the music", "close_music", ""},
        {"play syntheticgod", "play_track", "syntheticgod"},
        {"Play the song night drive please", "play_track", "night drive"},
        {"pause the music", "pause_music", ""},
        {"resume music", "resume_music", ""},
        {"play some music", "resume_music", ""},
        {"open session review", "open_review", ""},
        {"show needs me", "filter_needs_me", ""},
        {"show me only what needs me", "filter_needs_me", ""},
        {"show all cards", "filter_all", ""},
        {"reset layout", "reset_layout", ""},
        {"open settings", "open_settings", ""},
        {"switch to lapis blue", "set_theme", "blue"},
        {"switch to gold theme", "set_theme", "gold"},
        {"show code diffs", "show_diffs", "1"},
        {"hide code diffs", "show_diffs", "0"},
        {"open briefing", "open_briefing", ""},
        {"Fire off an agent to fix the flaky tile test", "spawn_agent", "fix the flaky tile test"},
        {"what changed in the sweep?", "", ""},
        {"why is the music overlay slow to open?", "", ""},
        {"show me the p99 for the fp4 build", "", ""},
        {"is gold better than blue for charts?", "", ""},
    };
    QStringList wrong;
    for (const Case &c : cases) {
        const StudioCommand got = parseStudioCommand(QString::fromUtf8(c.text));
        if (got.action != QLatin1String(c.action) || got.arg != QString::fromUtf8(c.arg))
            wrong << QStringLiteral("'%1' -> %2:%3").arg(c.text, got.action, got.arg);
    }
    check(wrong.isEmpty(), "studio cmd: voice-typed UI phrasings map to actions; questions stay questions",
          wrong.join(" | "));
    check(parseStudioCommand("tell the agent to rerun the bench").steer, "studio cmd: 'tell the agent' steers the live peer");
    {
        QString a = QStringLiteral("[[ACTION:open_music]] [[ACTION:bogus:1]] Opening it.");
        const QList<StudioCommand> tags = extractActionTags(&a);
        check(tags.size() == 1 && tags[0].action == "open_music" && a == "Opening it.",
              "studio cmd: [[ACTION:..]] tags parse, unknown names drop, text stays clean", a);
        QString b = QStringLiteral("[[ACTION:play_track:night drive]]");
        const QList<StudioCommand> t2 = extractActionTags(&b);
        check(t2.size() == 1 && t2[0].arg == "night drive" && b.isEmpty(), "studio cmd: tag with an argument");
    }
    // --- Visible computer use: cua_click / cua_type / cua_scroll tags (mock backend never actuates) ---
    {
        QString a = QStringLiteral("[[ACTION:cua_click:Send button]] [[ACTION:cua_type:Search field|hello: world]] "
                                   "[[ACTION:cua_scroll:results list|-5]] Done.");
        const QList<StudioCommand> tags = extractActionTags(&a);
        check(tags.size() == 3 && tags[0].action == "cua_click" && tags[0].arg == "Send button"
                  && tags[1].action == "cua_type" && tags[1].arg == "Search field|hello: world"
                  && tags[2].action == "cua_scroll" && tags[2].arg == "results list|-5" && a == "Done.",
              "cua: [[ACTION:cua_click/cua_type/cua_scroll]] tags are known actions", a);
        const ParsedModelOutput p = parseStudioModelOutput(QStringLiteral("Sure. [[ACTION:cua_type:To: field|a|b]]"));
        check(p.actions.size() == 1 && p.actions[0].name == "cua_type" && p.actions[0].arg == "To: field|a|b"
                  && p.cleanText == "Sure.",
              "cua: compound parser keeps ':' and '|' inside the arg");

        const CuaCommand click = parseCuaAction("cua_click", "  Send   button ");
        check(click.valid && click.kind == "click" && click.target == "Send button", "cua: click target simplified");
        check(!parseCuaAction("cua_click", "  ").valid, "cua: empty click target rejected");
        const CuaCommand type = parseCuaAction("cua_type", "To: field|a|b ");
        check(type.valid && type.kind == "type" && type.target == "To: field" && type.text == "a|b ",
              "cua: type splits at the first '|', text verbatim", type.target + " / " + type.text);
        check(!parseCuaAction("cua_type", "no separator").valid && !parseCuaAction("cua_type", "field|").valid,
              "cua: type without a target|text pair rejected");
        const CuaCommand sc = parseCuaAction("cua_scroll", "a|b list|-7");
        check(sc.valid && sc.kind == "scroll" && sc.target == "a|b list" && sc.deltaY == -7,
              "cua: scroll splits at the last '|'");
        check(parseCuaAction("cua_scroll", "page|down").deltaY == -5 && parseCuaAction("cua_scroll", "up").deltaY == 5
                  && parseCuaAction("cua_scroll", "up").target.isEmpty(),
              "cua: scroll up/down words, bare amount scrolls the display");
        check(!parseCuaAction("cua_scroll", "list|0").valid && !parseCuaAction("cua_scroll", "list|lots").valid
                  && parseCuaAction("cua_scroll", "list|9999").deltaY == 100,
              "cua: scroll rejects zero / junk and clamps huge deltas");

        Backend b(Backend::Mode::Mock);
        QStringList finished;
        QObject::connect(&b, &Backend::cuaActionFinished, [&](const QString &act, const QString &target, bool ok,
                                                            const QString &tier, const QString &) {
            finished << QStringLiteral("%1:%2:%3:%4").arg(act, target, ok ? "1" : "0", tier);
        });
        const QString r1 = b.dispatchStudioAction("cua_click", "Send button");
        const QString r2 = b.dispatchStudioAction("cua_type", "Search|lapis");
        const QString r3 = b.dispatchStudioAction("cua_scroll", "results|-3");
        check(r1.contains("Clicking Send button") && r1.contains("not run") && r2.contains("Typing into Search")
                  && r3.contains("Scrolling results"),
              "cua: mock dispatch confirms without running", r1 + " | " + r2 + " | " + r3);
        check(b.cuaLog() == QStringList({"cua_click:Send button", "cua_type:Search|lapis", "cua_scroll:results|-3"}),
              "cua: mock backend records every action", b.cuaLog().join(" | "));
        check(finished.size() == 3 && finished[0] == "cua_click:Send button:1:MOCK" && b.cuaPending() == 0,
              "cua: mock emits cuaActionFinished with tier MOCK, nothing pending", finished.join(" | "));
        check(b.dispatchStudioAction("cua_type", "no bar").isEmpty() && b.cuaLog().size() == 3,
              "cua: an invalid cua arg is not dispatched");
        check(kCompatSystemPrompt.contains("cua_click:<on-screen target>"), "cua: the system prompt teaches the cua tags");
    }

    // --- Desktop hotkeys and click-drags: cua_hotkey / cua_drag tags --------------------------------
    {
        QString a = QStringLiteral("[[ACTION:cua_hotkey:cmd+l]] [[ACTION:cua_hotkey:Chrome|cmd+t]] "
                                   "[[ACTION:cua_drag:slider handle|volume max]] Done.");
        const QList<StudioCommand> tags = extractActionTags(&a);
        check(tags.size() == 3 && tags[0].action == "cua_hotkey" && tags[0].arg == "cmd+l" && tags[1].action == "cua_hotkey"
                  && tags[1].arg == "Chrome|cmd+t" && tags[2].action == "cua_drag" && tags[2].arg == "slider handle|volume max"
                  && a == "Done.",
              "cua keys: [[ACTION:cua_hotkey]] and [[ACTION:cua_drag]] are known tags", a);
        const CuaCommand k = parseCuaAction("cua_hotkey", "cmd+l");
        check(k.valid && k.kind == "hotkey" && k.target.isEmpty() && k.text == "cmd+l", "cua keys: cua_hotkey:cmd+l has no app");
        const CuaCommand kt = parseCuaAction("cua_hotkey", " Chrome | cmd + t ");
        check(kt.valid && kt.kind == "hotkey" && kt.target == "Chrome" && kt.text == "cmd+t",
              "cua keys: cua_hotkey:Chrome|cmd+t targets Chrome", kt.target + " / " + kt.text);
        check(!parseCuaAction("cua_hotkey", "").valid && !parseCuaAction("cua_hotkey", "cmd+banana").valid
                  && !parseCuaAction("cua_hotkey", "Chrome|").valid && !parseCuaAction("cua_hotkey", "cmd+shift").valid,
              "cua keys: empty, unknown and modifier-only combos rejected");
        const CuaCommand d = parseCuaAction("cua_drag", " slider   handle | volume max ");
        check(d.valid && d.kind == "drag" && d.target == "slider handle" && d.text == "volume max",
              "cua keys: cua_drag:slider handle|volume max parses from / to");
        check(!parseCuaAction("cua_drag", "slider handle").valid && !parseCuaAction("cua_drag", "|volume max").valid
                  && !parseCuaAction("cua_drag", "slider handle| ").valid,
              "cua keys: a drag needs both ends");

        Backend b(Backend::Mode::Mock);
        QStringList finished;
        QObject::connect(&b, &Backend::cuaActionFinished, [&](const QString &act, const QString &target, bool ok,
                                                            const QString &tier, const QString &) {
            finished << QStringLiteral("%1:%2:%3:%4").arg(act, target, ok ? "1" : "0", tier);
        });
        const QString r1 = b.dispatchStudioAction("cua_hotkey", "cmd+l");
        const QString r2 = b.dispatchStudioAction("cua_hotkey", "Chrome|cmd+t");
        const QString r3 = b.dispatchStudioAction("cua_drag", "slider handle|volume max");
        check(r1.contains("Pressing Cmd+L") && r2.contains("Pressing Cmd+T in Chrome") && r3.contains("Dragging slider handle to volume max")
                  && r3.contains("not run"),
              "cua keys: mock dispatch confirms without running", r1 + " | " + r2 + " | " + r3);
        check(b.cuaLog() == QStringList({"cua_hotkey:cmd+l", "cua_hotkey:Chrome|cmd+t", "cua_drag:slider handle|volume max"}),
              "cua keys: mock backend records hotkeys and drags", b.cuaLog().join(" | "));
        check(finished == QStringList({"cua_hotkey:cmd+l:1:MOCK", "cua_hotkey:Chrome|cmd+t:1:MOCK",
                                       "cua_drag:slider handle|volume max:1:MOCK"}),
              "cua keys: finished signals name the combo and both drag ends", finished.join(" | "));
        check(b.dispatchStudioAction("cua_hotkey", "cmd+banana").isEmpty() && b.cuaLog().size() == 3,
              "cua keys: an invalid hotkey is not dispatched");
        check(kCompatSystemPrompt.contains("cua_hotkey:<app>|<combo>") && kCompatSystemPrompt.contains("cua_drag:<from>|<to>"),
              "cua keys: the system prompt teaches cua_hotkey and cua_drag");
    }

    // --- cua_open: apps and URLs without the omnibox; page-sized scroll words ----------------------
    {
        QString a = QStringLiteral("[[ACTION:cua_open:Chrome|https://x.com/elonmusk]] [[ACTION:cua_open:Finder]] Opening.");
        const QList<StudioCommand> tags = extractActionTags(&a);
        check(tags.size() == 2 && tags[0].action == "cua_open" && tags[0].arg == "Chrome|https://x.com/elonmusk"
                  && tags[1].action == "cua_open" && tags[1].arg == "Finder" && a == "Opening.",
              "cua open: [[ACTION:cua_open:...]] is a known tag and keeps the URL's ':'", a);
        const CuaCommand app = parseCuaAction("cua_open", " Chrome ");
        check(app.valid && app.kind == "open" && app.target == "Chrome" && app.text.isEmpty(), "cua open: cua_open:Chrome opens the app");
        const CuaCommand url = parseCuaAction("cua_open", "https://x.com/elonmusk");
        check(url.valid && url.target.isEmpty() && url.text == "https://x.com/elonmusk", "cua open: a bare URL goes to the default browser");
        const CuaCommand both = parseCuaAction("cua_open", "Chrome|x.com/elonmusk");
        check(both.valid && both.target == "Chrome" && both.text == "https://x.com/elonmusk",
              "cua open: Chrome|x.com/elonmusk adds https://", both.target + " / " + both.text);
        const CuaCommand bare = parseCuaAction("cua_open", "x.com/pmarca");
        check(bare.valid && bare.target.isEmpty() && bare.text == "https://x.com/pmarca", "cua open: a bare domain path is a URL");
        check(!parseCuaAction("cua_open", "").valid && !parseCuaAction("cua_open", "Chrome|not a url").valid
                  && parseCuaAction("cua_open", "Notion.app").target == "Notion.app",
              "cua open: empty and junk URLs rejected, a .app name stays an app");
        check(parseCuaAction("cua_scroll", "Chrome|pagedown").deltaY == -12 && parseCuaAction("cua_scroll", "Chrome|pageup").deltaY == 12
                  && parseCuaAction("cua_scroll", "Chrome|down").deltaY == -5 && parseCuaAction("cua_scroll", "Chrome|down").target == "Chrome",
              "cua open: scroll pageup / pagedown words");

        Backend b(Backend::Mode::Mock);
        const QString r1 = b.dispatchStudioAction("cua_open", "Chrome|x.com/elonmusk");
        const QString r2 = b.dispatchStudioAction("cua_open", "Chrome");
        const QString r3 = b.dispatchStudioAction("cua_open", "https://github.com");
        check(r1.contains("Opening https://x.com/elonmusk in Chrome") && r2.contains("Opening Chrome") && r3.contains("Opening https://github.com")
                  && r1.contains("not run"),
              "cua open: mock dispatch confirms without running", r1 + " | " + r2 + " | " + r3);
        check(b.cuaLog() == QStringList({"cua_open:Chrome|https://x.com/elonmusk", "cua_open:Chrome", "cua_open:https://github.com"}),
              "cua open: mock backend records app and URL", b.cuaLog().join(" | "));
        check(b.dispatchStudioAction("cua_open", "Chrome|not a url").isEmpty() && b.cuaLog().size() == 3,
              "cua open: an invalid URL is not dispatched");
        check(studioActionNames().contains("cua_open") && kCompatSystemPrompt.contains("cua_open:<app>|<url>")
                  && kCompatSystemPrompt.contains("[[ACTION:cua_open:Chrome|https://example.com]]")
                  && kCompatSystemPrompt.contains("[[ACTION:cua_scroll:Chrome|down]]")
                  && kCompatSystemPrompt.contains("never ask permission"),
              "cua open: the system prompt teaches cua_open, page scrolling and acting without asking");
    }

    // --- Closed-loop computer use: cua_observe and the bounded follow-up turns ---------------------
    {
        QString a = QStringLiteral("[[ACTION:cua_observe:Discord]] [[ACTION:cua_observe]] Looking.");
        const QList<StudioCommand> tags = extractActionTags(&a);
        check(tags.size() == 2 && tags[0].action == "cua_observe" && tags[0].arg == "Discord" && tags[1].action == "cua_observe"
                  && tags[1].arg.isEmpty() && a == "Looking.",
              "cua loop: [[ACTION:cua_observe:Discord]] and bare cua_observe are known tags", a);
        const CuaCommand obs = parseCuaAction("cua_observe", " Discord ");
        check(obs.valid && obs.kind == "observe" && obs.target == "Discord", "cua loop: cua_observe:Discord parses to observe Discord");
        check(parseCuaAction("cua_observe", "").valid && parseCuaAction("cua_observe", "").target.isEmpty(),
              "cua loop: bare cua_observe reads the front display");
        check(kCompatSystemPrompt.contains("cua_observe:<app>") && kCompatSystemPrompt.contains("[[ACTION:cua_observe:Discord]]"),
              "cua loop: the system prompt teaches cua_observe");

        const CuaStepResult seen{"cua_observe", "Discord", true, "L1_OBS", "12 AX nodes", "{\"title\":\"John Gout\"}"};
        const CuaStepResult clicked{"cua_click", "Send", true, "L1_AX", "AXButton \"Send\"", "{}"};
        const CuaStepResult missed{"cua_click", "Ghost", false, "ERR", "No AX match", "{}"};
        check(cuaShouldContinue({seen}, 0) && cuaShouldContinue({clicked, missed}, 3) && cuaShouldContinue({clicked}, 0)
                  && !cuaShouldContinue({}, 0),
              "cua loop: any non-empty cua batch continues until the model stops emitting tags or hits depth cap");
        check(!cuaShouldContinue({seen}, kMaxCuaLoopDepth) && !cuaShouldContinue({missed}, kMaxCuaLoopDepth + 3) && kMaxCuaLoopDepth == 7
                  && cuaShouldContinue({seen}, 6),
              "cua loop: never past depth 7");
        const QString p = cuaFollowupPrompt({seen});
        check(p.startsWith("[CUA observation after cua_observe:Discord (L1_OBS, ok=1, detail=12 AX nodes)]:\n{\"title\":\"John Gout\"}")
                  && p.endsWith("Continue only if more actions are needed; otherwise give a concise final answer without cua_* tags."),
              "cua loop: the follow-up prompt carries the action, tier, ok, detail and observation", p);
        check(cuaFollowupPrompt({missed}).contains("(ERR, ok=0, detail=No AX match)"), "cua loop: a failure reads ok=0");

        Backend b(Backend::Mode::Mock);
        b.setShadowRecordEnabled(true);
        check(!b.shadowRecordEnabled() && !b.shadowRecording(), "shadow recorder: always off in Mode::Mock, setter is a no-op");
        check(b.runModelAnswer("[[ACTION:cua_click:Send button]]").contains("Clicking Send button") && b.cuaLoopDepth() == 1
                  && b.cuaFollowups().size() == 1
                  && b.cuaFollowups()[0].startsWith("[CUA observation after cua_click:Send button (MOCK, ok=1"),
              "cua loop: a clean click batch earns one follow-up (the model sees the result)", b.cuaFollowups().join(" | "));
        b.clearAnswers();
        b.runModelAnswer("[[ACTION:cua_observe:Discord]]");
        check(b.cuaLoopDepth() == 1 && b.cuaFollowups().size() == 1
                  && b.cuaFollowups()[0].startsWith("[CUA observation after cua_observe:Discord (MOCK, ok=1")
                  && b.cuaLog().contains("cua_observe:Discord"),
              "cua loop: cua_observe earns one follow-up turn", b.cuaFollowups().join(" | "));
        check(!b.quickBusy() && b.quickQuestion().isEmpty(), "cua loop: mock records the follow-up and never sends it");
        int maxDepth = b.cuaLoopDepth();
        for (int i = 0; i < 8; ++i) {
            b.runModelAnswer("[[ACTION:cua_observe:Discord]] [[ACTION:cua_type:Discord search|John]]");
            maxDepth = std::max(maxDepth, b.cuaLoopDepth());
        }
        check(maxDepth <= kMaxCuaLoopDepth && b.cuaLoopDepth() == 7 && b.cuaFollowups().size() == 7,
              "cua loop: m_cuaLoopDepth caps at 7 follow-ups per prompt", QString::number(b.cuaLoopDepth()));
        b.runModelAnswer("Done: John Gout's DM is open.");
        check(b.cuaLoopDepth() == 7 && b.cuaFollowups().size() == 7, "cua loop: an answer without cua tags ends the loop");
        b.clearAnswers();
        check(b.cuaLoopDepth() == 0 && b.cuaFollowups().isEmpty(), "cua loop: dismissing the answer resets the depth");
        b.runModelAnswer("[[ACTION:cua_observe:Discord]]");
        check(b.cuaLoopDepth() == 1, "cua loop: a fresh prompt loops again");
        b.ask(QString(), -1, QStringLiteral("what is the weather like on mars"));
        check(b.cuaLoopDepth() == 0 && b.cuaFollowups().isEmpty(), "cua loop: a new user question resets the depth");
    }
    // --- Cmd+W guard: never unprompted, at most one per model reply ------------------------------
    {
        check(cuaCloseTabBlockReason("open my inbox", "cmd+w", 0).contains("unless the user asks to close")
                  && cuaCloseTabBlockReason("open my inbox", " Command + W ", 0).contains("unless the user asks to close")
                  && cuaCloseTabBlockReason("Close the duplicate tab", "CMD+W", 0).isEmpty()
                  && cuaCloseTabBlockReason("Close the duplicate tab", "cmd+w", 1).contains("only one Cmd+W")
                  && cuaCloseTabBlockReason("open my inbox", "cmd+t", 0).isEmpty()
                  && cuaCloseTabBlockReason("open my inbox", "cmd+shift+w", 0).isEmpty(),
              "cmd+w guard: pure reason (unprompted blocked, prompted once, other combos untouched)");
        Backend b(Backend::Mode::Mock);
        b.ask(QString(), -1, QStringLiteral("find the pricing page in chrome"));
        const QString unprompted = b.runModelAnswer("[[ACTION:cua_hotkey:Chrome|cmd+w]]");
        check(!b.cuaLog().contains("cua_hotkey:Chrome|cmd+w") && b.cuaLog().size() == 1
                  && b.cuaLog()[0].startsWith("cua_hotkey:Chrome|cmd+w -> BLOCKED closing tabs is blocked")
                  && unprompted.startsWith("Not pressing Cmd+W") && b.cuaFollowups().size() == 1
                  && b.cuaFollowups()[0].contains("(BLOCKED, ok=0, detail=closing tabs is blocked unless the user asks to close"),
              "cmd+w guard: unprompted cmd+w is blocked and the follow-up turn sees why", b.cuaLog().join(" | "));
        b.clearAnswers();
        b.ask(QString(), -1, QStringLiteral("Close the duplicate pricing tab"));
        b.runModelAnswer("[[ACTION:cua_hotkey:Chrome|cmd+w]] [[ACTION:cua_hotkey:cmd + W]] [[ACTION:cua_hotkey:cmd+t]]");
        const QStringList log = b.cuaLog().mid(1);
        check(log.size() == 3 && log[0] == "cua_hotkey:Chrome|cmd+w"
                  && log[1].startsWith("cua_hotkey:cmd+W -> BLOCKED only one Cmd+W runs per reply") && log[2] == "cua_hotkey:cmd+t",
              "cmd+w guard: a prompted reply runs its first cmd+w, blocks the second, other hotkeys still run", log.join(" | "));
        b.runModelAnswer("[[ACTION:cua_hotkey:cmd+w]]");
        check(b.cuaLog().last() == "cua_hotkey:cmd+w", "cmd+w guard: the next reply may close one more", b.cuaLog().join(" | "));
    }
    check(kCompatSystemPrompt.contains("[[ACTION:") && kCompatSystemPrompt.contains("1-3 crisp sentences"),
          "studio cmd: the system prompt teaches [[ACTION:..]] and short answers");

    // --- Prefs: a mock backend never writes the real studio_prefs.json --------------------------
    {
        const QString real = Backend::defaultPrefsPath();
        QFile rf(real);
        const bool existed = rf.exists();
        const QByteArray before = existed && rf.open(QIODevice::ReadOnly) ? rf.readAll() : QByteArray();
        rf.close();
        const QDateTime mtime = existed ? QFileInfo(real).lastModified() : QDateTime();
        {
            Backend m(Backend::Mode::Mock);
            m.setPrefsPath(real);  // even pointed straight at it
            m.setUiScale(1.37);
            m.setOpenaiCompatUrl(QString());
            const bool wrote = m.savePrefs();
            check(!wrote, "prefs: Mode::Mock savePrefs() refuses the real studio_prefs.json");
            m.setPrefsPath(QString());
        }
        QFile af(real);
        const QByteArray after = af.exists() && af.open(QIODevice::ReadOnly) ? af.readAll() : QByteArray();
        check(QFileInfo::exists(real) == existed && after == before
                  && (!existed || QFileInfo(real).lastModified() == mtime),
              "prefs: the real studio_prefs.json is byte- and mtime-identical after a mock run");
    }
    {
        QTemporaryDir tmp;
        const QString path = tmp.path() + "/studio_prefs.json";
        Backend m(Backend::Mode::Mock);
        m.setPrefsPath(path);
        m.completeOnboarding("controller");
        // Another writer adds the endpoint and a key this build does not know.
        {
            QFile f(path);
            QJsonObject o;
            if (f.open(QIODevice::ReadOnly))
                o = QJsonDocument::fromJson(f.readAll()).object();
            f.close();
            o.insert("openaiCompatUrl", "http://127.0.0.1:9/v1");
            o.insert("futureKey", 7);
            check(f.open(QIODevice::WriteOnly | QIODevice::Truncate) && f.write(QJsonDocument(o).toJson()) > 0,
                  "prefs: an outside writer adds keys to the temp prefs file");
        }
        m.setUiScale(1.2);
        m.savePrefs();
        auto readBack = [&] {
            QFile f(path);
            return f.open(QIODevice::ReadOnly) ? QJsonDocument::fromJson(f.readAll()).object() : QJsonObject();
        };
        QJsonObject o = readBack();
        check(o.value("openaiCompatUrl").toString() == "http://127.0.0.1:9/v1" && o.value("futureKey").toInt() == 7
                  && qFuzzyCompare(o.value("uiScale").toDouble(), 1.2),
              "prefs: saving merges over the file, openaiCompatUrl and unknown keys survive",
              QString::fromUtf8(QJsonDocument(o).toJson(QJsonDocument::Compact)));
        m.setOpenaiCompatUrl("http://127.0.0.1:8/v1");
        m.setOpenaiCompatUrl(QString());
        m.savePrefs();
        o = readBack();
        check(!o.contains("openaiCompatUrl") && o.value("futureKey").toInt() == 7,
              "prefs: only an explicit clear drops openaiCompatUrl");
    }

    // --- UI: commands, the chord and the pad inside the Music card --------------------------------
    using S = GamepadSnapshot;
    Backend backend(Backend::Mode::Mock);
    QTemporaryDir tmp;
    backend.setPrefsPath(tmp.path() + "/studio_prefs.json");
    backend.setHistoryPath(tmp.path() + "/studio_history.jsonl");
    backend.completeOnboarding("controller");
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    check(win != nullptr, "studio ui: QML loads");
    if (!win)
        return;
    QObject *app = win->findChild<QObject *>("studioApp");
    QObject *overlay = win->findChild<QObject *>("slowmixOverlay");
    QObject *omni = win->findChild<QObject *>("floatingOmnibar");
    SlowmixController *player = backend.slowmixController();
    Gamepad *pad = backend.gamepad();
    if (!app || !overlay || !omni)
        return;
    settle(450);
    QStringList actions;
    QObject::connect(&backend, &Backend::uiActionRequested,
                     [&actions](const QString &a, const QString &arg) { actions << (arg.isEmpty() ? a : a + ":" + arg); });

    check(backend.tryHandleStudioCommand("Can you open the Slowmix tab inside of this?") && actions.value(0) == "open_music",
          "studio ui: 'Can you open the Slowmix tab inside of this?' emits uiActionRequested(open_music)", actions.join(','));
    settle(250);
    check(overlay->property("open").toBool(), "studio ui: ... and opens the SlowmixOverlay");
    check(backend.studioLiveState().contains("Open overlays: Music (Slowmix)") && backend.studioLiveState().contains("Cards (5):"),
          "studio ui: live state names the open overlay and every card", backend.studioLiveState().left(200));
    check(!backend.tryHandleStudioCommand("what changed in the sweep?") && actions.size() == 1,
          "studio ui: a real question is not a command");
    backend.tryHandleStudioCommand("close the music");
    settle(250);
    check(!overlay->property("open").toBool(), "studio ui: 'close the music' closes it");

    // "play syntheticgod" finds the song even outside the 30-day rotation (archive), dry run.
    {
        QTemporaryDir clips;
        for (const char *f : {"syntheticgod_extended.wav", "night_drive.wav"}) {
            QFile w(clips.path() + "/" + f);
            if (w.open(QIODevice::WriteOnly))
                w.write("RIFF");
        }
        player->setLibrary(clips.path(), clips.path() + "/.plays.tsv");
        actions.clear();
        const bool ok = backend.tryHandleStudioCommand("play syntheticgod");
        check(ok && player->playing() && player->currentTrack().contains("syntheticgod")
                  && actions.value(0).startsWith("play_track:") && player->spawnCount() == 0 && backend.spawnCount() == 0,
              "studio ui: 'play syntheticgod' matches and plays it (mock: no process)",
              player->currentTrack() + " / " + actions.join(','));
        check(!backend.tryHandleStudioCommand("play devil's advocate"), "studio ui: 'play <not a song>' falls through to the model");
        check(backend.tryHandleStudioCommand("pause the music") && !player->playing(), "studio ui: 'pause the music' pauses");
        check(backend.tryHandleStudioCommand("resume music") && player->playing(), "studio ui: 'resume music' resumes");
        player->togglePlay();
        const QString fx = ensureSlowmixFixture();
        player->setLibrary(fx, fx + "/.plays.tsv");
    }
    backend.tryHandleStudioCommand("switch to lapis blue");
    settle(60);
    check(backend.theme() == "lapis", "studio ui: 'switch to lapis blue' switches the theme", backend.theme());
    backend.tryHandleStudioCommand("switch to gold theme");
    check(backend.theme() == "gold", "studio ui: 'switch to gold theme' switches back");
    backend.tryHandleStudioCommand("show needs me");
    settle(60);
    check(app->property("islandFilter").toString() == "needs", "studio ui: 'show needs me' filters the Constellation");
    backend.tryHandleStudioCommand("show all cards");
    settle(60);
    check(app->property("islandFilter").toString() == "all", "studio ui: 'show all cards' clears the filter");

    // The composer path: the confirmation is the quick answer, no model involved.
    const QString key0 = backend.projects().value(0).toMap().value("key").toString();
    backend.ask(key0, 0, "open settings");
    settle(250);
    QObject *settingsOv = win->findChild<QObject *>("settingsOverlay");
    check(backend.quickAnswer() == "Opened Settings." && !backend.quickBusy() && settingsOv
              && settingsOv->property("open").toBool(),
          "studio ui: ask('open settings') opens Settings and confirms in the answer strip", backend.quickAnswer());
    QMetaObject::invokeMethod(settingsOv, "hide");
    settle(250);

    // The ask card path: Enter on a UI command streams back its confirmation.
    sendKey(win, Qt::Key_K, Qt::ControlModifier);
    settle(60);
    typeText(win, "show code diffs");
    sendKey(win, Qt::Key_Return);
    settle(60);
    check(backend.showCodeDiffs() && omni->property("answer").toString() == "Code diffs are on."
              && !omni->property("busy").toBool(),
          "studio ui: the ask card runs a UI command and shows its 1-line confirmation",
          omni->property("answer").toString());
    sendKey(win, Qt::Key_Escape);
    backend.setShowCodeDiffs(false);
    settle(60);

    // Agent dispatch: lapis-peers new in the project's real folder (mock: no process).
    backend.setFocusedProject(key0);
    actions.clear();
    const bool spawned = backend.tryHandleStudioCommand("fire off an agent to fix the flaky tile test");
    check(spawned && actions.contains("spawn_agent:fix the flaky tile test")
              && std::any_of(actions.cbegin(), actions.cend(), [](const QString &a) { return a.startsWith("agent_spawned:"); })
              && backend.spawnCount() == 0,
          "studio ui: 'fire off an agent to ...' starts a session in the project folder, confirms its id",
          actions.join(','));
    backend.setFocusedProject(QString());
    QMetaObject::invokeMethod(app, "goLens", Q_ARG(QVariant, QVariant(0)));
    settle(450);

    // Shared history across a model switch, logged to studio_history.jsonl.
    {
        const QString m0 = backend.chatModel();
        backend.ask(key0, 0, "first question on sonnet");
        backend.advance(2.0);
        backend.setChatModel("claude-opus-5-5");
        backend.ask(key0, 0, "second question on opus");
        backend.advance(2.0);
        const QVector<ChatTurn> h = backend.sharedHistory();
        bool sawFirst = false, sawSecond = false;
        for (const ChatTurn &t : h) {
            sawFirst = sawFirst || t.content == "first question on sonnet";
            sawSecond = sawSecond || t.content == "second question on opus";
        }
        const TieredPrompt p = backend.tieredPrompt(key0, QString(), "What were my last 5 questions?", h);
        const QByteArray body = buildTieredChatRequest(p);
        check(sawFirst && sawSecond && body.contains("first question on sonnet") && body.contains("Studio live state:"),
              "studio ui: shared history spans a model switch and rides as Tier 2, with live state in Tier 3",
              QString::number(h.size()));
        QFile log(tmp.path() + "/studio_history.jsonl");
        QStringList models;
        if (log.open(QIODevice::ReadOnly))
            for (const QByteArray &line : log.readAll().split('\n'))
                if (!line.isEmpty())
                    models << QJsonDocument::fromJson(line).object().value("model").toString();
        check(models.contains(m0) && models.contains("claude-opus-5-5") && models.contains("studio"),
              "studio ui: every prompt + reply lands in studio_history.jsonl with its model", models.join(','));
        backend.setChatModel(m0);
    }

    // --- DualShock: L1 + R1 opens Music anywhere, then the pad drives the card ------------------
    pad->setConnected(true, QStringLiteral("DualShock 4"));
    settle(60);
    qint64 t = 100000;
    auto snap = [&](auto set) {
        S s;
        set(s);
        pad->processSnapshot(s, t);
        t += 20;
        settle(20);
    };
    auto tap = [&](auto set) {
        snap(set);
        snap([](S &) {});
        settle(30);
    };
    int steps = 0, accepts = 0, toggles = 0;
    QObject::connect(pad, &Gamepad::step, [&steps](int) { ++steps; });
    QObject::connect(pad, &Gamepad::acceptPrediction, [&accepts] { ++accepts; });
    QObject::connect(pad, &Gamepad::toggleMusic, [&toggles] { ++toggles; });
    snap([](S &s) { s.l1 = true; s.r1 = true; });
    snap([](S &) {});
    settle(250);
    check(toggles == 1 && steps == 0 && accepts == 0 && overlay->property("open").toBool(),
          "pad music: L1 + R1 emits toggleMusic (no L1 / R1 actions) and opens the SlowmixOverlay",
          QStringLiteral("%1 toggles, %2 steps, %3 accepts").arg(toggles).arg(steps).arg(accepts));
    check(pad->musicMode(), "pad music: the pad is in music mode while the card is open");
    QQuickItem *hints = firstItem(win, "slowmixHints");
    check(hints && hints->property("text").toString() == "✕ play · □ pause · ◄► speed · △ reverb · ○ close",
          "pad music: the footer speaks controller glyphs", hints ? hints->property("text").toString() : QString());
    const int row0 = overlay->property("row").toInt();
    tap([](S &s) { s.dpadY = -1; });
    check(overlay->property("row").toInt() == row0 + 1, "pad music: d-pad down moves the selected track");
    tap([](S &s) { s.a = true; });
    const QVariantList tracks = player->tracks();
    const QString want = tracks.value(row0 + 1).toMap().value("name").toString();
    check(player->playing() && player->currentTrack() == want, "pad music: ✕ plays the selected track",
          player->currentTrack() + " vs " + want);
    tap([](S &s) { s.x = true; });
    check(!player->playing(), "pad music: □ pauses");
    tap([](S &s) { s.x = true; });
    check(player->playing(), "pad music: □ resumes");
    {
        const QString r0 = player->reverb();
        tap([](S &s) { s.y = true; });
        const QString r1 = player->reverb();
        tap([](S &s) { s.y = true; });
        tap([](S &s) { s.y = true; });
        check(r1 != r0 && player->reverb() == r0, "pad music: △ cycles reverb Off → Hall → Heavy", r0 + " -> " + r1);
    }
    {
        const double s0 = player->speed();
        tap([](S &s) { s.dpadX = 1; });
        const double s1 = player->speed();
        tap([](S &s) { s.l1 = true; });
        const double s2 = player->speed();
        tap([](S &s) { s.r1 = true; });
        check(s1 > s0 && s2 < s1 && player->speed() > s2, "pad music: ◄► / L1 / R1 step the speed",
              QStringLiteral("%1 %2 %3 %4").arg(s0).arg(s1).arg(s2).arg(player->speed()));
    }
    {
        const bool loop0 = player->looping();
        snap([](S &s) { s.l2 = 1.0; });
        snap([](S &) {});
        check(player->looping() != loop0 && !pad->voiceActive(), "pad music: L2 toggles loop (never the voice key)");
        const double b0 = player->bass();
        tap([](S &s) { s.r2 = 1.0; });
        check(player->bass() != b0, "pad music: R2 cycles bass", QString::number(player->bass()));
    }
    {
        // R3 flips the card to Spotify (no zoom); there □ plays / pauses, R1 next, L1 prev, ↕ volume.
        SpotifyController *sp = backend.spotifyController();
        const double scale0 = backend.uiScale();
        int r3s = 0;
        QObject::connect(pad, &Gamepad::r3Pressed, [&r3s] { ++r3s; });
        tap([](S &s) { s.r3 = true; });
        check(r3s == 1 && overlay->property("tab").toString() == "spotify" && std::abs(backend.uiScale() - scale0) < 1e-9,
              "pad music: R3 switches to the Spotify tab and does not zoom",
              overlay->property("tab").toString() + " " + QString::number(backend.uiScale()));
        QQuickItem *badge = nullptr;
        for (QQuickItem *it : findItems(qobject_cast<QQuickItem *>(overlay), "padBadge"))
            if (it->isVisible() && it->property("glyph").toString() == "R3" && it->property("kind").toString() == "r3")
                badge = it;
        check(badge != nullptr, "pad music: the header shows an R3 glyph badge in controller mode");
        const int n0 = int(sp->commandLog().size());
        tap([](S &s) { s.x = true; });
        check(sp->playing() && !player->playing(), "pad spotify: □ plays Spotify and pauses slowmix");
        tap([](S &s) { s.r1 = true; });
        tap([](S &s) { s.l1 = true; });
        tap([](S &s) { s.dpadY = 1; });  // up: volume one chip higher (unknown volume starts at 60)
        const QStringList log = sp->commandLog().mid(n0);
        check(log == QStringList{"api PUT /v1/me/player/play", "api POST /v1/me/player/next", "api POST /v1/me/player/previous",
                                 "api PUT /v1/me/player/volume?volume_percent=80"},
              "pad spotify: □ / R1 / L1 / ↑ map to play-pause / next / previous / volume", log.join(" | "));
        tap([](S &s) { s.r3 = true; });
        check(overlay->property("tab").toString() == "slowmix", "pad music: R3 again returns to Slowmix");
        tap([](S &s) { s.x = true; });
        check(player->playing() && !sp->playing(), "pad music: □ on Slowmix resumes it and pauses Spotify");
    }
    tap([](S &s) { s.b = true; });
    settle(250);
    check(!overlay->property("open").toBool() && player->playing() && !pad->musicMode(),
          "pad music: ○ closes the card, playback continues");
    const int acceptsBefore = accepts;
    snap([](S &s) { s.l1 = true; });
    snap([](S &s) { s.l1 = true; s.r1 = true; });
    snap([](S &) {});
    settle(250);
    check(overlay->property("open").toBool() && toggles == 2 && accepts == acceptsBefore,
          "pad music: L1 held, then R1: the chord still opens Music, R1's own action is suppressed");
    tap([](S &s) { s.b = true; });
    settle(250);
    player->togglePlay();
    check(backend.spawnCount() == 0 && player->spawnCount() == 0 && Gamepad::platformCalls() == 0,
          "studio ui: zero processes, zero GameController calls");
}

// --- Final wiring: Spotify, slowmix volume, no-think quick asks, Tier 0 context, ambient voice ---

void finalWiringTests()
{
    // spotify_player JSON.
    {
        const SpotifyPlayback pb = parseSpotifyPlayback(
            R"({"is_playing":true,"item":{"name":"Night Drive","artists":[{"name":"Lapis"},{"name":"B"}]},"device":{"volume_percent":55}})");
        check(pb.ok && pb.playing && pb.track == "Night Drive" && pb.artist == "Lapis" && pb.volume == 55,
              "spotify: playback JSON -> playing, track, first artist, volume", pb.track + " " + pb.artist);
        check(!parseSpotifyPlayback("null").ok && !parseSpotifyPlayback("").ok && !parseSpotifyPlayback("garbage").ok,
              "spotify: null / empty / garbage playback is not ok");
        check(parseSpotifyFirstTrackId(R"({"tracks":[{"id":"spotify:track:abc123","name":"x"},{"id":"zzz"}]})") == "abc123"
                  && parseSpotifyFirstTrackId(R"({"tracks":[{"id":"def456"}]})") == "def456"
                  && parseSpotifyFirstTrackId(R"({"tracks":{"items":[{"uri":"spotify:track:ghi789"}]}})") == "ghi789"
                  && parseSpotifyFirstTrackId(R"({"tracks":[]})").isEmpty(),
              "spotify: search JSON -> first track id (prefix stripped; id, items and uri shapes)");
    }

    // Dry run: optimistic state, recorded commands, zero processes.
    {
        SpotifyController sp;
        int started = 0;
        QObject::connect(&sp, &SpotifyController::playbackStarted, [&started] { ++started; });
        sp.togglePlay();
        sp.setVolume(145);
        sp.nextTrack();
        check(sp.playing() && started == 1 && sp.volume() == 100 && sp.spawnCount() == 0
                  && sp.commandLog()
                         == QStringList{"api PUT /v1/me/player/play", "api PUT /v1/me/player/volume?volume_percent=100",
                                        "api POST /v1/me/player/next"},
              "spotify: dry run updates state, clamps volume, logs commands, spawns nothing", sp.commandLog().join(" | "));
    }

    // Live path against a stub spotify_player: real QProcess round trips, search -> start --id.
    {
        QTemporaryDir tmp;
        const QString log = tmp.path() + "/args.log";
        const QString stub = tmp.path() + "/spotify_player";
        QFile f(stub);
        if (f.open(QIODevice::WriteOnly)) {
            f.write(QStringLiteral("#!/bin/sh\necho \"$*\" >> '%1'\ncase \"$1 $2 $3\" in\n"
                                   "  \"get key playback\") echo '{\"is_playing\":true,\"item\":{\"name\":\"Night Drive\","
                                   "\"artists\":[{\"name\":\"Lapis\"}]},\"device\":{\"volume_percent\":55}}' ;;\n"
                                   "  \"search \"*) echo '{\"tracks\":[{\"id\":\"spotify:track:abc123\"}]}' ;;\n"
                                   "esac\nexit 0\n")
                        .arg(log)
                        .toUtf8());
            f.close();
            f.setPermissions(QFileDevice::ReadOwner | QFileDevice::WriteOwner | QFileDevice::ExeOwner);
        }
        SpotifyController sp;
        sp.setBinary(stub);
        // CLI route only: no token file and no Spotify.app, so a live test never reaches the
        // real Web API or the user's Spotify.app (both exist on a dev Mac).
        sp.setTokenPath(tmp.path() + "/no-token.json");
        sp.setAppPath(tmp.path() + "/NoSpotify.app");
        sp.setSpawnAllowed(true);  // one refresh at startup
        const bool refreshed = waitUntil([&] { return sp.track() == "Night Drive"; }, 5000);
        check(refreshed && sp.playing() && sp.artist() == "Lapis" && sp.volume() == 55 && !sp.busy(),
              "spotify live: get key playback fills track / artist / volume", sp.track() + " " + sp.error());
        sp.searchAndPlay(QStringLiteral("night drive"));
        const bool started = waitUntil([&] {
            QFile l(log);
            return l.open(QIODevice::ReadOnly) && l.readAll().contains("playback start track --name night drive");
        }, 5000);
        QFile l(log);
        const QString args = l.open(QIODevice::ReadOnly) ? QString::fromUtf8(l.readAll()) : QString();
        // No token, no Spotify.app: no Web API search, spotify_player resolves the name itself.
        check(started, "spotify live: no token -> playback start track --name <query> via the CLI", args);
        waitUntil([&] { return !sp.busy(); }, 3000);
        sp.setBinary(tmp.path() + "/missing");
        sp.nextTrack();
        // Two real spawns so far (get key playback, playback start); the missing binary adds none.
        check(sp.error().contains("not found") && sp.spawnCount() == 2,
              "spotify live: a missing binary reports instead of spawning", sp.error());
        check(SpotifyController::findBinary().isEmpty() || QFileInfo(SpotifyController::findBinary()).isExecutable(),
              "spotify: findBinary returns an executable or nothing", SpotifyController::findBinary());
    }

    Backend b(Backend::Mode::Mock);
    SpotifyController *sp = b.spotifyController();
    SlowmixController *mix = b.slowmixController();

    // One source of sound at a time.
    {
        b.dispatchStudioAction("resume_music");
        const bool mixOn = mix->playing();
        b.dispatchStudioAction("spotify_play");
        const bool handedOver = sp->playing() && !mix->playing();
        b.dispatchStudioAction("resume_music");
        check(mixOn && handedOver && mix->playing() && !sp->playing() && sp->commandLog().last() == "api PUT /v1/me/player/pause",
              "exclusive: Spotify starting pauses slowmix, slowmix starting pauses Spotify", sp->commandLog().join(" | "));
    }

    // Natural-language commands and their dispatch.
    {
        auto cmd = [](const char *t) {
            const StudioCommand c = parseStudioCommand(QString::fromUtf8(t));
            return c.arg.isEmpty() ? c.action : c.action + ":" + c.arg;
        };
        const QStringList got = {cmd("pause spotify"), cmd("hey can you skip this song on spotify"),
                                 cmd("previous track on spotify"), cmd("play night drive on spotify"),
                                 cmd("resume spotify please"), cmd("set the volume to 60 percent"),
                                 cmd("spotify volume 40"), cmd("volume 30 on spotify"), cmd("open spotify"),
                                 cmd("pause the music"), cmd("open music")};
        const QStringList want = {"spotify_pause", "spotify_next", "spotify_prev", "spotify_play:night drive",
                                  "spotify_play", "set_volume:60", "spotify_volume:40", "spotify_volume:30",
                                  "open_spotify", "pause_music", "open_music"};
        check(got == want, "nl: spotify / volume phrases map to their actions; slowmix phrases unchanged", got.join(", "));
        QString last;
        QObject::connect(&b, &Backend::uiActionRequested, [&last](const QString &a, const QString &arg) {
            last = arg.isEmpty() ? a : a + ":" + arg;
        });
        const QString r1 = b.dispatchStudioAction("set_volume", "60%");
        const QString r2 = b.dispatchStudioAction("spotify_volume", "0.35");
        check(std::abs(mix->volume() - 0.6) < 1e-9 && sp->volume() == 35 && r1 == "Slowmix volume 60%."
                  && r2 == "Spotify volume 35%." && last == "spotify_volume:35",
              "nl: set_volume:60% and spotify_volume:0.35 land as 0.60 / 35", r1 + " " + r2);
        const QVariantMap p0 = b.projects().value(0).toMap();
        const QString name = p0.value("name").toString(), key = p0.value("key").toString();
        const QString r3 = b.dispatchStudioAction("open_stage", name.toUpper());
        check(r3.contains(name) && last == "open_stage:" + key, "nl: open_stage:<name> finds the project and emits its key",
              r3 + " / " + last);
        check(b.dispatchStudioAction("focus_project", "zzz-no-such-project").isEmpty(),
              "nl: focus_project with no match does nothing");
        b.dispatchStudioAction("go_home");
        check(last == "go_home", "nl: go_home reaches the QML side");
    }

    // No-think by default for Qwen; thinking stays opt-in.
    {
        b.setChatModel("qwen3.8-flash-next");
        check(b.chatEffort() == "fast" && b.compatEffort() == "fast" && b.chatEfforts() == QStringList{"fast", "low", "medium"},
              "effort: Qwen3.8 Flash offers fast / low / medium and defaults to fast", b.chatEfforts().join(","));
        TieredPrompt p;
        p.model = "qwen3.8-flash-next";
        p.systemPrompt = kCompatSystemPrompt;
        p.question = "hi";
        p.effort = "fast";
        const QJsonObject fast = QJsonDocument::fromJson(buildTieredChatRequest(p)).object();
        p.effort = "none";
        const QJsonObject none = QJsonDocument::fromJson(buildTieredChatRequest(p)).object();
        p.effort = "low";
        const QJsonObject low = QJsonDocument::fromJson(buildTieredChatRequest(p)).object();
        check(fast.value("chat_template_kwargs").toObject().value("enable_thinking") == QJsonValue(false)
                  && !fast.contains("reasoning_effort") && none.value("chat_template_kwargs").toObject().contains("enable_thinking")
                  && low.value("reasoning_effort").toString() == "low" && !low.contains("chat_template_kwargs"),
              "effort: fast / none send enable_thinking false; low sends reasoning_effort");
        b.setChatEffort("medium");
        check(b.compatEffort() == "medium", "effort: picking medium on Qwen opts into thinking");
        b.setChatModel("claude-sonnet-5-5");
        check(b.compatEffort() == "fast" && b.chatEffort() == "medium",
              "effort: card streams under a CLI model stay no-think; the CLI model keeps its effort", b.chatEffort());
        // Old prefs (no effortSchema) saved "low" as Qwen's floor: they start no-think once.
        QTemporaryDir tmp;
        auto loadWith = [&](const QByteArray &json) {
            QFile pf(tmp.path() + "/prefs.json");
            if (!pf.open(QIODevice::WriteOnly | QIODevice::Truncate))
                return QStringLiteral("unwritable");
            pf.write(json);
            pf.close();
            Backend c(Backend::Mode::Mock);
            c.setPrefsPath(pf.fileName());
            return c.chatEffort();
        };
        const QString old = loadWith(R"({"defaultModel":"qwen3.8-flash-next","defaultEffort":"low"})");
        const QString kept = loadWith(R"({"defaultModel":"qwen3.8-flash-next","defaultEffort":"low","effortSchema":2})");
        check(old == "fast" && kept == "low", "effort: pre-schema prefs migrate low -> fast; schema 2 keeps low", old + " " + kept);
    }

    // Tier 0 local studio context rides in every compat request, byte-identical to the warm.
    {
        b.setStudioContext("CTX-MARKER gpu-host 4x RTX PRO 6000");
        const QString key = b.projects().value(0).toMap().value("key").toString();
        const TieredPrompt p = b.tieredPrompt(key, QString(), "what is gpu-host?");
        const QJsonObject body = QJsonDocument::fromJson(buildTieredChatRequest(p)).object();
        const QString sys = body.value("messages").toArray().first().toObject().value("content").toString();
        const QJsonObject warm = QJsonDocument::fromJson(buildTieredWarmRequest(p)).object();
        check(sys.contains("Local studio context:\nCTX-MARKER") && sys.contains("Your model ID is")
                  && warm.value("messages").toArray().first() == body.value("messages").toArray().first(),
              "tier0: studio context sits in the system message, same bytes in ask and warm", sys.right(80));
        b.setStudioContext(QString());
    }

    // Ambient voice over a local SSE server: compound tags in one turn, [[IGNORE]] silent, fast path local.
    {
        QTcpServer server;
        QList<QByteArray> bodies;
        QList<QByteArray> replies = {"[[ACTION:set_volume:40]] [[ACTION:spotify_next]] Done.", "[[IGNORE]]",
                                     "[[ACTION:set_volume:20]]", "[[ACTION:set_volume:10]]"};
        QObject::connect(&server, &QTcpServer::newConnection, [&] {
            while (QTcpSocket *s = server.nextPendingConnection()) {
                auto buf = std::make_shared<QByteArray>();
                QObject::connect(s, &QTcpSocket::readyRead, s, [&, s, buf] {
                    *buf += s->readAll();
                    const qsizetype he = buf->indexOf("\r\n\r\n");
                    if (he < 0)
                        return;
                    const QByteArray head = buf->left(he).toLower();
                    const qsizetype cl = head.indexOf("content-length:");
                    const qsizetype eol = head.indexOf("\r\n", cl);
                    const qsizetype len = cl < 0 ? 0 : head.mid(cl + 15, eol < 0 ? -1 : eol - cl - 15).trimmed().toLongLong();
                    if (buf->size() < he + 4 + len)
                        return;
                    bodies << buf->mid(he + 4, len);
                    const QByteArray answer = replies.isEmpty() ? QByteArray("ok") : replies.takeFirst();
                    const QByteArray body = "data: {\"choices\":[{\"delta\":{\"content\":\"" + answer + "\"}}]}\n\n"
                                            "data: [DONE]\n\n";
                    s->write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\nContent-Length: "
                             + QByteArray::number(body.size()) + "\r\n\r\n" + body);
                    s->disconnectFromHost();
                });
                QObject::connect(s, &QTcpSocket::disconnected, s, &QObject::deleteLater);
            }
        });
        check(server.listen(QHostAddress::LocalHost, 0), "ambient: local SSE server listens");
        LlmRouter router;
        b.setLlmRouter(&router);
        b.setOpenaiCompatUrl(QStringLiteral("http://127.0.0.1:%1/v1").arg(server.serverPort()));
        b.setStudioContext("CTX-AMBIENT");
        mix->setVolume(0.8);
        const int n0 = int(sp->commandLog().size());

        b.feedAmbientTail("0:01 You: turn it down and skip this one");  // off: ignored
        b.setAmbientVoiceEnabled(true);
        b.feedAmbientTail("0:01 You: turn it down and skip this one\n0:03 Call: sure\n0:04 You: turn it down and skip this one");
        QElapsedTimer t;
        t.start();
        const bool sent = waitUntil([&] { return !bodies.isEmpty(); }, 3000);
        const qint64 debounceMs = t.elapsed();
        const bool done = waitUntil([&] { return !b.ambientBusy() && !b.ambientAnswer().isEmpty(); }, 5000);
        const QJsonObject req = bodies.isEmpty() ? QJsonObject() : QJsonDocument::fromJson(bodies.first()).object();
        const QString sys = req.value("messages").toArray().first().toObject().value("content").toString();
        const QString leaf = req.value("messages").toArray().last().toObject().value("content").toString();
        check(sent && debounceMs >= 700 && req.value("chat_template_kwargs").toObject().value("enable_thinking") == QJsonValue(false)
                  && sys.contains("CTX-AMBIENT") && leaf.endsWith("Spoken: turn it down and skip this one"),
              "ambient: only the new You: line goes out, after the 800 ms debounce, no-think, Tier 0 context",
              QString::number(debounceMs) + "ms " + leaf.right(60));
        check(done && b.hasPendingProposal() && b.pendingProposalText() == "Done." && std::abs(mix->volume() - 0.8) < 1e-9
                  && !sp->commandLog().mid(n0).contains("api POST /v1/me/player/next")
                  && b.ambientAnswer() == "Done.  ·  L3+R3 Run",
              "ambient: action tags stage a pending proposal and run nothing yet", b.ambientAnswer());
        // L3 + R3 on the pad confirms it: both actions run in one go, the proposal clears.
        {
            Gamepad *pad = b.gamepad();
            int l3Zoom = 0;
            QObject::connect(pad, &Gamepad::zoomReset, [&l3Zoom] { ++l3Zoom; });
            GamepadSnapshot chord;
            chord.l3 = chord.r3 = true;
            pad->processSnapshot(chord, 100000);
            pad->processSnapshot(GamepadSnapshot(), 100200);
            check(!b.hasPendingProposal() && b.pendingProposalText().isEmpty() && l3Zoom == 0
                      && std::abs(mix->volume() - 0.4) < 1e-9
                      && sp->commandLog().mid(n0).contains("api POST /v1/me/player/next") && !b.ambientAnswer().isEmpty(),
                  "ambient: L3+R3 runs the staged compound proposal and clears it", b.ambientAnswer());
            check(!b.confirmPendingProposal(), "ambient: confirming with nothing staged is a no-op");
        }

        const QString answerBefore = b.ambientAnswer();
        const int n1 = int(sp->commandLog().size());
        b.feedAmbientTail("0:04 You: turn it down and skip this one\n0:09 You: lol yeah totally");
        b.flushAmbient();
        waitUntil([&] { return bodies.size() == 2 && !b.ambientBusy(); }, 5000);
        check(bodies.size() == 2 && sp->commandLog().size() == n1 && std::abs(mix->volume() - 0.4) < 1e-9
                  && b.ambientHeard().isEmpty() && b.ambientAnswer() == answerBefore,
              "ambient: [[IGNORE]] runs nothing and shows nothing new");

        b.feedAmbientTail("0:09 You: lol yeah totally\n0:12 You: pause spotify");
        b.flushAmbient();
        settle(50);
        check(bodies.size() == 2 && sp->commandLog().last() == "api PUT /v1/me/player/pause" && b.ambientAnswer() == "Paused Spotify.",
              "ambient: a plain UI command runs locally, no model call", b.ambientAnswer());

        // An unconfirmed proposal expires on its own (15 s live; 150 ms here) and runs nothing.
        b.setProposalTimeoutMs(150);
        b.feedAmbientTail("0:12 You: pause spotify\n0:13 You: make it quieter");
        b.flushAmbient();
        const bool staged = waitUntil([&] { return bodies.size() == 3 && !b.ambientBusy() && b.hasPendingProposal(); }, 5000);
        const QString stagedText = b.pendingProposalText();
        const bool expired = waitUntil([&] { return !b.hasPendingProposal(); }, 3000);
        check(staged && stagedText == "set_volume 20" && expired && b.pendingProposalText().isEmpty()
                  && std::abs(mix->volume() - 0.4) < 1e-9 && !b.confirmPendingProposal(),
              "ambient: an unconfirmed proposal expires and never runs", stagedText);
        // dismissPendingProposal drops it at once; a later L3+R3 does nothing.
        b.setProposalTimeoutMs(15000);
        b.feedAmbientTail("0:13 You: make it quieter\n0:14 You: quieter still");
        b.flushAmbient();
        const bool staged2 = waitUntil([&] { return bodies.size() == 4 && !b.ambientBusy() && b.hasPendingProposal(); }, 5000);
        b.dismissPendingProposal();
        GamepadSnapshot chord;
        chord.l3 = chord.r3 = true;
        b.gamepad()->processSnapshot(chord, 200000);
        b.gamepad()->processSnapshot(GamepadSnapshot(), 200200);
        check(staged2 && !b.hasPendingProposal() && std::abs(mix->volume() - 0.4) < 1e-9,
              "ambient: dismissPendingProposal clears it and L3+R3 then runs nothing");

        b.setAmbientVoiceEnabled(false);
        b.feedAmbientTail("0:14 You: quieter still\n0:15 You: open music");
        check(!b.ambientBusy() && bodies.size() == 4, "ambient: off means nothing is sent");
        b.setLlmRouter(nullptr);
    }
}

namespace {

// --- Constellation map: clusters, free cursor, flick physics, marquee ----------------------

void mapUnitTests()
{
    // Cluster classifier: Lapis category first, then ~tab: / scratch, else a repo project.
    check(mapClusterFor("cluster-serving", "infinity", "~/dev/cluster-serving") == "Work"
              && mapClusterFor("lapis", "apps", "~/dev/lapis") == "Apps"
              && mapClusterFor("~tab:14e5ecb8", "", "~") == "Quick Qs"
              && mapClusterFor("notes", "quick Qs", "~/dev/notes") == "Quick Qs"
              && mapClusterFor("scratch", "", "/tmp") == "Quick Qs"
              && mapClusterFor("minecraft", "minecraft", "~/dev/minecraft") == "Projects"
              && mapClusterFor("yoctogpt", "", "~/dev/yoctogpt") == "Projects",
          "map: categories classify into Work / Apps / Quick Qs / Projects");
    check(normalizeClusterName("quick qs") == "Quick Qs" && normalizeClusterName("app") == "Apps"
              && normalizeClusterName("WORK") == "Work" && normalizeClusterName("banana").isEmpty(),
          "map: cluster names normalize");

    // move_cluster on the two-column cluster grid (Work Projects / Apps Quick Qs).
    const QStringList d = defaultClusterOrder();
    check(moveClusterInOrder(d, d, "Apps", "top") == QStringList{"Apps", "Projects", "Work", "Quick Qs"},
          "map: Apps to the top swaps it with Work (same column)",
          moveClusterInOrder(d, d, "Apps", "top").join(","));
    check(moveClusterInOrder(d, d, "Work", "right") == QStringList{"Projects", "Work", "Apps", "Quick Qs"},
          "map: Work to the right swaps within its row");
    check(moveClusterInOrder(d, d, "Work", "left") == d && moveClusterInOrder(d, d, "Quick Qs", "bottom") == d
              && moveClusterInOrder(d, d, "nope", "left") == d,
          "map: a blocked or unknown move leaves the order");
    check(moveClusterInOrder(d, {"Work", "Apps"}, "apps", "left") == QStringList{"Apps", "Projects", "Work", "Quick Qs"},
          "map: moves count only the clusters on screen (empty ones keep their place)",
          moveClusterInOrder(d, {"Work", "Apps"}, "apps", "left").join(","));

    // Natural language / voice.
    auto pc = [](const char *t) {
        const StudioCommand c = parseStudioCommand(QString::fromUtf8(t));
        return c.action + (c.arg.isEmpty() ? QString() : ":" + c.arg);
    };
    check(pc("sort the map by recent") == "sort_map:recent" && pc("can you sort the cards by activity please") == "sort_map:activity"
              && pc("group the islands by category") == "sort_map:category",
          "map: 'sort the map by ...' parses to sort_map",
          pc("sort the map by recent") + " | " + pc("can you sort the cards by activity please"));
    check(pc("move the work cluster to the left") == "move_cluster:Work:left" && pc("put apps on top") == "move_cluster:Apps:top"
              && pc("move quick qs down") == "move_cluster:Quick Qs:bottom",
          "map: 'move the work cluster to the left' parses to move_cluster",
          pc("move the work cluster to the left") + " | " + pc("put apps on top") + " | " + pc("move quick qs down"));
    {
        QString answer = QStringLiteral("[[ACTION:move_cluster:Apps:top]] [[ACTION:sort_map:recent]] Done.");
        const QList<StudioCommand> tags = extractActionTags(&answer);
        check(tags.size() == 2 && tags[0].action == "move_cluster" && tags[0].arg == "Apps:top" && tags[1].arg == "recent"
                  && answer == "Done.",
              "map: model [[ACTION:move_cluster:Apps:top]] tags are known actions", answer);
    }

    // Backend: the actions re-sort, persist, and reset dragged cards.
    {
        QTemporaryDir tmp;
        Backend b(Backend::Mode::Mock);
        b.setPrefsPath(tmp.path() + "/studio_prefs.json");
        b.mockSetCategory("cluster-serving", "infinity");
        b.mockSetCategory("lapis", "apps");
        b.mockAddProject("~tab:aa11bb22", "Why is the fn key slow", "~", "quick Qs", "idle");
        QStringList clusters;
        for (const QVariant &v : b.projects())
            clusters << v.toMap().value("cluster").toString();
        check(clusters.contains("Work") && clusters.contains("Apps") && clusters.contains("Quick Qs") && clusters.contains("Projects"),
              "map: projectMap carries the cluster", clusters.join(","));
        check(b.presentClusters() == defaultClusterOrder(), "map: all four clusters present", b.presentClusters().join(","));
        b.setCardPosition("lapis", 0.5, 0.5);
        const QString r1 = b.dispatchStudioAction("move_cluster", "Apps:top");
        check(b.clusterOrder() == QStringList{"Apps", "Projects", "Work", "Quick Qs"} && b.cardPositions().isEmpty()
                  && r1.contains("Moved Apps"),
              "map: move_cluster reorders clusters and re-flows dragged cards", r1);
        const QString r2 = b.dispatchStudioAction("sort_map", "recent");
        check(b.mapSort() == "recent" && r2.contains("recent"), "map: sort_map:recent", r2);
        check(b.dispatchStudioAction("move_cluster", "Apps:top").contains("already"), "map: a blocked move says so");
        check(b.savePrefs(), "map: prefs flush");
        Backend r(Backend::Mode::Mock);
        r.setPrefsPath(b.prefsPath());
        check(r.mapSort() == "recent" && r.clusterOrder() == b.clusterOrder(), "map: sort and cluster order reload from prefs",
              r.mapSort() + " " + r.clusterOrder().join(","));
    }

    // Gamepad: analog left stick (0.14 radial deadzone, quadratic) and the ✕ hold edge.
    {
        double x = 9, y = 9;
        GamepadState::shapeStick(0.13, 0.0, &x, &y);
        const bool dead = x == 0 && y == 0;
        GamepadState::shapeStick(1.0, 0.0, &x, &y);
        const bool full = std::abs(x - 1.0) < 1e-9 && y == 0;
        GamepadState::shapeStick(0.57, 0.0, &x, &y);
        const bool quad = std::abs(x - 0.25) < 1e-9;
        GamepadState::shapeStick(0.0, 1.0, &x, &y);
        const bool up = std::abs(y + 1.0) < 1e-9;
        check(dead && full && quad && up, "pad: analog stick deadzone 0.14, quadratic curve, stick up is screen up");

        Gamepad g;
        QList<QPair<double, double>> moves;
        int navs = 0;
        QList<bool> held;
        QObject::connect(&g, &Gamepad::cursorMove, &g, [&](double vx, double vy) { moves.append({vx, vy}); });
        QObject::connect(&g, &Gamepad::nav, &g, [&](int, int) { ++navs; });
        QObject::connect(&g, &Gamepad::confirmHeld, &g, [&](bool down) { held.append(down); });
        g.setAnalogCursor(true);
        GamepadSnapshot s;
        s.lx = 0.8;
        s.ly = 0.3;
        for (int t = 0; t <= 80; t += 8)
            g.processSnapshot(s, t);
        const bool streaming = moves.size() == 11 && moves.last().first > 0 && moves.last().second < 0 && navs == 0
                               && g.leftX() > 0 && g.leftY() < 0;
        g.processSnapshot(GamepadSnapshot(), 90);
        g.processSnapshot(GamepadSnapshot(), 98);
        check(streaming && moves.size() == 12 && moves.last().first == 0 && moves.last().second == 0 && g.leftX() == 0,
              "pad: analog cursor mode streams stick velocity per poll (no nav steps), one 0,0 on release",
              QStringLiteral("%1 moves, %2 navs").arg(moves.size()).arg(navs));
        GamepadSnapshot dp;
        dp.dpadX = 1;
        g.processSnapshot(dp, 200);
        g.processSnapshot(GamepadSnapshot(), 210);
        check(navs == 1, "pad: the d-pad still hops in analog cursor mode", QString::number(navs));
        GamepadSnapshot a;
        a.a = true;
        g.processSnapshot(a, 300);
        g.processSnapshot(GamepadSnapshot(), 400);
        check(held == QList<bool>{true, false}, "pad: ✕ reports its hold edges (down, up)");
        g.setAnalogCursor(false);
        s.ly = 0;
        g.processSnapshot(s, 500);
        check(navs == 2, "pad: out of cursor mode the stick steps again", QString::number(navs));
    }
}

QList<QQuickItem *> mapIslands(QQuickWindow *win)
{
    QList<QQuickItem *> out;
    for (QQuickItem *it : findItems(win->contentItem(), "island"))
        if (it->isVisible() && it->property("draggable").toBool())
            out << it;
    return out;
}
QRectF itemRect(QQuickItem *it) { return QRectF(it->x(), it->y(), it->width(), it->height()); }
int overlapPairs(const QList<QQuickItem *> &isl, QStringList *pairs = nullptr)
{
    int n = 0;
    for (int i = 0; i < isl.size(); ++i)
        for (int j = i + 1; j < isl.size(); ++j)
            if (itemRect(isl[i]).intersects(itemRect(isl[j]))) {
                ++n;
                if (pairs)
                    *pairs << isl[i]->property("key").toString() + " x " + isl[j]->property("key").toString();
            }
    return n;
}
QQuickItem *islandByKey(QQuickWindow *win, const QString &key)
{
    for (QQuickItem *it : mapIslands(win))
        if (it->property("key").toString() == key)
            return it;
    return nullptr;
}
// The cluster scene: the five fixtures across Work / Projects / Apps plus two Quick Qs tabs.
void seedClusterScene(Backend &b)
{
    b.mockSetCategory("cluster-serving", "infinity");
    b.mockSetCategory("lapis", "apps");
    b.mockSetCategory("phonon", "apps");
    b.mockAddProject("~tab:aa11bb22", "Why is the fn key slow", "~", "quick Qs", "idle");
    b.mockAddProject("~tab:cc33dd44", "Best eSIM for Japan", "~", "quick Qs", "finished");
}

void mapUiTests()
{
    Backend backend(Backend::Mode::Mock);
    QTemporaryDir tmp;
    backend.setPrefsPath(tmp.path() + "/studio_prefs.json");
    backend.completeOnboarding("controller");
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    check(win != nullptr, "map ui: QML loads");
    if (!win)
        return;
    QObject *app = win->findChild<QObject *>("studioApp");
    QObject *map = win->findChild<QObject *>("constellation");
    TargetRegistry *reg = backend.targets();
    Gamepad *pad = backend.gamepad();
    if (!app || !map || !pad) {
        check(false, "map ui: app, constellation and pad present");
        return;
    }
    settle(450);
    seedClusterScene(backend);
    settle(300);

    // Clusters: tidy, labelled, zero overlap between cards and between clusters.
    QList<QQuickItem *> isl = mapIslands(win);
    {
        QStringList pairs;
        check(isl.size() == 7 && overlapPairs(isl, &pairs) == 0, "map ui: seven cards, zero overlapping pairs",
              QString::number(isl.size()) + " " + pairs.join("; "));
        QHash<QString, QRectF> box;
        for (QQuickItem *it : isl) {
            const QString c = it->property("proj").toMap().value("cluster").toString();
            box[c] = box.contains(c) ? box[c].united(itemRect(it)) : itemRect(it);
        }
        bool apart = box.size() == 4;
        for (auto a = box.cbegin(); a != box.cend(); ++a)
            for (auto b = std::next(a); b != box.cend(); ++b)
                apart = apart && !a.value().intersects(b.value());
        check(apart, "map ui: four category clusters, their bounding boxes never overlap", QStringList(box.keys()).join(","));
        QStringList labels;
        for (QQuickItem *t : findItems(win->contentItem(), "clusterLabel"))
            if (t->isVisible())
                labels << t->property("text").toString();
        check(labels == QStringList{"WORK", "PROJECTS", "APPS", "QUICK QS"}, "map ui: uppercase micro-labels, in cluster order",
              labels.join(","));
        // Lines join cards of one cluster only.
        bool inside = true;
        const QVariantList projects = backend.projects();
        const QVariantList edges = map->property("edges").toList();
        for (const QVariant &e : edges) {
            const QVariantList ab = e.toList();
            inside = inside && projects.value(ab.value(0).toInt()).toMap().value("cluster")
                                   == projects.value(ab.value(1).toInt()).toMap().value("cluster");
        }
        check(!edges.isEmpty() && inside, "map ui: constellation lines stay inside each cluster", QString::number(edges.size()));
        const QImage shot = win->grabWindow();
        QStringList tp;
        check(forbiddenPixels(shot) == 0 && textOverlaps(win, &tp) == 0, "map ui: clustered map clean (palette, text overlaps)",
              tp.join("; "));
    }

    // Voice reorder: move Apps to the top, the cards spring there, still no overlaps.
    {
        QQuickItem *apps = islandByKey(win, "lapis");
        const double y0 = apps ? apps->y() : 0;
        backend.dispatchStudioAction("move_cluster", "Apps:top");
        settle(120);
        const double yMid = apps ? apps->y() : 0;
        const bool gliding = map->property("gliding").toBool();
        settle(1500);
        const double y1 = apps ? apps->y() : 0;
        QStringList pairs;
        check(apps && gliding && y1 < y0 - 100 && yMid > y1 + 1 && overlapPairs(mapIslands(win), &pairs) == 0,
              "map ui: move_cluster Apps:top glides the Apps cards up on a spring, no overlaps after",
              QStringLiteral("y %1 -> mid %2 -> %3 %4").arg(y0).arg(yMid).arg(y1).arg(pairs.join(";")));
        backend.dispatchStudioAction("move_cluster", "Apps:bottom");
        settle(1500);
    }

    // Controller mode hides the corner numbers; keyboard / mouse shows them.
    {
        auto numbersShown = [&] {
            int n = 0;
            for (QQuickItem *t : findItems(win->contentItem(), "islandNumber"))
                n += t->isVisible() && !t->property("text").toString().isEmpty();
            return n;
        };
        const int inController = numbersShown();
        backend.setNavMode("vim");
        settle(60);
        const int inVim = numbersShown();
        backend.setNavMode("controller");
        settle(60);
        check(inController == 0 && inVim >= 7, "map ui: card numbers hide in controller mode, show for the keyboard",
              QStringLiteral("%1 / %2").arg(inController).arg(inVim));
    }

    // Free cursor: a pad shows it; the stick glides it onto a card, which highlights, becomes
    // the selection and the active session target.
    pad->setConnected(true, QStringLiteral("DualShock 4"));
    settle(300);
    QQuickItem *cursor = firstItem(win, "mapCursor");
    check(cursor && cursor->isVisible() && pad->analogCursor(), "map ui: a connected pad shows the free cursor (analog mode)");
    {
        QQuickItem *target = islandByKey(win, "yoctogpt");
        const int ti = target ? target->property("projIndex").toInt() : -1;
        qint64 t = 10000;
        bool over = false;
        for (int k = 0; k < 400 && target && !over; ++k) {
            const QPointF c = itemRect(target).center();
            const QPointF at(map->property("cursorX").toReal(), map->property("cursorY").toReal());
            const QPointF d = c - at;
            const double len = std::hypot(d.x(), d.y());
            GamepadSnapshot s;
            s.lx = len > 1 ? d.x() / len : 0;
            s.ly = len > 1 ? -d.y() / len : 0;
            pad->processSnapshot(s, t);
            t += 8;
            settle(8);
            over = map->property("hoverIndex").toInt() == ti;
        }
        pad->processSnapshot(GamepadSnapshot(), t);
        settle(60);
        const QVariantMap at = reg->activeTarget();
        // The card reports itself (kind session); the selection it becomes then re-reports it
        // as the focus cursor's island target. Either way the active target is this project.
        check(over && app->property("selectedIsland").toInt() == ti && target->property("cursorOver").toBool()
                  && (at.value("kind") == "session" || at.value("kind") == "island") && at.value("sessionId") == "yoctogpt",
              "map ui: the stick glides the cursor onto a card: it highlights, selects and becomes the session target",
              at.value("kind").toString() + " " + at.value("sessionId").toString());
    }
    // Aim assist: the same stick moves the cursor 35% slower over a card.
    {
        QMetaObject::invokeMethod(map, "setStick", Q_ARG(QVariant, 1.0), Q_ARG(QVariant, 0.0));
        map->setProperty("cursorX", 30.0);
        map->setProperty("cursorY", 1050.0);
        QMetaObject::invokeMethod(map, "advance", Q_ARG(QVariant, 0.05));
        const double free = map->property("cursorX").toReal() - 30.0;
        QQuickItem *card = islandByKey(win, "kernel-search");
        const QPointF c = itemRect(card).center();
        map->setProperty("cursorX", c.x() - 60);
        map->setProperty("cursorY", c.y());
        QMetaObject::invokeMethod(map, "advance", Q_ARG(QVariant, 0.05));
        const double slow = map->property("cursorX").toReal() - (c.x() - 60);
        QMetaObject::invokeMethod(map, "setStick", Q_ARG(QVariant, 0.0), Q_ARG(QVariant, 0.0));
        check(free > 0 && std::abs(slow / free - 0.65) < 0.03, "map ui: aim assist slows the cursor ~35% over a card",
              QStringLiteral("%1 px free, %2 px over a card").arg(free).arg(slow));
    }

    // Quick ✕ tap on a card opens it.
    {
        QQuickItem *card = islandByKey(win, "kernel-search");
        const QPointF c = itemRect(card).center();
        map->setProperty("cursorX", c.x());
        map->setProperty("cursorY", c.y());
        settle(30);
        GamepadSnapshot a;
        a.a = true;
        pad->processSnapshot(a, 50000);
        pad->processSnapshot(GamepadSnapshot(), 50050);
        settle(450);
        check(app->property("lens").toInt() > 0 && app->property("projectKey").toString() == "kernel-search",
              "map ui: a quick ✕ tap on the card under the cursor opens it");
        GamepadSnapshot b;
        b.b = true;
        pad->processSnapshot(b, 51000);
        pad->processSnapshot(GamepadSnapshot(), 51050);
        settle(450);
        check(app->property("lens").toInt() == 0, "map ui: ○ comes back to the map");
    }

    // ✕ held on a card grabs it (lift + accent), the card tracks the cursor with zero lag, and a
    // release mid-motion flicks it: it coasts on, slows by friction and settles inside the
    // window, clear of other cards, persisted.
    {
        QQuickItem *card = islandByKey(win, "yoctogpt");
        const QPointF c0 = itemRect(card).center();
        map->setProperty("cursorX", c0.x());
        map->setProperty("cursorY", c0.y());
        settle(30);
        GamepadSnapshot a;
        a.a = true;
        pad->processSnapshot(a, 60000);
        settle(30);
        const bool grabbed = card->property("grabbed").toBool();
        // Down, into the open band between the Projects and Quick Qs clusters.
        QMetaObject::invokeMethod(map, "setStick", Q_ARG(QVariant, 0.0), Q_ARG(QVariant, 0.5));
        QMetaObject::invokeMethod(map, "advance", Q_ARG(QVariant, 0.04));
        const QPointF cur(map->property("cursorX").toReal(), map->property("cursorY").toReal());
        const QPointF mid = itemRect(card).center();
        const bool tracks = std::abs((mid - c0).y() - (cur.y() - c0.y())) < 0.5 && std::abs(mid.x() - c0.x()) < 0.5
                            && cur.y() > c0.y() + 20;
        const double vx = map->property("cursorVy").toReal();
        pad->processSnapshot(GamepadSnapshot(), 60300);  // ✕ up while the stick still pushes down
        QMetaObject::invokeMethod(map, "setStick", Q_ARG(QVariant, 0.0), Q_ARG(QVariant, 0.0));
        const bool flying = map->property("flightCount").toInt() == 1;
        const double releaseY = mid.y();
        const bool landed = waitUntil([&] { return map->property("flightCount").toInt() == 0; }, 5000);
        settle(60);
        const QRectF r = itemRect(card);
        QStringList pairs;
        check(grabbed && tracks, "map ui: ✕ held grabs the card and it tracks the cursor with zero lag",
              QStringLiteral("grabbed %1 card dy %2 cursor dy %3").arg(grabbed).arg((mid - c0).y()).arg(cur.y() - c0.y()));
        check(!card->property("grabbed").toBool() && vx > 400 && flying && landed,
              "map ui: releasing mid-motion launches a flick that comes to rest",
              QStringLiteral("vy %1 flying %2 landed %3").arg(vx).arg(flying).arg(landed));
        check(r.center().y() > releaseY + 40 && r.left() >= 0 && r.top() >= 64 && r.right() <= 1920 && r.bottom() <= 1080
                  && overlapPairs(mapIslands(win), &pairs) == 0 && backend.cardPositions().contains("yoctogpt"),
              "map ui: the flicked card coasts past the release point, settles inside the window, clear of others, persisted",
              QStringLiteral("release y %1 -> rest %2,%3 %4").arg(releaseY).arg(r.center().x()).arg(r.center().y()).arg(pairs.join(";")));
    }

    // Mouse flick: a fast drag released in motion carries the card further than the pointer.
    {
        QQuickItem *card = islandByKey(win, "kernel-search");
        const QPointF from = card->mapToScene(QPointF(card->width() / 2, card->height() / 2));
        mouseMove(win, from);
        mouseButton(win, QEvent::MouseButtonPress, from, Qt::LeftButton);
        for (int i = 1; i <= 8; ++i) {
            mouseMove(win, from + QPointF(0, 12.0 * i), Qt::LeftButton);
            settle(6);
        }
        mouseButton(win, QEvent::MouseButtonRelease, from + QPointF(0, 96), Qt::LeftButton);
        const bool flying = map->property("flightCount").toInt() == 1;
        waitUntil([&] { return map->property("flightCount").toInt() == 0; }, 5000);
        settle(60);
        const double travelled = itemRect(card).center().y() - from.y();
        check(flying && travelled > 110 && app->property("lens").toInt() == 0,
              "map ui: a mouse drag released in motion flicks the card on past the pointer",
              QStringLiteral("flying %1 travelled %2 px for a 96 px drag").arg(flying).arg(travelled));
    }
    backend.resetCardPositions();
    settle(200);

    // Rubber band (pad): ✕ held on empty space draws the marquee; release selects every card it
    // touches, pins them as session targets, and the ask card carries them all.
    {
        QQuickItem *a = islandByKey(win, "cluster-serving");
        QQuickItem *b = islandByKey(win, "kernel-search");
        const QRectF ra = itemRect(a), rb = itemRect(b);
        const QPointF start(ra.left() - 20, ra.top() - 40);
        map->setProperty("cursorX", start.x());
        map->setProperty("cursorY", start.y());
        settle(30);
        check(map->property("hoverIndex").toInt() == -1, "map ui: marquee starts on empty space");
        GamepadSnapshot x;
        x.a = true;
        pad->processSnapshot(x, 70000);
        settle(20);
        map->setProperty("cursorX", rb.center().x());
        map->setProperty("cursorY", ra.bottom() - 10);
        settle(40);
        QQuickItem *mq = firstItem(win, "marqueeRect");
        const bool drawn = mq && mq->isVisible() && mq->width() > 100 && mq->height() > 100;
        pad->processSnapshot(GamepadSnapshot(), 70400);
        settle(60);
        const QStringList sel = map->property("selectedKeys").toStringList();
        int rings = 0;
        for (QQuickItem *r : findItems(win->contentItem(), "islandSelectRing"))
            rings += r->isVisible();
        QStringList pinned;
        for (const QVariant &v : reg->pinnedTargets())
            pinned << v.toMap().value("sessionId").toString();
        check(drawn && sel.contains("cluster-serving") && sel.contains("kernel-search") && sel.size() >= 2
                  && rings == sel.size() && reg->pinnedCount() == sel.size() && pinned.contains("kernel-search"),
              "map ui: ✕ drag on empty space draws a marquee; the cards it covers get rings and pin into backend.targets",
              QStringLiteral("sel %1 rings %2 pins %3").arg(sel.join(",")).arg(rings).arg(pinned.join(",")));

        // Grabbing one selected card moves the whole selection together.
        QList<QPointF> before;
        for (const QString &k : sel)
            before << itemRect(islandByKey(win, k)).center();
        const QPointF ca = itemRect(a).center();
        map->setProperty("cursorX", ca.x());
        map->setProperty("cursorY", ca.y());
        settle(20);
        pad->processSnapshot(x, 71000);
        settle(20);
        map->setProperty("cursorX", ca.x() + 40);
        map->setProperty("cursorY", ca.y() + 30);
        QMetaObject::invokeMethod(map, "advance", Q_ARG(QVariant, 1.0 / 120));
        bool together = true;
        for (int i = 0; i < sel.size(); ++i) {
            const QPointF now = itemRect(islandByKey(win, sel[i])).center();
            together = together && std::abs(now.x() - before[i].x() - 40) < 0.5 && std::abs(now.y() - before[i].y() - 30) < 0.5;
        }
        pad->processSnapshot(GamepadSnapshot(), 71400);
        waitUntil([&] { return map->property("flightCount").toInt() == 0; }, 5000);
        settle(60);
        check(together, "map ui: dragging one selected card moves the whole selection together");

        // The ask card (Cmd+K) opens about every selected card at once.
        sendKey(win, Qt::Key_K, Qt::ControlModifier);
        settle(60);
        QObject *omni = win->findChild<QObject *>("floatingOmnibar");
        const QVariantList chips = omni ? omni->property("chips").toList() : QVariantList();
        const QString slice = reg->buildCombinedSlice(4000);
        check(omni && omni->property("open").toBool() && chips.size() == sel.size() && slice.contains("cluster-serving")
                  && slice.contains("kernel-search"),
              "map ui: Cmd+K asks about all selected cards (one chip each, all in the Tier 3 slice)",
              QString::number(chips.size()));
        sendKey(win, Qt::Key_Escape);
        settle(60);
        QMetaObject::invokeMethod(app, "takeFocus");
        sendKey(win, Qt::Key_Escape);
        settle(60);
        check(map->property("selectedKeys").toStringList().isEmpty() && reg->pinnedCount() == 0,
              "map ui: Esc clears the multi-selection and its pins");
    }
    backend.resetCardPositions();
    settle(200);

    // Rubber band (mouse) over empty space, then ○ clears it; a plain click on empty space too.
    {
        QQuickItem *q1 = islandByKey(win, "~tab:aa11bb22");
        const QRectF rq = itemRect(q1);
        const QPointF from(rq.left() - 16, rq.top() - 30);
        mouseDrag(win, from, QPointF(rq.width() + 40, rq.height() + 50));
        settle(60);
        const QStringList sel = map->property("selectedKeys").toStringList();
        check(sel.contains("~tab:aa11bb22") && reg->pinnedCount() == sel.size(), "map ui: a mouse drag on empty space selects too",
              sel.join(","));
        GamepadSnapshot b;
        b.b = true;
        pad->processSnapshot(b, 80000);
        pad->processSnapshot(GamepadSnapshot(), 80050);
        settle(60);
        check(map->property("selectedKeys").toStringList().isEmpty() && reg->pinnedCount() == 0 && app->property("lens").toInt() == 0,
              "map ui: ○ clears the selection (and stays on the map)");
        mouseDrag(win, from, QPointF(rq.width() + 40, rq.height() + 50));
        settle(60);
        mouseClick(win, QPointF(from.x(), from.y()), Qt::LeftButton);
        settle(60);
        check(map->property("selectedKeys").toStringList().isEmpty(), "map ui: clicking empty space clears it");
    }
    check(map->property("needsMeOnly").toBool() == false, "map ui: needsMeOnly mirrors the filter");
    pad->setConnected(false, QString());
    settle(60);
}

// --- Half-screen window (960x1080, 11 cards in 4 clusters) ------------------------------------
// The cluster scene plus four more cards: Work 2, Projects 3, Apps 3, Quick Qs 3.
void seedHalfScreenScene(Backend &b)
{
    seedClusterScene(b);
    b.mockAddProject("lapis-studio", "lapis-studio", "~/dev/lapis-studio", "apps", "working");
    b.mockAddProject("netherite-v2", "netherite-v2", "~/dev/netherite-v2", "", "idle");
    b.mockAddProject("infinity-agy", "infinity-agy", "~/dev/infinity-agy", "infinity", "working");
    b.mockAddProject("~tab:ee55ff66", "Benchmark", "~", "quick Qs", "idle");
    Turn t;
    t.role = QStringLiteral("assistant");
    t.text = QStringLiteral("Updated [app.toml](file:///Users/someone/dev/lapis-studio/app.toml) so the port is 8080.");
    b.mockAppendTurn("lapis-studio", t);
    t.text = QStringLiteral("Swept the [tile sizes](https://example.com/sweep?run=4) and the 128x64 tile wins.");
    b.mockAppendTurn("netherite-v2", t);
}
// Card positions saved in a full-screen session: as fractions they put cards on top of each other
// once the window is half as wide (the screenshot that started this: lapis-studio x netherite-v2).
void seedFullScreenPositions(Backend &b)
{
    b.setCardPosition("lapis-studio", 0.5824, 0.3381);
    b.setCardPosition("netherite-v2", 0.7267, 0.3381);
    b.setCardPosition("~tab:ee55ff66", 0.6007, 0.5123);
    b.setCardPosition("infinity-agy", 0.6007, 0.6717);
    b.setCardPosition("~tab:aa11bb22", 0.4913, 0.8630);
    b.setCardPosition("cluster-serving", 0.7988, 0.7268);
    b.setCardPosition("lapis", 0.2677, 0.7774);
    b.setCardPosition("phonon", 0.1608, 0.8850);
    b.setCardPosition("~tab:cc33dd44", 0.1743, 0.6736);
}
// Pairs of map cards closer than `gap` px (0: intersecting).
int crowdedPairs(const QList<QQuickItem *> &isl, qreal gap, QStringList *pairs)
{
    int n = 0;
    const qreal g = gap / 2 - 0.01;
    for (int i = 0; i < isl.size(); ++i)
        for (int j = i + 1; j < isl.size(); ++j)
            if (itemRect(isl[i]).adjusted(-g, -g, g, g).intersects(itemRect(isl[j]).adjusted(-g, -g, g, g))) {
                ++n;
                *pairs << isl[i]->property("key").toString() + " x " + isl[j]->property("key").toString();
            }
    return n;
}
// Inside every card: the title, job and state lines never share a vertical pixel, and none is
// cut by the card's bottom edge.
int islandLineCollisions(const QList<QQuickItem *> &isl, QStringList *bad)
{
    int n = 0;
    for (QQuickItem *card : isl) {
        QQuickItem *body = findItems(card, "islandBody").value(0);
        if (!body)
            continue;
        const QRectF b = body->mapRectToScene(QRectF(0, 0, body->width(), body->height()));
        QList<QPair<QString, QRectF>> lines;
        for (const char *name : {"islandTitle", "islandJob", "islandLast"})
            for (QQuickItem *t : findItems(card, QString::fromLatin1(name)))
                if (t->isVisible() && !t->property("text").toString().isEmpty())
                    lines << qMakePair(QString::fromLatin1(name), t->mapRectToScene(QRectF(0, 0, t->width(), t->height())));
        const QString key = card->property("key").toString();
        for (int i = 0; i < lines.size(); ++i) {
            if (lines[i].second.bottom() > b.bottom() + 0.5) {
                ++n;
                *bad << key + ": " + lines[i].first + " below the card";
            }
            for (int j = i + 1; j < lines.size(); ++j)
                if (std::min(lines[i].second.bottom(), lines[j].second.bottom())
                        - std::max(lines[i].second.top(), lines[j].second.top()) > 0.5) {
                    ++n;
                    *bad << key + ": " + lines[i].first + " x " + lines[j].first;
                }
        }
    }
    return n;
}

void halfScreenUiTests()
{
    Backend backend(Backend::Mode::Mock);
    QTemporaryDir tmp;
    backend.setPrefsPath(tmp.path() + "/studio_prefs.json");
    backend.completeOnboarding("controller");  // the longest nav pill label ("nav controller")
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(960, 1080));
    check(win != nullptr, "half: QML loads at 960x1080");
    if (!win)
        return;
    QObject *map = win->findChild<QObject *>("constellation");
    if (!map) {
        check(false, "half: constellation present");
        return;
    }
    settle(450);
    seedHalfScreenScene(backend);
    settle(400);

    auto audit = [&](const QString &where) {
        auto check = [](bool ok, const QString &name, const QString &detail) {
            studio::check(ok, qPrintable(name), detail);
        };
        const QList<QQuickItem *> isl = mapIslands(win);
        QStringList pairs, lines, tp;
        qreal minH = 1e9;
        for (QQuickItem *it : isl)
            minH = std::min(minH, it->height());
        check(isl.size() == 11, "half: 11 cards on the map (" + where + ")", QString::number(isl.size()));
        check(overlapPairs(isl) == 0 && crowdedPairs(isl, 14, &pairs) == 0,
              "half: zero card overlaps, every pair >= 14 px apart (" + where + ")", pairs.join("; "));
        check(minH >= 108, "half: every card >= 108 px tall (" + where + ")", QString::number(minH));
        check(islandLineCollisions(isl, &lines) == 0, "half: zero title / job / state line collisions in any card (" + where + ")",
              lines.join("; "));
        check(textOverlaps(win, &tp) == 0, "half: no overlapping text anywhere (" + where + ")", tp.join("; "));
    };

    // Focus chrome on the selected card: "Open Stage R1" tucked on its top border, "✕ Expand" on
    // its bottom border, neither touching a cluster label or a neighbouring card.
    auto chrome = [&](const QString &where) {
        settle(800);  // idle: the predictor badge fades in
        QQuickItem *badge = firstItem(win, "tabBadge");
        QQuickItem *hint = firstItem(win, "focusCursorHint");
        QStringList hits;
        studio::check(badge && badge->isVisible() && hint && hint->isVisible(),
                      qPrintable("half: Open Stage badge and Expand chip both shown (" + where + ")"));
        studio::check(mapChromeCollisions(win, &hits) == 0,
                      qPrintable("half: focus chrome clears cluster labels and neighbouring cards (" + where + ")"), hits.join("; "));
    };

    // Auto-layout.
    audit("auto layout");
    chrome("960x1080");
    std::printf("info  half: card %.0fx%.0f, %d cluster column(s), content %.0f px, scrollable %d\n",
                map->property("islandW").toReal(), map->property("islandH").toReal(),
                map->property("layout").toMap().value("ccols").toInt(), map->property("contentH").toReal(),
                map->property("scrollable").toBool());

    // Every card can be reached: scrolled to the far end, the last card sits fully in the window.
    if (map->property("scrollable").toBool()) {
        QMetaObject::invokeMethod(map, "scrollBy", Q_ARG(QVariant, 100000.0));
        settle(400);
        audit("scrolled to the end");
        qreal lowest = 0;
        QQuickItem *last = nullptr;
        for (QQuickItem *it : mapIslands(win)) {
            const QRectF r = it->mapRectToScene(QRectF(0, 0, it->width(), it->height()));
            if (r.bottom() > lowest) { lowest = r.bottom(); last = it; }
        }
        check(last && lowest <= 1080 - 40 + 0.5, "half: scrolled to the end, the lowest card is fully on screen",
              QString::number(lowest));
        QMetaObject::invokeMethod(map, "scrollBy", Q_ARG(QVariant, -100000.0));
        settle(400);
    }

    // Markdown links on a card read as their label.
    {
        QStringList shown;
        for (QQuickItem *it : mapIslands(win))
            for (const char *name : {"islandJob", "islandLast"})
                for (QQuickItem *t : findItems(it, QString::fromLatin1(name)))
                    if (t->isVisible())
                        shown << t->property("text").toString();
        const QString all = shown.join(" | ");
        check(all.contains("Updated app.toml so the port is 8080") && all.contains("Swept the tile sizes")
                  && !all.contains("](") && !all.contains("file://") && !all.contains("https://"),
              "half: card lines show link labels, never raw [label](url)", all.left(400));
    }

    // Positions saved full-screen, then the window halves: still no overlap, re-run on resize.
    seedFullScreenPositions(backend);
    settle(300);
    audit("full-screen positions at half width");
    win->resize(1920, 1080);
    settle(300);
    {
        QStringList pairs;
        check(crowdedPairs(mapIslands(win), 14, &pairs) == 0, "half: saved positions at 1920 wide stay clear", pairs.join("; "));
    }
    chrome("1920x1080");
    win->resize(960, 1080);
    settle(400);
    audit("resized 1920 -> 960 with saved positions");
    win->resize(960, 700);
    settle(400);
    audit("resized to 960x700");
    win->resize(1100, 1080);
    settle(400);
    audit("resized to 1100x1080");
    win->resize(720, 600);  // the window minimum
    settle(400);
    audit("resized to the 720x600 minimum");
    win->resize(960, 1080);
    settle(400);

    // Bottom-right nav pill: fully inside the window with its 18 px margin, text unclipped.
    if (QQuickItem *nav = firstItem(win, "navPill")) {
        const QRectF r = nav->mapRectToScene(QRectF(0, 0, nav->width(), nav->height()));
        bool textFits = true;
        for (QQuickItem *t : findItems(nav, "navOption"))
            if (t->isVisible()) {
                const QRectF tr = t->mapRectToScene(QRectF(0, 0, t->width(), t->height()));
                textFits = textFits && tr.right() <= r.right() + 0.5 && t->width() + 0.5 >= t->property("implicitWidth").toReal();
            }
        check(nav->isVisible() && r.right() <= 960 - 18 + 0.5 && r.left() >= 0 && textFits,
              "half: bottom-right nav pill keeps its 18 px margin, text unclipped",
              QStringLiteral("right %1").arg(r.right()));
    } else {
        check(false, "half: nav pill present");
    }
}

}  // namespace

// --snap-half OUT.png: the Constellation in a half-screen window (960x1080), 11 cards in four
// clusters, dark. The visual twin of halfScreenUiTests.
int snapHalf(const QString &outPath)
{
    Backend backend(Backend::Mode::Mock);
    QTemporaryDir tmp;
    backend.setPrefsPath(tmp.path() + "/studio_prefs.json");
    backend.completeOnboarding("controller");
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(960, 1080));
    if (!win) {
        std::fprintf(stderr, "snap-half: QML failed to load\n");
        return 2;
    }
    QDir().mkpath(QFileInfo(outPath).absolutePath());
    settle(450);
    seedHalfScreenScene(backend);
    mouseMove(win, QPointF(4, 600));
    settle(900);
    QImage img = win->grabWindow();
    bool saved = img.save(outPath);
    QStringList pairs, lines, tp;
    const int cards = int(mapIslands(win).size());
    const int crowded = crowdedPairs(mapIslands(win), 14, &pairs);
    const int collide = islandLineCollisions(mapIslands(win), &lines);
    const int textHits = textOverlaps(win, &tp);
    std::printf("snap-half: %s (%dx%d) cards=%d crowded=%d lineCollisions=%d textOverlaps=%d forbidden=%d\n",
                qPrintable(outPath), img.width(), img.height(), cards, crowded, collide, textHits, forbiddenPixels(img));
    // The same window with the full-screen saved positions (the screenshot's failure case).
    seedFullScreenPositions(backend);
    settle(600);
    const QString posPath = QFileInfo(outPath).absolutePath() + "/" + QFileInfo(outPath).completeBaseName() + "-saved.png";
    img = win->grabWindow();
    saved = img.save(posPath) && saved;
    pairs.clear();
    std::printf("snap-half: %s (saved positions) crowded=%d\n", qPrintable(posPath), crowdedPairs(mapIslands(win), 14, &pairs));
    return saved && crowded == 0 && collide == 0 && textHits == 0 ? 0 : 1;
}

// --snap-map OUT.png: the clustered Constellation with the free pad cursor mid-marquee and a
// finished multi-selection (rings), dark, 1920x1080.
int snapMap(const QString &outPath)
{
    Backend backend(Backend::Mode::Mock);
    QQmlApplicationEngine engine;
    QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
    if (!win) {
        std::fprintf(stderr, "snap-map: QML failed to load\n");
        return 2;
    }
    QObject *map = win->findChild<QObject *>("constellation");
    Gamepad *pad = backend.gamepad();
    if (!map || !pad)
        return 3;
    QDir().mkpath(QFileInfo(outPath).absolutePath());
    settle(450);
    seedPhase3Scene(backend);
    seedClusterScene(backend);
    pad->setConnected(true, QStringLiteral("DualShock 4"));
    settle(500);
    // A finished selection: the two Apps cards.
    map->setProperty("selectedKeys", QStringList{});
    QMetaObject::invokeMethod(map, "setSelection", Q_ARG(QVariant, QVariant(QStringList{"lapis", "phonon"})));
    // A marquee in progress from empty space over the Quick Qs cards, the cursor at its corner.
    QQuickItem *q1 = islandByKey(win, "~tab:aa11bb22");
    QQuickItem *q2 = islandByKey(win, "~tab:cc33dd44");
    if (q1 && q2) {
        const QRectF a = itemRect(q1), b = itemRect(q2);
        map->setProperty("cursorX", a.left() - 26);
        map->setProperty("cursorY", a.top() - 30);
        settle(30);
        QMetaObject::invokeMethod(map, "padGrab", Q_ARG(QVariant, true));
        map->setProperty("cursorX", b.center().x() + 20);
        map->setProperty("cursorY", b.center().y() + 10);
    }
    mouseMove(win, QPointF(4, 1076));
    settle(700);
    const QImage img = win->grabWindow();
    const bool saved = img.save(outPath);
    std::printf("snap-map: %s (%dx%d) forbidden=%d\n", qPrintable(outPath), img.width(), img.height(), forbiddenPixels(img));
    return saved ? 0 : 1;
}

// Quick polish: view-state persistence, $HOME spawn, Tier 3 clock, OpenRouter request.
void polishTests()
{
    // Card lines: markdown links read as their label, never raw [label](url).
    {
        check(stripMarkdownLinks("Edited [app.toml](file:///Users/x/dev/app.toml) and pushed") == "Edited app.toml and pushed",
              "links: file:// link -> label");
        check(stripMarkdownLinks("See [the PR](https://github.com/o/r/pull/12) and [docs](http://x.y/a_(b))")
                  == "See the PR and docs",
              "links: http(s) links, parens inside the URL", stripMarkdownLinks("See [the PR](https://github.com/o/r/pull/12) and [docs](http://x.y/a_(b))"));
        check(stripMarkdownLinks("![loss curve](out/loss.png) shows it") == "loss curve shows it", "links: image -> alt text");
        check(stripMarkdownLinks("Wrote [notes](<file:///a b/notes.md> \"t\")") == "Wrote notes", "links: <path with spaces> and title");
        check(stripMarkdownLinks("Updated [app.toml](file:///Use") == "Updated app.toml", "links: a link cut mid-URL -> label",
              stripMarkdownLinks("Updated [app.toml](file:///Use"));
        check(stripMarkdownLinks("open <https://x.y/z>") == "open https://x.y/z", "links: autolink brackets dropped");
        check(stripMarkdownLinks("arr[0] (the first) stays") == "arr[0] (the first) stays"
                  && stripMarkdownLinks("plain line") == "plain line",
              "links: brackets that are not links stay");
        const HumanFilterReply r = parseFilterReply(
            "STATE: Moved the port into [app.toml](file:///Users/x/dev/lapis/app.toml)\nFORM: summary\nPOINTS:\n- one");
        check(r.stateLine == "Moved the port into app.toml", "links: the human-filter state line is plain", r.stateLine);
        Extracted ex;
        ex.prose = "Updated [app.toml](file:///Users/x/dev/app.toml) so the port is 8080.";
        const IslandState st = islandState(ex, "idle");
        check(!st.now.contains("](") && st.now.contains("app.toml"), "links: the island state line is plain", st.now);
    }

    // viewState: normalized, persisted, reloaded; junk never lands.
    QTemporaryDir tmp;
    const QString path = tmp.path() + "/studio_prefs.json";
    QString stageKey;
    int stageArts = 0;
    {
        Backend b(Backend::Mode::Mock);
        b.setPrefsPath(path);
        for (const QVariant &v : b.projects()) {
            const QVariantMap m = v.toMap();
            if (m.value("artifacts").toList().size() > stageArts) {
                stageArts = int(m.value("artifacts").toList().size());
                stageKey = m.value("key").toString();
            }
        }
        b.setViewState({{"lens", "nonsense"}, {"focusedProject", stageKey}});
        check(b.viewState().isEmpty(), "viewState: an unknown lens is dropped");
        b.setViewState({{"lens", "stage"}, {"focusedProject", stageKey}, {"artifactIndex", 99}, {"needsMeOnly", true}});
        check(b.savePrefs(), "viewState: prefs flush");
        QFile f(path);
        const bool opened = f.open(QIODevice::ReadOnly);
        check(opened, "viewState: prefs file opens", path);
        const QJsonObject vs = QJsonDocument::fromJson(f.readAll()).object().value("viewState").toObject();
        check(vs.value("lens").toString() == "stage" && vs.value("focusedProject").toString() == stageKey
                  && vs.value("artifactIndex").toInt() == 99 && vs.value("needsMeOnly").toBool(),
              "viewState: written to studio_prefs.json", QString::fromUtf8(QJsonDocument(vs).toJson(QJsonDocument::Compact)));
    }
    check(stageArts > 1, "viewState: a fixture with several artifacts exists", stageKey);
    {
        // Restore on startup: Stage, the saved project, artifact clamped, Needs Me filter back on.
        Backend b(Backend::Mode::Mock);
        b.setPrefsPath(path);
        check(b.viewState().value("lens").toString() == "stage", "viewState: reloads from prefs");
        QQmlApplicationEngine engine;
        QQuickWindow *win = createStudioWindow(engine, &b, false, QSize(1920, 1080));
        QObject *app = win ? win->findChild<QObject *>("studioApp") : nullptr;
        check(app && app->property("lens").toInt() == 2 && app->property("projectKey").toString() == stageKey
                  && app->property("artifactIndex").toInt() == stageArts - 1
                  && app->property("islandFilter").toString() == "needs",
              "viewState: restart lands on the saved Stage, project, artifact and filter",
              app ? QStringLiteral("lens %1 key %2 art %3 filter %4")
                        .arg(app->property("lens").toInt())
                        .arg(app->property("projectKey").toString())
                        .arg(app->property("artifactIndex").toInt())
                        .arg(app->property("islandFilter").toString())
                  : QStringLiteral("no app"));
        if (app) {
            QMetaObject::invokeMethod(app, "goLens", Q_ARG(QVariant, QVariant(1)));
            QCoreApplication::processEvents();
            check(b.viewState().value("lens").toString() == "ladder" && b.viewState().value("focusedProject") == stageKey,
                  "viewState: a lens change goes back to the backend", b.viewState().value("lens").toString());
        }
    }
    {
        // A saved project that no longer exists falls back to the Constellation.
        QFile f(path);
        const bool opened = f.open(QIODevice::WriteOnly | QIODevice::Truncate);
        check(opened, "viewState: prefs file opens for writing", path);
        f.write(R"({"viewState":{"lens":"ladder","focusedProject":"gone-project","artifactIndex":0,"needsMeOnly":false}})");
        f.close();
        Backend b(Backend::Mode::Mock);
        b.setPrefsPath(path);
        QQmlApplicationEngine engine;
        QQuickWindow *win = createStudioWindow(engine, &b, false, QSize(1920, 1080));
        QObject *app = win ? win->findChild<QObject *>("studioApp") : nullptr;
        check(app && app->property("lens").toInt() == 0 && app->property("islandFilter").toString() == "all",
              "viewState: a missing project falls back to the Constellation");
    }

    // spawn_agent from a $HOME scratch tab runs in $HOME; a /tmp session is still refused.
    {
        Backend b(Backend::Mode::Mock);
        b.mockAddProject("~tab:aa11bb22", "Why is the fn key slow", "~", "quick Qs", "idle");
        b.mockAddProject("~tab:ee55ff66", "Untitled scratch", "", "quick Qs", "idle");
        b.mockAddProject("tmp-scratch", "Scratch in tmp", "/tmp", "quick Qs", "idle");
        b.setFocusedProject("~tab:aa11bb22");
        const QString r1 = b.dispatchStudioAction("spawn_agent", "write a disk usage script");
        check(r1.startsWith("Spawning an Opus 5.5 agent in ~:"), "spawn: a $HOME tab spawns in $HOME", r1);
        b.setFocusedProject("~tab:ee55ff66");
        const QString r2 = b.dispatchStudioAction("spawn_agent", "tidy my downloads");
        check(r2.startsWith("Spawning an Opus 5.5 agent in ~:"), "spawn: a tab with no folder defaults to $HOME", r2);
        b.setFocusedProject("tmp-scratch");
        const QString r3 = b.dispatchStudioAction("spawn_agent", "do work");
        check(r3.contains("no project folder"), "spawn: /tmp is still refused", r3);
    }
    check(kCompatSystemPrompt.contains("Never say you are steering, messaging, spawning, or starting an agent unless the "
                                       "same answer emits [[ACTION:steer_agent:...]] or [[ACTION:spawn_agent:<task>]]"),
          "spawn: Tier 0 forbids prose-only 'Spawning an agent'");

    // Wall clock rides in Tier 3 only: the system prompt and the trunk never carry it.
    {
        const QString line = currentTimeLine(QDateTime(QDate(2026, 11, 19), QTime(18, 16)));
        const QString midnight = currentTimeLine(QDateTime(QDate(2026, 11, 20), QTime(0, 5)));
        check(line == "Current time: 6:16 pm" && midnight == "Current time: 12:05 am", "clock: 12-hour local time line",
              line + " | " + midnight);
        TieredPrompt p;
        p.model = "qwen3.8-flash-next";
        p.systemPrompt = kCompatSystemPrompt;
        p.trunk = "TRUNK";
        p.currentTime = line;
        p.question = "is the run still going?";
        const QString leaf = tieredLeafText(p);
        check(leaf.endsWith("Current time: 6:16 pm\n\nQuestion: is the run still going?"), "clock: Tier 3 leaf carries the time",
              leaf);
        const QJsonArray msgs = QJsonDocument::fromJson(buildTieredChatRequest(p)).object().value("messages").toArray();
        const QJsonArray warm = QJsonDocument::fromJson(buildTieredWarmRequest(p)).object().value("messages").toArray();
        bool prefixClean = true;
        for (int i = 0; i + 1 < msgs.size(); ++i)
            prefixClean = prefixClean && !msgs[i].toObject().value("content").toString().contains("Current time");
        check(prefixClean && !QJsonDocument(warm).toJson().contains("Current time")
                  && msgs.last().toObject().value("content").toString().contains("Current time: 6:16 pm"),
              "clock: Tiers 0/1 and the warm prefix never carry the time (APC stays hot)");
    }

    // OpenRouter body: DeepSeek, reasoning none on fast, provider routing, no Qwen-only kwargs.
    {
        TieredPrompt p;
        p.model = "deepseek/deepseek-v4.1-flash-1017";
        p.effort = "fast";
        p.systemPrompt = kCompatSystemPrompt;
        p.question = "hi";
        const QJsonObject o = QJsonDocument::fromJson(buildOpenRouterChatRequest(p)).object();
        const QJsonObject prov = o.value("provider").toObject();
        check(o.value("model").toString() == "deepseek/deepseek-v4.1-flash-1017" && o.value("stream").toBool()
                  && o.value("reasoning").toObject().value("effort").toString() == "none"
                  && prov.value("order").toArray() == QJsonArray({"Wafer", "InferenceNet", "Nebius"})
                  && prov.value("sort").toString() == "latency" && prov.value("allow_fallbacks").toBool()
                  && !o.contains("chat_template_kwargs") && !o.contains("reasoning_effort"),
              "openrouter: fast -> reasoning none, Wafer/InferenceNet/Nebius by latency",
              QString::fromUtf8(QJsonDocument(o).toJson(QJsonDocument::Compact)).left(400));
        const QString sys = o.value("messages").toArray().first().toObject().value("content").toString();
        check(sys.contains("served via OpenRouter") && !sys.contains("vLLM"), "openrouter: identity line names OpenRouter");
        p.effort = "medium";
        check(QJsonDocument::fromJson(buildOpenRouterChatRequest(p)).object().value("reasoning").toObject().value("effort")
                  == "medium",
              "openrouter: a thinking effort rides as reasoning.effort");
    }
}

// L2 (Phonon push-to-talk) -> R2 (send), turn after turn: the in-Studio ask card and the
// MacBook-wide Spotlight bar (Offscreen here: a pure state machine, no window, no keys posted).
void voiceSpotlightTests()
{
    using S = GamepadSnapshot;
    auto l2 = [](double v) { S s; s.l2 = v; return s; };
    // Gamepad alone: the prepare hook and voiceHold both run before the fn key goes out.
    {
        Gamepad g;
        QStringList ev;
        int delay = 0;
        g.setPrepareVoiceTarget([&](bool on) { ev << QStringLiteral("prepare:%1").arg(on); return on ? delay : 0; });
        g.setVoiceKeySinkForTest([&](bool down) { ev << QStringLiteral("key:%1").arg(down); });
        QObject::connect(&g, &Gamepad::voiceHold, &g, [&](bool on) { ev << QStringLiteral("hold:%1").arg(on); });
        g.processSnapshot(l2(1.0), 0);
        g.processSnapshot(S(), 40);
        check(ev == QStringList({"prepare:1", "hold:1", "key:1", "prepare:0", "hold:0", "key:0"}),
              "voice order: prepare -> voiceHold -> fn key, on press and release", ev.join(' '));
        ev.clear();
        delay = 45;
        g.processSnapshot(l2(1.0), 100);
        check(ev == QStringList({"prepare:1", "hold:1"}) && !g.voiceKeySent(),
              "voice delay: a 45 ms prepare holds the fn key-down back", ev.join(' '));
        QElapsedTimer t;
        t.start();
        while (!g.voiceKeySent() && t.elapsed() < 1000)
            settle(5);
        check(g.voiceKeySent() && ev.last() == "key:1" && t.elapsed() >= 30,
              "voice delay: the key-down goes out once the timer fires", QString::number(t.elapsed()) + " ms");
        g.processSnapshot(S(), 300);
        check(ev.last() == "key:0" && ev.count("key:0") == 1, "voice delay: release after the delayed down sends the up");
        ev.clear();
        g.processSnapshot(l2(1.0), 400);
        g.processSnapshot(S(), 410);  // released inside the 45 ms window
        settle(80);
        check(!ev.contains("key:1") && !ev.contains("key:0") && !g.voiceKeySent(),
              "voice delay: a release before the timer cancels it (no key-down, no orphan key-up)", ev.join(' '));
    }

    // Outside Studio: L2 opens the Spotlight bar over the frontmost app; R2 sends; turns persist.
    {
        Backend b(Backend::Mode::Mock);
        b.setTestStreamCapture(true);
        bool studioActive = false;
        b.setStudioActiveProbeForTest([&] { return studioActive; });
        Gamepad *pad = b.gamepad();
        QStringList keys;
        pad->setVoiceKeySinkForTest([&](bool down) { keys << (down ? "down" : "up"); });
        studio::cua::CuaSpotlight *sp = b.spotlight();
        sp->setFrontmostProbe([](int *pid, QString *name) { *pid = 4242; *name = QStringLiteral("Google Chrome"); });
        int openChanges = 0;
        QObject::connect(&b, &Backend::spotlightOpenChanged, &b, [&] { ++openChanges; });
        qint64 t = 1000;
        auto tap = [&](auto set) {
            S s;
            set(s);
            pad->processSnapshot(s, t);
            pad->processSnapshot(S(), t + 16);
            t += 40;
            settle(10);  // the Backend's pad routes are queued
        };
        check(!b.spotlightOpen() && b.property("spotlightOpen").toBool() == false, "spotlight: closed at start");
        pad->processSnapshot(l2(1.0), t += 20);
        studio::cua::SpotlightState st = sp->state();
        check(b.spotlightOpen() && openChanges == 1 && st.listening && st.targetApp == "Google Chrome" && st.targetPid == 4242,
              "spotlight: L2 outside Studio opens the bar over Chrome, listening");
        check(keys.isEmpty() && !pad->voiceKeySent(), "spotlight: the fn key waits 45 ms for the bar to take focus");
        settle(80);
        check(keys == QStringList({"down"}), "spotlight: ...then goes down", keys.join(' '));
        sp->setInputText(QStringLiteral("summarize this page"));  // what Phonon pasted
        pad->processSnapshot(S(), t += 20);
        check(keys == QStringList({"down", "up"}) && !sp->state().listening && b.spotlightOpen(),
              "spotlight: L2 release lifts the key, stops listening, bar stays up", keys.join(' '));
        tap([](S &s) { s.r2 = 1.0; });
        check(b.testStreamStarts().value(0) == "spotlight|summarize this page"
                  && b.testLastSlice().startsWith("Active macOS app behind Spotlight: Google Chrome")
                  && sp->state().busy && sp->state().lastQuestion == "summarize this page" && sp->inputText().isEmpty(),
              "spotlight: R2 sends the prompt with the app behind as the slice", b.testStreamStarts().join(';'));
        b.testStreamDelta(QStringLiteral("spotlight"), QStringLiteral("It is a pricing page "));
        b.testStreamDelta(QStringLiteral("spotlight"), QStringLiteral("with three tiers. [[ACTION:go_ho"));
        check(sp->state().answer == "It is a pricing page with three tiers.",
              "spotlight: streaming answer, a half-streamed tag never shows", sp->state().answer);
        b.testStreamFinish(QStringLiteral("spotlight"));
        st = sp->state();
        check(st.open && !st.busy && st.actionRuns == 0 && st.answer.startsWith("It is a pricing page") && b.spotlightOpen(),
              "spotlight: a pure Q&A turn keeps the bar up showing the answer", st.answer);

        // Turn 2 (L2 again while open: focus is already there, no delay) acts on Chrome.
        keys.clear();
        pad->processSnapshot(l2(1.0), t += 20);
        check(keys == QStringList({"down"}) && sp->state().lastQuestion == "summarize this page",
              "spotlight: L2 on the open bar sends the key at once, previous turn still shown", keys.join(' '));
        sp->setInputText(QStringLiteral("click sign in"));
        pad->processSnapshot(S(), t += 20);
        tap([](S &s) { s.r2 = 1.0; });
        check(b.testStreamStarts().size() == 2 && b.testStreamStarts().value(1) == "spotlight|click sign in",
              "spotlight: the second prompt goes out on the same stream", b.testStreamStarts().join(';'));
        b.testStreamDelta(QStringLiteral("spotlight"), QStringLiteral("Clicking it. [[ACTION:cua_click:Sign In]]"));
        b.testStreamFinish(QStringLiteral("spotlight"));
        st = sp->state();
        check(st.actionRuns == 1 && !st.open && !b.spotlightOpen() && b.cuaLog().contains("cua_click:Sign In")
                  && st.lastQuestion == "click sign in" && st.answer == "Clicking it." && st.statusLine.startsWith("Done"),
              "spotlight: a cua_* turn ghosts the bar, runs, then closes keeping the exchange",
              st.statusLine + " | " + b.cuaLog().join(';'));
        check(b.streamHistory(QStringLiteral("spotlight")).size() == 4, "spotlight: both turns are in the stream's Tier 2");

        // Still in Chrome: the next L2 pops the bar back with that exchange in its drawer.
        keys.clear();
        pad->processSnapshot(l2(1.0), t += 20);
        st = sp->state();
        check(b.spotlightOpen() && st.open && st.drawerVisible && st.lastQuestion == "click sign in" && st.opens == 2
                  && keys.isEmpty(),
              "spotlight: the next L2 reopens it with the previous Q&A (key waits for focus)");
        settle(80);
        pad->processSnapshot(S(), t += 20);
        check(keys == QStringList({"down", "up"}), "spotlight: ...and dictation runs normally", keys.join(' '));
        tap([](S &s) { s.b = true; });
        check(!b.spotlightOpen() && !sp->isOpen(), "spotlight: ○ dismisses the bar");

        // Inside Studio: L2 never opens the bar.
        studioActive = true;
        keys.clear();
        pad->processSnapshot(l2(1.0), t += 20);
        check(!b.spotlightOpen() && keys == QStringList({"down"}), "spotlight: L2 inside Studio leaves the bar closed, key at once");
        pad->processSnapshot(S(), t += 20);
    }

    // In-Studio: L2 -> R2 -> answer -> focus moves away -> L2 again refocuses the same card.
    {
        Backend backend(Backend::Mode::Mock);
        backend.setTestStreamCapture(true);
        Gamepad *pad = backend.gamepad();
        QStringList ev;
        pad->setVoiceKeySinkForTest([&](bool down) { ev << (down ? "key:down" : "key:up"); });
        QObject::connect(pad, &Gamepad::voiceHold, pad, [&](bool on) { ev << (on ? "hold:1" : "hold:0"); });
        QQmlApplicationEngine engine;
        QQuickWindow *win = createStudioWindow(engine, &backend, false, QSize(1920, 1080));
        check(win != nullptr, "studio voice: QML loads");
        if (!win)
            return;
        QObject *app = win->findChild<QObject *>("studioApp");
        QQuickItem *omni = firstItem(win, "floatingOmnibar");
        QQuickItem *card = firstItem(win, "omnibarCard");
        QQuickItem *input = firstItem(win, "omnibarInput");
        if (!app || !omni || !card || !input) {
            check(false, "studio voice: omnibar present");
            return;
        }
        settle(450);
        backend.setNavMode(QStringLiteral("controller"));
        pad->setConnected(true, QStringLiteral("DualShock 4"));
        settle(100);
        auto focusName = [&] {
            QQuickItem *f = win->activeFocusItem();
            return f ? f->objectName() : QString();
        };
        qint64 t = 0;
        // Raycast position: opened with no target, the card sits upper centre.
        if (QObject *reg = backend.targets())
            QMetaObject::invokeMethod(reg, "clearHoverTarget", Q_ARG(QString, QString()));
        QMetaObject::invokeMethod(omni, "openAt", Q_ARG(QVariant, -1), Q_ARG(QVariant, -1));
        settle(30);
        check(std::abs(card->x() - (1920 - card->width()) / 2) < 1 && std::abs(card->y() - 173) < 1 && card->width() == 620,
              "studio voice: openAt(-1, -1) with no target sits upper centre (Raycast), 620 wide",
              QStringLiteral("%1,%2 %3").arg(card->x()).arg(card->y()).arg(card->width()));
        QMetaObject::invokeMethod(omni, "close");
        settle(30);

        pad->processSnapshot(l2(1.0), t += 20);
        settle(30);
        QQuickItem *hint = firstItem(win, "omnibarVoiceHint");
        check(omni->property("open").toBool() && focusName() == "omnibarInput" && omni->property("voiceListening").toBool()
                  && hint && hint->isVisible() && hint->property("text").toString().contains("Listening"),
              "studio voice: L2 opens the ask card, prompt focused, Listening pill", focusName());
        check(ev.mid(0, 2) == QStringList({"hold:1", "key:down"}), "studio voice: the card is focused before the fn key",
              ev.join(' '));
        input->setProperty("text", QStringLiteral("what changed in this run"));
        pad->processSnapshot(S(), t += 20);
        settle(20);
        check(hint && hint->property("text").toString() == QStringLiteral("R2 send"), "studio voice: typed text shows the R2 hint",
              hint ? hint->property("text").toString() : QString());
        pad->processSnapshot([] { S s; s.r2 = 1.0; return s; }(), t += 20);
        pad->processSnapshot(S(), t += 20);
        settle(30);
        const QString sid = omni->property("streamId").toString();
        check(backend.testStreamStarts().value(0) == sid + "|what changed in this run" && omni->property("busy").toBool(),
              "studio voice: R2 sends turn 1", backend.testStreamStarts().join(';'));
        backend.testStreamDelta(sid, QStringLiteral("The loss dropped to 2.5."));
        backend.testStreamFinish(sid);
        settle(30);
        check(omni->property("open").toBool() && !omni->property("busy").toBool() && focusName() == "omnibarInput"
                  && omni->property("answer").toString() == "The loss dropped to 2.5.",
              "studio voice: turn 1 answered, card stays open, prompt refocused", focusName());
        QMetaObject::invokeMethod(app, "takeFocus");  // focus wanders off the card
        settle(30);
        check(focusName() != "omnibarInput" && omni->property("open").toBool(), "studio voice: focus moved away, card still open",
              focusName());
        ev.clear();
        pad->processSnapshot(l2(1.0), t += 20);
        settle(30);
        check(omni->property("open").toBool() && focusName() == "omnibarInput"
                  && omni->property("question").toString() == "what changed in this run"
                  && omni->property("answer").toString() == "The loss dropped to 2.5." && ev.value(1) == "key:down",
              "studio voice: the second L2 refocuses the open card, turn 1 still shown (the old !omnibar.open bug)",
              focusName() + " " + ev.join(' '));
        input->setProperty("text", QStringLiteral("and the step before"));
        pad->processSnapshot(S(), t += 20);
        pad->processSnapshot([] { S s; s.r2 = 1.0; return s; }(), t += 20);
        pad->processSnapshot(S(), t += 20);
        settle(30);
        check(backend.testStreamStarts().value(1) == sid + "|and the step before" && omni->property("sends").toInt() == 2,
              "studio voice: R2 sends turn 2 on the same card", backend.testStreamStarts().join(';'));
        check(!backend.spotlightOpen(), "studio voice: the Spotlight bar never opened inside Studio");
        backend.testStreamFinish(sid);
        settle(20);
        QMetaObject::invokeMethod(omni, "close");
        pad->setConnected(false, QString());
        settle(30);
    }
}

// The Jarvis bar (jarvisBarMode "overlay"): L2 hold or tap, auto-send on Phonon's paste, the
// instant interrupt, and the unified overlay over Studio itself. Mock + Offscreen bar: no window,
// no keys posted, no model.
// Bars only for fundamental metrics: disk space, file sizes and counts stay a table.
void chartGateTests()
{
    auto md = [](const QStringList &rows) { return tableFromMarkdown(rows); };
    struct Case {
        const char *name;
        QStringList rows;
    };
    const QList<Case> rejected = {
        {"disk space Mount | Used | Avail", {"| Mount | Used | Avail |", "|---|---|---|", "| / | 609 GB | 261 GB |", "| /data | 1200 GB | 800 GB |"}},
        {"2-row disk Used", {"| Disk | Used |", "|---|---|", "| internal | 609 GB |", "| external | 261 GB |"}},
        {"df Use%", {"| Filesystem | Size | Use% |", "|---|---|---|", "| disk3s1 | 994 | 61% |", "| disk5s2 | 1863 | 14% |"}},
        {"plain-number free space", {"| volume | free |", "|---|---|", "| Macintosh HD | 261 |", "| Backup | 1400 |"}},
        {"file sizes", {"| file | size |", "|---|---|", "| main.cpp | 12 KB |", "| backend.cpp | 240 KB |", "| tests.cpp | 310 KB |"}},
        {"bare byte cells", {"| artifact | on disk |", "|---|---|", "| model.gguf | 4.2G |", "| tokenizer | 12M |"}},
        {"file counts", {"| dir | files |", "|---|---|", "| src | 120 |", "| qml | 45 |"}},
        {"line counts", {"| module | lines |", "|---|---|", "| backend | 6286 |", "| transcript | 2319 |"}},
        {"PIDs", {"| process | PID |", "|---|---|", "| vllm | 4121 |", "| sglang | 5120 |"}},
        {"2-row VRAM is a lookup, not a sweep", {"| Model | VRAM (GB) |", "|---|---|", "| Llama 8B | 18 |", "| Llama 70B | 141 |"}},
    };
    for (const Case &c : rejected) {
        const QVariantMap t = md(c.rows);
        check(!t.isEmpty() && chartFromTable(t).isEmpty(), "chart gate: inventory table stays a table", c.name);
    }
    const QList<Case> charted = {
        {"Config | tok/s", {"| Config | tok/s |", "|---|---|", "| c1 | 126 |", "| c8 | 610 |"}},
        {"Kernel | Speedup (x)", {"| Kernel | Speedup (x) |", "|---|---|", "| naive | 1.00x |", "| tiled | 2.63x |"}},
        {"BW (GB/s)", {"| Copy | BW (GB/s) |", "|---|---|", "| H2D | 25.1 |", "| D2D | 1450 |"}},
        {"disk read MB/s is throughput", {"| Disk | Read |", "|---|---|", "| nvme0 | 7000 MB/s |", "| sata | 550 MB/s |"}},
        {"latency ms", {"| build | latency ms |", "|---|---|", "| a | 142 |", "| b | 38 |"}},
        {"VRAM sweep by model", {"| Model | VRAM (GB) |", "|---|---|", "| Llama 8B | 18 |", "| Qwen 32B | 70 |", "| Llama 70B | 141 |"}},
        {"peak memory by batch", {"| batch | peak mem |", "|---|---|", "| bs 1 | 20 GB |", "| bs 8 | 34 GB |", "| bs 32 | 71 GB |"}},
    };
    for (const Case &c : charted) {
        const QVariantMap ch = chartFromTable(md(c.rows));
        check(ch.value("kind") == "chart" && ch.value("values").toList().size() == c.rows.size() - 2,
              "chart gate: fundamental metric still charts", c.name);
    }
    // End to end: a disk-space write-up yields a table artifact and no chart.
    {
        const QList<Turn> turns = {
            mkTurn("user", "how much disk is left?"),
            mkTurn("assistant", "Here is the current disk usage on both volumes after the cleanup run finished.\n\n"
                                "| Mount | Used | Avail |\n|---|---|---|\n| / | 609 GB | 261 GB |\n| /data | 1.2 TB | 800 GB |\n")};
        const Extracted ex = extractArtifacts(turns, noFiles);
        check(!firstOfKind(ex.artifacts, "table").isEmpty() && firstOfKind(ex.artifacts, "chart").isEmpty(),
              "chart gate: a disk-space answer renders as a table, never a bar chart");
    }
}

// Jarvis zero-stall failover through the Backend: a hung leg 1 hands the same "spotlight"
// stream to leg 2 before any text reaches the bar, trips the 15 s breaker, and the next turn
// goes straight to leg 2.
void jarvisFailoverTests()
{
    check(Backend::jarvisEscalatedEffort("fast", 1, true) == "low" && Backend::jarvisEscalatedEffort("none", 3, true) == "low"
              && Backend::jarvisEscalatedEffort("", 1, true) == "low",
          "jarvis effort: no-think escalates to low after an ERR step on a follow-up");
    check(Backend::jarvisEscalatedEffort("fast", 0, true) == "fast" && Backend::jarvisEscalatedEffort("fast", 2, false) == "fast"
              && Backend::jarvisEscalatedEffort("medium", 2, true) == "medium",
          "jarvis effort: first turn, clean steps and thinking efforts are kept");

    Backend b(Backend::Mode::Mock);
    check(b.jarvisEndpoint().isEmpty() && b.jarvisModel().isEmpty() && b.jarvisLegs().isEmpty(),
          "jarvis legs: nothing configured (mock ignores env) -> the compat path as before");
    QString key;
    for (const QVariant &v : b.projects()) {
        key = v.toMap().value("key").toString();
        if (!key.isEmpty())
            break;
    }
    const QString planner = QDir::homePath() + "/.lapis-next/cua_traces/planner_turns.jsonl";
    const qint64 plannerBefore = QFileInfo(planner).exists() ? QFileInfo(planner).size() : -1;

    FakeSse hung;
    hung.hold = true;
    FakeSse ok;
    ok.answer = QStringLiteral("The run peaked at 126 tok/s.");
    check(hung.start() && ok.start(), "jarvis failover: two local SSE endpoints listen");
    LlmRouter router;
    b.setLlmRouter(&router);
    b.setJarvisLegsForTest(
        {{QStringLiteral("self"), hung.url() + "/chat/completions", QString(), QStringLiteral("Infatoshi/Inkling-2-Mini-heretic"),
          false, 200, 300},
         {QStringLiteral("inkling"), ok.url() + "/chat/completions", QString(), QStringLiteral("thinkingmachines/inkling-2-mini"),
          true, 3000, 2500}});
    QStringList finished;
    QString text;
    QObject::connect(&b, &Backend::streamFinished, [&](const QString &id, const QString &err) {
        if (id == "spotlight")
            finished << err;
    });
    QObject::connect(&b, &Backend::streamDelta, [&](const QString &id, const QString &t, const QString &) {
        if (id == "spotlight")
            text += t;
    });
    b.askStream(QStringLiteral("spotlight"), key, 0, QStringLiteral("what did the run peak at?"));
    const bool done = waitUntil([&] { return !finished.isEmpty(); }, 6000);
    check(done && finished == QStringList{QString()} && text == ok.answer,
          "jarvis failover: hung leg 1 -> leg 2 answers, one clean finish, no error shown", finished.join(';') + " / " + text);
    check(hung.bodies.size() == 1 && ok.bodies.size() == 1 && b.jarvisLastLeg() == "inkling" && b.jarvisTurnSeq() == 1,
          "jarvis failover: leg 1 tried once, leg 2 answered the same turn",
          QStringLiteral("%1/%2 %3").arg(hung.bodies.size()).arg(ok.bodies.size()).arg(b.jarvisLastLeg()));
    const QJsonObject body = QJsonDocument::fromJson(ok.bodies.value(0)).object();
    const QJsonObject hungBody = QJsonDocument::fromJson(hung.bodies.value(0)).object();
    check(body.value("model") == "thinkingmachines/inkling-2-mini"
              && body.value("provider").toObject().value("order").toArray() == QJsonArray{"PrimeIntellect"}
              && hungBody.value("model") == "Infatoshi/Inkling-2-Mini-heretic"
              && hungBody.value("chat_template_kwargs").toObject().value("reasoning_effort").toDouble(-1) == 0.0,
          "jarvis failover: each leg sent its own model and request shape");
    check(b.jarvisBreakerOpen() && b.jarvisLegs().size() == 1 && b.jarvisLegs().value(0).name == "inkling",
          "jarvis failover: leg 1's breaker is open, the next plan skips it");
    b.askStream(QStringLiteral("spotlight"), key, 0, QStringLiteral("and the p50 latency?"));
    check(waitUntil([&] { return finished.size() == 2; }, 6000) && hung.bodies.size() == 1 && ok.bodies.size() == 2
              && b.jarvisTurnSeq() == 2,
          "jarvis failover: while the breaker is open the turn goes straight to leg 2",
          QStringLiteral("%1/%2").arg(hung.bodies.size()).arg(ok.bodies.size()));
    const qint64 plannerAfter = QFileInfo(planner).exists() ? QFileInfo(planner).size() : -1;
    check(plannerBefore == plannerAfter, "jarvis failover: mock mode never writes planner_turns.jsonl");
    b.setLlmRouter(nullptr);
}

// Inkling everywhere: an island card runs on the Jarvis legs, a CUA follow-up's screenshot rides
// on leg 1's leaf only, and no history ever holds the image.
void inklingUniversalTests()
{
    Backend b(Backend::Mode::Mock);
    QString key;
    for (const QVariant &v : b.projects()) {
        key = v.toMap().value("key").toString();
        if (!key.isEmpty())
            break;
    }
    // Quick-ask gating: a CLI pick keeps its channel; a compat / OpenRouter pick rides the legs.
    const QVector<JarvisLeg> testLegs = {{QStringLiteral("self"), QStringLiteral("http://127.0.0.1:9/v1/chat/completions"), QString(),
                                          QStringLiteral("Infatoshi/Inkling-2-Mini-heretic"), false, 200, 1800}};
    check(b.chatLegs("__quick__").isEmpty() && b.chatLegs("card:x").isEmpty(),
          "inkling universal: no jarvisEndpoint (mock) -> every stream keeps its old path");
    b.setJarvisLegsForTest(testLegs);
    b.setChatModel(QStringLiteral("claude-sonnet-5-5"));
    check(b.chatLegs("__quick__").isEmpty() && b.chatLegs("card:x").size() == 1 && b.chatLegs("spotlight").size() == 1,
          "inkling universal: a Claude pick keeps the quick ask on its CLI; cards and Jarvis use the legs");
    b.setChatModel(QStringLiteral("deepseek/deepseek-v4.1-flash-1017"));
    check(b.chatLegs("__quick__").size() == 1, "inkling universal: a DeepSeek (OpenRouter) pick puts the quick ask on the legs");
    b.setChatModel(QStringLiteral("thinkingmachines/inkling-2-mini"));
    check(b.chatLegs("__quick__").size() == 1, "inkling universal: an Inkling pick puts the quick ask on the legs");
    b.setChatModel(QStringLiteral("qwen3.8-flash-next"));
    check(b.chatLegs("__quick__").size() == 1, "inkling universal: a compat pick puts the quick ask on the legs");

    FakeSse self;
    self.answer = QStringLiteral("The pinned post says the run peaked at 126 tok/s.");
    FakeSse backup;
    check(self.start() && backup.start(), "inkling universal: two local SSE endpoints listen");
    LlmRouter router;
    b.setLlmRouter(&router);
    b.setJarvisLegsForTest(
        {{QStringLiteral("self"), self.url() + "/chat/completions", QString(), QStringLiteral("Infatoshi/Inkling-2-Mini-heretic"),
          false, 200, 1800},
         {QStringLiteral("inkling"), backup.url() + "/chat/completions", QString(), QStringLiteral("thinkingmachines/inkling-2-mini"),
          true, 3000, 2500}});
    const QString card = QStringLiteral("card:inkling-universal");
    int finished = 0;
    QStringList errors;
    QObject::connect(&b, &Backend::streamFinished, [&](const QString &id, const QString &err) {
        if (id == card) {
            ++finished;
            errors << err;
        }
    });
    b.askStream(card, key, 0, QStringLiteral("what did the run peak at?"));
    check(waitUntil([&] { return finished == 1; }, 6000) && errors.value(0).isEmpty() && self.bodies.size() == 1
              && backup.bodies.isEmpty(),
          "inkling universal: an island card asks the self-hosted Inkling leg first",
          QStringLiteral("%1/%2 %3").arg(self.bodies.size()).arg(backup.bodies.size()).arg(errors.join(';')));
    const QJsonObject first = QJsonDocument::fromJson(self.bodies.value(0)).object();
    const QJsonArray firstMsgs = first.value("messages").toArray();
    const QJsonObject firstLeaf = firstMsgs.size() >= 2 ? firstMsgs.at(firstMsgs.size() - 2).toObject() : QJsonObject();
    check(first.value("model") == "Infatoshi/Inkling-2-Mini-heretic" && firstLeaf.value("role") == "user"
              && firstLeaf.value("content").isString() && first.value("continue_final_message").toBool()
              && first.value("chat_template_kwargs").toObject().value("reasoning_effort").toDouble(-1) == 0.0,
          "inkling universal: a plain card turn sends Inkling a text-only leaf at 0.0 forced zero-think");

    const QByteArray jpeg("\xff\xd8\xff\xe0 FAKE-SCREENSHOT-BYTES for the leaf", 40);
    const QString b64 = QString::fromLatin1(jpeg.toBase64());
    b.runCuaFollowupForTest(card, key,
                            QStringLiteral("[CUA observation after cua_open:Chrome|https://x.com (OPEN, ok=1, detail=Opened)]:\n- AXLink \"Posts\""),
                            QStringLiteral("[CUA observation after cua_open:Chrome|https://x.com (OPEN, ok=1, detail=Opened)]"), jpeg);
    check(waitUntil([&] { return finished == 2; }, 6000) && self.bodies.size() == 2,
          "inkling universal: the CUA follow-up streams on the same card");
    const QJsonArray fm = QJsonDocument::fromJson(self.bodies.value(1)).object().value("messages").toArray();
    QJsonArray leafParts;
    for (const QJsonValue &m : fm)
        if (m.toObject().value("content").isArray())
            leafParts = m.toObject().value("content").toArray();
    check(leafParts.size() == 2 && leafParts[0].toObject().value("image_url").toObject().value("url") == "data:image/jpeg;base64," + b64
              && leafParts[1].toObject().value("text").toString().contains("CUA observation after cua_open"),
          "inkling universal: the follow-up's leaf carries the screenshot then the observation text");
    {
        // legPrompt: every follow-up drops recall (its question is an AX dump); a card keeps its
        // trunk; a Spotlight desktop step with a screenshot drops the Studio-only tiers too.
        const QVector<JarvisLeg> legs = b.chatLegs(card);
        const TieredPrompt cardStep = b.legPrompt(card, legs, key, QString(), QStringLiteral("obs"), true, jpeg);
        const TieredPrompt spotStep = b.legPrompt(QStringLiteral("spotlight"), legs, key, QString(), QStringLiteral("obs"), true, jpeg);
        const TieredPrompt spotText = b.legPrompt(QStringLiteral("spotlight"), legs, key, QString(), QStringLiteral("obs"), true, {});
        check(cardStep.memory.isEmpty() && !cardStep.trunk.isEmpty() && cardStep.trunk == b.contextTrunk(key)
                  && spotStep.memory.isEmpty() && spotStep.trunk.isEmpty() && spotStep.liveState.isEmpty()
                  && spotStep.phononTail.isEmpty() && spotStep.recentActions.isEmpty() && spotStep.imageJpeg == jpeg
                  && spotText.trunk == b.contextTrunk(key) && spotText.imageJpeg.isEmpty(),
              "inkling universal: lean follow-ups (no recall; a Spotlight visual step drops the Studio tiers)");
    }
    bool clean = !b.sharedHistory().isEmpty() && !b.streamHistory(card).isEmpty();
    for (const ChatTurn &t : b.sharedHistory() + b.streamHistory(card))
        clean = clean && !t.content.contains(b64) && !t.content.contains(QLatin1String("data:image"));
    check(clean, "inkling universal: the screenshot never enters the shared or stream history");

    b.askStream(card, key, 0, QStringLiteral("and the p50?"));
    check(waitUntil([&] { return finished == 3; }, 6000) && self.bodies.size() == 3 && !self.bodies.value(2).contains("data:image"),
          "inkling universal: the next turn sends no image (it was the follow-up's leaf only)");
    b.setLlmRouter(nullptr);
}

void jarvisTests()
{
    using S = GamepadSnapshot;
    using studio::cua::SpotlightVoice;
    auto l2 = [](double v) { S s; s.l2 = v; return s; };

    // Gamepad: hold >= 260 ms stops on release; a quicker tap stays listening until the next L2.
    {
        Gamepad g;
        QStringList ev;
        g.setVoiceKeySinkForTest([&](bool down) { ev << (down ? "down" : "up"); });
        g.processSnapshot(l2(1.0), 0);
        g.processSnapshot(S(), 40);
        check(!g.voiceActive() && ev == QStringList({"down", "up"}), "jarvis pad: tap-toggle is off by default (release stops)",
              ev.join(' '));
        g.setVoiceTapToggle(true);
        ev.clear();
        g.processSnapshot(l2(1.0), 1000);
        g.processSnapshot(S(), 1000 + 300);
        check(!g.voiceActive() && !g.voiceToggled() && ev == QStringList({"down", "up"}),
              "jarvis pad: a 300 ms hold stops on release", ev.join(' '));
        ev.clear();
        int toggles = 0;
        QObject::connect(&g, &Gamepad::voiceToggledChanged, &g, [&] { ++toggles; });
        g.processSnapshot(l2(1.0), 2000);
        g.processSnapshot(S(), 2000 + 120);
        check(g.voiceActive() && g.voiceToggled() && toggles == 1 && ev == QStringList({"down"}),
              "jarvis pad: a 120 ms tap stays listening (toggled), no key-up", ev.join(' '));
        g.processSnapshot(l2(1.0), 4000);
        check(!g.voiceActive() && !g.voiceToggled() && ev == QStringList({"down", "up"}),
              "jarvis pad: the second tap's press stops listening (key-up)", ev.join(' '));
        g.processSnapshot(S(), 4000 + 90);
        check(!g.voiceActive() && ev.size() == 2, "jarvis pad: ...and its release is swallowed (no new hold)");
        g.processSnapshot(l2(1.0), 5000);
        g.processSnapshot(S(), 5000 + 259);
        check(g.voiceToggled(), "jarvis pad: 259 ms is still a tap");
        g.processSnapshot(l2(1.0), 6000);
        g.processSnapshot(S(), 6010);
        ev.clear();
        g.processSnapshot(l2(1.0), 7000);
        g.cancelVoice();  // ○ mid-hold
        check(!g.voiceActive() && ev == QStringList({"down", "up"}), "jarvis pad: cancelVoice mid-hold lifts the key", ev.join(' '));
        g.processSnapshot(S(), 7600);
        check(!g.voiceActive() && !g.voiceToggled() && ev.size() == 2, "jarvis pad: the held L2's release after a cancel is swallowed");
    }

    // Mode plumbing: Mock defaults to split; the pref round-trips through studio_prefs.json.
    {
        QTemporaryDir tmp;
        const QString prefs = tmp.path() + "/studio_prefs.json";
        Backend b(Backend::Mode::Mock);
        check(b.jarvisBarMode() == "split" && !b.jarvisOverlay() && !b.gamepad()->voiceTapToggle(),
              "jarvis mode: Mock defaults to split (the pad ui tests' routing)");
        b.setPrefsPath(prefs);
        b.setJarvisBarMode("bogus");
        check(b.jarvisBarMode() == "split", "jarvis mode: unknown values are ignored");
        b.setJarvisBarMode("overlay");
        check(b.property("jarvisBarMode").toString() == "overlay" && b.gamepad()->voiceTapToggle(),
              "jarvis mode: overlay turns L2 tap-toggle on (Q_PROPERTY)");
        b.savePrefs();
        Backend r(Backend::Mode::Mock);
        r.setPrefsPath(prefs);
        check(r.jarvisBarMode() == "overlay" && r.gamepad()->voiceTapToggle(), "jarvis mode: persisted in studio_prefs.json");
    }

    // Outside Studio, overlay mode: hold -> release auto-sends once the paste lands (no R2).
    {
        Backend b(Backend::Mode::Mock);
        b.setTestStreamCapture(true);
        b.setJarvisBarMode("overlay");
        bool studioActive = false;
        b.setStudioActiveProbeForTest([&] { return studioActive; });
        Gamepad *pad = b.gamepad();
        QStringList keys;
        pad->setVoiceKeySinkForTest([&](bool down) { keys << (down ? "down" : "up"); });
        studio::cua::CuaSpotlight *sp = b.spotlight();
        sp->setFrontmostProbe([](int *pid, QString *name) { *pid = 4242; *name = QStringLiteral("Google Chrome"); });
        qint64 t = 10000;
        pad->processSnapshot(l2(1.0), t);
        settle(80);
        check(b.spotlightOpen() && sp->state().voice == SpotlightVoice::Held && sp->state().contextLabel == "Chrome"
                  && keys == QStringList({"down"}),
              "jarvis send: L2 hold opens the bar over Chrome, [L2] release", keys.join(' '));
        pad->processSnapshot(S(), t += 400);
        check(keys == QStringList({"down", "up"}) && sp->state().voice == SpotlightVoice::Transcribing && b.awaitingAutoSend()
                  && sp->state().hint == "Transcribing…" && b.testStreamStarts().isEmpty(),
              "jarvis send: release lifts fn, shows Transcribing…, arms auto-send");
        settle(150);  // Phonon is still transcribing (~185 ms)
        check(b.testStreamStarts().isEmpty() && b.awaitingAutoSend(), "jarvis send: nothing sent before the paste");
        sp->setInputText(QStringLiteral("summarize this page"));  // Phonon's Cmd+V
        sp->notifyInputEdited();
        settle(20);
        check(b.testStreamStarts().isEmpty(), "jarvis send: the paste is debounced (65 ms), not sent mid-paste");
        settle(120);
        check(b.testStreamStarts().value(0) == "spotlight|summarize this page" && !b.awaitingAutoSend()
                  && sp->state().voice == SpotlightVoice::Idle && sp->state().busy && sp->inputText().isEmpty(),
              "jarvis send: the paste auto-submits after the debounce, zero R2 presses", b.testStreamStarts().join(';'));
        b.testStreamDelta("spotlight", "It is a pricing page.");
        b.testStreamFinish("spotlight");
        check(sp->state().open && sp->state().answer == "It is a pricing page." && sp->state().hint == "○ close",
              "jarvis send: the answer shows in the drawer, badge [○] close");

        // Tap-toggle: quick tap keeps listening; the second tap stops and sends.
        keys.clear();
        pad->processSnapshot(l2(1.0), t += 1000);
        pad->processSnapshot(S(), t += 110);
        settle(20);
        check(pad->voiceToggled() && sp->state().voice == SpotlightVoice::Toggled && keys == QStringList({"down"})
                  && sp->state().hint == "Listening · tap L2 to send" && sp->state().placeholder == "Tap L2 to send…"
                  && !b.awaitingAutoSend(),
              "jarvis toggle: a quick tap stays listening, Listening · tap L2 to send", keys.join(' '));
        settle(400);
        check(sp->state().voice == SpotlightVoice::Toggled && b.testStreamStarts().size() == 1,
              "jarvis toggle: still listening 400 ms later, nothing sent");
        pad->processSnapshot(l2(1.0), t += 3000);
        check(keys == QStringList({"down", "up"}) && b.awaitingAutoSend() && sp->state().voice == SpotlightVoice::Transcribing,
              "jarvis toggle: the second tap stops Phonon and arms auto-send", keys.join(' '));
        pad->processSnapshot(S(), t += 80);
        sp->setInputText(QStringLiteral("and the pricing for teams"));
        sp->notifyInputEdited();
        settle(130);
        check(b.testStreamStarts().value(1) == "spotlight|and the pricing for teams" && keys.size() == 2,
              "jarvis toggle: the paste auto-sends turn 2 (the stop tap's release is swallowed)", b.testStreamStarts().join(';'));
        b.testStreamFinish("spotlight");

        // Text already in the field and no new paste: sent as is after 260 ms.
        pad->processSnapshot(l2(1.0), t += 1000);
        sp->setInputText(QStringLiteral("open the docs"));  // typed while holding
        pad->processSnapshot(S(), t += 400);
        settle(150);
        check(b.testStreamStarts().size() == 2, "jarvis send: typed text waits 260 ms for a paste");
        settle(200);
        check(b.testStreamStarts().value(2) == "spotlight|open the docs", "jarvis send: ...then sends what is there",
              b.testStreamStarts().join(';'));
        b.testStreamFinish("spotlight");

        // Nothing said: the empty bar closes quietly after 1.3 s.
        pad->processSnapshot(l2(1.0), t += 1000);
        pad->processSnapshot(S(), t += 400);
        settle(900);
        check(b.spotlightOpen() && b.awaitingAutoSend(), "jarvis send: an empty field waits for a late paste");
        settle(550);
        check(!b.spotlightOpen() && !b.awaitingAutoSend() && b.testStreamStarts().size() == 3,
              "jarvis send: still empty at 1.3 s, the bar closes without sending");

        // ○ while listening: cancel, draft cleared, nothing sent.
        keys.clear();
        pad->processSnapshot(l2(1.0), t += 1000);
        settle(80);  // the reopened bar takes focus, then the fn key goes down
        pad->processSnapshot(S(), t += 100);  // tap: toggled
        sp->setInputText(QStringLiteral("half a thou"));
        {
            S s;
            s.b = true;
            pad->processSnapshot(s, t += 30);
            pad->processSnapshot(S(), t += 16);
        }
        settle(30);
        check(!b.spotlightOpen() && !pad->voiceActive() && !pad->voiceToggled() && keys == QStringList({"down", "up"})
                  && sp->inputText().isEmpty() && !b.awaitingAutoSend(),
              "jarvis cancel: ○ while toggled stops Phonon, clears the draft, closes", keys.join(' '));
        settle(400);
        check(b.testStreamStarts().size() == 3, "jarvis cancel: nothing was submitted");
    }

    // Instant interrupt: L2 while a turn streams and a cua_* step is in flight.
    {
        Backend b(Backend::Mode::Mock);
        b.setTestStreamCapture(true);
        b.setJarvisBarMode("overlay");
        b.setStudioActiveProbeForTest([] { return false; });
        Gamepad *pad = b.gamepad();
        pad->setVoiceKeySinkForTest([](bool) {});
        studio::cua::CuaSpotlight *sp = b.spotlight();
        sp->setFrontmostProbe([](int *pid, QString *name) { *pid = 4242; *name = QStringLiteral("Google Chrome"); });
        qint64 t = 50000;
        pad->processSnapshot(l2(1.0), t);
        sp->setInputText(QStringLiteral("open my latest pull request"));
        pad->processSnapshot(S(), t += 400);
        sp->notifyInputEdited();
        settle(120);
        check(b.testStreamStarts().value(0) == "spotlight|open my latest pull request" && b.jarvisBusy(),
              "jarvis interrupt: turn 1 is streaming");
        b.testBeginCuaAction();  // a live step mid-glide
        check(!studio::cua::cancelFlag.load() && b.cuaPending() == 1, "jarvis interrupt: a step in flight, no cancel yet");
        pad->processSnapshot(l2(1.0), t += 1000);  // the user cuts in
        check(b.jarvisInterrupts() == 1 && studio::cua::cancelFlag.load() && b.cuaPending() == 1,
              "jarvis interrupt: L2 latches cancelFlag while the step is still running");
        const QVector<ChatTurn> h = b.sharedHistory();
        check(h.size() >= 2 && h[h.size() - 2].content == "open my latest pull request"
                  && h.last().content.contains("Interrupted by user") && h.last().content.contains("superseded"),
              "jarvis interrupt: the interrupted request is in the shared history",
              h.isEmpty() ? QString() : h.last().content);
        check(sp->state().voice == SpotlightVoice::Held && !sp->state().busy && sp->state().open,
              "jarvis interrupt: the bar is listening for the correction");
        b.testFinishCuaAction();  // the aborted step returns
        check(!studio::cua::cancelFlag.load() && b.cuaPending() == 0, "jarvis interrupt: the latch clears once the step returns");
        b.testStreamDelta("spotlight", "late tokens from the cancelled turn");
        check(!sp->state().answer.contains("late tokens"),
              "jarvis interrupt: the cancelled stream's tokens never reach the bar");
        sp->setInputText(QStringLiteral("no, open it in a new tab instead"));
        pad->processSnapshot(S(), t += 400);
        sp->notifyInputEdited();
        settle(120);
        check(b.testStreamStarts().value(1) == "spotlight|no, open it in a new tab instead" && !studio::cua::cancelFlag.load(),
              "jarvis interrupt: the correction goes out as the next turn, cancelFlag clear", b.testStreamStarts().join(';'));
        {
            QStringList tier2;  // tieredPrompt sends m_sharedHistory as the next turn's Tier 2
            for (const ChatTurn &c : b.sharedHistory())
                tier2 << c.content;
            check(tier2.contains("open my latest pull request") && tier2.join('\n').contains("superseded by the next instruction"),
                  "jarvis interrupt: the next turn's Tier 2 carries what was interrupted", tier2.join(" | "));
        }
        b.testStreamFinish("spotlight");
        // Nothing running: L2 interrupts nothing and never latches.
        pad->processSnapshot(l2(1.0), t += 1000);
        check(b.jarvisInterrupts() == 1 && !studio::cua::cancelFlag.load(), "jarvis interrupt: an idle L2 cancels nothing");
        pad->processSnapshot(S(), t += 400);
        settle(10);
        // ○ while a step runs with the bar closed: the same clean cancel.
        b.dismissSpotlight(true);
        b.testBeginCuaAction();
        {
            S s;
            s.b = true;
            pad->processSnapshot(s, t += 1000);
            pad->processSnapshot(S(), t += 16);
        }
        settle(20);
        check(b.jarvisInterrupts() == 2 && studio::cua::cancelFlag.load(), "jarvis interrupt: ○ with the bar closed stops the step");
        b.testFinishCuaAction();
        check(!studio::cua::cancelFlag.load() && !b.jarvisBusy(), "jarvis interrupt: ...and the latch clears again");
        studio::cua::cancelFlag.store(false);
    }

    // Unified overlay over Studio itself: "Lapis" context, Studio + desktop actions in one turn.
    {
        Backend b(Backend::Mode::Mock);
        b.setTestStreamCapture(true);
        b.setJarvisBarMode("overlay");
        b.setStudioActiveProbeForTest([] { return true; });
        Gamepad *pad = b.gamepad();
        pad->setVoiceKeySinkForTest([](bool) {});
        studio::cua::CuaSpotlight *sp = b.spotlight();
        QStringList ui;
        QObject::connect(&b, &Backend::uiActionRequested, &b, [&](const QString &a, const QString &) { ui << a; });
        qint64 t = 90000;
        pad->processSnapshot(l2(1.0), t);
        const studio::cua::SpotlightState st = sp->state();
        check(b.spotlightOpen() && st.overStudio && st.contextLabel.startsWith("Lapis") && st.targetPid == 0,
              "jarvis studio: L2 inside Studio opens the Jarvis overlay with a Lapis pill", st.contextLabel);
        sp->setInputText(QStringLiteral("set the mood and play it in the browser"));
        pad->processSnapshot(S(), t += 400);
        sp->notifyInputEdited();
        settle(120);
        check(b.testStreamStarts().value(0) == "spotlight|set the mood and play it in the browser"
                  && b.testLastSlice().contains("Lapis Studio itself") && b.testLastSlice().contains("cua_*"),
              "jarvis studio: the turn's slice says it is over Studio, Studio + cua actions allowed",
              b.testLastSlice().left(120));
        b.testStreamDelta("spotlight", "On it. [[ACTION:open_music]] [[ACTION:cua_click:Play]]");
        b.testStreamFinish("spotlight");
        check(ui.contains("open_music") && b.cuaLog().contains("cua_click:Play"),
              "jarvis studio: one turn runs a Studio UI action and a desktop cua_* action", ui.join(',') + " | " + b.cuaLog().join(';'));
        // Split mode: L2 inside Studio never opens the bar (the ask card's path).
        b.dismissSpotlight(false);
        b.setJarvisBarMode("split");
        pad->processSnapshot(l2(1.0), t += 1000);
        check(!b.spotlightOpen() && !pad->voiceTapToggle(), "jarvis studio: split mode keeps L2 in Studio on the ask card");
        pad->processSnapshot(S(), t += 400);
    }
}

int runTests()
{
    unitTests();
    polishTests();
    mapUnitTests();
    studioControlTests();
    gamepadUnitTests();
    openAiCompatUnitTests();
    llmRouterBackendTests();
    formFactorTests();
    phase3UnitTests();
    phase3UiTests();
    mapUiTests();
    halfScreenUiTests();
    slowmixUiTests();
    omnibarUiTests();
    memoryTierTests();
    finalWiringTests();
    hotReloadTests();
    uiTests();
    gamepadUiTests();
    voiceSpotlightTests();
    jarvisTests();
    chartGateTests();
    jarvisFailoverTests();
    inklingUniversalTests();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

}  // namespace studio
