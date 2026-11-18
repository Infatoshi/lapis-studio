// --test: unit tests plus offscreen UI tests against the mock backend.
#include "backend.hpp"
#include "demo.hpp"
#include "fixtures.hpp"
#include "gamepad.hpp"
#include "pdfrender.hpp"
#include "transcript.hpp"

#include <QDir>
#include <QFileInfo>
#include <QFontInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSet>
#include <QDateTime>
#include <QTemporaryDir>
#include <QWheelEvent>
#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQuickItem>
#include <QQuickWindow>

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
    }
    // Never a blank island.
    check(elapsedLabel(30 * 1000) == "just now" && elapsedLabel(42 * 60000) == "42m"
              && elapsedLabel((3 * 60 + 12) * 60000LL) == "3h 12m" && elapsedLabel(2 * 3600000LL) == "2h"
              && elapsedLabel((5 * 24 + 3) * 3600000LL) == "5d 3h",
          "status: elapsed labels");
    check(statusMetric("working", (3 * 60 + 12) * 60000LL) == "live · 3h 12m", "status: live metric",
          statusMetric("working", (3 * 60 + 12) * 60000LL));
    check(statusMetric("needs-ok", 42 * 60000) == "live · 42m" && statusMetric("finished", 5 * 3600000LL) == "done · 5h"
              && statusMetric("idle", 0) == "idle",
          "status: done / idle metrics");

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
        b.cycleModel(-1);  // W / k / Up from the first model wraps to the last (qwen3.8-flash-next)
        check(b.chatModel() == "qwen3.8-flash-next" && b.chatEffort() == "high" && b.chatProvider() == "openai-compat",
              "picker: W wraps to Qwen3.8 Flash and clamps xhigh -> high", b.chatModel() + " " + b.chatEffort());
        b.cycleModel(-1);
        check(b.chatModel() == "gemini-4-argon" && b.chatEffort() == "high" && b.chatProvider() == "google",
              "picker: W steps to Gemini 4 Argon", b.chatModel() + " " + b.chatEffort());
        b.cycleModel(-1);
        check(b.chatModel() == "gemini-4-hydrogen" && b.chatEfforts().contains("xhigh"), "picker: hydrogen offers xhigh",
              b.chatEfforts().join(','));
        b.cycleEffort(1);
        check(b.chatEffort() == "xhigh", "picker: D on hydrogen -> xhigh", b.chatEffort());
        for (int i = 0; i < 3; ++i)
            b.cycleModel(-1);
        check(b.chatModel() == "gpt-6-luna" && b.chatEffort() == "high" && b.chatProvider() == "openai",
              "picker: W x3 -> GPT-6 Luna, xhigh clamps to high", b.chatModel() + " " + b.chatEffort());
        b.cycleEffort(-1);
        b.cycleEffort(-1);
        b.cycleEffort(-1);
        check(b.chatEffort() == "low", "picker: A x3 -> low (stops at low)", b.chatEffort());
        b.cycleModel(1);
        b.cycleModel(1);
        check(b.chatModel() == "gemini-4-carbon", "picker: S x2 crosses OpenAI -> Google", b.chatModel());
        int perProvider[4] = {0, 0, 0, 0};
        for (const ModelSpec &m : modelCatalog()) {
            const int idx = m.provider == "anthropic" ? 0
                          : m.provider == "openai"    ? 1
                          : m.provider == "google"    ? 2 : 3;
            perProvider[idx] += 1;
        }
        check(modelCatalog().size() == 12 && perProvider[0] == 4 && perProvider[1] == 3 && perProvider[2] == 4
                  && perProvider[3] == 1,
              "picker: 4 Claude + 3 GPT-6 + 4 Gemini + 1 OpenAI-compat models");
        b.setChatModel("gpt-5");
        check(b.chatModel() == "gemini-4-carbon", "picker: unknown model ignored");
        b.setChatEffort("xhigh");
        check(b.chatEffort() == "low", "picker: an effort the model lacks is ignored", b.chatEffort());
        check(clampEffort("claude-haiku-5-5", "xhigh") == "high" && clampEffort("gpt-6-astra", "xhigh") == "xhigh",
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
        const QByteArray line = codexRpc(7, "turn/start", codexTurnParams("th1", "hi", "gpt-6-luna", "xhigh"));
        check(line.endsWith('\n') && line.count('\n') == 1, "codex: one JSON object per line");
        const QJsonObject o = QJsonDocument::fromJson(line).object();
        const QJsonObject p = o.value("params").toObject();
        check(o.value("id").toInt() == 7 && o.value("method").toString() == "turn/start" && p.value("threadId").toString() == "th1"
                  && p.value("model").toString() == "gpt-6-luna" && p.value("effort").toString() == "high"
                  && p.value("summary").toString() == "concise"
                  && p.value("input").toArray().at(0).toObject().value("text").toString() == "hi",
              "codex: turn/start params (model, clamped effort, summary)", QString::fromUtf8(line));
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
        for (const char *m : {"gpt-6-astra", "gpt-6-sol", "gpt-6-luna"})
            check(channelKeyFor(m, "low") == "openai" && providerFor(m) == "openai",
                  "codex: astra / sol / luna all ride the one app-server", m);
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
    }

    // Honest speed classes in the catalog and picker.
    {
        const QHash<QString, QString> want = {
            {"claude-sonnet-5-5", "Sonnet 5.5 · Fast"}, {"gemini-3.8-flash", "Gemini 3.8 Flash · Fast"},
            {"gemini-4-hydrogen", "Gemini 4 Hydrogen · Deep Think"}, {"gemini-4-carbon", "Gemini 4 Carbon · Teach"},
            {"gpt-6-astra", "GPT-6 Astra · Frontier"}, {"gpt-6-luna", "GPT-6 Luna · Fast"}};
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
        b.mockAppendTurn("kernel-search", t);
        b.mockAppendTurn("kernel-search", t);
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
            r.setChatModel("claude-haiku-5-5");
            check(r.chatEffort() == "high", "prefs: switching to Haiku clamps xhigh -> high", r.chatEffort());
            check(r.savePrefs(), "prefs: save model + effort");
        }
        {
            Backend r(Backend::Mode::Mock);
            r.setPrefsPath(path);
            check(r.chatModel() == "claude-haiku-5-5" && r.chatEffort() == "high", "prefs: default model + effort round trip",
                  r.chatModel() + " " + r.chatEffort());
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
        check(w < 45, "ui: constellation stays sparse", QString::number(w));
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
        // One shared halo clock: every live island's halo pulses in lockstep.
        const auto halos = findItems(win->contentItem(), "islandHalo");
        QList<QQuickItem *> live;
        for (int tries = 0; tries < 20 && live.size() < 2; ++tries) {  // the clock rests 400 ms per beat
            settle(50);
            live.clear();
            for (QQuickItem *h : halos)
                if (h->isVisible())
                    live << h;
        }
        bool lockstep = live.size() >= 2;
        for (QQuickItem *h : live)
            lockstep = lockstep && h->scale() == live.first()->scale();
        check(lockstep, "spring: active islands share one halo pulse clock", QString::number(live.size()));
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
        sendKey(win, Qt::Key_K, Qt::ControlModifier);
        check(backend.navMode() == "vim" && sel() == 0, "nav: Cmd+K -> vim (Neovim)", backend.navMode());
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
        sendKey(win, Qt::Key_K, Qt::ControlModifier);
        check(backend.navMode() == "wasd", "nav: Cmd+K -> wasd (Gaming)", backend.navMode());
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
        sendKey(win, Qt::Key_K, Qt::ControlModifier);
        check(backend.navMode() == "mouse", "nav: Cmd+K -> mouse", backend.navMode());
        letter(Qt::Key_L, "l");
        letter(Qt::Key_D, "d");
        check(sel() == 0, "nav: mouse mode has no letter navigation");
        sendKey(win, Qt::Key_K, Qt::ControlModifier);
        check(backend.navMode() == "controller" && sel() == 0, "nav: Cmd+K wraps back to controller");
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
        check(all.contains("Controller") && all.contains("Neovim") && all.contains("Gaming") && all.contains("Mouse")
                  && !all.contains("arrows", Qt::CaseInsensitive),
              "settings: Mode shows Controller / Neovim / Gaming / Mouse only");
        check(all.contains("UI Scale") && all.contains("1.00x") && all.contains("0.75x – 1.60x") && all.contains("wheel"),
              "settings: UI scale with range and wheel hint");
        check(all.contains("Default Model") && all.contains("Sonnet 5.5") && all.contains("Default Effort"),
              "settings: default model & effort");
        check(all.contains("confirm") && all.contains("settings"),
              "settings: Controller keybind cheat-sheet");
        QStringList pairs;
        check(textOverlaps(win, &pairs) == 0, "settings: no overlapping text", pairs.join("; "));
        check(forbiddenPixels(win->grabWindow()) == 0, "settings: no blue/green pixels");
        sendKey(win, Qt::Key_L, Qt::NoModifier, "l");  // row 0: Mode -> Neovim
        settle(50);
        check(backend.navMode() == "vim", "settings: l on Mode switches to Neovim", backend.navMode());
        texts.clear();
        visibleWords(win, &texts);
        check(texts.join('\n').contains("h j k l"), "settings: cheat-sheet follows the mode");
        sendKey(win, Qt::Key_H, Qt::NoModifier, "h");
        sendKey(win, Qt::Key_J, Qt::NoModifier, "j");  // row 1: UI scale
        sendKey(win, Qt::Key_L, Qt::NoModifier, "l");
        check(qFuzzyCompare(backend.uiScale(), 1.1), "settings: l on UI Scale zooms 10%", QString::number(backend.uiScale()));
        sendKey(win, Qt::Key_H, Qt::NoModifier, "h");
        sendKey(win, Qt::Key_J, Qt::NoModifier, "j");  // row 2: model
        sendKey(win, Qt::Key_L, Qt::NoModifier, "l");
        check(backend.chatModel() == "claude-opus-5-5", "settings: l on Default Model -> Opus 5.5", backend.chatModel());
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

    // Tab predictor on the Constellation: gold ring + "Open Stage ⇥ Tab" on the predicted island.
    {
        settle(700);  // idle: the ring breathes in
        QQuickItem *ring = firstItem(win, "predictorRingShape");
        QQuickItem *badge = firstItem(win, "tabBadgeLabel");
        const QVariantMap p = app->property("predictedAction").toMap();
        const int target = p.value("index").toInt();
        const auto isl = findItems(win->contentItem(), "island");
        check(p.value("actionKind").toString() == "open" && p.value("targetId").toString().startsWith("island:"),
              "predictor: Constellation predicts an island", p.value("targetId").toString());
        check(ring && ring->isVisible() && firstItem(win, "predictorRing")->opacity() > 0.9, "predictor: gold ring visible when idle");
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

    // Stage across chart -> diff -> table -> pdf.
    sendKey(win, Qt::Key_3, Qt::ControlModifier);
    settle(350);
    const char *viewers[4] = {"chartView", "diffView", "tableView", "pdfView"};
    for (int i = 0; i < 4; ++i) {
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
        if (i == 2) {
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

    // Tab predictor on the Stage: rings the next artifact's pager dot ("View Diff" after a chart)
    // and Tab steps there; once everything is seen it suggests a prompt instead.
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
                  && p.value("targetId").toString() == "pager:1",
              "predictor: a chart predicts its diff", p.value("label").toString() + " " + p.value("targetId").toString());
        QQuickItem *ring = firstItem(win, "predictorRingShape");
        check(ring && ring->isVisible() && qAbs(ring->width() - ring->height()) < 0.5 && ring->property("radius").toReal() > ring->width() / 2 - 0.5,
              "predictor: a round gold circle around the pager dot");
        QStringList pairs;
        check(textOverlaps(win, &pairs) == 0, "predictor: Stage badge overlaps no text", pairs.join("; "));
        sendKey(win, Qt::Key_Tab);
        settle(400);
        check(idx() == 1 && firstItem(win, "diffView") && firstItem(win, "diffView")->isVisible(),
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
        sendKey(win, Qt::Key_M, Qt::ControlModifier);
        settle(60);
        check(strip && strip->isVisible() && in && in->hasActiveFocus(), "picker: Cmd+M opens the strip and focuses the composer");
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
        letter(Qt::Key_J, "j");
        letter(Qt::Key_J, "j");
        letter(Qt::Key_J, "j");
        const QString afterJ = backend.chatModel();
        letter(Qt::Key_H, "h");
        const QString afterH = backend.chatEffort();
        sendKey(win, Qt::Key_Down);
        sendKey(win, Qt::Key_Right);
        check(afterS == "claude-opus-5-5" && afterDD == "high" && afterW == "claude-sonnet-5-5" && afterJ == "gpt-6-astra"
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
        sendKey(win, Qt::Key_M, Qt::ControlModifier);
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
        QObject::connect(g, &Gamepad::scrollDelta, g, [this](double d) { evs.append({"scroll", 0, 0, d, now}); });
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
        check(log.names() == "confirm,back,tab,secondary,zoomReset,zoom,step,step,settings,newSession,newSession",
              "pad: ✕ ○ △ □ L3 R3 L1 R1 ☰ share touchpad map once per press", log.names());
        check(log.evs.value(6).a == -1 && log.evs.value(7).a == 1, "pad: L1 steps -1, R1 steps +1");
        log.evs.clear();
        S s;
        s.options = true;
        log.feed(&g, s, t);
        s.touchpad = true;
        log.feed(&g, s, t + 8);
        check(log.count("newSession") == 1, "pad: share + touchpad together toggle the dialog once", log.names());
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

    // Constellation: L1 / R1 and the stick move the island ring, △ accepts the prediction.
    app->setProperty("selectedIsland", 0);
    tap([](S &s) { s.r1 = true; });
    check(prop(app, "selectedIsland").toInt() == 1, "pad ui: R1 selects the next island");
    tap([](S &s) { s.l1 = true; });
    check(prop(app, "selectedIsland").toInt() == 0, "pad ui: L1 selects the previous island");
    tap([](S &s) { s.lx = 1.0; });
    check(prop(app, "selectedIsland").toInt() == 1, "pad ui: left stick right moves one island");
    tap([](S &s) { s.dpadX = -1; });
    check(prop(app, "selectedIsland").toInt() == 0, "pad ui: d-pad left moves back");
    {
        const int before = prop(app, "predictorAccepts").toInt();
        const bool predicted = !prop(app, "predictedAction").isNull();
        tap([](S &s) { s.y = true; });
        settle(400);
        check(predicted && prop(app, "predictorAccepts").toInt() == before + 1 && prop(app, "lens").toInt() > 0,
              "pad ui: △ accepts the Tab prediction on the constellation", QString::number(prop(app, "lens").toInt()));
        tap([](S &s) { s.b = true; });
        settle(400);
        check(prop(app, "lens").toInt() == 0, "pad ui: ○ zooms back out to the constellation");
    }
    // □ on the constellation opens Settings on the Default Model row; ☰ toggles Settings.
    tap([](S &s) { s.x = true; });
    check(prop(settings, "open").toBool() && prop(settings, "row").toInt() == 2, "pad ui: □ on constellation opens model settings");
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
        check(forbiddenPixels(win->grabWindow()) == 0, "pad ui: settings pad section has zero blue/green pixels");
    }
    tap([](S &s) { s.dpadY = -1; });
    check(prop(settings, "row").toInt() == 1, "pad ui: d-pad down moves the settings row");
    {
        const double before = backend.uiScale();
        tap([](S &s) { s.dpadX = 1; });
        check(backend.uiScale() > before, "pad ui: d-pad right changes the UI scale row");
        backend.setUiScale(1.0);
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

    // Share / touchpad: the new-session dialog; stick changes folder / harness; ○ closes.
    tap([](S &s) { s.touchpad = true; });
    check(prop(dialog, "open").toBool(), "pad ui: touchpad opens the new-session dialog");
    tap([](S &s) { s.dpadX = 1; });
    check(prop(dialog, "harnessIdx").toInt() == 1, "pad ui: d-pad right picks the next harness");
    tap([](S &s) { s.r1 = true; });
    check(prop(dialog, "modelIdx").toInt() == 1 || backend.harnessModels().value("codex").toList().size() < 2,
          "pad ui: R1 picks the next model in the dialog");
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
    tap([](S &s) { s.r1 = true; });
    check(arts < 2 || prop(app, "artifactIndex").toInt() == 1, "pad ui: R1 slides to the next artifact");
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

    // ✕ in the Stage focuses the composer; □ there toggles the model picker; stick drives it.
    tap([](S &s) { s.a = true; });
    check(prop(composer, "inputFocused").toBool(), "pad ui: ✕ in the Stage focuses the composer");
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
    tap([](S &s) { s.y = true; });
    check(!prop(composer, "inputText").toString().isEmpty() || prop(app, "promptSuggestion").toString().isEmpty(),
          "pad ui: △ in the composer accepts the suggested prompt", prop(composer, "inputText").toString());
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
}

}  // namespace

int runTests()
{
    unitTests();
    gamepadUnitTests();
    openAiCompatUnitTests();
    uiTests();
    gamepadUiTests();
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

}  // namespace studio
