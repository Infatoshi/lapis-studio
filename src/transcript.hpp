// Pure transcript parsing and artifact extraction. No I/O beyond optional
// file-existence checks, so every function here is unit-testable.
#pragma once

#include <QList>
#include <QString>
#include <QStringList>
#include <QVariantList>
#include <QVariantMap>

namespace studio {

enum class Harness { Claude, Codex, Agy };

Harness harnessFromName(const QString &name);

struct Turn {
    QString role;      // "user" | "assistant"
    QString text;      // prose
    QString thinking;  // reasoning / thinking blocks, when the harness records them
    QStringList patches;  // unified diffs synthesised from edit tools or apply_patch
    QStringList tools;    // one-line tool call summaries: "Bash: cmake --build build", "Read ~/dev/x.cpp"
    QStringList files;    // file paths those tool calls touched
    QString toolOutput;   // tool results (Bash stdout, lapis-viz "wrote /tmp/x.mp4"), scanned for media paths
    bool viz = false;     // a lapis-viz call produced some of that output
};

// Media paths in free text (prose or tool output): ~/ or absolute, ending in a media extension.
QStringList mediaPathsIn(const QString &text);
// "video" | "image" | "svg" | "audio" | "pdf" for a file extension.
QString mediaKindForExt(const QString &ext);

// Parse raw JSONL (possibly cut mid-line by a tail read) into turns, oldest first.
QList<Turn> parseTranscript(const QByteArray &raw, Harness harness);

struct Extracted {
    QVariantList artifacts;   // newest first; each a QVariantMap with a "kind"
    QString headline;         // e.g. "29.6 tok/s · 2.63x"
    QVariantList spark;       // up to 8 normalised bar heights 0..1
    int sparkBest = -1;       // index of the winning bar (lowest when lower is better)
    QString prose;            // latest assistant write-up
    QString thinking;         // latest thinking block
    QString goal;             // first user request
    QStringList assumptions;  // KEY=VALUE tokens the agent states
    QVariantList arms;        // [{name, value, unit, best}]
    QString winner;           // e.g. "128-reg · 29.6 tok/s"
    QStringList recentTurns;  // last 4 turns, "User: ..." / "Assistant: ...", <= 2,500 chars total
    QStringList files;        // files the session touched (tool inputs and diffs), newest last, <= 25
    QStringList toolCalls;    // latest tool call summaries, newest last, <= 15
    // Hyperparameters / configs and key technical terms, sparse (<= 6 params + 3 terms):
    // [{name, value, runs: [..values across runs, oldest first], detail, kind: "param"|"term", mentions}]
    QVariantList params;
    QString jobName;          // 4-7 word plain-English status: "Sweeping sm120 FP4 tile sizes"
    QList<uint> tailSigs;     // signatures of the last 64 turns (fresh-turn counting across rescans)
    int added = 0;            // turns appended since the previous parse of the same transcript
};

// Plain-English job name from the goal plus the latest tool calls and replies (4-7 words when
// there is enough to say, never padded; "" when nothing names the job). `title`: the Lapis tab
// title, used as the subject when it reads as a phrase.
QString synthesizeJobName(const QList<Turn> &turns, const QString &goal, const QString &title = QString());
// A folder or tab name that does not read as English ("cluster-serving", "q3_v2", "a1b2c3").
bool isCrypticName(const QString &name);
// Hyperparameters + key terms from turn text, tool calls and table columns.
QVariantList extractParams(const QList<Turn> &turns, const QVariantList &artifacts);
// One signature per turn (role + text + tools), for counting appended turns.
QList<uint> turnSignatures(const QList<Turn> &turns);
// Turns in `now` after the newest turn `before` also has (0 when nothing is shared).
int newTurnsSince(const QList<uint> &before, const QList<uint> &now);

// pathExists lets tests run without touching the filesystem.
using PathCheck = bool (*)(const QString &);
Extracted extractArtifacts(const QList<Turn> &turns, PathCheck pathExists = nullptr);

// Helpers shared with the UI layer.
QString clampWords(const QString &s, int maxWords);
QString shortModel(const QString &model);
QVariantList diffLines(const QString &diff, int maxLines = 400);
QString highlightCode(const QString &line);
QVariantMap tableFromMarkdown(const QStringList &lines);
QVariantMap tableFromCsv(const QString &csv);
QVariantMap chartFromTable(const QVariantMap &table);
QVariantList audioPeaks(const QString &wavPath, int buckets = 160);
QString stripMarkup(const QString &s);
int countWords(const QString &s);

}  // namespace studio
