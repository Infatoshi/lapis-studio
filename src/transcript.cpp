#include "transcript.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSet>
#include <QtEndian>

#include <algorithm>
#include <functional>
#include <cmath>

namespace studio {

Harness harnessFromName(const QString &name)
{
    if (name == QLatin1String("codex"))
        return Harness::Codex;
    if (name == QLatin1String("agy") || name == QLatin1String("antigravity"))
        return Harness::Agy;
    return Harness::Claude;
}

// --- transcript parsing ------------------------------------------------------

namespace {

void pushTurn(QList<Turn> &turns, Turn t)
{
    t.text = t.text.trimmed();
    t.thinking = t.thinking.trimmed();
    if (t.text.isEmpty() && t.thinking.isEmpty() && t.patches.isEmpty() && t.tools.isEmpty() && t.toolOutput.isEmpty())
        return;
    if (t.role == QLatin1String("user") && t.text.startsWith(QLatin1Char('<')))
        return;  // hook output, command echoes, system reminders
    if (!turns.isEmpty() && turns.last().role == t.role) {
        Turn &prev = turns.last();
        if (!t.text.isEmpty())
            prev.text = prev.text.isEmpty() ? t.text : prev.text + QStringLiteral("\n\n") + t.text;
        if (!t.thinking.isEmpty())
            prev.thinking = prev.thinking.isEmpty() ? t.thinking : prev.thinking + QStringLiteral("\n") + t.thinking;
        prev.patches += t.patches;
        prev.tools += t.tools;
        prev.files += t.files;
        if (!t.toolOutput.isEmpty())
            prev.toolOutput = prev.toolOutput.isEmpty() ? t.toolOutput : prev.toolOutput + QLatin1Char('\n') + t.toolOutput;
        prev.viz = prev.viz || t.viz;
        return;
    }
    turns.append(std::move(t));
}

QString patchFromEdit(const QString &file, const QString &oldText, const QString &newText)
{
    QString out = QStringLiteral("--- a/%1\n+++ b/%1\n@@ edit @@\n").arg(file);
    const QStringList oldLines = oldText.split(QLatin1Char('\n'));
    const QStringList newLines = newText.split(QLatin1Char('\n'));
    // Trim the shared prefix and suffix so the diff shows only what changed, with
    // up to two lines of context on each side.
    qsizetype pre = 0;
    while (pre < oldLines.size() && pre < newLines.size() && oldLines[pre] == newLines[pre])
        ++pre;
    qsizetype suf = 0;
    while (suf < oldLines.size() - pre && suf < newLines.size() - pre
           && oldLines[oldLines.size() - 1 - suf] == newLines[newLines.size() - 1 - suf])
        ++suf;
    const qsizetype ctxStart = std::max<qsizetype>(0, pre - 2);
    for (qsizetype i = ctxStart; i < pre; ++i)
        out += QLatin1Char(' ') + oldLines[i] + QLatin1Char('\n');
    for (qsizetype i = pre; i < oldLines.size() - suf; ++i)
        out += QLatin1Char('-') + oldLines[i] + QLatin1Char('\n');
    for (qsizetype i = pre; i < newLines.size() - suf; ++i)
        out += QLatin1Char('+') + newLines[i] + QLatin1Char('\n');
    const qsizetype ctxEnd = std::min<qsizetype>(oldLines.size(), oldLines.size() - suf + 2);
    for (qsizetype i = oldLines.size() - suf; i < ctxEnd; ++i)
        out += QLatin1Char(' ') + oldLines[i] + QLatin1Char('\n');
    return out;
}

QString patchFromApplyPatch(const QString &input)
{
    QString out;
    static const QRegularExpression fileRe(QStringLiteral("^\\*\\*\\* (Update|Add|Delete) File: (.+)$"));
    for (const QString &line : input.split(QLatin1Char('\n'))) {
        const auto m = fileRe.match(line);
        if (m.hasMatch()) {
            const QString f = m.captured(2).trimmed();
            out += QStringLiteral("--- a/%1\n+++ b/%1\n").arg(f);
            if (m.captured(1) != QLatin1String("Update"))
                out += QStringLiteral("@@ %1 @@\n").arg(m.captured(1).toLower());
            continue;
        }
        if (line.startsWith(QLatin1String("***")))
            continue;
        out += line + QLatin1Char('\n');
    }
    return out;
}

QString oneLine(const QString &s, int maxChars)
{
    QString t = s.simplified();
    if (t.size() > maxChars)
        t = t.left(maxChars - 1) + QChar(0x2026);
    return t;
}

// One line naming what a tool call did, plus the files it touched. Covers Claude Code tool_use
// inputs and codex function_call arguments (command arrays or strings).
QString toolSummary(const QString &name, const QJsonObject &in, QStringList *files)
{
    static const char *const pathKeys[] = {"file_path", "path", "notebook_path"};
    QString target;
    for (const char *k : pathKeys) {
        const QString p = in.value(QLatin1String(k)).toString();
        if (!p.isEmpty()) {
            if (files)
                *files << p;
            if (target.isEmpty())
                target = p;
        }
    }
    QString cmd;
    const QJsonValue c = in.contains(QLatin1String("command")) ? in.value(QLatin1String("command"))
                                                                : in.value(QLatin1String("cmd"));
    if (c.isArray()) {
        QStringList argv;
        for (const QJsonValue &x : c.toArray())
            argv << x.toString();
        // codex shells wrap the real command as ["bash", "-lc", "<cmd>"]
        cmd = argv.size() >= 3 && argv[1].startsWith(QLatin1Char('-')) ? argv.mid(2).join(QLatin1Char(' '))
                                                                       : argv.join(QLatin1Char(' '));
    } else {
        cmd = c.toString();
    }
    if (!cmd.isEmpty())
        return oneLine(name + QStringLiteral(": ") + cmd, 200);
    if (!target.isEmpty())
        return oneLine(name + QLatin1Char(' ') + target, 200);
    for (const char *k : {"pattern", "query", "url", "description", "prompt"}) {
        const QString v = in.value(QLatin1String(k)).toString();
        if (!v.isEmpty())
            return oneLine(name + QStringLiteral(": ") + v, 200);
    }
    return name;
}

QString textOf(const QJsonValue &content, QStringList *thinking = nullptr, QStringList *patches = nullptr,
               QStringList *tools = nullptr, QStringList *files = nullptr)
{
    if (content.isString())
        return content.toString();
    QStringList parts;
    for (const QJsonValue &v : content.toArray()) {
        const QJsonObject c = v.toObject();
        const QString type = c.value(QLatin1String("type")).toString();
        if (type == QLatin1String("text") || type == QLatin1String("input_text")
            || type == QLatin1String("output_text")) {
            parts << c.value(QLatin1String("text")).toString();
        } else if (type == QLatin1String("thinking") && thinking) {
            *thinking << c.value(QLatin1String("thinking")).toString();
        } else if (type == QLatin1String("tool_use") && patches) {
            const QString name = c.value(QLatin1String("name")).toString();
            const QJsonObject in = c.value(QLatin1String("input")).toObject();
            if (tools)
                *tools << toolSummary(name, in, files);
            const QString file = in.value(QLatin1String("file_path")).toString();
            if (name == QLatin1String("Edit")) {
                *patches << patchFromEdit(file, in.value(QLatin1String("old_string")).toString(),
                                          in.value(QLatin1String("new_string")).toString());
            } else if (name == QLatin1String("MultiEdit")) {
                for (const QJsonValue &e : in.value(QLatin1String("edits")).toArray()) {
                    const QJsonObject eo = e.toObject();
                    *patches << patchFromEdit(file, eo.value(QLatin1String("old_string")).toString(),
                                              eo.value(QLatin1String("new_string")).toString());
                }
            }
        }
    }
    return parts.join(QLatin1Char('\n'));
}

QString agyUserText(const QString &content)
{
    static const QRegularExpression req(QStringLiteral("<USER_REQUEST>\\s*(.*?)\\s*</USER_REQUEST>"),
                                        QRegularExpression::DotMatchesEverythingOption);
    const auto m = req.match(content);
    if (m.hasMatch())
        return m.captured(1).trimmed();
    static const QRegularExpression trailers(
        QStringLiteral("\\s*<(ADDITIONAL_METADATA|USER_SETTINGS_CHANGE)>.*?</\\1>\\s*"),
        QRegularExpression::DotMatchesEverythingOption);
    QString s = content;
    s.replace(trailers, QStringLiteral("\n"));
    return s.trimmed();
}

}  // namespace

QList<Turn> parseTranscript(const QByteArray &raw, Harness harness)
{
    QList<Turn> turns;
    QString pendingThinking;  // codex reasoning arrives before the message it explains
    QSet<QString> vizCalls;   // tool call ids whose command ran lapis-viz
    for (const QByteArray &line : raw.split('\n')) {
        if (line.trimmed().isEmpty())
            continue;
        QJsonParseError err{};
        const QJsonDocument doc = QJsonDocument::fromJson(line, &err);
        if (err.error != QJsonParseError::NoError || !doc.isObject())
            continue;  // a tail read cuts the first line
        const QJsonObject rec = doc.object();
        Turn t;
        if (harness == Harness::Agy) {
            const QString kind = rec.value(QLatin1String("type")).toString();
            const QJsonValue content = rec.value(QLatin1String("content"));
            if (!content.isString())
                continue;
            if (kind == QLatin1String("USER_INPUT")) {
                t.role = QStringLiteral("user");
                t.text = agyUserText(content.toString());
                if (t.text.startsWith(QLatin1Char('<')))
                    t.text.clear();
            } else if (kind == QLatin1String("PLANNER_RESPONSE")) {
                t.role = QStringLiteral("assistant");
                t.text = content.toString();
                t.thinking = rec.value(QLatin1String("thinking")).toString();
            } else {
                // Tool steps (commands and their output): only media paths are kept.
                const QStringList media = mediaPathsIn(content.toString());
                if (media.isEmpty())
                    continue;
                t.role = QStringLiteral("assistant");
                t.toolOutput = media.join(QLatin1Char('\n'));
                t.viz = content.toString().contains(QLatin1String("lapis-viz"));
            }
        } else if (harness == Harness::Codex) {
            if (rec.value(QLatin1String("type")).toString() != QLatin1String("response_item"))
                continue;
            const QJsonObject p = rec.value(QLatin1String("payload")).toObject();
            const QString ptype = p.value(QLatin1String("type")).toString();
            if (ptype == QLatin1String("reasoning")) {
                for (const QJsonValue &s : p.value(QLatin1String("summary")).toArray())
                    pendingThinking += s.toObject().value(QLatin1String("text")).toString() + QLatin1Char('\n');
                continue;
            }
            if (ptype == QLatin1String("custom_tool_call")
                && p.value(QLatin1String("name")).toString() == QLatin1String("apply_patch")) {
                t.role = QStringLiteral("assistant");
                t.patches << patchFromApplyPatch(p.value(QLatin1String("input")).toString());
                t.tools << QStringLiteral("apply_patch");
            } else if (ptype == QLatin1String("function_call_output") || ptype == QLatin1String("custom_tool_call_output")) {
                const QJsonValue out = p.value(QLatin1String("output"));
                const QString text = out.isString() ? out.toString()
                                                    : out.toObject().value(QLatin1String("output")).toString();
                const QStringList media = mediaPathsIn(text);
                if (media.isEmpty())
                    continue;
                t.role = QStringLiteral("assistant");
                t.toolOutput = media.join(QLatin1Char('\n'));
                t.viz = vizCalls.contains(p.value(QLatin1String("call_id")).toString());
            } else if (ptype == QLatin1String("function_call") || ptype == QLatin1String("custom_tool_call")) {
                t.role = QStringLiteral("assistant");
                const QString name = p.value(QLatin1String("name")).toString();
                const QJsonValue args = p.value(QLatin1String("arguments"));
                QJsonObject in = args.isObject() ? args.toObject()
                                                 : QJsonDocument::fromJson(args.toString().toUtf8()).object();
                if (in.isEmpty() && p.value(QLatin1String("input")).isString())
                    in.insert(QStringLiteral("cmd"), p.value(QLatin1String("input")).toString());
                t.tools << toolSummary(name, in, &t.files);
                if (t.tools.last().contains(QLatin1String("lapis-viz")))
                    vizCalls.insert(p.value(QLatin1String("call_id")).toString());
            } else if (ptype == QLatin1String("message")) {
                t.role = p.value(QLatin1String("role")).toString();
                t.text = textOf(p.value(QLatin1String("content")));
            } else {
                continue;
            }
            if (t.role == QLatin1String("assistant") && !pendingThinking.isEmpty()) {
                t.thinking = pendingThinking;
                pendingThinking.clear();
            }
        } else {
            const QString type = rec.value(QLatin1String("type")).toString();
            if ((type != QLatin1String("user") && type != QLatin1String("assistant"))
                || rec.value(QLatin1String("isMeta")).toBool())
                continue;
            t.role = type;
            const QJsonValue content = rec.value(QLatin1String("message")).toObject().value(QLatin1String("content"));
            for (const QJsonValue &v : content.toArray()) {
                const QJsonObject c = v.toObject();
                const QString ct = c.value(QLatin1String("type")).toString();
                if (ct == QLatin1String("tool_use")) {
                    if (c.value(QLatin1String("input")).toObject().value(QLatin1String("command")).toString().contains(
                            QLatin1String("lapis-viz")))
                        vizCalls.insert(c.value(QLatin1String("id")).toString());
                } else if (ct == QLatin1String("tool_result")) {
                    // Tool output arrives in a user record; its media paths belong to the assistant
                    // turn that ran the tool (pushTurn merges them into it).
                    const QJsonValue rc = c.value(QLatin1String("content"));
                    QString text = rc.toString();
                    for (const QJsonValue &x : rc.toArray())
                        text += x.toObject().value(QLatin1String("text")).toString() + QLatin1Char('\n');
                    const QStringList media = mediaPathsIn(text);
                    if (!media.isEmpty()) {
                        Turn o;
                        o.role = QStringLiteral("assistant");
                        o.toolOutput = media.join(QLatin1Char('\n'));
                        o.viz = vizCalls.contains(c.value(QLatin1String("tool_use_id")).toString());
                        pushTurn(turns, std::move(o));
                    }
                }
            }
            QStringList thinking;
            QStringList patches;
            t.text = textOf(rec.value(QLatin1String("message")).toObject().value(QLatin1String("content")),
                            &thinking, &patches, &t.tools, &t.files);
            t.thinking = thinking.join(QLatin1Char('\n'));
            t.patches = patches;
        }
        if (t.role != QLatin1String("user") && t.role != QLatin1String("assistant"))
            continue;
        pushTurn(turns, std::move(t));
    }
    return turns;
}

// --- small helpers ------------------------------------------------------------

QString clampWords(const QString &s, int maxWords)
{
    const QStringList words = s.simplified().split(QLatin1Char(' '), Qt::SkipEmptyParts);
    if (words.size() <= maxWords)
        return words.join(QLatin1Char(' '));
    return words.mid(0, maxWords).join(QLatin1Char(' '));
}

QString shortModel(const QString &model)
{
    QString m = model;
    m.remove(QStringLiteral("[1m]"));
    if (m.startsWith(QLatin1String("claude-")))
        m = m.mid(7);
    if (m.startsWith(QLatin1String("gemini-4-")))
        m = m.mid(9);
    static const QRegularExpression ver(QStringLiteral("-(\\d+)-(\\d+)$"));
    m.replace(ver, QStringLiteral("-\\1.\\2"));
    return m;
}

QString stripToolNoise(const QString &s)
{
    if (!s.contains(QLatin1Char('<')) && !s.contains(QLatin1Char('{')) && !s.contains(QLatin1Char('[')))
        return s;  // the common case: plain prose, no regex work at all
    using RE = QRegularExpression;
    static const RE blocks(
        QStringLiteral("<(?:\\w+:)?(tool_calls?|tool_use|tool_result|function_calls?|invoke)\\b[^>]*>"
                       ".*?(</(?:\\w+:)?\\1>|$)"),
        RE::DotMatchesEverythingOption | RE::CaseInsensitiveOption);
    static const RE specialBlocks(
        QStringLiteral("<[|\\x{FF5C}]tool[_\\x{2581}]calls?[_\\x{2581}](section[_\\x{2581}])?begin[|\\x{FF5C}]>"
                       ".*?(<[|\\x{FF5C}]tool[_\\x{2581}]calls?[_\\x{2581}](section[_\\x{2581}])?end[|\\x{FF5C}]>|$)"),
        RE::DotMatchesEverythingOption);
    static const RE specialTokens(QStringLiteral("<[|\\x{FF5C}][^|\\x{FF5C}<>\\n]{1,48}[|\\x{FF5C}]>"));
    static const RE strayTags(QStringLiteral("</?(?:\\w+:)?(tool_calls?|tool_use|tool_result|function_calls?|invoke|parameter)\\b[^>]*>"),
                              RE::CaseInsensitiveOption);
    static const RE jsonCall(QStringLiteral("^\\s*\\{\\s*\"(name|tool|tool_name|function)\"\\s*:.*\"(arguments|parameters|input|args)\"\\s*:"));
    static const RE toolLine(QStringLiteral("^\\s*(\\[(tool|tool_call|tool_use|function_call)\\b[^\\]]*\\]|(tool call|calling tool|tool_call)\\s*[:(])"),
                             RE::CaseInsensitiveOption);
    QString out = s;
    out.remove(blocks);
    out.remove(specialBlocks);
    out.remove(specialTokens);
    out.remove(strayTags);
    QStringList keep;
    for (const QString &l : out.split(QLatin1Char('\n'))) {
        if (jsonCall.match(l).hasMatch() || toolLine.match(l).hasMatch())
            continue;
        keep << l;
    }
    return keep.join(QLatin1Char('\n'));
}

QString stripMarkup(const QString &s)
{
    QString out = s;
    static const QRegularExpression tags(QStringLiteral("<[^>]*>"));
    out.remove(tags);
    out.replace(QStringLiteral("&nbsp;"), QStringLiteral(" "));
    out.replace(QStringLiteral("&lt;"), QStringLiteral("<"));
    out.replace(QStringLiteral("&gt;"), QStringLiteral(">"));
    out.replace(QStringLiteral("&amp;"), QStringLiteral("&"));
    return out;
}

QString stripMarkdownLinks(const QString &s)
{
    if (!s.contains(QLatin1Char('[')) && !s.contains(QLatin1Char('<')))
        return s;
    QString out = s;
    // [label](url "title"), ![alt](url), [label](<path with spaces>); one level of parens in the URL.
    static const QRegularExpression link(QStringLiteral(
        "!?\\[([^\\]\\n]*)\\]\\((?:<[^>\\n]*>|[^()\\s]*(?:\\([^()\\s]*\\)[^()\\s]*)*)(?:\\s+\"[^\"\\n]*\")?\\)"));
    out.replace(link, QStringLiteral("\\1"));
    // A link the line was cut inside: "[app.toml](file:///Use" or "[app.toml](file:///Use…".
    static const QRegularExpression dangling(QStringLiteral("!?\\[([^\\]\\n]*)\\]\\([^)\\s]*$"));
    out.replace(dangling, QStringLiteral("\\1"));
    // Autolinks: <https://x.y>, <file:///a/b>.
    static const QRegularExpression autolink(QStringLiteral("<((?:https?|file)://[^>\\s]*)>"));
    out.replace(autolink, QStringLiteral("\\1"));
    return out;
}

int countWords(const QString &s)
{
    int n = 0;
    static const QRegularExpression alnum(QStringLiteral("[\\p{L}\\p{N}]"));
    for (const QString &w : stripMarkup(s).split(QRegularExpression(QStringLiteral("\\s+")), Qt::SkipEmptyParts))
        if (w.contains(alnum))
            ++n;
    return n;
}

namespace {

QString esc(const QString &s)
{
    QString out;
    out.reserve(s.size() * 1.2);
    for (QChar c : s) {
        switch (c.unicode()) {
        case '&': out += QStringLiteral("&amp;"); break;
        case '<': out += QStringLiteral("&lt;"); break;
        case '>': out += QStringLiteral("&gt;"); break;
        case ' ': out += QStringLiteral("&nbsp;"); break;
        case '\t': out += QStringLiteral("&nbsp;&nbsp;&nbsp;&nbsp;"); break;
        default: out += c;
        }
    }
    return out;
}

QString span(const char *color, const QString &s)
{
    return QStringLiteral("<font color=\"%1\">%2</font>").arg(QLatin1String(color), esc(s));
}

const QSet<QString> &keywords()
{
    static const QSet<QString> k = {
        "auto", "bool", "break", "case", "class", "const", "constexpr", "continue", "def", "do",
        "double", "elif", "else", "enum", "extern", "false", "float", "fn", "for", "from", "half",
        "if", "import", "in", "inline", "int", "let", "mut", "namespace", "nullptr", "pub", "return",
        "self", "static", "struct", "switch", "template", "true", "typename", "uint32_t", "uint8_t",
        "unsigned", "using", "void", "while", "with", "__global__", "__device__", "__shared__",
        "__forceinline__", "__restrict__", "#include", "#define", "#pragma", "None", "True", "False",
        "lambda", "yield", "async", "await", "impl", "match", "use", "int32_t", "int64_t", "size_t"};
    return k;
}

}  // namespace

QString highlightCode(const QString &line)
{
    QString out;
    const qsizetype n = line.size();
    qsizetype i = 0;
    while (i < n) {
        const QChar c = line[i];
        if ((c == QLatin1Char('/') && i + 1 < n && line[i + 1] == QLatin1Char('/'))
            || (c == QLatin1Char('#') && (i + 1 >= n || !line[i + 1].isLetter()))) {
            out += span("#6e665e", line.mid(i));
            break;
        }
        if (c == QLatin1Char('"') || c == QLatin1Char('\'')) {
            qsizetype j = i + 1;
            while (j < n && line[j] != c) {
                if (line[j] == QLatin1Char('\\'))
                    ++j;
                ++j;
            }
            j = std::min(n, j + 1);
            out += span("#c9a27a", line.mid(i, j - i));
            i = j;
            continue;
        }
        if (c.isDigit()) {
            qsizetype j = i;
            while (j < n && (line[j].isLetterOrNumber() || line[j] == QLatin1Char('.')))
                ++j;
            out += span("#d08a5a", line.mid(i, j - i));
            i = j;
            continue;
        }
        if (c.isLetter() || c == QLatin1Char('_') || c == QLatin1Char('#')) {
            qsizetype j = i + 1;
            while (j < n && (line[j].isLetterOrNumber() || line[j] == QLatin1Char('_')))
                ++j;
            const QString word = line.mid(i, j - i);
            if (keywords().contains(word))
                out += span("#e09f3e", word);
            else if (j < n && line[j] == QLatin1Char('('))
                out += span("#e8d5b5", word);
            else
                out += esc(word);
            i = j;
            continue;
        }
        out += esc(QString(c));
        ++i;
    }
    return out;
}

namespace {

// Inline markdown on one line of prose: `code`, **bold**, *italic* / _italic_, [text](url).
// An unmatched trailing backtick (mid-stream) stays literal until its partner arrives.
QString inlineMarkdown(const QString &s, int codePx)
{
    static const QRegularExpression bold(QStringLiteral("\\*\\*(?=\\S)(.+?)(?<=\\S)\\*\\*|__(?=\\S)(.+?)(?<=\\S)__"));
    static const QRegularExpression ital(QStringLiteral("(?<![\\w*])\\*(?=[^\\s*])(.+?)(?<=[^\\s*])\\*(?![\\w*])"
                                                        "|(?<!\\w)_(?=\\S)(.+?)(?<=\\S)_(?!\\w)"));
    static const QRegularExpression link(QStringLiteral("\\[([^\\]]+)\\]\\(([^)\\s]+)\\)"));
    auto prose = [&](const QString &t) {
        QString h = t.toHtmlEscaped();
        h.replace(bold, QStringLiteral("<b>\\1\\2</b>"));
        h.replace(ital, QStringLiteral("<i>\\1\\2</i>"));
        h.replace(link, QStringLiteral("<a href=\"\\2\" style=\"color:#e09f3e; text-decoration:none\">\\1</a>"));
        return h;
    };
    QString out;
    qsizetype i = 0;
    while (i < s.size()) {
        const qsizetype open = s.indexOf(QLatin1Char('`'), i);
        const qsizetype close = open < 0 ? -1 : s.indexOf(QLatin1Char('`'), open + 1);
        if (open < 0 || close < 0) {
            out += prose(s.mid(i));
            break;
        }
        out += prose(s.mid(i, open - i));
        out += QStringLiteral("<span style=\"font-family:Menlo; font-size:%1px; color:#e8d5b5; background-color:#1d1915\">"
                              "&nbsp;%2&nbsp;</span>")
                   .arg(codePx)
                   .arg(s.mid(open + 1, close - open - 1).toHtmlEscaped());
        i = close + 1;
    }
    return out;
}

QStringList tableCells(const QString &line)
{
    QString t = line.trimmed();
    if (t.startsWith(QLatin1Char('|'))) t.remove(0, 1);
    if (t.endsWith(QLatin1Char('|'))) t.chop(1);
    QStringList cells = t.split(QLatin1Char('|'));
    for (QString &c : cells)
        c = c.trimmed();
    return cells;
}

}  // namespace

QString markdownToHtml(const QString &md, int codePx)
{
    static const QRegularExpression fence(QStringLiteral("^\\s*(```|~~~)\\s*([\\w+#.-]*)"));
    static const QRegularExpression heading(QStringLiteral("^(#{1,6})\\s+(.*?)\\s*#*\\s*$"));
    static const QRegularExpression bullet(QStringLiteral("^(\\s*)[-*+]\\s+(.*)$"));
    static const QRegularExpression numbered(QStringLiteral("^(\\s*)(\\d+)[.)]\\s+(.*)$"));
    static const QRegularExpression tableSep(QStringLiteral("^\\s*\\|?\\s*:?-{2,}:?\\s*(\\|\\s*:?-{2,}:?\\s*)*\\|?\\s*$"));
    static const QRegularExpression rule(QStringLiteral("^\\s*([-*_])(\\s*\\1){2,}\\s*$"));
    const QString para = QStringLiteral("<p style=\"margin-top:0px; margin-bottom:8px; line-height:130%\">");

    QString out;
    QStringList paraLines;
    QString listTag;  // "ul" / "ol" while a list is open
    auto flushPara = [&] {
        if (paraLines.isEmpty())
            return;
        out += para + paraLines.join(QStringLiteral("<br>")) + QStringLiteral("</p>");
        paraLines.clear();
    };
    auto closeList = [&] {
        if (listTag.isEmpty())
            return;
        out += QStringLiteral("</%1>").arg(listTag);
        listTag.clear();
    };
    auto openList = [&](const QString &tag, int start) {
        if (listTag == tag)
            return;
        closeList();
        listTag = tag;
        out += QStringLiteral("<%1 style=\"margin-top:0px; margin-bottom:8px; -qt-list-indent:1\"%2>")
                   .arg(tag, start > 1 ? QStringLiteral(" start=\"%1\"").arg(start) : QString());
    };

    const QStringList lines = md.split(QLatin1Char('\n'));
    for (qsizetype i = 0; i < lines.size(); ++i) {
        const QString &line = lines[i];
        const auto fm = fence.match(line);
        if (fm.hasMatch()) {
            flushPara();
            closeList();
            const QString marker = fm.captured(1);
            const QString lang = fm.captured(2);
            QStringList body;
            ++i;
            while (i < lines.size() && !lines[i].trimmed().startsWith(marker))
                body << lines[i++];
            while (!body.isEmpty() && body.last().trimmed().isEmpty())
                body.removeLast();
            QStringList html;
            for (const QString &b : body)
                html << highlightCode(b).replace(QStringLiteral("&nbsp;"), QStringLiteral(" "));
            out += QStringLiteral("<table width=\"100%\" cellspacing=\"0\" cellpadding=\"10\" bgcolor=\"#14120f\" "
                                  "style=\"margin-top:2px; margin-bottom:10px; border-style:none\"><tr><td>");
            if (!lang.isEmpty())
                out += QStringLiteral("<p style=\"margin:0px; font-family:Menlo; font-size:%1px; color:#6e665e\">%2</p>")
                           .arg(std::max(9, codePx - 3))
                           .arg(lang.toHtmlEscaped());
            out += QStringLiteral("<pre style=\"margin:0px; white-space:pre-wrap; font-family:Menlo; font-size:%1px; "
                                  "color:#d9cfc3\">%2</pre></td></tr></table>")
                       .arg(codePx)
                       .arg(html.join(QLatin1Char('\n')));
            continue;
        }
        if (line.trimmed().isEmpty()) {
            flushPara();
            closeList();
            continue;
        }
        if (line.trimmed().startsWith(QLatin1Char('|')) && i + 1 < lines.size() && tableSep.match(lines[i + 1]).hasMatch()) {
            flushPara();
            closeList();
            out += QStringLiteral("<table cellspacing=\"0\" cellpadding=\"5\" style=\"margin-bottom:10px; "
                                  "border-style:solid; border-color:#1d1915\" border=\"1\"><tr>");
            for (const QString &c : tableCells(line))
                out += QStringLiteral("<th align=\"left\" style=\"color:#e8d5b5\">%1</th>").arg(inlineMarkdown(c, codePx));
            out += QStringLiteral("</tr>");
            i += 2;
            for (; i < lines.size() && lines[i].trimmed().startsWith(QLatin1Char('|')); ++i) {
                out += QStringLiteral("<tr>");
                for (const QString &c : tableCells(lines[i]))
                    out += QStringLiteral("<td>%1</td>").arg(inlineMarkdown(c, codePx));
                out += QStringLiteral("</tr>");
            }
            --i;
            out += QStringLiteral("</table>");
            continue;
        }
        const auto hm = heading.match(line);
        if (hm.hasMatch()) {
            flushPara();
            closeList();
            const int level = hm.captured(1).size();
            out += QStringLiteral("<p style=\"margin-top:4px; margin-bottom:6px; color:#e8d5b5; font-size:%1px\"><b>%2</b></p>")
                       .arg(level <= 2 ? codePx + 6 : codePx + 3)
                       .arg(inlineMarkdown(hm.captured(2), codePx));
            continue;
        }
        if (rule.match(line).hasMatch()) {
            flushPara();
            closeList();
            out += QStringLiteral("<hr>");
            continue;
        }
        const auto bm = bullet.match(line);
        const auto nm = numbered.match(line);
        if (bm.hasMatch() || nm.hasMatch()) {
            flushPara();
            if (bm.hasMatch())
                openList(QStringLiteral("ul"), 1);
            else
                openList(QStringLiteral("ol"), nm.captured(2).toInt());
            out += QStringLiteral("<li style=\"line-height:130%\">%1</li>")
                       .arg(inlineMarkdown(bm.hasMatch() ? bm.captured(2) : nm.captured(3), codePx));
            continue;
        }
        if (line.startsWith(QLatin1Char('>'))) {
            flushPara();
            closeList();
            out += QStringLiteral("<p style=\"margin-top:0px; margin-bottom:8px; margin-left:12px; color:#6e665e\"><i>%1</i></p>")
                       .arg(inlineMarkdown(line.mid(1).trimmed(), codePx));
            continue;
        }
        closeList();
        paraLines << inlineMarkdown(line, codePx);
    }
    flushPara();
    closeList();
    // The last block's bottom margin would leave a gap under the answer.
    static const QRegularExpression trailing(QStringLiteral("margin-bottom:(8|10)px([^<]*>)(?!.*margin-bottom)"),
                                             QRegularExpression::DotMatchesEverythingOption);
    out.replace(trailing, QStringLiteral("margin-bottom:0px\\2"));
    return out;
}

QVariantList diffLines(const QString &diff, int maxLines)
{
    QVariantList out;
    int oldNo = 0;
    int newNo = 0;
    static const QRegularExpression hunk(QStringLiteral("^@@ -(\\d+)(?:,\\d+)? \\+(\\d+)"));
    for (const QString &line : diff.split(QLatin1Char('\n'))) {
        if (out.size() >= maxLines)
            break;
        QVariantMap m;
        QString kind;
        QString body = line;
        if (line.startsWith(QLatin1String("+++")) || line.startsWith(QLatin1String("---"))
            || line.startsWith(QLatin1String("diff ")) || line.startsWith(QLatin1String("index "))) {
            if (line.startsWith(QLatin1String("---")) || line.startsWith(QLatin1String("index ")))
                continue;  // one file header line is enough
            kind = QStringLiteral("f");
            body = line.mid(line.startsWith(QLatin1String("+++ b/")) ? 6 : 4);
            m[QStringLiteral("html")] = span("#e8d5b5", body);
        } else if (line.startsWith(QLatin1String("@@"))) {
            kind = QStringLiteral("@");
            const auto hm = hunk.match(line);
            if (hm.hasMatch()) {
                oldNo = hm.captured(1).toInt();
                newNo = hm.captured(2).toInt();
            }
            m[QStringLiteral("html")] = span("#6e665e", line);
        } else {
            QChar marker = line.isEmpty() ? QLatin1Char(' ') : line[0];
            if (marker != QLatin1Char('+') && marker != QLatin1Char('-') && marker != QLatin1Char(' '))
                marker = QLatin1Char(' ');
            else
                body = line.mid(1);
            kind = QString(marker);
            m[QStringLiteral("html")] = highlightCode(body);
            if (marker == QLatin1Char('+')) {
                m[QStringLiteral("no")] = newNo > 0 ? QString::number(newNo) : QString();
                if (newNo > 0) ++newNo;
            } else if (marker == QLatin1Char('-')) {
                m[QStringLiteral("no")] = oldNo > 0 ? QString::number(oldNo) : QString();
                if (oldNo > 0) ++oldNo;
            } else {
                m[QStringLiteral("no")] = newNo > 0 ? QString::number(newNo) : QString();
                if (newNo > 0) ++newNo;
                if (oldNo > 0) ++oldNo;
            }
        }
        m[QStringLiteral("k")] = kind;
        out << m;
    }
    while (!out.isEmpty() && out.last().toMap().value(QStringLiteral("k")).toString() == QLatin1String(" ")
           && stripMarkup(out.last().toMap().value(QStringLiteral("html")).toString()).trimmed().isEmpty())
        out.removeLast();
    return out;
}

// --- tables and charts ----------------------------------------------------------

namespace {

QStringList splitRow(const QString &line)
{
    QString s = line.trimmed();
    if (s.startsWith(QLatin1Char('|')))
        s = s.mid(1);
    if (s.endsWith(QLatin1Char('|')))
        s.chop(1);
    QStringList cells;
    for (const QString &c : s.split(QLatin1Char('|')))
        cells << c.trimmed().remove(QStringLiteral("**")).remove(QLatin1Char('`'));
    return cells;
}

bool parseNumber(const QString &cell, double *out)
{
    static const QRegularExpression num(QStringLiteral("^[~≈+]?(-?\\d[\\d,]*(?:\\.\\d+)?)"));
    const auto m = num.match(cell.trimmed());
    if (!m.hasMatch())
        return false;
    bool ok = false;
    const double v = QString(m.captured(1)).remove(QLatin1Char(',')).toDouble(&ok);
    if (ok && out)
        *out = v;
    return ok;
}

struct Unit {
    QString unit;
    bool lowerBetter = false;
    int score = 0;
};

Unit unitFor(const QString &header, const QStringList &samples)
{
    const QString h = header.toLower();
    Unit u;
    auto sampleHas = [&](const QString &suffix) {
        for (const QString &s : samples)
            if (s.trimmed().endsWith(suffix))
                return true;
        return false;
    };
    if (h.contains(QLatin1String("tok/s")) || h.contains(QLatin1String("tokens/s")) || h.contains(QLatin1String("throughput"))) {
        u = {QStringLiteral("tok/s"), false, 5};
    } else if (h.contains(QLatin1String("speedup")) || h == QLatin1String("x") || h.endsWith(QLatin1String("(x)")) || sampleHas(QStringLiteral("x"))) {
        u = {QStringLiteral("x"), false, 6};
    } else if (h.contains(QLatin1String("tflop")) || h.contains(QLatin1String("gb/s"))) {
        u = {h.contains(QLatin1String("tflop")) ? QStringLiteral("TFLOPS") : QStringLiteral("GB/s"), false, 4};
    } else if (h.contains(QLatin1String("ms")) || h.contains(QLatin1String("latency")) || h.contains(QLatin1String("time"))
               || h.contains(QLatin1String("us")) || sampleHas(QStringLiteral("ms"))) {
        u = {h.contains(QLatin1String("us")) && !h.contains(QLatin1String("ms")) ? QStringLiteral("us") : QStringLiteral("ms"), true, 3};
    } else if (h.contains(QLatin1String("loss")) || h.contains(QLatin1String("error"))) {
        u = {QString(), true, 3};
    } else if (h.contains(QLatin1Char('%')) || sampleHas(QStringLiteral("%"))) {
        u = {QStringLiteral("%"), false, 2};
    } else {
        u = {QString(), false, 1};
    }
    return u;
}

// Bars are for fundamental performance metrics (throughput, speedup, latency, loss, FLOPs, a
// memory scaling sweep). Inventory tables (disk / mount free and used, file sizes, counts of
// files / lines / rows, PIDs, ports) stay a table: comparing two disks as bars is noise.
bool fundamentalChart(const QStringList &header, const QVariantList &rows, int labelCol, int valueCol, const Unit &unit)
{
    const QString vh = header.value(valueCol).toLower();
    const QString lh = header.value(labelCol).toLower();
    QStringList valueCells, labelCells;
    for (const QVariant &r : rows) {
        const QStringList cells = r.toStringList();
        valueCells << cells.value(valueCol).trimmed();
        labelCells << cells.value(labelCol).trimmed();
    }
    // A rate (GB/s, MB/s, GiB/s, tok/s, req/s) is always a throughput metric, even next to disks.
    static const QRegularExpression rate(QStringLiteral("\\b(?:[kmgt]i?b|tok(?:en)?s?|req|it|samples?|img|ops)/s\\b"),
                                         QRegularExpression::CaseInsensitiveOption);
    if (rate.match(vh).hasMatch())
        return true;
    for (const QString &v : valueCells)
        if (rate.match(v).hasMatch())
            return true;
    if (unit.unit == QLatin1String("tok/s") || unit.unit == QLatin1String("x") || unit.unit == QLatin1String("TFLOPS")
        || unit.unit == QLatin1String("GB/s"))
        return true;

    // Storage / inventory vocabulary anywhere in the header row.
    static const QRegularExpression storageHeader(
        QStringLiteral("\\b(?:storage|disk|disks|filesystem|file ?system|fs|mount(?:ed)?(?: on)?|volume|partition|"
                       "capacity|free|used|avail|available|space|file ?size|bytes|inodes?)\\b|\\buse%"),
        QRegularExpression::CaseInsensitiveOption);
    // ...and in the value column's own header: sizes and plain counts.
    static const QRegularExpression countHeader(
        QStringLiteral("^(?:#|n|num|number|count|total)$|\\b(?:size|count|files|lines|loc|rows|entries|items|pid|port|"
                       "kb|mb|gb|gib|mib|tb|tib|kib|bytes)\\b"),
        QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression byteCell(QStringLiteral("^[~≈]?\\d[\\d,]*(?:\\.\\d+)?\\s*(?:[kmgtp]i?b?|bytes?)$"),
                                             QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression memoryHeader(QStringLiteral("\\b(?:vram|memory|mem|hbm|kv ?cache|peak|resident|rss)\\b"),
                                                 QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression sweepAxis(
        QStringLiteral("\\b(?:model|config|configuration|step|steps|batch|bs|seq|sequence|context|ctx|len|length|"
                       "layers?|rank|tp|dp|quant|precision|dtype|variant|arm|run|tile|kernel|size)\\b"),
        QRegularExpression::CaseInsensitiveOption);

    bool byteValues = !valueCells.isEmpty();
    for (const QString &v : valueCells)
        byteValues = byteValues && byteCell.match(v).hasMatch();
    // A memory scaling sweep (VRAM by model / batch / context) is a fundamental metric.
    if (memoryHeader.match(vh).hasMatch() && rows.size() >= 3 && sweepAxis.match(lh).hasMatch())
        return true;
    bool storage = byteValues || countHeader.match(vh).hasMatch();
    for (const QString &h : header)
        storage = storage || storageHeader.match(h).hasMatch();
    // Labels that are paths or devices (/, /home, /dev/disk3s1, ~/data) are a file listing.
    int pathLabels = 0;
    for (const QString &l : labelCells)
        if (l.startsWith(QLatin1Char('/')) || l.startsWith(QLatin1String("~/")) || l.startsWith(QLatin1String("./")))
            ++pathLabels;
    storage = storage || pathLabels * 2 > labelCells.size();
    if (!storage)
        return true;
    // Inventory vocabulary elsewhere in the row next to a real latency / loss column: still a metric.
    // (unitFor's "us" also fires on "used", so the value header itself must name the metric.)
    static const QRegularExpression latencyHeader(QStringLiteral("\\b(?:ms|us|µs|latency|ttft|p50|p90|p99|loss)\\b"),
                                                  QRegularExpression::CaseInsensitiveOption);
    return !byteValues && !countHeader.match(vh).hasMatch() && !storageHeader.match(vh).hasMatch()
           && latencyHeader.match(vh).hasMatch();
}

QString fmt(double v)
{
    if (std::fabs(v) >= 100)
        return QString::number(v, 'f', 0);
    if (std::fabs(v) >= 10)
        return QString::number(v, 'f', 1);
    return QString::number(v, 'f', 2);
}

QString withUnit(double v, const QString &unit)
{
    if (unit == QLatin1String("x"))
        return QString::number(v, 'f', 2) + unit;
    const QString n = (v == std::floor(v) && std::fabs(v) < 1e12) ? QString::number(qint64(v)) : fmt(v);
    if (unit == QLatin1String("%"))
        return n + unit;
    return unit.isEmpty() ? n : n + QLatin1Char(' ') + unit;
}

}  // namespace

QVariantMap tableFromMarkdown(const QStringList &lines)
{
    QVariantMap t;
    if (lines.size() < 3)
        return t;
    const QStringList header = splitRow(lines[0]);
    QVariantList rows;
    static const QRegularExpression sep(QStringLiteral("^[\\s|:\\-]+$"));
    for (qsizetype i = 1; i < lines.size(); ++i) {
        if (sep.match(lines[i]).hasMatch())
            continue;
        QStringList r = splitRow(lines[i]);
        while (r.size() < header.size())
            r << QString();
        rows << QVariant(r.mid(0, header.size()));
    }
    if (rows.isEmpty())
        return t;
    t[QStringLiteral("kind")] = QStringLiteral("table");
    t[QStringLiteral("header")] = header;
    t[QStringLiteral("rows")] = rows;
    return t;
}

QVariantMap tableFromCsv(const QString &csv)
{
    QStringList lines;
    for (const QString &l : csv.split(QLatin1Char('\n')))
        if (!l.trimmed().isEmpty())
            lines << l;
    QVariantMap t;
    if (lines.size() < 2)
        return t;
    auto cells = [](const QString &l) {
        QStringList out;
        for (const QString &c : l.split(QLatin1Char(',')))
            out << c.trimmed().remove(QLatin1Char('"'));
        return out;
    };
    const QStringList header = cells(lines[0]);
    QVariantList rows;
    for (qsizetype i = 1; i < lines.size(); ++i) {
        QStringList r = cells(lines[i]);
        while (r.size() < header.size())
            r << QString();
        rows << QVariant(r.mid(0, header.size()));
    }
    t[QStringLiteral("kind")] = QStringLiteral("table");
    t[QStringLiteral("header")] = header;
    t[QStringLiteral("rows")] = rows;
    return t;
}

QVariantMap chartFromTable(const QVariantMap &table)
{
    QVariantMap c;
    const QStringList header = table.value(QStringLiteral("header")).toStringList();
    const QVariantList rows = table.value(QStringLiteral("rows")).toList();
    if (header.size() < 2 || rows.size() < 2 || rows.size() > 400)
        return c;
    int labelCol = -1;
    int bestCol = -1;
    Unit bestUnit;
    for (int col = 0; col < header.size(); ++col) {
        QStringList samples;
        bool numeric = true;
        for (const QVariant &r : rows) {
            const QString cell = r.toStringList().value(col);
            samples << cell;
            if (!parseNumber(cell, nullptr))
                numeric = false;
        }
        if (!numeric) {
            if (labelCol < 0)
                labelCol = col;
            continue;
        }
        const Unit u = unitFor(header[col], samples);
        if (bestCol < 0 || u.score > bestUnit.score || (u.score == bestUnit.score && col > bestCol)) {
            bestCol = col;
            bestUnit = u;
        }
    }
    bool line = false;
    if (labelCol < 0 && bestCol > 0 && rows.size() >= 6) {
        line = true;  // numeric x (step, seq len) against a numeric y
        labelCol = 0;
    }
    if (labelCol < 0 || bestCol < 0 || (!line && rows.size() > 24))
        return c;
    if (!fundamentalChart(header, rows, labelCol, bestCol, bestUnit))
        return c;  // disk space, file sizes, counts: a plain table, never bars
    QStringList labels;
    QVariantList values;
    double best = 0;
    int bestIdx = 0;
    for (int i = 0; i < rows.size(); ++i) {
        const QStringList r = rows[i].toStringList();
        double v = 0;
        parseNumber(r.value(bestCol), &v);
        labels << clampWords(r.value(labelCol), 2);
        values << v;
        if (i == 0 || (bestUnit.lowerBetter ? v < best : v > best)) {
            best = v;
            bestIdx = i;
        }
    }
    if (!line && QSet<QString>(labels.begin(), labels.end()).size() != labels.size())
        return {};  // repeated labels (a sweep): the table says it better than bars
    // A chart needs at least two distinct measured points; identical values are filler, not a comparison.
    {
        QSet<double> distinct;
        for (const QVariant &v : values)
            distinct.insert(v.toDouble());
        if (distinct.size() < 2)
            return {};
    }
    QStringList valueText;
    for (const QVariant &v : values)
        valueText << withUnit(v.toDouble(), bestUnit.unit == QLatin1String("x") ? QStringLiteral("x") : QString());
    c[QStringLiteral("kind")] = line ? QStringLiteral("line") : QStringLiteral("chart");
    c[QStringLiteral("labels")] = labels;
    c[QStringLiteral("values")] = values;
    c[QStringLiteral("valueText")] = valueText;
    c[QStringLiteral("unit")] = bestUnit.unit;
    c[QStringLiteral("best")] = bestIdx;
    c[QStringLiteral("lowerBetter")] = bestUnit.lowerBetter;
    c[QStringLiteral("metric")] = withUnit(best, bestUnit.unit);
    QString title = clampWords(header[bestCol] + (line ? QStringLiteral(" over ") : QStringLiteral(" by ")) + header[labelCol], 5);
    if (!title.isEmpty())
        title[0] = title[0].toUpper();
    c[QStringLiteral("title")] = title;
    if (line)
        c[QStringLiteral("metric")] = withUnit(values.isEmpty() ? 0.0 : values.last().toDouble(), bestUnit.unit);
    c[QStringLiteral("winnerLabel")] = labels.value(bestIdx);
    c[QStringLiteral("valueLabel")] = header[bestCol];
    return c;
}

QVariantList audioPeaks(const QString &wavPath, int buckets)
{
    QVariantList out;
    QFile f(wavPath);
    if (!f.open(QIODevice::ReadOnly))
        return out;
    const QByteArray d = f.read(64 * 1024 * 1024);
    if (d.size() < 44 || !d.startsWith("RIFF") || d.mid(8, 4) != "WAVE")
        return out;
    int channels = 1;
    int bits = 16;
    qsizetype pos = 12;
    qsizetype dataPos = -1;
    qsizetype dataLen = 0;
    while (pos + 8 <= d.size()) {
        const QByteArray id = d.mid(pos, 4);
        const quint32 len = qFromLittleEndian<quint32>(d.constData() + pos + 4);
        if (id == "fmt " && pos + 24 <= d.size()) {
            channels = qFromLittleEndian<quint16>(d.constData() + pos + 10);
            bits = qFromLittleEndian<quint16>(d.constData() + pos + 22);
        } else if (id == "data") {
            dataPos = pos + 8;
            dataLen = std::min<qsizetype>(len, d.size() - dataPos);
            break;
        }
        pos += 8 + len + (len & 1);
    }
    if (dataPos < 0 || bits != 16 || channels < 1)
        return out;
    const qsizetype frames = dataLen / (2 * channels);
    if (frames < buckets)
        return out;
    const auto *s = reinterpret_cast<const qint16 *>(d.constData() + dataPos);
    double peakAll = 1;
    QList<double> peaks(buckets, 0.0);
    for (int b = 0; b < buckets; ++b) {
        const qsizetype a = frames * b / buckets;
        const qsizetype e = frames * (b + 1) / buckets;
        double m = 0;
        for (qsizetype i = a; i < e; ++i)
            m = std::max(m, std::fabs(double(qFromLittleEndian(s[i * channels]))));
        peaks[b] = m;
        peakAll = std::max(peakAll, m);
    }
    for (double p : peaks)
        out << p / peakAll;
    return out;
}

// --- artifact extraction ---------------------------------------------------------

namespace {

bool defaultExists(const QString &p)
{
    return QFileInfo::exists(p);
}

QString expandHome(const QString &p)
{
    if (p.startsWith(QLatin1String("~/")))
        return QDir::homePath() + p.mid(1);
    return p;
}

QString mediaKind(const QString &ext)
{
    const QString e = ext.toLower();
    if (e == QLatin1String("pdf")) return QStringLiteral("pdf");
    if (e == QLatin1String("svg")) return QStringLiteral("svg");
    if (e == QLatin1String("mp4") || e == QLatin1String("mov") || e == QLatin1String("webm")) return QStringLiteral("video");
    if (e == QLatin1String("wav") || e == QLatin1String("mp3")) return QStringLiteral("audio");
    return QStringLiteral("image");
}

const QRegularExpression &mediaPathRe()
{
    static const QRegularExpression re(
        QStringLiteral("(?<![\\w/])((?:~|/)[^\\s`'\"()<>\\[\\]*]+\\.(png|jpe?g|gif|svg|mp4|mov|webm|wav|mp3|pdf))(?![\\w])"),
        QRegularExpression::CaseInsensitiveOption);
    return re;
}

bool looksLikeDiff(const QString &body)
{
    int plus = 0;
    int minus = 0;
    bool header = false;
    for (const QString &l : body.split(QLatin1Char('\n'))) {
        if (l.startsWith(QLatin1String("@@")) || l.startsWith(QLatin1String("--- a/")) || l.startsWith(QLatin1String("diff --git")))
            header = true;
        else if (l.startsWith(QLatin1Char('+')))
            ++plus;
        else if (l.startsWith(QLatin1Char('-')))
            ++minus;
    }
    return header && plus + minus > 0;
}

QString diffTitle(const QString &diff)
{
    static const QRegularExpression f(QStringLiteral("^\\+\\+\\+ (?:b/)?(.+)$"), QRegularExpression::MultilineOption);
    const auto m = f.match(diff);
    const QString name = m.hasMatch() ? QFileInfo(m.captured(1).trimmed()).fileName() : QStringLiteral("patch");
    return clampWords(name + QStringLiteral(" diff"), 5);
}

QString diffMetric(const QString &diff)
{
    int plus = 0;
    int minus = 0;
    for (const QString &l : diff.split(QLatin1Char('\n'))) {
        if (l.startsWith(QLatin1String("+++")) || l.startsWith(QLatin1String("---")))
            continue;
        if (l.startsWith(QLatin1Char('+'))) ++plus;
        else if (l.startsWith(QLatin1Char('-'))) ++minus;
    }
    return QStringLiteral("+%1 −%2").arg(plus).arg(minus);
}

}  // namespace

QStringList mediaPathsIn(const QString &text)
{
    QStringList out;
    if (text.isEmpty())
        return out;
    auto it = mediaPathRe().globalMatch(text);
    while (it.hasNext()) {
        const QString p = it.next().captured(1);
        if (!out.contains(p))
            out << p;
    }
    return out;
}

QString mediaKindForExt(const QString &ext) { return mediaKind(ext); }

QStringList distillKeyPoints(const QString &prose, int maxPoints)
{
    QStringList bullets;
    QStringList paragraphs;
    QString para;
    bool inFence = false;
    static const QRegularExpression bulletRe(QStringLiteral("^\\s*(?:[-*•]|\\d+[.)])\\s+(.+)$"));
    static const QRegularExpression emph(QStringLiteral("\\*\\*|__|`"));
    auto clean = [](QString s) {
        s.remove(emph);
        return stripMarkup(s).simplified();
    };
    auto endPara = [&]() {
        if (!para.trimmed().isEmpty())
            paragraphs << clean(para);
        para.clear();
    };
    for (const QString &raw : prose.split(QLatin1Char('\n'))) {
        const QString l = raw.trimmed();
        if (l.startsWith(QLatin1String("```"))) {
            inFence = !inFence;
            endPara();
            continue;
        }
        if (inFence || l.startsWith(QLatin1Char('|')) || l.startsWith(QLatin1Char('#'))) {
            endPara();
            continue;
        }
        if (l.isEmpty()) {
            endPara();
            continue;
        }
        const auto m = bulletRe.match(l);
        if (m.hasMatch()) {
            endPara();
            const QString b = clean(m.captured(1));
            if (!b.isEmpty())
                bullets << b;
            continue;
        }
        para += QLatin1Char(' ') + l;
    }
    endPara();
    QStringList points;
    if (bullets.size() >= 2) {
        points = bullets.mid(0, maxPoints);
    } else {
        // Sentences from the opening paragraphs, plus a lone bullet if there was one.
        static const QRegularExpression sentenceEnd(QStringLiteral("(?<=[.!?])\\s+(?=[A-Z0-9\"'(])"));
        for (const QString &p : std::as_const(paragraphs)) {
            for (const QString &s : p.split(sentenceEnd, Qt::SkipEmptyParts)) {
                if (points.size() >= maxPoints)
                    break;
                if (countWords(s) >= 3)
                    points << s.trimmed();
            }
        }
        points += bullets;
        points = points.mid(0, maxPoints);
    }
    for (QString &p : points)
        if (p.size() > 240)
            p = p.left(239).trimmed() + QChar(0x2026);
    return points;
}

Extracted extractArtifacts(const QList<Turn> &turns, PathCheck pathExists)
{
    if (!pathExists)
        pathExists = defaultExists;
    Extracted ex;
    QSet<QString> seen;
    QVariantList turnArts;
    bool turnHasChart = false;
    auto add = [&](QVariantMap a, const QString &key) {
        if (seen.contains(key) || ex.artifacts.size() + turnArts.size() >= 12)
            return;
        const bool chart = a.value(QStringLiteral("kind")).toString() == QLatin1String("chart")
            || a.value(QStringLiteral("kind")).toString() == QLatin1String("line");
        if (chart && turnHasChart)
            return;  // one chart per turn; the table still carries the rest
        turnHasChart = turnHasChart || chart;
        seen.insert(key);
        turnArts << a;
    };
    auto flushTurn = [&]() {
        static const QStringList order = {QStringLiteral("chart"), QStringLiteral("line"), QStringLiteral("diff"),
                                          QStringLiteral("table")};
        auto rank = [&](const QVariant &v) {
            const qsizetype i = order.indexOf(v.toMap().value(QStringLiteral("kind")).toString());
            return i < 0 ? order.size() : i;
        };
        std::stable_sort(turnArts.begin(), turnArts.end(),
                         [&](const QVariant &a, const QVariant &b) { return rank(a) < rank(b); });
        // Tables from the same turn as a bar chart highlight the chart's winner rows.
        QString winnerLabel;
        for (const QVariant &v : turnArts)
            if (v.toMap().value(QStringLiteral("kind")).toString() == QLatin1String("chart"))
                winnerLabel = v.toMap().value(QStringLiteral("winnerLabel")).toString();
        if (!winnerLabel.isEmpty())
            for (QVariant &v : turnArts) {
                QVariantMap m = v.toMap();
                if (m.value(QStringLiteral("kind")).toString() == QLatin1String("table") && !m.contains(QStringLiteral("highlight"))) {
                    m[QStringLiteral("highlightLabel")] = winnerLabel;
                    v = m;
                }
            }
        ex.artifacts += turnArts;
        turnArts.clear();
        turnHasChart = false;
    };
    // A table becomes a chart when it can; small charted tables are not repeated.
    auto addTable = [&](QVariantMap tbl, const QString &key, const QString &fallbackTitle) {
        if (tbl.isEmpty())
            return;
        const QVariantMap chart = chartFromTable(tbl);
        const int rows = int(tbl.value(QStringLiteral("rows")).toList().size());
        tbl[QStringLiteral("title")] = chart.isEmpty() ? fallbackTitle : chart.value(QStringLiteral("title"));
        tbl[QStringLiteral("metric")] = QStringLiteral("%1 rows").arg(rows);
        bool charted = false;
        if (!chart.isEmpty()) {
            tbl[QStringLiteral("highlight")] = chart.value(QStringLiteral("best"));
            const qsizetype before = turnArts.size();
            add(chart, QStringLiteral("chart:") + key);
            charted = turnArts.size() > before;
        }
        if (!charted || rows > 6)
            add(tbl, QStringLiteral("table:") + key);
    };

    static const QRegularExpression fence(QStringLiteral("```([\\w+-]*)[^\\n]*\\n(.*?)```"),
                                          QRegularExpression::DotMatchesEverythingOption);
    const QRegularExpression &pathRe = mediaPathRe();
    static const QRegularExpression throughputRe(
        QStringLiteral("(\\d+(?:\\.\\d+)?)\\s?(tok/s|tokens/s|TFLOPS|GB/s)"), QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression speedupRe(QStringLiteral("(?<![\\w.])(\\d+(?:\\.\\d+)?)x(?![\\w])"));
    static const QRegularExpression assumeRe(QStringLiteral("(?<![\\w-])([A-Z][A-Z0-9_]{1,15})=([\\w.]{1,12})(?![\\w=])"));

    QString throughput;
    QString speedup;
    QStringList assumptionKeys;

    // Goal: the first prompt a person typed, not compaction summaries or peer relays.
    static const QStringList machinePrefixes = {
        QStringLiteral("This session is being continued"), QStringLiteral("[lapis-peers"),
        QStringLiteral("Caveat:"), QStringLiteral("[Request interrupted"), QStringLiteral("Continue from where"),
        QStringLiteral("<")};
    for (const Turn &t : turns) {
        if (t.role != QLatin1String("user") || !ex.goal.isEmpty())
            continue;
        const QString first = t.text.trimmed().section(QLatin1Char('\n'), 0, 0);
        if (first.isEmpty() || std::any_of(machinePrefixes.begin(), machinePrefixes.end(),
                                           [&](const QString &p) { return first.startsWith(p); }))
            continue;
        ex.goal = clampWords(first, 7);
    }

    // Artifacts [proseStart, proseEnd) came from the newest assistant turn that wrote prose.
    qsizetype proseStart = -1;
    qsizetype proseEnd = -1;
    int scanned = 0;
    for (qsizetype ti = turns.size() - 1; ti >= 0 && scanned < 60; --ti) {
        const Turn &t = turns[ti];
        ++scanned;
        // Assumptions come from the agent or the user (e.g. "DCP=4 MTP=3").
        auto am = assumeRe.globalMatch(t.text);
        while (am.hasNext() && assumptionKeys.size() < 4) {
            const auto m = am.next();
            if (!assumptionKeys.contains(m.captured(1))) {
                assumptionKeys << m.captured(1);
                ex.assumptions << m.captured(0);
            }
        }
        if (t.role != QLatin1String("assistant"))
            continue;
        if (proseStart >= 0 && proseEnd < 0)
            proseEnd = ex.artifacts.size();  // the newest written turn has flushed
        struct Flush {
            std::function<void()> f;
            ~Flush() { f(); }
        } flushAtEnd{flushTurn};
        if (ex.prose.isEmpty() && !t.text.isEmpty()) {
            ex.prose = t.text;
            proseStart = ex.artifacts.size();
        }
        if (ex.thinking.isEmpty() && !t.thinking.isEmpty())
            ex.thinking = t.thinking;

        if (throughput.isEmpty()) {
            const auto m = throughputRe.match(t.text);
            if (m.hasMatch())
                throughput = m.captured(1) + QLatin1Char(' ') + m.captured(2);
        }
        if (speedup.isEmpty()) {
            const auto m = speedupRe.match(t.text);
            if (m.hasMatch())
                speedup = m.captured(1) + QLatin1Char('x');
        }

        // Fenced blocks: diffs and CSVs.
        QString rest = t.text;
        auto fm = fence.globalMatch(t.text);
        while (fm.hasNext()) {
            const auto m = fm.next();
            const QString lang = m.captured(1).toLower();
            const QString body = m.captured(2);
            if (lang == QLatin1String("diff") || lang == QLatin1String("patch") || looksLikeDiff(body)) {
                QVariantMap a;
                a[QStringLiteral("kind")] = QStringLiteral("diff");
                a[QStringLiteral("title")] = diffTitle(body);
                a[QStringLiteral("metric")] = diffMetric(body);
                a[QStringLiteral("lines")] = diffLines(body);
                add(a, QStringLiteral("diff:") + body.left(200));
            } else if (lang == QLatin1String("csv")) {
                addTable(tableFromCsv(body), body.left(200), QStringLiteral("Sweep results"));
            }
            rest.replace(m.captured(0), QString());
        }

        // Markdown tables outside fences.
        QStringList block;
        auto flushTable = [&]() {
            if (block.size() >= 3) {
                const QVariantMap tbl = tableFromMarkdown(block);
                addTable(tbl, block.join(QLatin1Char('\n')).left(200),
                         clampWords(tbl.value(QStringLiteral("header")).toStringList().join(QLatin1Char(' ')), 4));
            }
            block.clear();
        };
        for (const QString &line : rest.split(QLatin1Char('\n'))) {
            if (line.trimmed().startsWith(QLatin1Char('|')))
                block << line;
            else
                flushTable();
        }
        flushTable();

        for (const QString &p : t.patches) {
            QVariantMap a;
            a[QStringLiteral("kind")] = QStringLiteral("diff");
            a[QStringLiteral("title")] = diffTitle(p);
            a[QStringLiteral("metric")] = diffMetric(p);
            a[QStringLiteral("lines")] = diffLines(p);
            add(a, QStringLiteral("diff:") + p.left(200));
        }

        // Media named in the prose or printed by a tool (lapis-viz writes an .mp4 or .png and prints
        // its path). Tool output comes first: it is what the turn actually produced.
        for (const QString *srcp : {&t.toolOutput, &t.text}) {
            const QString &src = *srcp;
            auto pm = pathRe.globalMatch(src);
            while (pm.hasNext()) {
                const auto m = pm.next();
                const QString path = expandHome(m.captured(1));
                if (seen.contains(QStringLiteral("file:") + path) || !pathExists(path))
                    continue;
                QVariantMap a;
                a[QStringLiteral("kind")] = mediaKind(m.captured(2));
                a[QStringLiteral("path")] = path;
                a[QStringLiteral("title")] = clampWords(QFileInfo(path).completeBaseName().replace(QLatin1Char('_'), QLatin1Char(' ')).replace(QLatin1Char('-'), QLatin1Char(' ')), 5);
                a[QStringLiteral("metric")] = m.captured(2).toLower();
                if (t.viz && srcp == &t.toolOutput) {
                    a[QStringLiteral("source")] = QStringLiteral("lapis-viz");
                    a[QStringLiteral("metric")] = QStringLiteral("viz · ") + m.captured(2).toLower();
                }
                add(a, QStringLiteral("file:") + path);
            }
        }
    }

    if (proseStart >= 0 && proseEnd < 0)
        proseEnd = ex.artifacts.size();

    // Form factor: a chart only for a genuine sweep (chartFromTable needs >= 2 distinct measured
    // points). When the newest write-up is qualitative, architectural or a decision, Stage leads
    // with its distilled key points instead of forcing a graph out of incidental numbers.
    if (proseStart >= 0 && countWords(ex.prose) >= 12) {
        bool sweep = false;
        bool media = false;
        for (qsizetype i = proseStart; i < proseEnd && i < ex.artifacts.size(); ++i) {
            const QString k = ex.artifacts[i].toMap().value(QStringLiteral("kind")).toString();
            sweep = sweep || k == QLatin1String("chart") || k == QLatin1String("line");
            media = media || k == QLatin1String("video") || k == QLatin1String("image") || k == QLatin1String("svg")
                || k == QLatin1String("pdf") || k == QLatin1String("audio");
        }
        const QStringList points = distillKeyPoints(ex.prose);
        if (!sweep && !points.isEmpty()) {
            static const QRegularExpression heading(QStringLiteral("^#{1,4}\\s+(.+)$"), QRegularExpression::MultilineOption);
            const auto hm = heading.match(ex.prose);
            QString title = hm.hasMatch() ? clampWords(stripMarkup(hm.captured(1)).remove(QLatin1String("**")).simplified(), 5)
                                          : QStringLiteral("Key points");
            QVariantMap s;
            s[QStringLiteral("kind")] = QStringLiteral("summary");
            s[QStringLiteral("title")] = title;
            s[QStringLiteral("metric")] = QString();
            s[QStringLiteral("points")] = points;
            s[QStringLiteral("text")] = points.join(QLatin1Char('\n'));
            // A visual the turn produced still leads; the key points come right after it.
            const qsizetype at = media ? std::min(proseEnd, ex.artifacts.size()) : proseStart;
            ex.artifacts.insert(std::max<qsizetype>(0, at), s);
            while (ex.artifacts.size() > 12)
                ex.artifacts.removeLast();
        }
    }

    QStringList head;
    if (!throughput.isEmpty())
        head << throughput;
    if (!speedup.isEmpty())
        head << speedup;
    ex.headline = head.join(QStringLiteral(" · "));

    for (const QVariant &v : ex.artifacts) {
        const QVariantMap a = v.toMap();
        const QString kind = a.value(QStringLiteral("kind")).toString();
        if (kind != QLatin1String("chart") && kind != QLatin1String("line"))
            continue;
        const QVariantList values = a.value(QStringLiteral("values")).toList();
        const QStringList labels = a.value(QStringLiteral("labels")).toStringList();
        const QStringList valueText = a.value(QStringLiteral("valueText")).toStringList();
        double mx = 0;
        for (const QVariant &x : values)
            mx = std::max(mx, x.toDouble());
        for (int i = 0; i < values.size() && i < 8; ++i)
            ex.spark << (mx > 0 ? values[i].toDouble() / mx : 0.0);
        const int best = a.value(QStringLiteral("best")).toInt();
        ex.sparkBest = best < ex.spark.size() ? best : -1;
        for (int i = 0; kind == QLatin1String("chart") && i < labels.size() && i < 6; ++i) {
            QVariantMap arm;
            arm[QStringLiteral("name")] = labels[i];
            arm[QStringLiteral("value")] = valueText.value(i);
            arm[QStringLiteral("progress")] = 1.0;
            arm[QStringLiteral("best")] = i == best;
            ex.arms << arm;
        }
        if (kind == QLatin1String("chart"))
            ex.winner = labels.value(best) + QStringLiteral(" · ") + a.value(QStringLiteral("metric")).toString();
        if (ex.headline.isEmpty()) {
            // A bare number means nothing on an island: name what it measures.
            ex.headline = a.value(QStringLiteral("metric")).toString();
            if (a.value(QStringLiteral("unit")).toString().isEmpty())
                ex.headline += QLatin1Char(' ') + clampWords(a.value(QStringLiteral("valueLabel")).toString().toLower(), 2);
        }
        break;
    }

    // What the session did, for the quick Q&A context: the last four turns and what they touched.
    {
        constexpr int maxTurns = 4;
        constexpr int budget = 2500;
        const qsizetype first = std::max<qsizetype>(0, turns.size() - maxTurns);
        int used = 0;
        for (qsizetype i = first; i < turns.size(); ++i) {
            const Turn &t = turns[i];
            QString body = stripMarkup(t.text).simplified();
            if (body.isEmpty() && !t.tools.isEmpty())
                body = QStringLiteral("[tools] ") + t.tools.mid(std::max<qsizetype>(0, t.tools.size() - 4)).join(QStringLiteral("; "));
            if (body.isEmpty())
                continue;
            const QString who = t.role == QLatin1String("user") ? QStringLiteral("User: ") : QStringLiteral("Assistant: ");
            const int left = budget - used;
            const int share = std::max(160, left / int(turns.size() - i));  // later turns keep their share
            QString line = who + body;
            if (line.size() > share)
                line = line.left(share - 1) + QChar(0x2026);
            if (used + line.size() > budget)
                break;
            used += int(line.size());
            ex.recentTurns << line;
        }
        static const QRegularExpression plusFile(QStringLiteral("^\\+\\+\\+ b/(.+)$"), QRegularExpression::MultilineOption);
        QStringList files;
        QStringList tools;
        for (const Turn &t : turns) {
            QStringList tf = t.files;
            for (const QString &p : t.patches)
                for (auto it = plusFile.globalMatch(p); it.hasNext();)
                    tf << it.next().captured(1).trimmed();
            for (const QString &f : tf) {
                if (f.isEmpty())
                    continue;
                files.removeAll(f);
                files << f;
            }
            tools += t.tools;
        }
        ex.files = files.mid(std::max<qsizetype>(0, files.size() - 25));
        ex.toolCalls = tools.mid(std::max<qsizetype>(0, tools.size() - 15));
    }
    // The readable conversation (Ladder): the last 16 turns that still say something once tool
    // noise is stripped; machine-written user records (compaction summaries) are not the user.
    {
        static const QStringList machine = {
            QStringLiteral("This session is being continued"), QStringLiteral("Caveat:"),
            QStringLiteral("[Request interrupted"), QStringLiteral("Continue from where"), QStringLiteral("Project: ")};
        QVariantList thread;
        for (qsizetype i = turns.size() - 1; i >= 0 && thread.size() < 16; --i) {
            const Turn &t = turns[i];
            const QString raw = t.text.trimmed();
            if (raw.isEmpty())
                continue;
            if (t.role == QLatin1String("user")
                && std::any_of(machine.begin(), machine.end(), [&](const QString &p) { return raw.startsWith(p); }))
                continue;
            const QString text = cleanTurnText(raw, 800);
            if (text.isEmpty())
                continue;
            // A relay from another agent arrives as a user record; it is not the user talking.
            const bool peer = t.role == QLatin1String("user") && raw.startsWith(QLatin1String("[lapis-peers"));
            thread.prepend(QVariantMap{{QStringLiteral("role"), peer ? QStringLiteral("peer") : t.role},
                                       {QStringLiteral("text"), text}});
        }
        ex.threadTurns = thread;
        for (const Turn &t : turns) {
            if (t.role != QLatin1String("user"))
                continue;
            const QString first = t.text.trimmed().section(QLatin1Char('\n'), 0, 0);
            if (first.isEmpty() || first.startsWith(QLatin1Char('<'))
                || std::any_of(machine.begin(), machine.end(), [&](const QString &p) { return first.startsWith(p); }))
                continue;
            ex.promptName = promptTitle(first, 5);
            if (!ex.promptName.isEmpty())
                break;
        }
    }
    ex.params = extractParams(turns, ex.artifacts);
    ex.jobName = synthesizeJobName(turns, ex.goal);
    ex.tailSigs = turnSignatures(turns);
    return ex;
}

// --- plain-English job names, hyperparameters, key terms ------------------------------------

namespace {

// Technical tokens keep their real casing in names and pills.
QString properCase(const QString &w)
{
    static const QHash<QString, QString> fixed = {
        {QStringLiteral("fp4"), QStringLiteral("FP4")}, {QStringLiteral("nvfp4"), QStringLiteral("NVFP4")},
        {QStringLiteral("fp8"), QStringLiteral("FP8")}, {QStringLiteral("bf16"), QStringLiteral("BF16")},
        {QStringLiteral("fp16"), QStringLiteral("FP16")}, {QStringLiteral("int4"), QStringLiteral("INT4")},
        {QStringLiteral("int8"), QStringLiteral("INT8")}, {QStringLiteral("kv"), QStringLiteral("KV")},
        {QStringLiteral("kv-cache"), QStringLiteral("KV-cache")}, {QStringLiteral("lora"), QStringLiteral("LoRA")},
        {QStringLiteral("gpu"), QStringLiteral("GPU")}, {QStringLiteral("gpus"), QStringLiteral("GPUs")},
        {QStringLiteral("cpu"), QStringLiteral("CPU")}, {QStringLiteral("tp"), QStringLiteral("TP")},
        {QStringLiteral("tp4"), QStringLiteral("TP4")}, {QStringLiteral("dcp"), QStringLiteral("DCP")},
        {QStringLiteral("mtp"), QStringLiteral("MTP")}, {QStringLiteral("gemm"), QStringLiteral("GEMM")},
        {QStringLiteral("rmsnorm"), QStringLiteral("RMSNorm")}, {QStringLiteral("llm"), QStringLiteral("LLM")},
        {QStringLiteral("cuda"), QStringLiteral("CUDA")}, {QStringLiteral("api"), QStringLiteral("API")},
        {QStringLiteral("ui"), QStringLiteral("UI")}, {QStringLiteral("h3"), QStringLiteral("H3")},
        {QStringLiteral("moe"), QStringLiteral("MoE")}, {QStringLiteral("vllm"), QStringLiteral("vLLM")},
        {QStringLiteral("sglang"), QStringLiteral("SGLang")}, {QStringLiteral("hz"), QStringLiteral("Hz")},
        {QStringLiteral("fn"), QStringLiteral("fn")}, {QStringLiteral("12m"), QStringLiteral("12M")},
        {QStringLiteral("65b"), QStringLiteral("65B")}, {QStringLiteral("tma"), QStringLiteral("TMA")},
    };
    const QString l = w.toLower();
    return fixed.value(l, w);
}

const QSet<QString> &stopWords()
{
    static const QSet<QString> s = {
        "a", "an", "the", "and", "or", "of", "on", "in", "into", "to", "for", "with", "at", "by", "from", "my", "our",
        "your", "this", "that", "these", "it", "its", "is", "are", "be", "please", "can", "could", "you", "we", "i",
        "lets", "let's", "let", "us", "me", "some", "all", "any", "now", "then", "so", "just", "also", "up", "out",
        "new", "make", "do", "get", "use", "using", "via", "vs", "versus", "against", "over", "about", "what", "how",
        "why", "when", "where", "which", "should", "would", "want", "need", "see", "look", "try", "keep", "still",
        "more", "less", "same", "one", "two", "cards", "there", "here", "if", "not", "no", "yes", "ok", "okay",
        "session", "sessions", "did", "does", "done", "they", "them", "their", "he", "she", "his", "her", "was",
        "were", "has", "have", "had", "will", "like", "really", "very", "thing", "things", "stuff", "go", "going",
        "help", "understand", "explain", "tell", "show", "know", "think", "noticed", "saw", "seen", "said",
        "wondering", "quick", "question", "existing", "got", "get", "gets", "ping", "pinged", "hey", "hi"};
    return s;
}

struct VerbRule {
    const char *pattern;  // case-insensitive regex over goal / recent activity
    const char *gerund;
};

// First match wins; the latest activity is checked before the goal.
const QList<VerbRule> &verbRules()
{
    static const QList<VerbRule> r = {
        {"\\bsweep|\\bfor (tile|t|b|bs|lr) in\\b|\\bgrid search", "Sweeping"},
        {"\\btrain|\\bfine-?tun|\\blora\\b|\\bepoch", "Training"},
        {"\\bprofil|\\bncu\\b|\\bnsys\\b", "Profiling"},
        {"\\bblind review|\\breview", "Reviewing"},
        {"\\bbench|\\btok/s\\b|\\blatency\\b|\\bthroughput", "Benchmarking"},
        {"\\bserv(e|ing)\\b|\\bvllm\\b|\\bsglang\\b", "Serving"},
        {"\\bsearch", "Searching"},
        {"\\brecord|\\bffmpeg\\b|\\brender", "Rendering"},
        {"\\bfix|\\bdebug|\\bbug\\b|\\bbroken|\\bcrash", "Fixing"},
        {"\\bpytest\\b|\\bctest\\b|--test\\b|\\btests?\\b", "Testing"},
        {"\\bcmake\\b|\\bcompil|\\bbuild", "Building"},
        {"\\bwrite|\\bdraft", "Writing"},
        {"\\bport\\b|\\bmigrat", "Porting"},
        {"\\bupgrade|\\bimprov|\\boptimi", "Improving"},
    };
    return r;
}

// Gerunds a goal's leading verb maps to ("sweep" -> drop it, the verb slot says Sweeping).
bool isVerbWord(const QString &w)
{
    static const QSet<QString> v = {"sweep", "train", "profile", "review", "bench", "benchmark", "serve", "search",
                                    "record", "render", "fix", "debug", "test", "build", "compile", "write", "draft",
                                    "port", "migrate", "upgrade", "improve", "optimize", "optimise", "run", "add",
                                    "compare", "finetune", "fine-tune", "investigate", "check", "find", "create",
                                    "implement", "speed", "tune"};
    return v.contains(w.toLower());
}

QString cleanWord(QString w)
{
    while (!w.isEmpty() && !w.back().isLetterOrNumber())
        w.chop(1);
    while (!w.isEmpty() && !w.front().isLetterOrNumber() && w.front() != QLatin1Char('-'))
        w.remove(0, 1);
    return w;
}

}  // namespace

namespace {

// A line that can name a job: URLs, code spans and quotes stripped; at least two content words.
QString cleanSubjectLine(QString s)
{
    static const QRegularExpression url(QStringLiteral("https?://\\S+|www\\.\\S+"));
    static const QRegularExpression junk(QStringLiteral("[`\"“”*_#>\\[\\]()]"));
    s.remove(url);
    s.replace(junk, QStringLiteral(" "));
    return s.simplified();
}

int contentWords(const QString &s)
{
    int n = 0;
    for (const QString &raw : s.split(QLatin1Char(' '), Qt::SkipEmptyParts)) {
        const QString w = cleanWord(raw).toLower();
        if (!w.isEmpty() && !stopWords().contains(w) && std::any_of(w.begin(), w.end(), [](QChar c) { return c.isLetter(); }))
            ++n;
    }
    return n;
}

// Requests that are about the conversation, not the job ("Okay, you are replacing the Phonon
// session.", "Collapse the Lapis session into here", "Look at the H3 session and ...").
bool isMetaRequest(const QString &line)
{
    static const QRegularExpression meta(
        QStringLiteral("^(ok(ay)?\\b|you are\\b|you're\\b|collapse\\b|look at\\b|continue\\b|resume\\b|hey\\b|hi\\b|"
                       "thanks\\b|thank you\\b|yes\\b|no\\b|sure\\b|go\\b|do it\\b|status\\b|what's up\\b|"
                       "this session\\b|\\[|<|/)"),
        QRegularExpression::CaseInsensitiveOption);
    return meta.match(line.trimmed()).hasMatch() || line.contains(QLatin1String("session"), Qt::CaseInsensitive);
}

// A request's leading verb and its gerund: "fix the fn key toggle" -> Fixing.
const QHash<QString, QString> &leadVerbs()
{
    static const QHash<QString, QString> v = {
        {"fix", "Fixing"}, {"debug", "Debugging"}, {"sweep", "Sweeping"}, {"train", "Training"},
        {"finetune", "Training"}, {"fine-tune", "Training"}, {"profile", "Profiling"}, {"review", "Reviewing"},
        {"bench", "Benchmarking"}, {"benchmark", "Benchmarking"}, {"serve", "Serving"}, {"search", "Searching"},
        {"record", "Rendering"}, {"render", "Rendering"}, {"test", "Testing"}, {"build", "Building"},
        {"compile", "Building"}, {"write", "Writing"}, {"draft", "Writing"}, {"port", "Porting"},
        {"migrate", "Porting"}, {"upgrade", "Improving"}, {"improve", "Improving"}, {"optimize", "Optimizing"},
        {"optimise", "Optimizing"}, {"tune", "Tuning"}, {"compare", "Comparing"}, {"investigate", "Investigating"},
        {"add", "Adding"}, {"implement", "Implementing"}, {"find", "Finding"}, {"check", "Checking"},
        {"create", "Creating"}, {"run", "Running"}, {"update", "Updating"}, {"clean", "Cleaning"},
        {"install", "Installing"}, {"deploy", "Deploying"}, {"refactor", "Refactoring"}, {"merge", "Merging"},
        {"rewrite", "Rewriting"}, {"explore", "Exploring"}, {"analyze", "Analyzing"}, {"analyse", "Analyzing"},
        {"measure", "Measuring"}, {"wire", "Wiring"}, {"launch", "Launching"}, {"start", "Starting"},
        {"stop", "Stopping"}, {"kill", "Stopping"}, {"delete", "Deleting"}, {"remove", "Removing"},
        {"move", "Moving"}, {"rename", "Renaming"}, {"convert", "Converting"}, {"generate", "Generating"},
        {"design", "Designing"}, {"plan", "Planning"}, {"research", "Researching"}, {"summarize", "Summarizing"},
        {"send", "Sending"}, {"ship", "Shipping"}, {"publish", "Publishing"}, {"read", "Reading"},
        {"configure", "Configuring"}, {"restore", "Restoring"}, {"upload", "Uploading"},
        {"download", "Downloading"}, {"translate", "Translating"}, {"transcribe", "Transcribing"},
        {"make", "Making"}, {"setup", "Setting up"}, {"set", "Setting"}, {"audit", "Auditing"},
        {"evaluate", "Evaluating"}, {"eval", "Evaluating"}, {"recreate", "Recreating"}, {"edit", "Editing"},
        {"listen", "Listening"}, {"watch", "Watching"}, {"open", "Opening"}, {"triage", "Triaging"},
        {"organize", "Organizing"}, {"summarise", "Summarizing"}, {"collect", "Collecting"}};
    return v;
}

// The clause of a request line that can name its job, word order kept:
//   verb   the leading imperative verb as typed ("fix"), "" for a noun phrase ("Kneron H3 blind review")
//   words  its subject, articles / possessives / filler dropped, connectors ("for", "on") kept between
//          content words, at most `maxContent` content words
// ok is false for chat that does not name a job: questions, "I mean it's ...", "dont even ask ...".
struct RequestClause {
    QString verb;
    QStringList words;
    QString text;  // the clause the words came from, for verb inference
    bool ok = false;
};

// `title`: the line is a Lapis tab title, a label by construction, so any length of noun phrase
// names the job; a request line names one as a noun phrase only when short ("roblox game 2"):
// a long non-imperative sentence is conversation ("Another agent stated the mismatch plainly").
// `minContent`: content words a clause needs ("Fix bug" names nothing; "Fix fn key toggle" does).
RequestClause requestClause(const QString &line, int maxContent, bool title = false, int minContent = 2)
{
    static const QSet<QString> filler = {"ok", "okay", "so", "hey", "hi", "yo", "please", "pls", "can", "could",
                                         "would", "will", "you", "u", "i", "we", "want", "wanna", "need", "to",
                                         "lets", "let's", "let", "us", "now", "also", "then", "just", "go", "ahead",
                                         "and", "quickly", "kindly", "first", "next", "finally", "alright", "right"};
    // A noun phrase never holds these; a line that does is conversation, not a job name.
    static const QSet<QString> chatty = {
        "i", "i'm", "im", "i've", "i'd", "me", "you", "you're", "your", "we", "he", "she", "they", "it", "it's",
        "its", "dont", "don't", "didnt", "didn't", "doesnt", "doesn't", "isnt", "isn't", "cant", "can't", "wont",
        "won't", "mean", "how", "what", "why", "who", "whats", "what's", "did", "does", "do", "is", "are", "was",
        "were", "am", "be", "been", "these", "those", "this", "that", "there", "here", "if", "because", "but",
        "even", "really", "should", "would", "could", "have", "has", "had", "got", "my", "our", "ask", "asked"};
    static const QSet<QString> skip = {"a", "an", "the", "my", "our", "your", "his", "her", "their", "some", "any",
                                       "all", "this", "these", "those", "that", "its", "own", "most", "though",
                                       "anyway", "too", "again", "everything", "something", "anything", "stuff",
                                       "actually", "basically", "maybe", "probably"};
    // An embedded question ends the request ("check what jobs the server is running"): what follows is
    // not the object, so a clause with nothing before it names nothing.
    static const QSet<QString> embedded = {"what", "how", "why", "whether", "who", "whom", "whose", "what's", "whats"};
    static const QSet<QString> connectors = {"for", "on", "in", "to", "with", "at", "from", "into", "of", "about",
                                             "vs", "via", "across", "over", "under", "per", "by", "using", "against"};
    static const QSet<QString> breakers = {"and", "then", "so", "but", "or", "because", "which", "while", "plus",
                                           "after", "before", "if", "when", "where", "until", "since", "like"};
    static const QRegularExpression numUnit(QStringLiteral("\\b(\\d+(?:\\.\\d+)?) (Hz|fps|ms|GB|MB|k)\\b"),
                                            QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression sentenceEnd(QStringLiteral("[.?!;]+(\\s+|$)"));
    static const QRegularExpression bareInt(QStringLiteral("^[0-9,]+$"));
    // "240 Hz" stays one word (a no-break space, set per clause below), so the number survives the
    // bare-number filter.
    const QString s = cleanSubjectLine(line);
    // Sentences with their terminators ("6.2" never splits: an end needs whitespace or the line end).
    QList<QPair<QString, bool>> sentences;
    {
        qsizetype from = 0;
        for (auto it = sentenceEnd.globalMatch(s); it.hasNext();) {
            const auto m = it.next();
            sentences.append({s.mid(from, m.capturedStart() - from), m.captured(0).contains(QLatin1Char('?'))});
            from = m.capturedEnd();
        }
        if (from < s.size())
            sentences.append({s.mid(from), false});
    }
    int tried = 0;
    for (const auto &[sentence, question] : std::as_const(sentences)) {
        if (sentence.trimmed().isEmpty())
            continue;
        if (++tried > 3)
            break;
        // Leading clauses that are pure filler ("Okay, ...") give way to the next one.
        for (const QString &part : sentence.split(QLatin1Char(','), Qt::SkipEmptyParts)) {
            // simplified() first: it would turn the "240 Hz" no-break space back into a space.
            QString clause = part.simplified();
            clause.replace(numUnit, QStringLiteral("\\1") + QChar(0x00a0) + QStringLiteral("\\2"));
            QStringList toks = clause.split(QLatin1Char(' '), Qt::SkipEmptyParts);
            while (!toks.isEmpty() && filler.contains(cleanWord(toks.first()).toLower()))
                toks.removeFirst();
            if (toks.isEmpty())
                continue;
            RequestClause c;
            c.text = toks.join(QLatin1Char(' '));
            // Chat about the chat ("you are replacing the Phonon session", "[lapis-peers ...").
            if (isMetaRequest(c.text))
                break;
            const QString first = cleanWord(toks.first()).toLower();
            if (leadVerbs().contains(first)) {
                c.verb = cleanWord(toks.takeFirst());
                if (first == QLatin1String("set") && !toks.isEmpty() && cleanWord(toks.first()).toLower() == QLatin1String("up")) {
                    toks.removeFirst();
                    c.verb = QStringLiteral("setup");
                }
            } else {
                // Only a line's opening sentence can be a label; later ones are context.
                if (question || (!title && (toks.size() > 6 || tried > 1)))
                    break;
                bool chat = false;
                for (const QString &t : std::as_const(toks))
                    chat = chat || chatty.contains(cleanWord(t).toLower());
                if (chat)
                    break;
            }
            QStringList pending;
            int content = 0;
            for (const QString &raw : std::as_const(toks)) {
                const QString w = cleanWord(raw);
                if (w.isEmpty())
                    continue;
                const QString l = w.toLower();
                // "fix the fact that it's unreachable": a relative clause or a pronoun after the
                // object starts a new thought; the object ends there.
                if (content > 0 && (l == QLatin1String("that") || (chatty.contains(l) && !skip.contains(l))))
                    break;
                if (skip.contains(l) || bareInt.match(l).hasMatch())
                    continue;
                if (embedded.contains(l))
                    break;
                if (breakers.contains(l)) {
                    if (content > 0)
                        break;
                    continue;
                }
                if (connectors.contains(l)) {
                    // "listen to ...": a connector straight after the verb belongs to it.
                    if (content > 0 || (!c.verb.isEmpty() && c.words.isEmpty() && pending.isEmpty()))
                        pending << l;
                    continue;
                }
                if (stopWords().contains(l) || chatty.contains(l))
                    continue;
                if (content >= maxContent) {
                    // Cut mid-phrase ("support to PS4 controller with obvious | keybinds"): fall
                    // back to the last connector so the name ends on a whole phrase.
                    if (pending.isEmpty())
                        for (qsizetype k = c.words.size() - 1; k > 0; --k)
                            if (connectors.contains(c.words[k]) && k >= 2) {
                                content -= int(c.words.size() - k - 1);
                                c.words = c.words.mid(0, k);
                                break;
                            }
                    break;
                }
                c.words += pending;
                pending.clear();
                c.words << properCase(w);
                ++content;
            }
            c.ok = content >= std::max(1, minContent);
            if (c.ok)
                return c;
            break;  // the sentence's first real clause names nothing: try the next sentence
        }
    }
    return {};
}

// Connectors left hanging at the end of a cut name ("Sweeping tile sizes for FP4 GEMM on").
void trimTrailingConnectors(QStringList &words)
{
    static const QSet<QString> hang = {"for", "on", "in", "to", "with", "at", "from", "into", "of", "about", "vs",
                                       "via", "across", "over", "under", "per", "by", "using", "against", "up"};
    while (words.size() > 1 && hang.contains(words.last().toLower()))
        words.removeLast();
}

}  // namespace

QString promptTitle(const QString &request, int maxWords)
{
    const QString line = request.trimmed().section(QLatin1Char('\n'), 0, 0);
    if (line.isEmpty() || line.startsWith(QLatin1Char('[')) || line.startsWith(QLatin1Char('<')))
        return {};
    const RequestClause c = requestClause(line, std::max(1, maxWords - 1));
    if (!c.ok)
        return {};
    QStringList words;
    if (!c.verb.isEmpty())
        words << (c.verb.toLower() == QLatin1String("setup") ? QStringLiteral("Set up") : c.verb);
    words += c.words;
    // maxWords counts meaningful words; connectors between them ("for", "on") ride along.
    while (words.size() > maxWords + 2)
        words.removeLast();
    trimTrailingConnectors(words);
    if (words.size() < 2)
        return {};
    QString out = words.join(QLatin1Char(' '));
    out[0] = out.at(0).toUpper();
    return out;
}

QString cleanTurnText(const QString &text, int maxChars)
{
    static const QRegularExpression harnessBlocks(
        QStringLiteral("<(system-reminder|command-[a-z]+|local-command-[a-z]+|ADDITIONAL_METADATA|USER_SETTINGS_CHANGE|"
                       "user-prompt-submit-hook)>.*?(</\\1>|$)"),
        QRegularExpression::DotMatchesEverythingOption);
    static const QRegularExpression fenceLine(QStringLiteral("^\\s*```[\\w+-]*\\s*$"), QRegularExpression::MultilineOption);
    static const QRegularExpression blankRuns(QStringLiteral("\\n{3,}"));
    static const QRegularExpression trailingWs(QStringLiteral("[ \\t]+\\n"));
    QString s = text;
    s.remove(harnessBlocks);
    s = stripToolNoise(s);
    if (s.contains(QLatin1Char('<')))
        s = stripMarkup(s);
    s.remove(fenceLine);
    s.remove(QStringLiteral("**"));
    s.remove(QLatin1Char('`'));
    s.replace(trailingWs, QStringLiteral("\n"));
    s.replace(blankRuns, QStringLiteral("\n\n"));
    s = s.trimmed();
    if (s.size() > maxChars) {
        qsizetype cut = s.lastIndexOf(QLatin1Char(' '), maxChars - 1);
        if (cut < maxChars / 2)
            cut = maxChars - 1;
        s = s.left(cut).trimmed() + QChar(0x2026);
    }
    return s;
}

QString synthesizeJobName(const QList<Turn> &turns, const QString &goal, const QString &title)
{
    // Recent activity, newest first: tool calls (what is running now) apart from reply prose.
    // Paths and table rows are dropped: "fixtures/" is not a fix, a "| build |" column not a build.
    QString recentTools, recent;
    static const QRegularExpression pathTok(QStringLiteral("\\S*/\\S*"));
    for (qsizetype i = turns.size() - 1, n = 0; i >= 0 && n < 6; --i, ++n) {
        recentTools += turns[i].tools.join(QLatin1Char(' ')).remove(pathTok) + QLatin1Char(' ');
        QStringList lines;
        for (const QString &l : turns[i].text.left(600).split(QLatin1Char('\n')))
            if (!l.trimmed().startsWith(QLatin1Char('|')))
                lines << l;
        recent += lines.join(QLatin1Char(' ')).remove(pathTok) + QLatin1Char(' ');
    }
    recent = recentTools + recent;
    // Progress: the newest "step N" (training) or "N/M" counter in the recent turns.
    QString progress;
    static const QRegularExpression stepRe(QStringLiteral("\\bstep[ =:]+(\\d{2,7})\\b"), QRegularExpression::CaseInsensitiveOption);
    for (qsizetype i = turns.size() - 1, n = 0; i >= 0 && n < 6 && progress.isEmpty(); --i, ++n) {
        const QString hay = turns[i].text + QLatin1Char(' ') + turns[i].toolOutput.right(4000);
        QString last;
        for (auto it = stepRe.globalMatch(hay); it.hasNext();)
            last = it.next().captured(1);
        if (!last.isEmpty())
            progress = QStringLiteral("step ") + last;
    }

    // Subject source, first that reads as a job: a Lapis tab title phrase ("GPT 6.2 Astra
    // KernelBench run"), the goal, then each request in order. Word order is kept ("Fixing fn key
    // toggle latency"), so a name never reads as a bag of content words; chat that names no job
    // ("dont even ask ...", "I mean it's ...", questions) is skipped rather than mangled.
    const int maxContent = progress.isEmpty() ? 5 : 2;
    QStringList candidates;
    const QString t = cleanSubjectLine(title);
    const bool titled = t.contains(QLatin1Char(' ')) && contentWords(t) >= 2;
    if (titled)
        candidates << title;
    candidates << goal;
    for (const Turn &u : turns)
        if (u.role == QLatin1String("user"))
            candidates << u.text.trimmed().section(QLatin1Char('\n'), 0, 0);
    RequestClause clause;
    for (qsizetype i = 0; i < candidates.size(); ++i) {
        const QString c = candidates[i].trimmed();
        // "Project: quick-qs (~). On screen: ..." is Lapis Studio's own context preamble, not a request.
        if (c.isEmpty() || c.startsWith(QLatin1Char('[')) || c.startsWith(QLatin1Char('<'))
            || c.startsWith(QLatin1String("Project: ")))
            continue;
        // A training step counter carries a one-word subject ("Training LoRA step 400").
        clause = requestClause(c, maxContent, titled && i == 0, progress.isEmpty() ? 2 : 1);
        if (clause.ok)
            break;
    }
    if (!clause.ok)
        return {};

    // Verb: the request's own verb when it has one (the intent), else what the latest turns are doing.
    static const QList<QRegularExpression> verbRes = [] {
        QList<QRegularExpression> out;
        for (const VerbRule &r : verbRules())
            out << QRegularExpression(QString::fromLatin1(r.pattern), QRegularExpression::CaseInsensitiveOption);
        return out;
    }();
    QString verb = leadVerbs().value(clause.verb.toLower());
    for (int pass = 0; pass < 2 && verb.isEmpty(); ++pass)
        for (int k = 0; k < verbRes.size(); ++k)
            if (verbRes[k].match(pass == 0 ? clause.text : recent).hasMatch()) {
                verb = QString::fromLatin1(verbRules()[k].gerund);
                break;
            }
    QStringList subject;
    for (const QString &w : std::as_const(clause.words)) {
        const QString l = w.toLower();
        // An inferred verb already says the action ("Kneron H3 blind review" -> Reviewing ...).
        const bool echoesVerb = clause.verb.isEmpty() && l.size() >= 3
            && verb.toLower().startsWith(l.left(qMin<qsizetype>(l.size(), 5)));
        if ((echoesVerb && isVerbWord(l)) || subject.contains(w, Qt::CaseInsensitive))
            continue;
        subject << w;
    }
    trimTrailingConnectors(subject);
    if (verb.isEmpty())
        verb = QStringLiteral("Working on");
    QStringList words = verb.split(QLatin1Char(' ')) + subject;
    if (!progress.isEmpty())
        words += progress.split(QLatin1Char(' '));
    // Too short to say anything: borrow the newest technical token from the replies.
    if (words.size() < 4) {
        static const QRegularExpression techRe(
            // Names only (configs, formats, model sizes), never metrics like "226 tok/s".
            QStringLiteral("\\b(sm_?\\d{2,3}|NVFP4|FP4|FP8|BF16|KV[- ]cache|LoRA|TP\\d|DCP=?\\d|MTP|GEMM|RMSNorm|\\d+B)\\b"),
            QRegularExpression::CaseInsensitiveOption);
        for (auto it = techRe.globalMatch(recent); it.hasNext() && words.size() < 4;) {
            const QString w = properCase(it.next().captured(1));
            if (!words.contains(w, Qt::CaseInsensitive))
                words << w;
        }
    }
    if (subject.isEmpty() && progress.isEmpty())
        return {};  // a verb alone ("Fixing") names nothing: better no job line than a vague one
    if (words.size() > 7) {
        // Cut on a phrase boundary: a word past the cap that is not a connector means the 7th
        // word sits mid-phrase ("... controller with obvious | keybinds"), so back off to the
        // last connector.
        static const QSet<QString> conn = {"for", "on", "in", "to", "with", "at", "from", "into", "of", "about",
                                           "vs", "via", "across", "over", "under", "per", "by", "using", "against"};
        const bool midPhrase = !conn.contains(words[7].toLower());
        words = words.mid(0, 7);
        if (midPhrase)
            for (qsizetype k = words.size() - 1; k >= 3; --k)
                if (conn.contains(words[k].toLower())) {
                    words = words.mid(0, k);
                    break;
                }
    }
    trimTrailingConnectors(words);
    return words.join(QLatin1Char(' '));
}

bool isCrypticName(const QString &name)
{
    const QString n = name.trimmed();
    if (n.isEmpty())
        return true;
    if (n.contains(QLatin1Char(' ')))
        return false;  // a Lapis tab title already reads as English
    static const QRegularExpression hex(QStringLiteral("^[0-9a-f]{6,}$"));
    if (hex.match(n).hasMatch())
        return true;
    if (n.contains(QLatin1Char('-')) || n.contains(QLatin1Char('_')) || n.contains(QLatin1Char('.')))
        return true;
    if (std::any_of(n.begin(), n.end(), [](QChar c) { return c.isDigit(); }))
        return true;
    return n.size() <= 4;
}

namespace {

struct ParamDef {
    const char *name;     // pill label
    const char *mention;  // regex; capture 1 = value
    const char *columns;  // regex over table headers whose column holds this param's values
    const char *detail;   // one plain-English line
};

const QList<ParamDef> &paramDefs()
{
    static const QList<ParamDef> d = {
        {"tile", "\\btile(?:[ _-]?size)?\\s*(?:=|:)?\\s*(\\d+x\\d+(?:x\\d+)?)\\b|\\b(\\d+x\\d+(?:x\\d+)?)\\s+tile\\b",
         "^(tile|tile_size|block|cta)$",
         "Tile shape each thread block computes; bigger tiles reuse more data but need more registers."},
        {"regs", "\\b(?:regs|registers|maxrregcount)\\s*(?:=|:)\\s*(\\d{2,3})\\b|\\b(\\d{2,3})[- ](?:reg|regs|register)\\b",
         "^(regs|registers|reg)$",
         "Registers per thread; more keeps data on-chip, too many spill to slow local memory."},
        {"spills", "\\bspill(?:s|ed)?\\s*(?:=|:)?\\s*(\\d{1,4})\\b|\\b(\\d{1,4}) (?:regs? )?spill", "^(spills|spill)$",
         "Registers that overflowed to local memory per thread; 0 is the goal."},
        {"DCP", "\\bDCP\\s*(?:=|:)?\\s*(\\d{1,2})\\b|--dcp[ =](\\d{1,2})\\b", "^(dcp)$",
         "Decode context parallel: the KV cache is sharded across this many GPUs while decoding."},
        {"TP", "\\bTP\\s*(?:=|:)?\\s*(\\d{1,2})\\b|--tp[ =](\\d{1,2})\\b|tensor[- ]parallel(?:ism)?\\s*(?:=|:)?\\s*(\\d{1,2})\\b",
         "^(tp)$", "Tensor parallel: each layer's weights are split across this many GPUs."},
        {"MTP", "\\bMTP\\s*(?:=|:)\\s*(\\d{1,2})\\b|--mtp(?:[ =](\\d{1,2}))?\\b", "^(mtp)$",
         "Multi-token prediction: extra tokens drafted per decode step and verified in one pass."},
        {"batch", "\\bbatch(?:[ _-]?size)?\\s*(?:=|:)?\\s*(\\d{1,5})\\b|\\bbs\\s*=\\s*(\\d{1,5})\\b|--batch-size[ =](\\d{1,5})",
         "^(batch|bs|batch_size)$", "Sequences processed together per step; higher fills the GPU but adds latency."},
        {"lr", "\\b(?:lr|learning[ _]rate)\\s*(?:=|:)?\\s*(\\d(?:\\.\\d+)?e-?\\d+|0\\.\\d+)\\b", "^(lr|learning_rate)$",
         "Learning rate: how big each optimizer update is; too high diverges, too low crawls."},
        {"steps", "\\b(?:max_)?steps\\s*(?:=|:)\\s*(\\d{2,7})\\b|\\bstep[ =:]+(\\d{2,7})\\b", "^(step|steps)$",
         "Optimizer steps; the latest step is how far training has got."},
        {"ctx", "\\b(?:ctx|context(?: length)?|max[-_]model[-_]len|seq[-_]?len)\\s*(?:=|:)?\\s*(\\d+k|\\d{3,7})\\b|\\b(\\d+k) context\\b",
         "^(ctx|seq|seqlen|seq_len|context)$", "Context length in tokens the model attends over at once."},
        {"effort", "\\b(?:reasoning[_ ])?effort\\s*(?:=|:)?\\s*(low|medium|high|xhigh)\\b", "^(effort)$",
         "Reasoning effort: how long the model may think before it answers."},
        {"rank", "\\blora[_ ]rank\\s*(?:=|:)?\\s*(\\d{1,4})\\b|\\brank\\s*=\\s*(\\d{1,4})\\b", "^(rank|lora_rank|r)$",
         "LoRA rank: size of the low-rank adapter; higher learns more and costs more."},
        {"stages", "\\b(?:num_)?stages\\s*(?:=|:)\\s*(\\d{1,2})\\b", "^(stages|num_stages)$",
         "Pipeline stages: async copies in flight to hide memory latency."},
        {"warps", "\\b(?:num_)?warps\\s*(?:=|:)\\s*(\\d{1,2})\\b", "^(warps|num_warps)$",
         "Warps per thread block (32 threads each)."},
        {"arch", "\\b(sm_?1[0-2]\\d|sm_?[89]\\d)\\b", "^(arch|sm)$",
         "GPU architecture the kernel is compiled for; sm_120 is consumer Blackwell (RTX 50, RTX Pro 6000)."},
        {"dtype", "\\b(NVFP4|FP4|FP8|BF16|FP16|INT4|INT8)\\b", "^(dtype|precision|quant)$",
         "Number format for weights or KV cache; fewer bits mean less memory traffic and more rounding error."},
    };
    return d;
}

struct TermDef {
    const char *name;
    const char *pattern;
    const char *detail;
};

const QList<TermDef> &termDefs()
{
    static const QList<TermDef> d = {
        {"KV cache", "\\bKV[- ]cache\\b", "Keys and values of past tokens, kept so decode never recomputes them."},
        {"dequant", "\\bdequant(?:ize|ization)?\\b", "Unpacking low-bit numbers back to full precision right before the math."},
        {"spec decoding", "\\bspec(?:ulative)? decod", "A small draft proposes tokens; the big model verifies several at once."},
        {"occupancy", "\\boccupancy\\b", "How many warps an SM keeps resident; more hides latency."},
        {"LoRA", "\\bLoRA\\b", "Low-rank adapter: trains a small add-on instead of the whole model."},
        {"TMA", "\\bTMA\\b", "Tensor Memory Accelerator: hardware bulk copies from global to shared memory."},
        {"RMSNorm", "\\bRMSNorm\\b", "Normalizes each vector by its root mean square; cheap, used in most LLMs."},
        {"p99", "\\bp99\\b", "The 99th-percentile time: only 1 in 100 frames or requests is slower."},
        {"parity", "\\bparity\\b", "Same outputs as the reference within a tolerance: the correctness gate."},
    };
    return d;
}

}  // namespace

QVariantList extractParams(const QList<Turn> &turns, const QVariantList &artifacts)
{
    struct Acc {
        QStringList runs;
        QString value;
        int mentions = 0;
        int lastTurn = -1;
        bool fromWinner = false;  // value came from a table's winning row: prose adds runs only
    };
    const QList<ParamDef> &defs = paramDefs();
    // Compiled once per process; QRegularExpression matching is thread-safe (worker thread).
    static const QList<QRegularExpression> mentionRes = [] {
        QList<QRegularExpression> out;
        for (const ParamDef &d : paramDefs()) {
            const QString n = QString::fromLatin1(d.name);
            // Upper-case pills (DCP, TP, MTP) match case-sensitively: "tp" and "dcp" are too common in words.
            const bool upper = n.at(0).isUpper();
            out << QRegularExpression(QString::fromLatin1(d.mention),
                                      upper ? QRegularExpression::NoPatternOption : QRegularExpression::CaseInsensitiveOption);
        }
        return out;
    }();
    static const QList<QRegularExpression> colRes = [] {
        QList<QRegularExpression> out;
        for (const ParamDef &d : paramDefs())
            out << QRegularExpression(QString::fromLatin1(d.columns), QRegularExpression::CaseInsensitiveOption);
        return out;
    }();
    QList<Acc> acc(defs.size());
    auto note = [&](int d, QString v, int turn) {
        v = v.trimmed();
        if (v.isEmpty())
            return;
        if (defs[d].name == QLatin1String("arch")) {
            v = v.toLower();
            if (!v.contains(QLatin1Char('_')))
                v.insert(2, QLatin1Char('_'));
        } else if (defs[d].name == QLatin1String("dtype")) {
            v = v.toUpper();
        }
        Acc &a = acc[d];
        if (!a.runs.contains(v)) {  // first-seen order: the order the runs tried them
            a.runs << v;
            if (a.runs.size() > 8)
                a.runs.removeFirst();
        }
        if (!a.fromWinner)
            a.value = v;
        a.mentions += 1;
        a.lastTurn = std::max(a.lastTurn, turn);
    };
    // Tables: a column named like a param lists its value in every run (oldest row first).
    for (const QVariant &v : artifacts) {
        const QVariantMap t = v.toMap();
        if (t.value(QStringLiteral("kind")).toString() != QLatin1String("table"))
            continue;
        const QStringList header = t.value(QStringLiteral("header")).toStringList();
        const QVariantList rows = t.value(QStringLiteral("rows")).toList();
        // The winning row: the chart's best index, or every row named like the chart's winner.
        const QString winLabel = t.value(QStringLiteral("highlightLabel")).toString();
        const int winIndex = t.contains(QStringLiteral("highlight")) ? t.value(QStringLiteral("highlight")).toInt() : -1;
        for (int d = 0; d < defs.size(); ++d) {
            for (int c = 0; c < header.size(); ++c) {
                if (!colRes[d].match(header[c].trimmed()).hasMatch())
                    continue;
                QString winner;
                for (int r = 0; r < rows.size(); ++r) {
                    const QStringList cells = rows[r].toStringList();
                    note(d, cells.value(c), -1);
                    if (r == winIndex || (!winLabel.isEmpty() && cells.value(0) == winLabel))
                        winner = cells.value(c);
                }
                if (!winner.isEmpty()) {
                    acc[d].value = winner.trimmed();
                    acc[d].fromWinner = true;
                }
            }
        }
    }
    // Mentions in prose, tool calls and the tail of tool output, oldest turn first so the
    // newest mention ends up as the value.
    const qsizetype from = std::max<qsizetype>(0, turns.size() - 80);
    for (qsizetype i = from; i < turns.size(); ++i) {
        const Turn &t = turns[i];
        static const QRegularExpression fenceRe(QStringLiteral("```.*?```"), QRegularExpression::DotMatchesEverythingOption);
        QString hay = t.text;
        hay.remove(fenceRe);  // code blocks: their tables are counted above, their code is noise
        hay += QLatin1Char('\n') + t.tools.join(QLatin1Char('\n')) + QLatin1Char('\n') + t.toolOutput.right(3000);
        for (int d = 0; d < defs.size(); ++d) {
            for (auto it = mentionRes[d].globalMatch(hay); it.hasNext();) {
                const auto m = it.next();
                QString v;
                for (int g = 1; g <= m.lastCapturedIndex() && v.isEmpty(); ++g)
                    v = m.captured(g);
                if (v.isEmpty() && defs[d].name == QLatin1String("MTP"))
                    v = QStringLiteral("on");
                note(d, v, int(i));
            }
        }
    }
    QList<int> order;
    for (int d = 0; d < defs.size(); ++d)
        if (acc[d].mentions > 0)
            order << d;
    // Knobs first (tile, regs, DCP, MTP, lr, ...), then outcomes and context (spills, arch, dtype);
    // within a tier, params that varied across runs (what a sweep is about), then mentions.
    auto tier = [&](int d) {
        const QString n = QString::fromLatin1(defs[d].name);
        return n == QLatin1String("spills") || n == QLatin1String("arch") || n == QLatin1String("dtype") ? 1 : 0;
    };
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (tier(a) != tier(b))
            return tier(a) < tier(b);
        const bool va = acc[a].runs.size() > 1, vb = acc[b].runs.size() > 1;
        if (va != vb)
            return va;
        return acc[a].mentions > acc[b].mentions;
    });
    QVariantList out;
    for (int d : order) {
        if (out.size() >= 6)
            break;
        out << QVariantMap{{QStringLiteral("name"), QString::fromLatin1(defs[d].name)},
                           {QStringLiteral("value"), acc[d].value},
                           {QStringLiteral("runs"), acc[d].runs},
                           {QStringLiteral("detail"), QString::fromLatin1(defs[d].detail)},
                           {QStringLiteral("kind"), QStringLiteral("param")},
                           {QStringLiteral("mentions"), acc[d].mentions}};
    }
    // Key terms: the glossary words this session leans on, most-mentioned first.
    QList<QPair<int, int>> terms;  // (mentions, def index)
    const QList<TermDef> &tdefs = termDefs();
    static const QList<QRegularExpression> termRes = [] {
        QList<QRegularExpression> out;
        for (const TermDef &t : termDefs())
            out << QRegularExpression(QString::fromLatin1(t.pattern), QRegularExpression::CaseInsensitiveOption);
        return out;
    }();
    for (int d = 0; d < tdefs.size(); ++d) {
        int n = 0;
        for (qsizetype i = from; i < turns.size(); ++i)
            for (auto it = termRes[d].globalMatch(turns[i].text); it.hasNext(); it.next())
                ++n;
        bool dup = false;
        for (const QVariant &p : out)
            dup = dup || p.toMap().value(QStringLiteral("name")).toString().compare(QString::fromLatin1(tdefs[d].name), Qt::CaseInsensitive) == 0;
        if (n > 0 && !dup)
            terms << qMakePair(n, d);
    }
    std::stable_sort(terms.begin(), terms.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
    for (int k = 0; k < terms.size() && k < 3; ++k) {
        const TermDef &t = tdefs[terms[k].second];
        out << QVariantMap{{QStringLiteral("name"), QString::fromLatin1(t.name)},
                           {QStringLiteral("value"), QString()},
                           {QStringLiteral("runs"), QStringList()},
                           {QStringLiteral("detail"), QString::fromLatin1(t.detail)},
                           {QStringLiteral("kind"), QStringLiteral("term")},
                           {QStringLiteral("mentions"), terms[k].first}};
    }
    return out;
}

QList<uint> turnSignatures(const QList<Turn> &turns)
{
    QList<uint> out;
    for (qsizetype i = std::max<qsizetype>(0, turns.size() - 64); i < turns.size(); ++i) {
        const Turn &t = turns[i];
        out << qHash(t.role + QLatin1Char('\x1f') + t.text.left(2000) + QLatin1Char('\x1f') + t.tools.join(QLatin1Char('\x1e')));
    }
    return out;
}

int newTurnsSince(const QList<uint> &before, const QList<uint> &now)
{
    if (before.isEmpty() || now.isEmpty())
        return 0;
    // The last turn of the previous parse may have grown (a streaming reply merges into it), so
    // match on the newest earlier turn that both parses share.
    for (qsizetype b = before.size() - 1; b >= 0; --b) {
        const qsizetype at = now.lastIndexOf(before[b]);
        if (at >= 0)
            return int(std::max<qsizetype>(0, now.size() - 1 - at - (before.size() - 1 - b)));
    }
    return 0;
}

}  // namespace studio
