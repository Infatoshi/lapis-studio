// Live Lapis bridge and the clean mock backend used by --test / --record-demo.
#pragma once

#include "gamepad.hpp"
#include "transcript.hpp"

#include <QElapsedTimer>
#include <QFileSystemWatcher>
#include <QHash>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QTimer>
#include <QVariantList>
#include <QVariantMap>

namespace studio {

struct SessionInfo {
    QString id;
    QString title;
    QString harness;
    QString category;
    QString directory;
    QString ident;
    QString transcript;
    qint64 mtime = 0;
    QString status;
    qint64 started = 0;     // ms since epoch: birth time of the Lapis endpoint socket
    bool redacted = false;  // infinity category: never read its transcript
};

// Which island a session belongs to. Real project folders group by basename; sessions opened
// in $HOME or /tmp are unrelated scratch work, so their Lapis tab title names the island.
struct ProjectId {
    QString key;
    QString name;
    bool fromTitle = false;
};
bool isScratchDir(const QString &dir);
QString slugFromTitle(const QString &title);
ProjectId projectIdFor(const QString &directory, const QString &title, const QString &category);
// "42m", "3h 12m", "2d 4h".
QString elapsedLabel(qint64 ms);
// The island metric when a project has nothing to chart: "live · 3h 12m".
QString statusMetric(const QString &status, qint64 elapsedMs);

// The chat model catalog: every model the inline picker cycles through, with its provider
// ("anthropic" | "openai" | "google") and the efforts that model accepts (low..xhigh subset).
struct ModelSpec {
    QString id;
    QString provider;
    QString label;
    QStringList efforts;
    // Honest speed class, shown next to the label in the picker: "Fast" | "Balanced" |
    // "Frontier" | "Deep Think" | "Teach". gemini-4-hydrogen is Deep Think, not Flash.
    QString speed;
};
// "Gemini 4 Hydrogen · Deep Think"
QString modelDisplayLabel(const QString &model);
const QList<ModelSpec> &modelCatalog();
QString providerFor(const QString &model);    // "" for an unknown model
QStringList effortsFor(const QString &model);
// `effort` when the model accepts it, else the nearest accepted effort (by rank low<medium<high<xhigh).
QString clampEffort(const QString &model, const QString &effort);
// Channel key: one persistent process per key. claude and agy fix model+effort at spawn, so each
// (model, effort) is its own warm process; codex app-server takes model+effort per turn: one daemon.
QString channelKeyFor(const QString &model, const QString &effort);
QString agyModelId(const QString &model, const QString &effort);  // gemini-4-carbon + medium -> gemini-4-carbon-medium

// Exact helper command lines (unit-tested).
QStringList quickHelperArgs(const QString &sessionId, const QString &model = QStringLiteral("claude-sonnet-5-5"),
                            const QString &effort = QStringLiteral("low"));  // claude, stripped of hooks/MCP/settings
QStringList agyChatArgs(const QString &model, const QString &effort);  // persistent agy stdio chat process
QByteArray cswapHookPayload(const QString &sessionId);  // SessionStart for cswap-private hook
QByteArray claudeUserLine(const QString &text);         // one stream-json user message
QStringList traceHelperArgs();                          // warm agy stdio process
QByteArray agyUserLine(const QString &text);
QStringList codexAppServerArgs();                       // warm codex app-server (JSON-RPC lines)
QByteArray codexRpc(int id, const QString &method, const QJsonObject &params);
QJsonObject codexThreadStartParams(const QString &instructions);
QJsonObject codexTurnParams(const QString &threadId, const QString &text, const QString &model, const QString &effort);
// One parsed line from the codex app-server stdout (JSON-RPC 2.0 framing, one object per line).
struct CodexEvent {
    enum Kind { None, Response, AgentDelta, ReasoningDelta, TurnCompleted, Error, ItemStarted };
    Kind kind = None;
    int id = -1;           // Response / Error: the request id it answers
    QString threadId;      // Response to thread/start: the new thread; notifications: their thread
    QString text;          // deltas; Error: message; ItemStarted: the item type ("reasoning", ...)
};
CodexEvent parseCodexLine(const QByteArray &line);

// The live status of a quick ask before (and while) answer tokens arrive, from one provider
// stream line: "thinking", "running run_command", "" when the line says nothing new.
// Also returns any thinking text the line carries and, at a turn's end, its thinking-token count.
struct StreamProgress {
    QString phase;
    QString thinking;
    QString text;
    int thinkingTokens = -1;
};
StreamProgress agyProgress(const QJsonObject &line);     // agy stream-json (step_update / result)
StreamProgress claudeProgress(const QJsonObject &line);  // claude stream-json (stream_event deltas)
// The newest `maxWords` words of a stream (the live thinking line scrolls, it never freezes).
QString tailWords(const QString &s, int maxWords);

// OpenAI-compatible HTTP/SSE fast-feedback helpers (unit-tested).
struct OpenAiSseEvent {
    enum Kind { None, Delta, Done, Error };
    Kind kind = None;
    QString thinking;
    QString text;
    QString error;
};
QByteArray buildOpenAiCompatRequest(const QString &model, const QString &systemPrompt,
                                      const QString &context, const QString &question,
                                      const QString &effort = QString());
OpenAiSseEvent parseOpenAiSseLine(const QByteArray &line);

struct ParsedCache {
    qint64 mtime = 0;
    qint64 size = 0;
    Extracted ex;
};

class Backend : public QObject {
    Q_OBJECT
    Q_PROPERTY(QVariantList projects READ projects NOTIFY projectsChanged)
    Q_PROPERTY(bool live READ live CONSTANT)
    Q_PROPERTY(QStringList folders READ folders NOTIFY foldersChanged)
    Q_PROPERTY(QString defaultFolder READ defaultFolder NOTIFY foldersChanged)
    Q_PROPERTY(QVariantMap harnessModels READ harnessModels CONSTANT)
    Q_PROPERTY(QString quickQuestion READ quickQuestion NOTIFY quickChanged)
    Q_PROPERTY(QString quickAnswer READ quickAnswer NOTIFY quickChanged)
    Q_PROPERTY(bool quickBusy READ quickBusy NOTIFY quickChanged)
    Q_PROPERTY(double quickSeconds READ quickSeconds NOTIFY quickChanged)
    Q_PROPERTY(QString traceQuestion READ traceQuestion NOTIFY traceChanged)
    Q_PROPERTY(QString traceThinking READ traceThinking NOTIFY traceChanged)
    Q_PROPERTY(QString traceAnswer READ traceAnswer NOTIFY traceChanged)
    Q_PROPERTY(bool traceBusy READ traceBusy NOTIFY traceChanged)
    Q_PROPERTY(double traceSeconds READ traceSeconds NOTIFY traceChanged)
    Q_PROPERTY(QString toast READ toast NOTIFY toastChanged)
    Q_PROPERTY(double uiScale READ uiScale WRITE setUiScale NOTIFY prefsChanged)
    Q_PROPERTY(QString navMode READ navMode WRITE setNavMode NOTIFY prefsChanged)
    Q_PROPERTY(bool firstLaunch READ firstLaunch NOTIFY prefsChanged)
    Q_PROPERTY(QString chatModel READ chatModel WRITE setChatModel NOTIFY prefsChanged)
    Q_PROPERTY(QString chatEffort READ chatEffort WRITE setChatEffort NOTIFY prefsChanged)
    Q_PROPERTY(QString chatProvider READ chatProvider NOTIFY prefsChanged)
    Q_PROPERTY(QStringList chatEfforts READ chatEfforts NOTIFY prefsChanged)
    Q_PROPERTY(QVariantList models READ models CONSTANT)
    Q_PROPERTY(bool modelSuggestions READ modelSuggestions WRITE setModelSuggestions NOTIFY prefsChanged)
    Q_PROPERTY(QString suggestion READ suggestion NOTIFY suggestionChanged)
    Q_PROPERTY(QString suggestionKey READ suggestionKey NOTIFY suggestionChanged)
    Q_PROPERTY(QString quickThinking READ quickThinking NOTIFY quickChanged)
    Q_PROPERTY(QString quickModel READ quickModel NOTIFY quickChanged)
    Q_PROPERTY(QString quickPhase READ quickPhase NOTIFY quickChanged)
    Q_PROPERTY(double quickElapsed READ quickElapsed NOTIFY quickTick)
    Q_PROPERTY(int quickThinkingTokens READ quickThinkingTokens NOTIFY quickChanged)
    Q_PROPERTY(QString focusedProject READ focusedProject WRITE setFocusedProject NOTIFY projectsChanged)
    Q_PROPERTY(QString openaiCompatUrl READ openaiCompatUrl WRITE setOpenaiCompatUrl NOTIFY prefsChanged)
    Q_PROPERTY(QString openaiCompatModel READ openaiCompatModel WRITE setOpenaiCompatModel NOTIFY prefsChanged)
    // PS4 DualShock 4 / extended gamepad. Always present; only main()'s live window enables it.
    Q_PROPERTY(QObject *gamepad READ gamepadObject CONSTANT)

public:
    enum class Mode { Live, Mock };
    // warmHelpers=false: live reads only (registry, transcripts, board), no model helpers.
    explicit Backend(Mode mode, QObject *parent = nullptr, bool warmHelpers = true);
    ~Backend() override;

    QVariantList projects() const { return m_projectList; }
    bool live() const { return m_mode == Mode::Live; }
    QStringList folders() const { return m_folders; }
    QString defaultFolder() const { return m_defaultFolder; }
    QVariantMap harnessModels() const;
    QString quickQuestion() const { return m_quickQuestion; }
    QString quickAnswer() const { return m_quickAnswer; }
    bool quickBusy() const { return m_quickBusy; }
    double quickSeconds() const { return m_quickSeconds; }
    QString traceQuestion() const { return m_traceQuestion; }
    QString traceThinking() const { return m_traceThinking; }
    QString traceAnswer() const { return m_traceAnswer; }
    bool traceBusy() const { return m_traceBusy; }
    double traceSeconds() const { return m_traceSeconds; }
    QString toast() const { return m_toast; }
    QString quickThinking() const { return m_quickThinking; }
    QString quickModel() const { return m_quickModel; }
    QString quickPhase() const { return m_quickPhase; }
    double quickElapsed() const { return m_quickElapsed; }
    int quickThinkingTokens() const { return m_quickThinkTokens; }
    // Stream timing of the newest quick ask (ms from send; -1 = not yet): first sign of life
    // (phase or thinking), first thinking text, first answer token. --probe-chat prints them.
    qint64 quickFirstEventMs() const { return m_quickFirstEventMs; }
    qint64 quickFirstThinkingMs() const { return m_quickFirstThinkMs; }
    qint64 quickFirstTokenMs() const { return m_quickFirstTokenMs; }
    int quickThinkingDeltas() const { return m_quickThinkDeltas; }

    // The project on screen (Ladder / Stage). New turns appended to it never raise its +N badge;
    // every other project's island counts them until it is opened.
    QString focusedProject() const { return m_focusedProject; }
    Gamepad *gamepad() { return &m_gamepad; }
    QObject *gamepadObject() { return &m_gamepad; }
    void setFocusedProject(const QString &key);
    int freshTurns(const QString &key) const { return m_fresh.value(key); }
    // Mock only: append one turn to a fixture project's transcript and re-extract it, as if the
    // agent had written while the user looked elsewhere (tests, demo).
    void mockAppendTurn(const QString &key, const Turn &t);
    // Kill every daemon now, closing stdin first so it can exit on its own (the destructor runs
    // this). Returns the PIDs that were running.
    QList<qint64> shutdownChannels();

    // UI preferences: font/zoom scale (0.75..1.60), input mode (vim | wasd | mouse; there is no
    // arrows mode: arrow keys are a silent fallback, and an old "arrows" file value becomes wasd),
    // the default chat model + effort, and opt-in model suggestions for the Tab predictor.
    // Live mode persists them to ~/.lapis-next/studio_prefs.json (atomic, 0600); Mock mode keeps
    // them in memory unless a test points setPrefsPath() at a temp file.
    static constexpr double kMinScale = 0.75;
    static constexpr double kMaxScale = 1.60;
    static double clampScale(double s);
    static QString defaultPrefsPath();
    double uiScale() const { return m_uiScale; }
    QString navMode() const { return m_navMode; }
    void setUiScale(double s);
    void setNavMode(const QString &mode);
    Q_INVOKABLE void zoomBy(double factor) { setUiScale(m_uiScale * factor); }
    Q_INVOKABLE void resetZoom() { setUiScale(1.0); }
    Q_INVOKABLE void cycleNavMode();
    static QStringList navModes() { return {QStringLiteral("controller"), QStringLiteral("vim"), QStringLiteral("wasd"), QStringLiteral("mouse")}; }
    // True when the interactive launch found no prefs file: the onboarding modal shows.
    bool firstLaunch() const { return m_firstLaunch; }
    Q_INVOKABLE void completeOnboarding(const QString &mode);  // picks the mode and writes the file now
    Q_INVOKABLE void showOnboarding();                         // explicit trigger (tests, demo)

    QString chatModel() const { return m_chatModel; }
    QString chatEffort() const { return m_chatEffort; }
    QString chatProvider() const { return providerFor(m_chatModel); }
    QStringList chatEfforts() const { return effortsFor(m_chatModel); }
    QVariantList models() const;
    void setChatModel(const QString &model);    // unknown ids are ignored; effort is clamped to the model
    void setChatEffort(const QString &effort);  // ignored unless the current model accepts it
    Q_INVOKABLE void cycleModel(int delta);     // W/S, k/j, Up/Down in the picker: across all providers
    Q_INVOKABLE void cycleEffort(int delta);    // A/D, h/l, Left/Right: within the model's efforts, no wrap
    Q_INVOKABLE void warmChannel();             // live: start the selected model's daemon now (no-op in mock)
    Q_INVOKABLE QString shortModelLabel(const QString &model) const;
    Q_INVOKABLE QString modelSpeed(const QString &model) const;  // "Fast", "Deep Think", ...
    bool modelSuggestions() const { return m_modelSuggestions; }
    void setModelSuggestions(bool on);
    QString suggestion() const { return m_suggestion; }
    QString suggestionKey() const { return m_suggestionKey; }
    // Opt-in warm model suggestion for the composer's Tab prediction (codex app-server, own thread).
    Q_INVOKABLE void requestSuggestion(const QString &projectKey, int artifactIndex);
    // What the latest ask was routed to (tests): model, effort and channel key.
    QString lastAskModel() const { return m_lastAskModel; }
    QString lastAskEffort() const { return m_lastAskEffort; }
    QString lastAskChannel() const { return m_lastAskChannel; }
    int channelCount() const { return int(m_channels.size()); }
    // PID of the daemon behind KEY, 0 if none is running (lets --probe-chat prove reuse).
    qint64 channelPid(const QString &key) const;
    QString prefsPath() const { return m_prefsPath; }
    void setPrefsPath(const QString &path);  // loads from it; empty disables persistence
    bool savePrefs();                        // write now (normally debounced 400 ms after a change)
    QString openaiCompatUrl() const { return m_openaiCompatUrl; }
    QString openaiCompatModel() const { return m_openaiCompatModel; }
    void setOpenaiCompatUrl(const QString &url);
    void setOpenaiCompatModel(const QString &model);

    // The prompt preamble for quick asks and traces: on-screen artifact, last turns, files, tools, write-up.
    QString contextFor(const QString &projectKey, int artifactIndex) const;

    // Every external process goes through spawn(); in Mock mode it refuses.
    int spawnCount() const { return m_spawnCount; }
    int blockedSpawns() const { return m_blockedSpawns; }
    QStringList transcriptsRead() const { return m_transcriptsRead; }

    Q_INVOKABLE QString startSession(const QString &folder, const QString &harness, const QString &model,
                                     const QString &goal);
    Q_INVOKABLE void ask(const QString &projectKey, int artifactIndex, const QString &question);
    Q_INVOKABLE void think(const QString &projectKey, int artifactIndex, const QString &question);
    Q_INVOKABLE void steer(const QString &projectKey, const QString &text);
    Q_INVOKABLE void editAssumption(const QString &projectKey, int index, const QString &text);
    Q_INVOKABLE void togglePin(const QString &projectKey, int index);
    Q_INVOKABLE void clearAnswers();
    Q_INVOKABLE QString shortModel(const QString &model) const { return studio::shortModel(model); }
    Q_INVOKABLE int projectIndex(const QString &key) const;
    Q_INVOKABLE int pdfPages(const QString &path) const;
    Q_INVOKABLE QVariantList audioPeaks(const QString &path) const { return studio::audioPeaks(path, 160); }
    Q_INVOKABLE int wordCount(const QString &s) const { return studio::countWords(s); }
    static bool isRedactedCategory(const QString &category);

    // --harness codex: quick asks default to gpt-6-astra (not persisted; the picker still switches).
    void setQuickHarness(const QString &h);

    // Mock clock: advances scripted replies by dt seconds (the demo steps it per frame).
    void advance(double dt);
    void setAutoAdvance(bool on);

    // Live: (re)scan the Lapis registry and transcripts.
    void refresh();

    // --bench-fps instrumentation: wall time each backend step spends on the GUI thread, and the
    // transcript scan time spent on the worker thread (off the GUI thread by construction).
    struct GuiStall {
        double refreshMs = 0, applyMs = 0, publishMs = 0, boardMs = 0, workerMs = 0;
        double guiParseMs = 0;  // transcript scan time that ran on the GUI thread (should be 0)
        double maxMs = 0;  // longest single GUI-thread slice
        int refreshes = 0, applies = 0, publishes = 0, boards = 0, emits = 0, parsed = 0;
    };
    const GuiStall &guiStall() const { return m_stall; }
    void resetGuiStall() { m_stall = {}; }
    bool scanning() const { return m_scanning; }
    // Forget parsed transcripts so the next refresh() re-reads every one (a burst of live updates).
    void benchInvalidateCache() { m_cache.clear(); }
    // Re-run applyScan() on the current registry and cache: what a landed live scan costs.
    void benchReapply();

signals:
    void projectsChanged();
    void foldersChanged();
    void quickChanged();
    void traceChanged();
    void toastChanged();
    void prefsChanged();
    void suggestionChanged();
    void sessionStarted(const QString &projectKey);
    void quickTick();

private:
    struct Ladder {
        QString goal;
        QVariantList assumptions;  // [{text, pinned}]
        QVariantList arms;         // [{name, value, progress, best}]
        QString armsLabel;
        QString winner;
    };
    struct Project {
        QString key;
        QString name;
        QString dir;
        QList<SessionInfo> sessions;
        Extracted ex;
        Ladder ladder;
        bool ladderOverride = false;  // a kickoff from this app, until the transcript speaks
        qint64 mtime = 0;
        bool pending = false;
        QList<Turn> mockTurns;  // mock only: the fixture transcript, so appended turns re-extract
    };
    struct MockScript {
        QString projectKey;
        double t = -1;  // seconds since kickoff, <0 inactive
    };
    struct MockReply {
        double t = -1;
        double duration = 0;
        QString question;
        QString thinking;
        QString answer;
    };

    QProcess *spawn(const QString &program, const QStringList &args, const QString &cwd = QString(),
                    const QProcessEnvironment &env = QProcessEnvironment());
    void publish();
    QVariantMap projectMap(const Project &p) const;
    Project *find(const QString &key);
    void setToast(const QString &t);
    void loadPrefs();

    // live
    void loadWorkspace();
    void scanTranscripts();
    // added: transcript path -> turns appended since its previous parse (this scan only).
    void applyScan(const QList<SessionInfo> &sessions, const QHash<QString, ParsedCache> &cache,
                   const QHash<QString, int> &added = {});
    void notePhase(const QString &phase);
    void noteThinking(const QString &delta);
    void noteAnswer(const QString &answer);   // the whole answer so far (clamped)
    void appendAnswer(const QString &delta);  // one raw text delta
    void pollBoard();
    // Persistent provider channels (live only).
    struct Channel {
        QString key, provider, model, effort;
        QPointer<QProcess> proc;
        QByteArray buf;
        bool starting = false;     // claude: the cswap hook runs before the process spawns
        QByteArray pending;        // lines written once the process is up
        QString sessionId;
        qint64 lastUsed = 0;
        int inFlight = 0;          // turns written but not yet finished
        // codex app-server
        int nextId = 1;
        int threadReq = -1, suggestReq = -1;
        QString thread, suggestThread;
        QList<QJsonObject> queue;         // quick turns waiting for the thread
        QList<QJsonObject> suggestQueue;  // suggestion turns waiting for their thread
    };
    Channel *ensureChannel(const QString &model, const QString &effort);
    void spawnChannel(Channel *c);
    void onChannelOutput(const QString &key);
    void flushCodex(Channel *c);
    void finishQuick();
    void evictChannels();
    void ensureTraceHelper();
    void onTraceOutput();
    void finishTrace();
    void askOpenAiCompat(const QString &projectKey, int artifactIndex, const QString &question);
    void onOpenAiCompatReadyRead();
    void onOpenAiCompatFinished();
    void sendWithRetry(const QString &agent, const QString &text, int attempt);
    void loadFolders();

    // mock
    void loadFixtures();
    void advanceKickoff(Project &p, double t);

    Mode m_mode;
    QList<Project> m_projects;
    QVariantList m_projectList;
    QStringList m_folders;
    QString m_defaultFolder;
    QString m_toast;
    double m_uiScale = 1.0;
    QString m_navMode = QStringLiteral("controller");
    bool m_firstLaunch = false;
    QString m_chatModel = QStringLiteral("claude-sonnet-5-5");
    QString m_chatEffort = QStringLiteral("low");
    bool m_modelSuggestions = false;
    QString m_suggestion, m_suggestionKey, m_suggestPendingKey;
    QString m_lastAskModel, m_lastAskEffort, m_lastAskChannel;
    QString m_quickThinking, m_quickModel, m_quickPhase;
    QString m_quickAnswerRaw, m_quickThinkRaw;  // unclamped streams the shown lines derive from
    double m_quickElapsed = 0;
    int m_quickThinkTokens = -1;
    qint64 m_quickFirstEventMs = -1, m_quickFirstThinkMs = -1, m_quickFirstTokenMs = -1;
    int m_quickThinkDeltas = 0;
    QTimer m_quickTicker;  // live: 10 Hz elapsed ticker while an ask is in flight
    QString m_focusedProject;
    QHash<QString, int> m_fresh;  // project key -> turns appended while it was not on screen
    QString m_prefsPath;
    QTimer m_prefsSave;
    QString m_openaiCompatUrl;
    QString m_openaiCompatModel = QStringLiteral("qwen3.8-flash-next");
    QString m_openaiCompatKey;
    QNetworkAccessManager m_nam;
    QPointer<QNetworkReply> m_compatReply;
    QByteArray m_compatBuf;
    Gamepad m_gamepad;

    QString m_quickQuestion, m_quickAnswer;
    bool m_quickBusy = false;
    double m_quickSeconds = 0;
    QString m_traceQuestion, m_traceThinking, m_traceAnswer;
    bool m_traceBusy = false;
    double m_traceSeconds = 0;

    mutable QHash<QString, QPair<qint64, int>> m_pdfPages;  // path -> (mtime, page count)
    GuiStall m_stall;
    void noteStall(double ms) { m_stall.maxMs = std::max(m_stall.maxMs, ms); }
    QList<SessionInfo> m_lastSessions;

    int m_spawnCount = 0;
    int m_blockedSpawns = 0;
    QStringList m_transcriptsRead;

    // live state
    QFileSystemWatcher m_watcher;
    QTimer m_rescanDebounce;
    QTimer m_boardTimer;
    bool m_boardBusy = false;  // one board poll in flight at a time
    bool m_scanning = false;
    bool m_rescanQueued = false;
    QHash<QString, ParsedCache> m_cache;
    QHash<QString, QString> m_statusById;
    QHash<QString, Ladder> m_kickoffs;  // project key -> ladder seeded from the new-session dialog
    QTimer m_ageTimer;  // keeps "live · 3h 12m" island metrics current
    QHash<QString, Channel *> m_channels;  // owned; deleted in the destructor or when the process dies
    QString m_quickChannel;                // the channel the newest quick ask went to
    QElapsedTimer m_quickClock;
    // warm agy
    QPointer<QProcess> m_traceProc;
    QByteArray m_traceBuf;
    QElapsedTimer m_traceClock;
    int m_traceInFlight = 0;  // turns written but not yet resulted; only the newest one is shown
    bool m_traceSawThinking = false;
    QString m_traceRaw;

    // mock state
    MockScript m_kickoff;
    MockReply m_mockQuick;
    MockReply m_mockTrace;
    int m_mockQuickCount = 0;
    double m_mockSuggestT = -1;
    QTimer m_autoTimer;
    QElapsedTimer m_autoClock;
};

}  // namespace studio
