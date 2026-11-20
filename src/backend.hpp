// Live Lapis bridge and the clean mock backend used by --test / --record-demo.
#pragma once

#include "gamepad.hpp"
#include "transcript.hpp"
#include "core/types.hpp"
#include "core/window_geometry.hpp"
#include "ingest/memory.hpp"
#include "ingest/phonon.hpp"
#include "ingest/slowmix.hpp"
#include "ingest/spotify.hpp"
#include "llm/prefix.hpp"
#include "llm/router.hpp"
#include "llm/sse.hpp"
#include "targets/target_registry.hpp"

#include <QElapsedTimer>
#include <QFileSystemWatcher>
#include <QHash>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QSet>
#include <QThreadPool>
#include <QTimer>
#include <QVariantList>
#include <QVariantMap>
#include <memory>
#include "mac/cua_shadow.hpp"

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
// The island's plain-English 1-to-2-turn state, no metrics and no uptime:
//   now     what the agent is on ("Sweeping sm120 FP4 tile sizes"), from the job name or goal
//   last    the latest completed step or receipt, or (blocker) the decision the agent waits on
//   lastLong  the same step at briefing length (<= 16 words; the island line keeps <= 9)
struct IslandState {
    QString now;
    QString last;
    QString lastLong;
    bool blocker = false;
};
IslandState islandState(const Extracted &ex, const QString &status);

// Morning Briefing, local fallback: only changed / active projects (needs-ok, fresh turns,
// live), two lines each. Line 1: what happened + concrete proof. Line 2: one next action.
// Takes the published project maps; returns [{key, name, index, line1, line2, blocker}].
QVariantList localBriefing(const QVariantList &projects);
// Merge a model reply ("key | line1 | line2" per line) over the local briefing; rows the model
// skipped or mangled keep their local lines. Returns how many rows the model rewrote.
int mergeBriefingReply(QVariantList &rows, const QString &reply);

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
// OpenAI-compatible HTTP/SSE fast-feedback helpers (OpenAiSseEvent, parseOpenAiSseLine,
// buildOpenAiCompatRequest, buildOpenAiWarmRequest, kCompatSystemPrompt) live in lapis_llm
// (llm/sse.hpp, llm/prefix.hpp); readPhononTail / readPhononSnapshot live in lapis_ingest
// (ingest/phonon.hpp). All are re-exported through the includes at the top of this header.

// Human-interest filter: the request asking the fast model to keep only what matches the human's
// direction (goal), pinned params / terms and what they are inspecting right now (recent actions),
// dropping boilerplate and routine steps, and to say whether a chart or summary text fits better.
// tasteRules: the human's saved standing rules (session_reviews.jsonl), listed for the model.
QByteArray buildFilterPrompt(const QString &model, const QString &goal, const QVariantList &params,
                             const QStringList &recentActions, const QString &latestTurn,
                             const QStringList &tasteRules = {});

// Visual-first Stage deck: every non-diff artifact (chart, table, image, pdf, media, summary, ...)
// keeps its extracted order and leads; diffs follow only when showDiffs (the Code pill), else the
// deck drops them. The backend indexes artifacts in this same order (contextSlice, asks).
QVariantList stageArtifacts(const QVariantList &artifacts, bool showDiffs);
// A project that waits on the human (the Needs Me filter): needs-ok / waiting / gate status, or
// the agent's last turn is a question for them.
bool projectNeedsMe(const QString &status, bool blocker);

// End-of-session reflection: original intent vs what matched, the human's corrections, and 1-3
// personal taste rules to keep.
struct SessionReflection {
    QString intent;
    QString corrections;
    QStringList rules;
    bool isEmpty() const { return intent.isEmpty() && corrections.isEmpty() && rules.isEmpty(); }
};
// Local fallback (mock / offline): from each non-redacted project's goal, latest step and the
// user turns that steer or correct the agent. Takes [{name, goal, last, userTurns: [..]}].
SessionReflection localReflection(const QVariantList &sessions);
// Parses "INTENT: ... / CORRECTIONS: ... / RULES: - ..." (tolerant of case and stray markdown).
SessionReflection parseReflectionReply(const QString &reply);
// A user turn that steers or corrects the agent ("no, lead with the chart", "don't ...").
bool isSteerTurn(const QString &text);
struct HumanFilterReply {
    QString stateLine;  // one plain-English line, what matters now
    QString summary;    // key points, one per line, no bullets
    QString form;       // "chart" | "summary" | "" when the model did not say
};
// Parses "STATE: ... / FORM: chart|summary / POINTS: - ..." (tolerant of case and stray markdown).
HumanFilterReply parseFilterReply(const QString &reply);

struct ParsedCache {
    qint64 mtime = 0;
    qint64 size = 0;
    Extracted ex;
};

// A Lapis tab title that names nothing: empty, "tmp", or still the $HOME folder's basename.
bool isGenericTabTitle(const QString &title);
// The island a live session joins. projectIdFor(), except a scratch-dir tab whose title is still
// generic: each such tab is its own "~tab:<shortId>" island (distinct $HOME sessions never pool
// into one category bucket), named from its first request, else its job name, else its
// lapis-peers board snippet, else "tab <shortId>".
ProjectId sessionProjectId(const SessionInfo &s, const Extracted &ex, const QString &boardSnippet = QString());

// A Studio UI action, either from the fast natural-language matcher or a model's
// "[[ACTION:name]]" / "[[ACTION:name:arg]]" tag. Actions: open_music, close_music,
// play_track:<query>, pause_music, resume_music, toggle_music, open_review, filter_needs_me,
// filter_all, reset_layout, open_settings, open_briefing, set_theme:blue|gold, show_diffs:1|0,
// spawn_agent:<task>, set_volume:<pct>, open_spotify, spotify_play[:<song>], spotify_pause,
// spotify_next, spotify_prev, spotify_volume:<pct>, focus_project:<name>, open_stage:<name>,
// open_ladder:<name>, go_home, sort_map:category|recent|activity,
// move_cluster:<cluster>:left|right|top|bottom, cua_click:<target>, cua_type:<target>|<text>,
// cua_scroll:<target>|<deltaY>. A model reply may carry several; all run in one turn.
struct StudioCommand {
    QString action;
    QString arg;
    bool steer = false;  // spawn_agent phrased "tell the agent to ...": steer the live peer first
    bool isEmpty() const { return action.isEmpty(); }
};
QStringList studioActionNames();
// Plain-English UI commands ("can you open the slowmix tab", "play syntheticgod", "switch to
// lapis blue"); an empty command when the text is a question for the model. Case-insensitive,
// tolerant of voice-typed filler ("can you", "please", "inside of this").
StudioCommand parseStudioCommand(const QString &text);
// Strips every leading [[ACTION:...]] tag from *answer (and the whitespace after it) and returns
// the known actions in order. Unknown action names are dropped (still stripped).
QList<StudioCommand> extractActionTags(QString *answer);
// A visible computer-use action from a cua_click / cua_type / cua_scroll tag. cua_type's arg is
// "<target>|<text>" (split at the first '|'); cua_scroll's is "<target>|<deltaY>" (split at the
// last '|'; deltaY is an integer line count, positive scrolls up, or "up" / "down" for +-5).
struct CuaCommand {
    QString kind;  // "click", "type", "scroll"
    QString target;
    QString text;
    int deltaY = 0;
    bool valid = false;
};
CuaCommand parseCuaAction(const QString &action, const QString &arg);
// Closed-loop computer use: one finished cua_* action as the model sees it next turn.
struct CuaStepResult {
    QString action;  // "cua_click", "cua_observe", ...
    QString target;
    bool ok = false;
    QString tier, detail, observation;
};
// At most this many automatic follow-up turns per user prompt.
constexpr int kMaxCuaLoopDepth = 4;
// "[CUA observation after <action> (<tier>, ok=<0|1>, detail=<detail>)]:\n<observation>".
QString cuaObservationPrompt(const QString &action, const QString &tier, bool ok, const QString &detail,
                             const QString &observation);
// A batch earns a follow-up turn when it read the screen (cua_observe) or an action failed,
// and fewer than kMaxCuaLoopDepth follow-ups have run for this prompt.
bool cuaShouldContinue(const QVector<CuaStepResult> &steps, int depth);
// Every step's observation block, then the "Continue only if more actions are needed" line.
QString cuaFollowupPrompt(const QVector<CuaStepResult> &steps);
// The Constellation cluster a project sits in: "Work" (the infinity category), "Apps" (apps),
// "Quick Qs" (a ~tab: island, a scratch-dir session, or the quick Qs category), else "Projects".
QString mapClusterFor(const QString &key, const QString &category, const QString &dir);
// "work" / "quick qs" / "quick questions" / "app" -> the canonical cluster name; "" when unknown.
QString normalizeClusterName(const QString &name);
QStringList defaultClusterOrder();  // Work, Projects, Apps, Quick Qs
// One step of `cluster` on the two-column cluster grid the map draws (only the clusters in
// `present` take a cell, in `order`): left / right swap within a row, top / bottom within a
// column. A blocked or unknown move returns `order` unchanged.
QStringList moveClusterInOrder(const QStringList &order, const QStringList &present, const QString &cluster,
                               const QString &dir);
// The track (index into SlowmixController::tracks maps) a spoken query names best, -1 for none.
// Matches on the name and file name with spaces, underscores and punctuation ignored.
int matchSlowmixTrack(const QVariantList &tracks, const QString &query);

// One agent row of `lapis-peers board --json`: its status and latest reply snippet ("" when the
// board withholds it, as for infinity).
struct BoardRow {
    QString id;
    QString status;
    QString snippet;
};
QList<BoardRow> parseBoardJson(const QByteArray &json);

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
    Q_PROPERTY(QString theme READ theme WRITE setTheme NOTIFY prefsChanged)
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
    Q_PROPERTY(QString activeCategory READ activeCategory NOTIFY projectsChanged)
    Q_PROPERTY(QString phononTail READ phononTail NOTIFY phononTailChanged)
    Q_PROPERTY(bool phononLive READ phononLive NOTIFY phononTailChanged)
    Q_PROPERTY(QString openaiCompatUrl READ openaiCompatUrl WRITE setOpenaiCompatUrl NOTIFY prefsChanged)
    Q_PROPERTY(QString openaiCompatModel READ openaiCompatModel WRITE setOpenaiCompatModel NOTIFY prefsChanged)
    // Last IDE view, restored on the first projects publish: {lens: "constellation"|"stage"|"ladder",
    // focusedProject, artifactIndex, needsMeOnly}. Persisted in studio_prefs.json (live window only).
    Q_PROPERTY(QVariantMap viewState READ viewState NOTIFY viewStateChanged)
    // PS4 DualShock 4 / extended gamepad. Always present; only main()'s live window enables it.
    Q_PROPERTY(QObject *gamepad READ gamepadObject CONSTANT)
    // slowmix music player (SlowmixController). Spawns `slowmix --stdio` only after main()'s live
    // window calls setSpawnAllowed(true); Mock and every offscreen mode are dry runs.
    Q_PROPERTY(QObject *slowmix READ slowmix CONSTANT)
    // Spotify via the spotify_player CLI (SpotifyController); same spawn rule as slowmix. Starting
    // one pauses the other.
    Q_PROPERTY(QObject *spotify READ spotify CONSTANT)
    // Ambient voice: new "You:" lines in the Phonon tail become a no-think intent request (800 ms
    // debounce): [[IGNORE]] stays silent, action tags run, a short answer lands in ambientAnswer.
    Q_PROPERTY(bool ambientVoiceEnabled READ ambientVoiceEnabled WRITE setAmbientVoiceEnabled NOTIFY ambientChanged)
    Q_PROPERTY(bool ambientBusy READ ambientBusy NOTIFY ambientChanged)
    // Passive human-click shadow recorder (~/.lapis-next/cua_traces/human_traces.jsonl): on in the
    // real live window unless LAPIS_SHADOW_RECORD=0 or "shadowRecordEnabled": false in prefs;
    // always false in Mode::Mock and every offscreen mode (--test, --demo, --bench).
    Q_PROPERTY(bool shadowRecordEnabled READ shadowRecordEnabled WRITE setShadowRecordEnabled NOTIFY shadowRecordChanged)
    Q_PROPERTY(QString ambientAnswer READ ambientAnswer NOTIFY ambientChanged)
    Q_PROPERTY(QString ambientHeard READ ambientHeard NOTIFY ambientChanged)
    // Ambient action tags are proposed, not run: they wait here (and on the all-Spaces HUD bar)
    // for L3 + R3 (confirmPendingProposal) and expire after 15 s.
    Q_PROPERTY(bool hasPendingProposal READ hasPendingProposal NOTIFY pendingProposalChanged)
    Q_PROPERTY(QString pendingProposalText READ pendingProposalText NOTIFY pendingProposalChanged)
    // Hovered / pinned semantic targets (TargetRegistry); viewers push, FloatingOmnibar reads.
    Q_PROPERTY(QObject *targets READ targets CONSTANT)
    // Morning Briefing: changed / active projects in two lines each (see localBriefing()).
    Q_PROPERTY(QVariantList briefing READ briefing NOTIFY briefingChanged)
    Q_PROPERTY(bool briefingBusy READ briefingBusy NOTIFY briefingChanged)
    Q_PROPERTY(QString briefingSource READ briefingSource NOTIFY briefingChanged)
    // Visual-first Stage: diffs stay out of the deck unless the Code pill (or Settings) turns them on.
    Q_PROPERTY(bool showCodeDiffs READ showCodeDiffs WRITE setShowCodeDiffs NOTIFY prefsChanged)
    // Freeform Constellation: {projectKey: {x, y}}, the card centre as a fraction (0..1) of the
    // Constellation's width / height. Projects without an entry keep their auto-layout slot.
    Q_PROPERTY(QVariantMap cardPositions READ cardPositions NOTIFY cardPositionsChanged)
    // Constellation clusters: card order inside each cluster ("category" keeps the project order,
    // "recent" newest first, "activity" needs-me then live work first) and the cluster order on
    // the two-column cluster grid. Both persist; sort_map / move_cluster voice actions set them.
    Q_PROPERTY(QString mapSort READ mapSort WRITE setMapSort NOTIFY mapLayoutChanged)
    Q_PROPERTY(QStringList clusterOrder READ clusterOrder NOTIFY mapLayoutChanged)
    // End-of-session reflection: {intent, corrections, rules: [..]} plus where it came from.
    Q_PROPERTY(QVariantMap reflection READ reflection NOTIFY reflectionChanged)
    Q_PROPERTY(bool reflectionBusy READ reflectionBusy NOTIFY reflectionChanged)
    Q_PROPERTY(QString reflectionSource READ reflectionSource NOTIFY reflectionChanged)
    // Saved taste rules (rating > 0), newest last; they ride along in the fast-model filter prompt.
    Q_PROPERTY(QStringList tasteRules READ tasteRules NOTIFY tasteRulesChanged)

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
    QString traceThinking() const { return stripToolNoise(m_traceThinking); }
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
    // Mock only: tag a fixture project's session with a Lapis category, or add a bare project
    // (the cluster tests and the --snap-map frame).
    void mockSetCategory(const QString &key, const QString &category);
    void mockAddProject(const QString &key, const QString &name, const QString &dir, const QString &category,
                        const QString &status);
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
    // Accent theme: "gold" (default, #e09f3e / #b5653b) | "lapis" (Lapis Blue, #4f8cff / #2d5aa8).
    QString theme() const { return m_theme; }
    void setTheme(const QString &theme);  // unknown ids are ignored
    static QStringList themes() { return {QStringLiteral("gold"), QStringLiteral("lapis")}; }
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
    // Morning Briefing: the local lines land at once; with an OpenAI-compatible endpoint
    // configured (live mode) qwen3.8-flash-next rewrites them when it answers.
    QVariantList briefing() const { return m_briefing; }
    bool briefingBusy() const { return m_briefingReply != nullptr; }
    QString briefingSource() const { return m_briefingSource; }
    Q_INVOKABLE void requestBriefing();
    bool showCodeDiffs() const { return m_showCodeDiffs; }
    void setShowCodeDiffs(bool on);
    QVariantMap cardPositions() const { return m_cardPositions; }
    // x, y: the card centre as a fraction of the Constellation (clamped to 0..1). Persisted.
    Q_INVOKABLE void setCardPosition(const QString &projectKey, double x, double y);
    Q_INVOKABLE void resetCardPositions();
    QString mapSort() const { return m_mapSort; }
    void setMapSort(const QString &sort);
    QStringList clusterOrder() const { return m_clusterOrder; }
    // Moves one cluster a step (left|right|top|bottom) among the clusters on screen. Dragged card
    // positions reset so the whole map re-flows. False when the move is blocked or unknown.
    Q_INVOKABLE bool moveCluster(const QString &cluster, const QString &dir);
    QStringList presentClusters() const;  // the clusters with at least one project, in clusterOrder
    // Session reflection: local lines land at once; with an OpenAI-compatible endpoint and a
    // router, one "reflection:1" stream to the fast model (qwen3.8-flash-next) rewrites them.
    QVariantMap reflection() const;
    bool reflectionBusy() const { return m_reflectionBusy; }
    QString reflectionSource() const { return m_reflectionSource; }
    Q_INVOKABLE void runSessionReflection();
    // Appends {ts, rule, rating} to session_reviews.jsonl; rating > 0 keeps the rule in force,
    // rating <= 0 retires it. True when the line was written (false with no reviews path).
    Q_INVOKABLE bool saveTasteRule(const QString &rule, int rating = 1);
    QStringList tasteRules() const { return m_tasteRules; }
    static QString defaultReviewsPath();
    QString reviewsPath() const { return m_reviewsPath; }
    void setReviewsPath(const QString &path);  // loads saved rules from it; empty: memory only
    // Live only, with an OpenAI-compatible endpoint: qwen3.8-flash-next distills the project's
    // latest turn down to what the human cares about and updates its summary / stateLine.
    // Async, one request in flight; repeats for the same project + transcript mtime are dropped.
    Q_INVOKABLE void filterForHuman(const QString &projectKey);
    int filterRequests() const { return m_filterRequests; }
    // Live only, with an OpenAI-compatible endpoint: opening an island prefills the fast model's
    // KV cache with exactly the prefix the next ask sends: Tier 0 system prompt + Tier 1 trunk
    // (contextTrunk(key)), sent through LlmRouter::warmPrefix.
    // Async, fire-and-forget; repeats for the same project + transcript mtime are dropped.
    Q_INVOKABLE void warmProjectPrefix(const QString &projectKey);
    int warmRequests() const { return m_warmRequests; }
    // Concurrent OpenAI-compatible conversations keyed by streamId, all multiplexed by LlmRouter
    // (no 6-per-host cap). A stream keeps its prior turns as Tier 2; restarting an active id
    // cancels the old request. Emits streamDelta / streamFinished.
    Q_INVOKABLE void askStream(const QString &streamId, const QString &projectKey, int artifactIndex,
                               const QString &question);
    // askStream with an explicit Tier 3 target slice (hovered / pinned targets from
    // TargetRegistry::buildCombinedSlice) in place of the on-screen artifact slice. Tier 0 + 1
    // stay byte-identical to the warmed prefix. An empty slice falls back to the artifact's.
    Q_INVOKABLE void askStreamWithSlice(const QString &streamId, const QString &projectKey, int artifactIndex,
                                        const QString &targetSlice, const QString &question);
    Q_INVOKABLE void cancelStream(const QString &streamId);
    Q_INVOKABLE void clearStreamHistory(const QString &streamId);
    // Hover history, pins and the active semantic target (created on first use).
    TargetRegistry *targets();
    // Cross-session memory (past Claude Code / Codex / Antigravity / Phonon turns). Live mode
    // rescans in the background; Mock mode stays empty unless a test calls setTurns().
    MemoryIndex *memoryIndex() { return &m_memory; }
    // The ask's Tier 3 "Relevant past conversations:" block: top 4 turns, <= 3,200 chars, never
    // from the project's own (Tier 1) session. "" when nothing matches.
    QString memoryContext(const QString &projectKey, const QString &question) const;
    // The session whose transcript feeds the project's Tier 1 trunk (newest), by harness ident
    // (its Lapis id when it has none).
    QString activeSessionId(const QString &projectKey) const;
    // The OpenAI-compatible prompt every quick ask and stream sends (Tiers 0-3).
    TieredPrompt tieredPrompt(const QString &projectKey, const QString &slice, const QString &question,
                              const QVector<ChatTurn> &history = {}) const;
    // Live window placement ("windowState" in studio_prefs.json). Setting it writes prefs at once.
    WindowState windowState() const { return m_windowState; }
    void setWindowState(const WindowState &state);
    QObject *slowmix() { return &m_slowmix; }
    SlowmixController *slowmixController() { return &m_slowmix; }
    QObject *spotify() { return &m_spotify; }
    SpotifyController *spotifyController() { return &m_spotify; }

    // Ambient voice (see the Q_PROPERTYs). feedAmbientTail is what pollPhonon calls with each new
    // tail; tests call it directly. flushAmbient skips the 800 ms debounce.
    bool ambientVoiceEnabled() const { return m_ambientEnabled; }
    void setAmbientVoiceEnabled(bool on);
    bool shadowRecordEnabled() const { return m_shadowAllowed && m_shadowPref; }
    void setShadowRecordEnabled(bool on);
    bool shadowRecording() const;  // the tap is actually installed (permission granted)
    Q_INVOKABLE void toggleAmbientVoice() { setAmbientVoiceEnabled(!m_ambientEnabled); }
    bool ambientBusy() const { return m_ambientBusy; }
    QString ambientAnswer() const { return m_ambientAnswer; }
    QString ambientHeard() const { return m_ambientHeard; }
    Q_INVOKABLE void clearAmbientAnswer();
    void feedAmbientTail(const QString &tail);
    void flushAmbient();
    bool hasPendingProposal() const { return !m_pendingProposalActions.isEmpty(); }
    QString pendingProposalText() const { return m_pendingProposalSummary; }
    // Runs the staged Ambient actions in their project and clears the proposal; false when none.
    Q_INVOKABLE bool confirmPendingProposal();
    Q_INVOKABLE void dismissPendingProposal();
    void setProposalTimeoutMs(int ms) { m_proposalTimer.setInterval(ms); }  // tests: default 15 s
    // The Tier 0 local studio context (loadLocalStudioContext, live mode only; tests may set it).
    QString studioContext() const { return m_studioContext; }
    void setStudioContext(const QString &ctx) { m_studioContext = ctx; }
    // The effort an OpenAI-compatible request actually sends: the picker's effort when the chat
    // model is the compat model, else "fast" (no thinking). "fast" / "none" mean enable_thinking off.
    QString compatEffort() const;
    QVector<ChatTurn> streamHistory(const QString &streamId) const { return m_streamHistory.value(streamId); }
    // Live mode owns a router (created lazily); tests may inject one (not owned) in any mode.
    LlmRouter *llmRouter();
    void setLlmRouter(LlmRouter *router);
    // Tier 1 trunk (stable project context, no on-screen artifact) and the on-screen slice.
    QString contextTrunk(const QString &projectKey) const;
    Q_INVOKABLE QString contextSlice(const QString &projectKey, int artifactIndex) const;
    // Island metrics stay hidden unless pinned for that project.
    Q_INVOKABLE void toggleMetricPin(const QString &projectKey);
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
    QVariantMap viewState() const { return m_viewState; }
    // Normalizes and stores the view; the debounced prefs save writes it (no path = memory only).
    Q_INVOKABLE void setViewState(const QVariantMap &state);
    Q_INVOKABLE void setGamepadBinding(const QString &inputId, const QString &actionId);
    Q_INVOKABLE void resetGamepadBindings();
    Q_INVOKABLE void recordAction(const QString &action);
    QStringList recentActions() const { return m_recentActions; }
    QString activeCategory() const { return m_activeCategory; }
    QString phononTail() const { return m_phononTail; }
    bool phononLive() const { return m_phononLive; }  // fresh turns and Status not "complete"

    // The prompt preamble for quick asks and traces: on-screen artifact, last turns, files, tools, write-up.
    QString contextFor(const QString &projectKey, int artifactIndex) const;

    // Every external process goes through spawn(); in Mock mode it refuses.
    int spawnCount() const { return m_spawnCount; }
    int blockedSpawns() const { return m_blockedSpawns; }
    QStringList transcriptsRead() const { return m_transcriptsRead; }

    Q_INVOKABLE QString startSession(const QString &folder, const QString &harness, const QString &model,
                                     const QString &goal);
    Q_INVOKABLE void ask(const QString &projectKey, int artifactIndex, const QString &question);
    // Natural-language Studio UI control. True when the text was a UI command: the action ran
    // (backend-owned state) and uiActionRequested fired for the QML side. ask() and every
    // conversation stream call it first, so "open music" never reaches a model.
    Q_INVOKABLE bool tryHandleStudioCommand(const QString &text);
    // Runs one action (fast matcher or a model's [[ACTION:..]] tag) and returns its 1-line
    // confirmation ("Opened Music (Slowmix)."); "" when the action is unknown or cannot run.
    Q_INVOKABLE QString dispatchStudioAction(const QString &action, const QString &arg = QString());
    // What is on screen, pushed by StudioApp: {view, highlighted, overlays: [..]}. Rides in the
    // Tier 3 leaf as "Studio live state:" (with every card and the playing track).
    Q_INVOKABLE void setUiState(const QVariantMap &state);
    QString studioLiveState() const;
    // Every Studio exchange, newest last, across models and surfaces (last 10 pairs). It is the
    // Tier 2 of every OpenAI-compatible ask and rides in the CLI channels' prompt too.
    QVector<ChatTurn> sharedHistory() const { return m_sharedHistory; }
    static QString defaultHistoryPath();  // ~/.lapis-next/studio_history.jsonl
    QString historyPath() const { return m_historyPath; }
    void setHistoryPath(const QString &path) { m_historyPath = path; }  // empty: no log
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
    Q_INVOKABLE QString markdownHtml(const QString &md, int codePx = 14) const { return studio::markdownToHtml(md, codePx); }
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
    // Tests: land a registry scan or a board poll exactly as the live paths do, without
    // spawning lapis-peers or reading the registry.
    void testApplyScan(const QList<SessionInfo> &sessions, const QHash<QString, ParsedCache> &cache)
    {
        applyScan(sessions, cache);
    }
    void applyBoard(const QList<BoardRow> &rows);

signals:
    void projectsChanged();
    void viewStateChanged();
    void foldersChanged();
    void quickChanged();
    void traceChanged();
    void toastChanged();
    void prefsChanged();
    void suggestionChanged();
    void briefingChanged();
    void cardPositionsChanged();
    void mapLayoutChanged();
    void reflectionChanged();
    void tasteRulesChanged();
    void phononTailChanged();
    void ambientChanged();
    void shadowRecordChanged();
    void pendingProposalChanged();
    void sessionStarted(const QString &projectKey);
    void quickTick();
    void streamDelta(const QString &streamId, const QString &textDelta, const QString &thinkingDelta);
    void streamFinished(const QString &streamId, const QString &error);
    // A Studio UI action for the QML side to carry out (see StudioCommand).
    void uiActionRequested(const QString &action, const QString &arg);
    // A cua_* action finished on the worker (Live) or was recorded (Mock).
    void cuaActionFinished(const QString &action, const QString &target, bool ok, const QString &tier,
                           const QString &detail);

public:
    // Mock mode: every cua_* action recorded instead of run ("cua_click:Send"); Live: finished ones.
    QStringList cuaLog() const { return m_cuaLog; }
    int cuaPending() const { return m_cuaPending; }
    // Closed-loop computer use: automatic follow-up turns run for the current user prompt
    // (<= kMaxCuaLoopDepth; every new ask, clearAnswers or a cancel resets it to 0).
    int cuaLoopDepth() const { return m_cuaLoopDepth; }
    // The follow-up prompts issued for the current user prompt. Mock records them and never
    // sends: no model, no network, no screen.
    QStringList cuaFollowups() const { return m_cuaFollowups; }
    // Runs a finished quick answer's [[ACTION:..]] tags exactly as finishQuick does (the closed
    // CUA loop included) and returns the cleaned answer.
    QString runModelAnswer(const QString &answer);

private:
    // One model answer's cua_* actions, gathered until the last one finishes.
    struct CuaBatch {
        bool open = false;  // runAnswerTags is still dispatching it
        QString streamId;   // the conversation stream that answered; "" = the quick ask
        QString projectKey;
        int outstanding = 0;
        QVector<CuaStepResult> steps;
        QString goal;  // the user's own prompt (follow-ups keep it), for the CUA trace log
    };
    CuaBatch m_cuaBatch;
    quint64 m_cuaBatchId = 0;  // bumped per batch and per reset: a stale action's result is dropped
    int m_cuaLoopDepth = 0;
    QStringList m_cuaFollowups;
    bool m_cuaFollowupAsk = false;  // ask() / startConversation are carrying a follow-up, not the user
    QString m_cuaFollowupNote;      // the compact user turn a follow-up leaves in the history
    QString m_cuaFollowupGoal;      // the user prompt a follow-up continues (trace log)
    QString m_cuaAnswerGoal;        // the goal of the stream runAnswerTags is handling
    QString m_cuaAnswerStream;      // which stream runAnswerTags is handling ("" = the quick ask)
    int m_quickArtifact = -1;       // the user ask's artifact, reused by its follow-ups
    QString m_quickNote;            // the user turn finishQuick records (follow-ups: compact header)
    void resetCuaLoop();
    void noteCuaResult(quint64 batchId, const CuaStepResult &step);
    void maybeFinishCuaBatch();
    // Live: runs a CuaExecutor on m_cuaPool (one action at a time, never the GUI thread).
    QString startCuaAction(const QString &action, const CuaCommand &c);
    QThreadPool *m_cuaPool = nullptr;
    QStringList m_cuaLog;
    int m_cuaPending = 0;
    bool m_cuaHud = false;  // the real app: voice and cua_* actions drive the floating HUD bar
    // Human-click shadow recorder: allowed only in the real live window (warm helpers) without
    // LAPIS_SHADOW_RECORD=0; m_shadowPref is the prefs toggle. syncShadowRecorder starts / stops it.
    std::unique_ptr<studio::cua::CuaShadowRecorder> m_shadow;
    bool m_shadowAllowed = false;
    bool m_shadowPref = true;
    void syncShadowRecorder();
    void refreshShadowExcludes();
    // tryHandleStudioCommand for a known project (spawn_agent runs in its folder).
    bool handleStudioCommand(const QString &text, const QString &projectKey, QString *reply);
    QString spawnAgent(const QString &projectKey, const QString &task, bool steerLive);
    void noteExchange(const QString &projectKey, const QString &model, const QString &prompt,
                      const QString &answer);
    // Model tags in a finished answer: dispatched, stripped from *answer (filled with the
    // confirmation when the tag was all the model said).
    void runAnswerTags(QString *answer);
    QString recentConversationBlock() const;  // the shared ring as plain text (CLI prompts)
    QVector<ChatTurn> m_sharedHistory;
    QString m_historyPath;
    QVariantMap m_uiState;
    QString m_commandProject;  // project of the ask being handled (spawn_agent's folder)
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
    void pollPhonon();
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
    void finishQuick(bool ok = true);  // ok: a real answer (tags run, the exchange is shared + logged)
    void evictChannels();
    void ensureTraceHelper();
    void onTraceOutput();
    void finishTrace();
    void askOpenAiCompat(const QString &projectKey, int artifactIndex, const QString &question);
    // DeepSeek V4.1 Flash via OpenRouter: same tiers, reasoning effort "none" on fast.
    void askOpenRouter(const QString &projectKey, int artifactIndex, const QString &question);
    QString openrouterKey() const;  // m_openrouterKey (prefs openrouterApiKey), then env OPENROUTER_API_KEY
    void onRouterDelta(const QString &streamId, const QString &text, const QString &thinking);
    void onRouterFinished(const QString &streamId, const QString &error);
    // contextFor() = head + slice + rest; contextTrunk() = head + rest. False if no such project.
    bool contextPieces(const QString &projectKey, int artifactIndex, QString *head, QString *slice,
                       QString *rest) const;
    QString compatModel() const;  // m_openaiCompatModel, env, then the chat model
    QString compatKey() const;    // m_openaiCompatKey, then env
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
    QString m_theme = QStringLiteral("gold");
    bool m_firstLaunch = false;
    QString m_chatModel = QStringLiteral("claude-sonnet-5-5");
    QString m_chatEffort = QStringLiteral("low");
    QString m_effortBeforeFast;  // the thinking effort in use before a no-think ("fast") model
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
    QString m_openrouterKey;  // prefs openrouterApiKey (read only; never written back)
    QVariantMap m_viewState;  // prefs viewState: {lens, focusedProject, artifactIndex, needsMeOnly}
    bool m_compatUrlCleared = false;  // setOpenaiCompatUrl("") ran: savePrefs drops the file's URL
    QNetworkAccessManager m_nam;  // suggestion, filter and briefing (one request each at most)
    std::unique_ptr<LlmRouter> m_ownedRouter;
    QPointer<LlmRouter> m_router;
    struct StreamAsk {
        QString question;
        QString answer;
        QString projectKey;
        QString note;  // the user turn recorded in history (a CUA follow-up's compact header)
        QString goal = {};  // the user's own prompt, kept across CUA follow-ups
    };
    QHash<QString, StreamAsk> m_streamAsks;              // in-flight askStream ids
    QHash<QString, QVector<ChatTurn>> m_streamHistory;    // Tier 2 per conversation stream
    QPointer<QNetworkReply> m_suggestReply;
    QPointer<QNetworkReply> m_briefingReply;
    struct HumanFilter {
        qint64 mtime = 0;
        QString stateLine, summary, form;
    };
    QHash<QString, HumanFilter> m_humanFilter;  // project key -> latest filtered view
    QPointer<QNetworkReply> m_filterReply;      // concurrency 1
    QString m_filterInFlight;                   // "key@mtime" of the request in flight
    QSet<QString> m_filterDone;                 // "key@mtime" already asked (no respamming)
    QStringList m_filterQueue;                  // project keys waiting for the slot
    int m_filterRequests = 0;
    int m_warmRequests = 0;  // dedup of "key@mtime@hash" lives in LlmRouter::warmPrefix
    QString compatEndpoint() const;             // "" when no OpenAI-compatible URL is configured
    QVariantList m_briefing;
    QString m_briefingSource;
    bool m_showCodeDiffs = false;
    QVariantMap m_cardPositions;
    QString m_mapSort = QStringLiteral("category");
    QStringList m_clusterOrder = defaultClusterOrder();
    SessionReflection m_reflection;
    QString m_reflectionSource;
    QString m_reflectionRaw;  // the streamed model reply so far
    bool m_reflectionBusy = false;
    QStringList m_tasteRules;
    QString m_reviewsPath;
    void loadTasteRules();
    QVariantList reflectionSessions() const;  // localReflection() input, redacted projects left out
    QSet<QString> m_pinnedMetrics;
    QString m_activeCategory;
    QString m_phononTail;
    bool m_phononLive = false;
    qint64 m_phononMtime = 0;
    QStringList m_selectedAgentIds;
    QStringList m_recentActions;
    QStringList m_projectOrder;  // locks card slots so background updates never shuffle muscle memory
    Gamepad m_gamepad;
    std::unique_ptr<TargetRegistry> m_targets;  // lazy: targets()
    SlowmixController m_slowmix;  // its destructor only hangs up; the daemon keeps playing
    SpotifyController m_spotify;
    bool m_slowmixWasPlaying = false;
    // Ambient voice.
    void runAmbient(const QString &spoken);
    bool m_ambientEnabled = false;
    bool m_ambientBusy = false;
    QString m_ambientPrevTail;
    QStringList m_ambientPending;
    QString m_ambientRaw;
    QString m_ambientAnswer;
    QString m_ambientHeard;
    QString m_ambientProject;
    QTimer m_ambientDebounce;
    void stageProposal(const QVector<StudioAction> &actions, const QString &summary, const QString &project);
    QStringList runStudioActions(const QVector<StudioAction> &actions);
    QVector<StudioAction> m_pendingProposalActions;
    QString m_pendingProposalSummary;
    QString m_pendingProposalProject;
    QTimer m_proposalTimer;  // single-shot 15 s: an unconfirmed proposal expires
    QString m_studioContext;  // Tier 0 tail, loaded once in live mode
    MemoryIndex m_memory;
    QTimer m_memoryTimer;  // live: periodic background rescan
    void refreshMemory();
    WindowState m_windowState;
    void startConversation(const QString &streamId, const QString &projectKey, const QString &slice,
                           const QString &question);

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
    // Latest `lapis-peers board` reply snippet per agent: the only context a remote (SSH) session
    // with no local transcript has.
    QHash<QString, QString> m_snippetById;
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
