// The Jarvis bar: the MacBook-wide floating command bar (L2 over any app or Space, and over
// Lapis Studio itself in the unified "overlay" Jarvis mode).
//
// One sleek 56 pt capsule (status mark, context pill, inline prompt, one controller badge) with a
// compact answer drawer below a hairline. It pops up over whatever app is frontmost, takes Phonon's dictation
// (Phonon pastes with Cmd+V into the focused field of the frontmost app, so this app has to be
// active with the bar's text field focused before the fn key goes down), and shows the model's
// answer below the prompt. While a cua_* action runs on the app behind it, the bar ghosts out
// (alpha 0, click-through); after the action it closes but keeps the last exchange, so the next
// L2 hold pops it back with the previous question and answer still visible.
//
// Mode::Offscreen never creates a window: a pure state machine for tests (zero screen effects).
// All methods, construction and destruction run on the AppKit main thread.
#pragma once

#include <QImage>
#include <QRectF>
#include <QString>
#include <cstdint>
#include <functional>
#include <memory>

namespace studio::cua {

constexpr double kSpotlightWidth = 580;
constexpr double kSpotlightBaseHeight = 56;   // the single command capsule, no drawer
constexpr double kSpotlightAnswerMaxHeight = 190;
constexpr double kSpotlightDrawerChrome = 46;  // hairline + gaps + question line + bottom margin
constexpr double kSpotlightMaxHeight = kSpotlightBaseHeight + kSpotlightDrawerChrome + kSpotlightAnswerMaxHeight;

// What the voice side of the bar is doing. Held: L2 is down (release sends). Toggled: a quick L2
// tap left Phonon listening (the next tap sends). Transcribing: listening stopped and the bar is
// waiting for Phonon's paste, which auto-sends.
enum class SpotlightVoice { Idle, Held, Toggled, Transcribing };

// Pure: the panel's frame (AppKit coordinates, origin bottom-left) on a screen whose AppKit frame
// is `screen`: kSpotlightWidth wide, centred horizontally, its top edge 20% below the screen top.
// `height` is clamped to [kSpotlightBaseHeight, kSpotlightMaxHeight].
QRectF spotlightFrame(const QRectF &screen, double height);
// Pure: the panel height for a drawer holding an answer body `answerHeight` points tall (the
// body itself capped at kSpotlightAnswerMaxHeight); no drawer gives kSpotlightBaseHeight.
double spotlightHeight(bool drawer, double answerHeight);
// Pure: the badge on the bar's right, as text (state, accessibility). On screen it is one vector
// controller icon plus a word: Held "L2 release", Toggled "Listening · tap L2 to send" (drawn
// [L2] send), Transcribing "Transcribing…", an answer showing "○ close", typed text "R2 send",
// else "L2 voice".
QString spotlightHint(SpotlightVoice voice, bool answerShown = false, bool hasText = false);
// Pure: the prompt placeholder: "Speak or type…" (held), "Tap L2 to send…" (toggled),
// "Transcribing…", else "Ask or control <context>…" ("Ask or control anything…" with none).
QString spotlightPlaceholder(const QString &context, SpotlightVoice voice = SpotlightVoice::Idle);
// Pure: the context pill's short name for an app ("Google Chrome" -> "Chrome").
QString spotlightContextLabel(const QString &appName);

// [NSApp isActive] (false without an NSApp).
bool studioAppIsActive();
// The live (Visible) bar's window number, 0 when none: captures and the shadow recorder skip it.
uint32_t spotlightWindowNumber();

// The bar's accent: Amber Gold #e09f3e (default) or lapis blue #4f8cff, read on every render.
// Normally set through setOverlayTheme (cua_executor.hpp). Any thread (an atomic flag).
void setSpotlightLapisBlue(bool lapisBlue);
bool spotlightLapisBlue();
// Main thread: restyles the live bar (if one exists and has a panel) with the current flag.
void refreshSpotlightTheme();

// --snap-spotlight OUT.png: the bar rendered offscreen (never a window) in six states stacked on
// #000000: idle over Chrome, held listening with dictated text, tap-toggled listening over
// "Lapis · infinity", transcribing, a drawer with a question and a two-paragraph answer, and the
// same drawer in lapis blue. 0 on success.
int snapSpotlight(const QString &outPath);

struct SpotlightState {
    bool open = false;          // the panel is on screen (alpha may be 0 while an action runs)
    bool key = false;           // its text field has keyboard focus (Offscreen: == open)
    bool listening = false;     // voice is Held or Toggled
    SpotlightVoice voice = SpotlightVoice::Idle;
    bool toggled = false;       // voice == Toggled
    bool transcribing = false;  // voice == Transcribing
    bool busy = false;
    bool actionRunning = false; // ghosted: alpha 0, click-through
    double alpha = 0;
    QString inputText;
    QString lastQuestion;
    QString answer;
    QString statusLine;
    QString targetApp;
    int targetPid = 0;
    bool overStudio = false;    // opened over Lapis Studio itself (unified Jarvis overlay)
    QString contextLabel;       // the pill: "Chrome", "Lapis · infinity"; empty: no pill
    bool escapeWatch = false;   // the global Esc monitor is on (a turn is acting)
    QString hint;               // spotlightHint(voice, answer shown, typed text)
    QString placeholder;
    bool drawerVisible = false;
    QRectF frame;               // AppKit points (Offscreen: on a virtual 1440x900 screen)
    int opens = 0;              // openOrFocus calls that put it on screen
    int actionRuns = 0;         // setActionRunning(true) calls
    int dismissals = 0;
};

class CuaSpotlight {
public:
    enum class Mode { Visible, Offscreen };
    explicit CuaSpotlight(Mode mode = Mode::Visible);
    ~CuaSpotlight();
    CuaSpotlight(const CuaSpotlight &) = delete;
    CuaSpotlight &operator=(const CuaSpotlight &) = delete;

    // Pops the bar up (or re-focuses it) over the frontmost app: snapshots that app as the target
    // when this app is not active, orders Studio windows on other Spaces out (so activating never
    // slides to them), activates this app and focuses the prompt with the caret at the end.
    // Keeps typed text and the previous exchange. Returns true when it was already open and key
    // (focus is already there: the voice key needs no delay).
    bool openOrFocus();
    // The unified Jarvis overlay over Lapis Studio itself: the context pill reads contextLabel
    // ("Lapis · infinity"), there is no other app to target or re-activate, and no Studio window
    // is ordered out (the bar is on Studio's own Space). Same return as openOrFocus.
    bool openOrFocusOverStudio(const QString &contextLabel);
    // Closes the bar. restoreTarget: re-activate the app it was opened over. The last exchange
    // stays in memory for the next openOrFocus.
    void dismiss(bool restoreTarget = true);
    bool isOpen() const;

    // Held listening (on) or Idle; setVoice for Toggled / Transcribing.
    void setListening(bool on);
    void setVoice(SpotlightVoice voice);
    void setInputText(const QString &text);
    QString inputText() const;
    QString takeInputText();  // returns the prompt and clears it
    // The prompt changed from outside the API (Phonon's Cmd+V paste, typing): fires
    // onInputEdited. The live field calls it from controlTextDidChange:; tests call it after
    // setInputText to play a paste.
    void notifyInputEdited();
    // While a turn acts on the desktop (even with the bar closed or ghosted), a global Esc key
    // (from the user, never this process's own synthetic keys) fires onGlobalEscape. Visible
    // mode installs an NSEvent global monitor; Offscreen only records the flag.
    void setEscapeWatch(bool on);
    // A new turn: "› question" in the drawer, "Thinking…" as the status, busy.
    void beginTurn(const QString &question);
    void setAnswer(const QString &displayText);  // tags already stripped
    void setStatusLine(const QString &status);
    // Done streaming (pure Q&A): stays open showing the answer, prompt focused again.
    void setBusy(bool busy);
    // cua_* actions on the app behind: running ghosts the bar (alpha 0, click-through) and hands
    // focus back to the target app; not running brings it back.
    void setActionRunning(bool running, const QString &status = {});
    // The turn's cua_* actions are done: the bar closes (the target app stays frontmost) and the
    // exchange, with `summary` as its status, waits for the next L2.
    void finishAfterAction(const QString &summary);
    // Re-activates the target app (false when there is none or it quit).
    bool reactivateTarget();
    // Order front every Studio window openOrFocus ordered out.
    void restoreMainWindows();
    // D-pad summon: closes the bar (no target re-activation), restores the Studio windows and
    // activates Studio with its main window key.
    void summonStudio();

    QString targetApp() const;
    int targetPid() const;
    // Offscreen (tests) / injection: what openOrFocus treats as the frontmost other app.
    void setFrontmostProbe(std::function<void(int *pid, QString *appName)> probe);
    uint32_t windowNumber() const;  // 0 in Offscreen mode or before the first open
    SpotlightState state() const;
    // The bar for the current state, drawn by the same painter the live card uses (plus the hint,
    // prompt and answer the live controls draw), on transparency at `scale` px/pt. Works in both
    // modes and never creates or touches a window. Null for a non-positive or absurd scale.
    QImage render(double scale = 2) const;

    std::function<void(const QString &text)> onSubmit;  // Return / Enter in the prompt
    std::function<void()> onDismiss;                    // Esc
    std::function<void()> onInputEdited;                // the prompt text changed (paste, typing)
    std::function<void()> onGlobalEscape;               // Esc anywhere while setEscapeWatch(true)

    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

}  // namespace studio::cua
