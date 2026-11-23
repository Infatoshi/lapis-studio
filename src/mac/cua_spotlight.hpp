// The MacBook-wide floating command bar (L2 / R2 outside Lapis Studio, over any app or Space).
//
// A Raycast-style card that pops up over whatever app is frontmost, takes Phonon's dictation
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

constexpr double kSpotlightWidth = 620;
constexpr double kSpotlightBaseHeight = 96;   // top row + prompt, no drawer
constexpr double kSpotlightMaxHeight = 340;   // with question + answer
constexpr double kSpotlightAnswerMaxHeight = 190;

// Pure: the panel's frame (AppKit coordinates, origin bottom-left) on a screen whose AppKit frame
// is `screen`: kSpotlightWidth wide, centred horizontally, its top edge 20% below the screen top.
// `height` is clamped to [kSpotlightBaseHeight, kSpotlightMaxHeight].
QRectF spotlightFrame(const QRectF &screen, double height);
// Pure: the panel height for a drawer holding an answer body `answerHeight` points tall (the
// body itself capped at kSpotlightAnswerMaxHeight); no drawer gives kSpotlightBaseHeight.
double spotlightHeight(bool drawer, double answerHeight);
// Pure: the hint on the top row's right, as text (state, accessibility). On screen the bar draws
// it as vector controller icons: [L2] voice  [R2] send  [○] close, or a pulsing dot + Listening….
QString spotlightHint(bool listening);
// Pure: the prompt placeholder ("Command or ask about Google Chrome…").
QString spotlightPlaceholder(const QString &targetApp);

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

// --snap-spotlight OUT.png: the bar rendered offscreen (never a window) in four states stacked on
// #000000: idle with the "in Google Chrome" pill, listening with dictated text, a drawer with a
// question and a two-paragraph answer, and the same drawer in lapis blue. 0 on success.
int snapSpotlight(const QString &outPath);

struct SpotlightState {
    bool open = false;          // the panel is on screen (alpha may be 0 while an action runs)
    bool key = false;           // its text field has keyboard focus (Offscreen: == open)
    bool listening = false;
    bool busy = false;
    bool actionRunning = false; // ghosted: alpha 0, click-through
    double alpha = 0;
    QString inputText;
    QString lastQuestion;
    QString answer;
    QString statusLine;
    QString targetApp;
    int targetPid = 0;
    QString hint;               // spotlightHint(listening)
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
    // Closes the bar. restoreTarget: re-activate the app it was opened over. The last exchange
    // stays in memory for the next openOrFocus.
    void dismiss(bool restoreTarget = true);
    bool isOpen() const;

    void setListening(bool on);
    void setInputText(const QString &text);
    QString inputText() const;
    QString takeInputText();  // returns the prompt and clears it
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

    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

}  // namespace studio::cua
