// PS4 DualShock 4 (and any MFi / extended gamepad) input for lapis_studio.
//
// Two layers:
//   GamepadState  pure C++ debouncer: one Snapshot of the pad + a clock in, discrete actions out
//                 (stick / d-pad nav with 320 ms delay + 110 ms repeat, edge-triggered buttons,
//                 trigger hysteresis, smooth right-stick scroll on an 8 ms tick). Fully unit-tested.
//   Gamepad       QObject exposed to QML as backend.gamepad. Emits one signal per action. Only
//                 setEnabled(true) (interactive live window) ever touches GameController.framework;
//                 every offscreen mode leaves it disabled, so --test / --bench-fps stay hermetic.
#pragma once

#include <QHash>
#include <QList>
#include <QObject>
#include <QString>
#include <QVariantList>
#include <QVariantMap>

#include <functional>
#include <memory>

class QTimer;

namespace studio {

// One poll of the pad. Axes follow GameController conventions: stick / d-pad up is +1.
struct GamepadSnapshot {
    double lx = 0, ly = 0;          // left stick
    double rx = 0, ry = 0;          // right stick
    int dpadX = 0, dpadY = 0;       // -1, 0, +1
    bool a = false;                 // Cross ✕
    bool b = false;                 // Circle ○
    bool x = false;                 // Square □
    bool y = false;                 // Triangle △
    bool l1 = false, r1 = false;    // shoulders
    double l2 = 0, r2 = 0;          // triggers 0..1
    bool l3 = false, r3 = false;    // stick clicks
    bool menu = false;              // Options ☰ (GCExtendedGamepad.buttonMenu)
    bool options = false;           // Share (GCExtendedGamepad.buttonOptions)
    bool touchpad = false;          // touchpad click (GCDualShockGamepad.touchpadButton)
    double touchpadX = 0;           // touchpad primary finger X (-1..+1, 0 = untouched/center)
    double touchpadY = 0;           // touchpad primary finger Y (-1..+1, 0 = untouched/center)
};

struct GamepadAction {
    enum Kind {
        Nav,
        Confirm,
        Back,
        BackHeld,
        AcceptPrediction,
        Secondary,
        Step,
        Zoom,
        ZoomReset,
        ToggleSettings,
        ToggleNewSession,
        ToggleBriefing,
        ToggleMusic,
        ToggleReview,
        Scroll,
        TouchpadPan,
        HorizontalInspect,
        VoiceHold,
        Send,
        RawCaptured,
        ConfirmHeld,  // value 1.0 on the confirm input's press edge, 0.0 on its release
        Cursor,       // vx, vy: the shaped left stick (screen y down), every poll while deflected
        ConfirmProposal,  // L3 + R3 clicked together (within kStickChordMs): run the staged proposal
        R3Raw             // R3 press that resolved as a single click (not part of the L3 + R3 chord)
    };
    Kind kind;
    int dx = 0, dy = 0;   // Nav: screen direction (dy +1 = down); Step: dx = -1 / +1
    double value = 0;     // Zoom: uiScale delta; Scroll: pixels; VoiceHold: 1.0 down / 0.0 up
    QString inputId = {}; // RawCaptured: physical slot id ("a", "l2", ...)
    double vx = 0, vy = 0;  // Cursor: -1..+1 per axis after the deadzone and response curve
    static GamepadAction make(Kind k, int dx = 0, int dy = 0, double v = 0, const QString &in = {})
    {
        return {k, dx, dy, v, in};
    }
};

class GamepadState {
public:
    static constexpr double kStickDeadzone = 0.35;   // left stick engages nav past this
    static constexpr double kStickRelease = 0.25;    // ...and lets go below this (no chatter)
    static constexpr double kScrollDeadzone = 0.20;  // right stick
    static constexpr double kTriggerPress = 0.55;
    static constexpr double kTriggerRelease = 0.45;
    static constexpr qint64 kRepeatDelayMs = 320;
    static constexpr qint64 kRepeatIntervalMs = 110;
    static constexpr qint64 kScrollTickMs = 8;
    static constexpr double kScrollPxPerSec = 2000;  // full deflection
    static constexpr double kZoomStep = 0.10;
    static constexpr double kCursorDeadzone = 0.14;  // analog cursor: radial deadzone
    // L3 / R3 presses wait this long for the other stick click before firing alone, so the
    // L3 + R3 chord never also fires zoom_reset / zoom_in.
    static constexpr qint64 kStickChordMs = 80;

    // The analog cursor's response: a radial deadzone, then a quadratic curve on the magnitude
    // (fine aim near the centre, a fast sweep at full tilt), direction kept. Screen y is down.
    static void shapeStick(double lx, double ly, double *outX, double *outY);

    static QHash<QString, QString> defaultBindings();
    static QStringList validInputs();
    static QStringList validActions();

    GamepadState() : m_bindings(defaultBindings()) {}

    void setBindings(const QHash<QString, QString> &bindings);
    const QHash<QString, QString> &bindings() const { return m_bindings; }
    void setCaptureMode(bool on) { m_capture = on; }
    bool captureMode() const { return m_capture; }
    // Analog cursor mode (the Constellation's free cursor owns the left stick): the stick stops
    // emitting discrete Nav and emits Cursor velocity every poll instead; the d-pad still navs.
    void setAnalogCursor(bool on);
    bool analogCursor() const { return m_analog; }
    // The shaped left stick from the latest poll (0, 0 inside the deadzone). Always tracked.
    double leftX() const { return m_leftX; }
    double leftY() const { return m_leftY; }

    QList<GamepadAction> process(const GamepadSnapshot &s, qint64 nowMs);
    void reset();

private:
    void navDirection(const GamepadSnapshot &s, int *dx, int *dy) const;
    void emitMappedAction(const QString &inputId, bool pressed, QList<GamepadAction> &out) const;
    void processStickClicks(const GamepadSnapshot &s, qint64 nowMs, QList<GamepadAction> &out);

    QHash<QString, QString> m_bindings;
    bool m_capture = false;
    GamepadSnapshot m_prev;
    int m_navDx = 0, m_navDy = 0;
    bool m_navFromStick = false;
    qint64 m_nextRepeatMs = 0;
    bool m_l2 = false, m_r2 = false;
    bool m_scrolling = false;
    qint64 m_lastScrollMs = 0;
    int m_rNavDx = 0;
    qint64 m_nextRNavMs = 0;
    bool m_chord = false;  // L1 + R1 held together: one ToggleMusic, no single-shoulder actions
    // L3 / R3: a press is pending (not yet emitted) for up to kStickChordMs; m_stickChord holds
    // until both sticks are released after a ConfirmProposal.
    bool m_stickChord = false;
    bool m_l3Pending = false, m_r3Pending = false;
    bool m_l3Down = false, m_r3Down = false;  // the single press was emitted; its release is owed
    qint64 m_l3Since = 0, m_r3Since = 0;
    bool m_analog = false;
    double m_leftX = 0, m_leftY = 0;
};

class Gamepad : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool connected READ connected NOTIFY connectedChanged)
    Q_PROPERTY(QString controllerName READ controllerName NOTIFY connectedChanged)
    Q_PROPERTY(bool enabled READ enabled NOTIFY enabledChanged)
    Q_PROPERTY(bool captureMode READ captureMode WRITE setCaptureMode NOTIFY captureModeChanged)
    Q_PROPERTY(bool voiceActive READ voiceActive NOTIFY voiceActiveChanged)
    Q_PROPERTY(QVariantList bindingSlots READ bindingSlots NOTIFY bindingsChanged)
    Q_PROPERTY(QVariantList availableActions READ availableActions CONSTANT)
    // True while the Music overlay owns the pad (StudioApp binds it): L2 then toggles loop
    // instead of holding the Phonon voice key.
    Q_PROPERTY(bool musicMode READ musicMode WRITE setMusicMode NOTIFY musicModeChanged)
    // True while the Constellation's free cursor owns the left stick (StudioApp binds it).
    Q_PROPERTY(bool analogCursor READ analogCursor WRITE setAnalogCursor NOTIFY analogCursorChanged)
    // The shaped left stick (-1..+1, screen y down; 0 inside the 0.14 deadzone), per poll.
    Q_PROPERTY(double leftX READ leftX NOTIFY leftStickChanged)
    Q_PROPERTY(double leftY READ leftY NOTIFY leftStickChanged)

public:
    using Snapshot = GamepadSnapshot;

    explicit Gamepad(QObject *parent = nullptr);
    ~Gamepad() override;

    bool connected() const { return m_connected; }
    QString controllerName() const { return m_name; }
    bool enabled() const { return m_enabled; }
    bool captureMode() const { return m_state.captureMode(); }
    bool voiceActive() const { return m_voiceActive; }
    bool musicMode() const { return m_musicMode; }
    void setMusicMode(bool on);
    bool analogCursor() const { return m_state.analogCursor(); }
    void setAnalogCursor(bool on);
    double leftX() const { return m_leftX; }
    double leftY() const { return m_leftY; }
    // Interactive live window only: starts watching GCController connect / disconnect.
    void setEnabled(bool on);
    void setCaptureMode(bool on);

    // Remapping API (persisted by Backend).
    QHash<QString, QString> bindingsMap() const { return m_state.bindings(); }
    void setBindingsMap(const QHash<QString, QString> &map);
    Q_INVOKABLE QString actionFor(const QString &inputId) const;
    Q_INVOKABLE QString actionLabel(const QString &actionId) const;
    Q_INVOKABLE QString inputLabel(const QString &inputId) const;
    // The compact glyph ("R1", "✕") of the first input bound to actionId, empty when unbound.
    Q_INVOKABLE QString glyphFor(const QString &actionId) const;
    QVariantList bindingSlots() const;
    QVariantList availableActions() const;

    // Feed one poll (the platform layer calls this every 8 ms; tests call it directly).
    void processSnapshot(const Snapshot &s, qint64 nowMs);
    // Platform layer (and tests): a controller came or went. Disconnect resets the debouncer.
    void setConnected(bool on, const QString &name);

    // Calls into GameController.framework so far (process-wide). Stays 0 in every offscreen mode.
    static int platformCalls();

    // Backend (live window): called on every L2 edge after voiceActive updates and before the
    // Phonon fn key goes out. It moves focus to the right text field (the in-Studio ask card, or
    // the MacBook-wide Spotlight bar) and returns how long the key-down should wait for that focus
    // to land (ms; <= 0 sends at once). Only consulted while the voice key is armed.
    void setPrepareVoiceTarget(std::function<int(bool active)> fn) { m_prepareVoiceTarget = std::move(fn); }
    // Tests: replaces the CGEvent post with a sink and arms the voice key without
    // GameController.framework (platformCalls stays 0).
    void setVoiceKeySinkForTest(std::function<void(bool down)> sink) { m_voiceKeySink = std::move(sink); }
    // Platform layer: a background d-pad press summons Studio through this when set (Backend
    // closes the Spotlight bar and restores ordered-out windows); unset, it just activates NSApp.
    void setSummonStudio(std::function<void()> fn) { m_summonStudio = std::move(fn); }
    // Platform layer: true while an overlay panel (the Spotlight bar) has NSApp active but the
    // Studio window is not what the user is looking at, so the d-pad still summons Studio.
    void setOverlayActive(std::function<bool()> fn) { m_overlayActive = std::move(fn); }
    // The fn key-down went out for the current hold (false while a delayed down is pending).
    bool voiceKeySent() const { return m_voiceKeySent; }

signals:
    void connectedChanged();
    void enabledChanged();
    void captureModeChanged();
    void voiceActiveChanged();
    void bindingsChanged();
    void musicModeChanged();
    void analogCursorChanged();
    void leftStickChanged();
    void nav(int dx, int dy);
    void confirm();
    // The confirm input (✕ by default) went down / up: hold-to-grab on the Constellation.
    void confirmHeld(bool down);
    // Analog cursor mode: shaped left-stick velocity, every poll while deflected, then one 0, 0.
    void cursorMove(double vx, double vy);
    void back();
    void backHeld(bool down);
    void acceptPrediction();
    void secondary();
    void step(int delta);
    void zoomStep(double delta);
    void zoomReset();
    void toggleSettings();
    void toggleNewSession();
    void toggleBriefing();
    void toggleMusic();
    void toggleReview();  // "toggle_review": unbound by default, mappable in the controller map
    void scrollDelta(double dy);
    void touchpadPan(double dx, double dy);
    void horizontalInspect(int dir);
    void voiceHold(bool active);
    void sendPressed();
    // Raw R3 press, whatever it is bound to (the Music card uses it to switch tabs). Fires once
    // the press resolves as a single click, never for the L3 + R3 chord.
    void r3Pressed();
    // L3 + R3 clicked together: confirm the staged Ambient proposal. Background monitoring is on,
    // so this fires from any Space or app.
    void confirmProposalRequested();
    void rawInputCaptured(const QString &inputId);

private:
    void triggerVoiceKey(bool down);
    void setVoiceActive(bool active);
    bool voiceKeyArmed() const { return m_enabled || bool(m_voiceKeySink); }
    struct Platform;
    std::function<int(bool)> m_prepareVoiceTarget;
    std::function<void(bool)> m_voiceKeySink;
    std::function<void()> m_summonStudio;
    std::function<bool()> m_overlayActive;
    QTimer *m_voiceKeyTimer = nullptr;  // a delayed fn key-down (focus still landing)
    bool m_voiceKeySent = false;
    GamepadState m_state;
    bool m_connected = false;
    bool m_enabled = false;
    bool m_voiceActive = false;
    bool m_musicMode = false;
    double m_leftX = 0, m_leftY = 0;
    QString m_name;
    std::unique_ptr<Platform> m_platform;
    friend struct Platform;
};

}  // namespace studio
