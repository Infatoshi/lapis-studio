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

#include <memory>

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
};

struct GamepadAction {
    enum Kind {
        Nav,
        Confirm,
        Back,
        AcceptPrediction,
        Secondary,
        Step,
        Zoom,
        ZoomReset,
        ToggleSettings,
        ToggleNewSession,
        Scroll,
        VoiceHold,
        Send,
        RawCaptured
    };
    Kind kind;
    int dx = 0, dy = 0;   // Nav: screen direction (dy +1 = down); Step: dx = -1 / +1
    double value = 0;     // Zoom: uiScale delta; Scroll: pixels; VoiceHold: 1.0 down / 0.0 up
    QString inputId = {}; // RawCaptured: physical slot id ("a", "l2", ...)
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

    static QHash<QString, QString> defaultBindings();
    static QStringList validInputs();
    static QStringList validActions();

    GamepadState() : m_bindings(defaultBindings()) {}

    void setBindings(const QHash<QString, QString> &bindings);
    const QHash<QString, QString> &bindings() const { return m_bindings; }
    void setCaptureMode(bool on) { m_capture = on; }
    bool captureMode() const { return m_capture; }

    QList<GamepadAction> process(const GamepadSnapshot &s, qint64 nowMs);
    void reset();

private:
    void navDirection(const GamepadSnapshot &s, int *dx, int *dy) const;
    void emitMappedAction(const QString &inputId, bool pressed, QList<GamepadAction> &out) const;

    QHash<QString, QString> m_bindings;
    bool m_capture = false;
    GamepadSnapshot m_prev;
    int m_navDx = 0, m_navDy = 0;
    bool m_navFromStick = false;
    qint64 m_nextRepeatMs = 0;
    bool m_l2 = false, m_r2 = false;
    bool m_scrolling = false;
    qint64 m_lastScrollMs = 0;
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

public:
    using Snapshot = GamepadSnapshot;

    explicit Gamepad(QObject *parent = nullptr);
    ~Gamepad() override;

    bool connected() const { return m_connected; }
    QString controllerName() const { return m_name; }
    bool enabled() const { return m_enabled; }
    bool captureMode() const { return m_state.captureMode(); }
    bool voiceActive() const { return m_voiceActive; }
    // Interactive live window only: starts watching GCController connect / disconnect.
    void setEnabled(bool on);
    void setCaptureMode(bool on);

    // Remapping API (persisted by Backend).
    QHash<QString, QString> bindingsMap() const { return m_state.bindings(); }
    void setBindingsMap(const QHash<QString, QString> &map);
    Q_INVOKABLE QString actionFor(const QString &inputId) const;
    Q_INVOKABLE QString actionLabel(const QString &actionId) const;
    Q_INVOKABLE QString inputLabel(const QString &inputId) const;
    QVariantList bindingSlots() const;
    QVariantList availableActions() const;

    // Feed one poll (the platform layer calls this every 8 ms; tests call it directly).
    void processSnapshot(const Snapshot &s, qint64 nowMs);
    // Platform layer (and tests): a controller came or went. Disconnect resets the debouncer.
    void setConnected(bool on, const QString &name);

    // Calls into GameController.framework so far (process-wide). Stays 0 in every offscreen mode.
    static int platformCalls();

signals:
    void connectedChanged();
    void enabledChanged();
    void captureModeChanged();
    void voiceActiveChanged();
    void bindingsChanged();
    void nav(int dx, int dy);
    void confirm();
    void back();
    void acceptPrediction();
    void secondary();
    void step(int delta);
    void zoomStep(double delta);
    void zoomReset();
    void toggleSettings();
    void toggleNewSession();
    void scrollDelta(double dy);
    void voiceHold(bool active);
    void sendPressed();
    void rawInputCaptured(const QString &inputId);

private:
    void triggerVoiceKey(bool down);
    struct Platform;
    GamepadState m_state;
    bool m_connected = false;
    bool m_enabled = false;
    bool m_voiceActive = false;
    QString m_name;
    std::unique_ptr<Platform> m_platform;
    friend struct Platform;
};

}  // namespace studio
