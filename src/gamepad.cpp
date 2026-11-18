// Pure C++ half of the gamepad bridge: the debouncer and the QObject's signal fan-out.
// No GameController calls here (see gamepad_mac.mm).
#include "gamepad.hpp"

#include <algorithm>
#include <cmath>

namespace studio {

namespace {
struct InputMeta {
    const char *id;
    const char *label;
    const char *group;
    const char *glyph;  // compact on-screen hint ("R1", "✕")
};
const InputMeta kInputs[] = {
    {"l2", "L2 Trigger", "Triggers & Bumpers", "L2"},
    {"r2", "R2 Trigger", "Triggers & Bumpers", "R2"},
    {"l1", "L1 Bumper", "Triggers & Bumpers", "L1"},
    {"r1", "R1 Bumper", "Triggers & Bumpers", "R1"},
    {"a", "✕ Cross", "Face Buttons", "✕"},
    {"b", "○ Circle", "Face Buttons", "○"},
    {"x", "□ Square", "Face Buttons", "□"},
    {"y", "△ Triangle", "Face Buttons", "△"},
    {"l3", "L3 Stick Click", "Sticks & System", "L3"},
    {"r3", "R3 Stick Click", "Sticks & System", "R3"},
    {"menu", "Options ☰", "Sticks & System", "☰"},
    {"options", "Share / Touchpad", "Sticks & System", "Share"},
};

struct ActionMeta {
    const char *id;
    const char *label;
    const char *desc;
};
const ActionMeta kActions[] = {
    {"voice_hold", "Phonon Voice (Hold)", "Hold to dictate with Phonon, release to insert"},
    {"send", "Send / Submit", "Send prompt in composer or confirm dialog"},
    {"confirm", "Interact / Confirm", "Zoom into island/stage or focus composer"},
    {"back", "Back / Zoom Out", "Step back or close current overlay"},
    {"secondary", "Write-up / Picker", "Toggle write-up or open model & effort picker"},
    {"accept_prediction", "Tab / Full Send", "Accept the Tab prediction: open it, or fill the suggested prompt"},
    {"step_prev", "Previous View", "Step to the previous island, artifact, or page"},
    {"step_next", "Next View", "Step to the next island, artifact, or page"},
    {"zoom_in", "Zoom In (+10%)", "Increase UI / detail zoom by 10%"},
    {"zoom_out", "Zoom Out (-10%)", "Decrease UI / detail zoom by 10%"},
    {"zoom_reset", "Reset Zoom (1.00x)", "Reset UI / detail zoom back to 1.00x"},
    {"toggle_settings", "Settings & Controls", "Open Settings / Controller Mapping"},
    {"toggle_new_session", "New Session", "Open New Session dialog"},
    {"none", "Unbound", "No action"},
};
}  // namespace

QHash<QString, QString> GamepadState::defaultBindings()
{
    return {
        {QStringLiteral("l2"), QStringLiteral("voice_hold")},
        {QStringLiteral("r2"), QStringLiteral("send")},
        {QStringLiteral("l1"), QStringLiteral("step_prev")},
        {QStringLiteral("r1"), QStringLiteral("accept_prediction")},
        {QStringLiteral("a"), QStringLiteral("confirm")},
        {QStringLiteral("b"), QStringLiteral("back")},
        {QStringLiteral("x"), QStringLiteral("secondary")},
        {QStringLiteral("y"), QStringLiteral("step_next")},
        {QStringLiteral("l3"), QStringLiteral("zoom_reset")},
        {QStringLiteral("r3"), QStringLiteral("zoom_in")},
        {QStringLiteral("menu"), QStringLiteral("toggle_settings")},
        {QStringLiteral("options"), QStringLiteral("toggle_new_session")},
    };
}

QStringList GamepadState::validInputs()
{
    QStringList out;
    for (const InputMeta &m : kInputs)
        out << QString::fromLatin1(m.id);
    return out;
}

QStringList GamepadState::validActions()
{
    QStringList out;
    for (const ActionMeta &m : kActions)
        out << QString::fromLatin1(m.id);
    return out;
}

void GamepadState::setBindings(const QHash<QString, QString> &bindings)
{
    m_bindings = defaultBindings();
    const QStringList acts = validActions();
    for (auto it = bindings.cbegin(); it != bindings.cend(); ++it) {
        if (m_bindings.contains(it.key()) && acts.contains(it.value()))
            m_bindings.insert(it.key(), it.value());
    }
}

void GamepadState::emitMappedAction(const QString &inputId, bool down, QList<GamepadAction> &out) const
{
    const QString act = m_bindings.value(inputId, QStringLiteral("none"));
    if (act == QLatin1String("voice_hold")) {
        out.append({GamepadAction::VoiceHold, 0, 0, down ? 1.0 : 0.0, {}});
        return;
    }
    if (act == QLatin1String("back")) {
        out.append(GamepadAction::make(GamepadAction::BackHeld, 0, 0, down ? 1.0 : 0.0));
        if (down)
            out.append(GamepadAction::make(GamepadAction::Back));
        return;
    }
    if (!down)
        return;
    if (act == QLatin1String("send"))
        out.append(GamepadAction::make(GamepadAction::Send));
    else if (act == QLatin1String("confirm"))
        out.append(GamepadAction::make(GamepadAction::Confirm));
    else if (act == QLatin1String("secondary"))
        out.append(GamepadAction::make(GamepadAction::Secondary));
    else if (act == QLatin1String("accept_prediction"))
        out.append(GamepadAction::make(GamepadAction::AcceptPrediction));
    else if (act == QLatin1String("step_prev"))
        out.append(GamepadAction::make(GamepadAction::Step, -1));
    else if (act == QLatin1String("step_next"))
        out.append(GamepadAction::make(GamepadAction::Step, 1));
    else if (act == QLatin1String("zoom_in"))
        out.append(GamepadAction::make(GamepadAction::Zoom, 0, 0, kZoomStep));
    else if (act == QLatin1String("zoom_out"))
        out.append(GamepadAction::make(GamepadAction::Zoom, 0, 0, -kZoomStep));
    else if (act == QLatin1String("zoom_reset"))
        out.append(GamepadAction::make(GamepadAction::ZoomReset));
    else if (act == QLatin1String("toggle_settings"))
        out.append(GamepadAction::make(GamepadAction::ToggleSettings));
    else if (act == QLatin1String("toggle_new_session"))
        out.append(GamepadAction::make(GamepadAction::ToggleNewSession));
}

void GamepadState::reset()
{
    const QHash<QString, QString> keepBindings = m_bindings;
    *this = GamepadState();
    m_bindings = keepBindings;
}

// The direction the user holds, as one discrete screen step. The d-pad wins over the stick;
// the stick snaps to its dominant axis (no diagonal nav) with release hysteresis.
void GamepadState::navDirection(const GamepadSnapshot &s, int *dx, int *dy) const
{
    *dx = 0;
    *dy = 0;
    if (s.dpadX != 0 || s.dpadY != 0) {
        if (s.dpadX != 0)
            *dx = s.dpadX > 0 ? 1 : -1;
        else
            *dy = s.dpadY > 0 ? -1 : 1;  // pad up (+1) is screen up (-1)
        return;
    }
    const double ax = std::abs(s.lx), ay = std::abs(s.ly);
    // Already holding a stick direction: keep it while that axis stays past the release line
    // and the other axis has not clearly taken over.
    if (m_navFromStick && (m_navDx != 0 || m_navDy != 0)) {
        const double along = m_navDx != 0 ? s.lx * m_navDx : -s.ly * m_navDy;
        const double across = m_navDx != 0 ? ay : ax;
        if (along > kStickRelease && !(across > kStickDeadzone && across > along)) {
            *dx = m_navDx;
            *dy = m_navDy;
            return;
        }
    }
    if (std::max(ax, ay) <= kStickDeadzone)
        return;
    if (ax >= ay)
        *dx = s.lx > 0 ? 1 : -1;
    else
        *dy = s.ly > 0 ? -1 : 1;
}

QList<GamepadAction> GamepadState::process(const GamepadSnapshot &s, qint64 nowMs)
{
    QList<GamepadAction> out;
    const GamepadSnapshot &p = m_prev;

    // Discrete navigation: fire on engage, then after 320 ms every 110 ms while held.
    int dx = 0, dy = 0;
    navDirection(s, &dx, &dy);
    if (dx == 0 && dy == 0) {
        m_navDx = m_navDy = 0;
        m_navFromStick = false;
    } else if (dx != m_navDx || dy != m_navDy) {
        m_navDx = dx;
        m_navDy = dy;
        m_navFromStick = s.dpadX == 0 && s.dpadY == 0;
        m_nextRepeatMs = nowMs + kRepeatDelayMs;
        out.append(GamepadAction::make(GamepadAction::Nav, dx, dy));
    } else if (nowMs >= m_nextRepeatMs) {
        // On the nominal 110 ms grid; a stalled poll fires once and re-bases instead of bursting.
        m_nextRepeatMs += kRepeatIntervalMs;
        if (m_nextRepeatMs <= nowMs)
            m_nextRepeatMs = nowMs + kRepeatIntervalMs;
        out.append(GamepadAction::make(GamepadAction::Nav, dx, dy));
    }

    // Face, shoulder, stick-click and system buttons.
    auto pressed = [](bool now, bool before) { return now && !before; };
    auto released = [](bool now, bool before) { return !now && before; };

    bool l2EdgeOn = false, l2EdgeOff = false;
    if (!m_l2 && s.l2 > kTriggerPress) {
        m_l2 = true;
        l2EdgeOn = true;
    } else if (m_l2 && s.l2 < kTriggerRelease) {
        m_l2 = false;
        l2EdgeOff = true;
    }
    bool r2EdgeOn = false, r2EdgeOff = false;
    if (!m_r2 && s.r2 > kTriggerPress) {
        m_r2 = true;
        r2EdgeOn = true;
    } else if (m_r2 && s.r2 < kTriggerRelease) {
        m_r2 = false;
        r2EdgeOff = true;
    }

    const bool shareOrPadNow = s.options || s.touchpad;
    const bool shareOrPadPrev = p.options || p.touchpad;

    if (m_capture) {
        const QPair<const char *, bool> checks[] = {
            {"a", pressed(s.a, p.a)},
            {"b", pressed(s.b, p.b)},
            {"x", pressed(s.x, p.x)},
            {"y", pressed(s.y, p.y)},
            {"l1", pressed(s.l1, p.l1)},
            {"r1", pressed(s.r1, p.r1)},
            {"l2", l2EdgeOn},
            {"r2", r2EdgeOn},
            {"l3", pressed(s.l3, p.l3)},
            {"r3", pressed(s.r3, p.r3)},
            {"menu", pressed(s.menu, p.menu)},
            {"options", pressed(shareOrPadNow, shareOrPadPrev)},
        };
        for (const auto &c : checks) {
            if (c.second) {
                out.append({GamepadAction::RawCaptured, 0, 0, 0, QString::fromLatin1(c.first)});
                break;
            }
        }
        m_prev = s;
        return out;
    }

    auto handleDigital = [&](const char *id, bool now, bool before) {
        if (pressed(now, before))
            emitMappedAction(QString::fromLatin1(id), true, out);
        else if (released(now, before))
            emitMappedAction(QString::fromLatin1(id), false, out);
    };
    handleDigital("a", s.a, p.a);
    handleDigital("b", s.b, p.b);
    handleDigital("x", s.x, p.x);
    handleDigital("y", s.y, p.y);
    handleDigital("l1", s.l1, p.l1);
    handleDigital("r1", s.r1, p.r1);
    handleDigital("l3", s.l3, p.l3);
    handleDigital("r3", s.r3, p.r3);
    handleDigital("menu", s.menu, p.menu);
    // Split touchpad click: left half opens Morning Briefing; right half (or Share) runs mapped "options" action.
    if (pressed(s.touchpad, p.touchpad) && !s.options && !p.options && s.touchpadX < -0.15)
        out.append(GamepadAction::make(GamepadAction::ToggleBriefing));
    else
        handleDigital("options", shareOrPadNow, shareOrPadPrev);

    // Touchpad finger drag: smooth 2D pan for wide tables / charts.
    const bool touchingNow = std::abs(s.touchpadX) > 1e-4 || std::abs(s.touchpadY) > 1e-4;
    const bool touchingPrev = std::abs(p.touchpadX) > 1e-4 || std::abs(p.touchpadY) > 1e-4;
    if (touchingNow && touchingPrev) {
        const int pdx = int(std::lround((s.touchpadX - p.touchpadX) * 420.0));
        const int pdy = int(std::lround((p.touchpadY - s.touchpadY) * 420.0));
        if (pdx != 0 || pdy != 0)
            out.append(GamepadAction::make(GamepadAction::TouchpadPan, pdx, pdy));
    }

    if (l2EdgeOn)
        emitMappedAction(QStringLiteral("l2"), true, out);
    else if (l2EdgeOff)
        emitMappedAction(QStringLiteral("l2"), false, out);
    if (r2EdgeOn)
        emitMappedAction(QStringLiteral("r2"), true, out);
    else if (r2EdgeOff)
        emitMappedAction(QStringLiteral("r2"), false, out);

    // Right stick: continuous scroll, quadratic past the deadzone, one delta per 8 ms tick
    // scaled by the real elapsed time (capped so a stalled poll never jumps a page).
    const double ar = std::abs(s.ry);
    if (ar <= kScrollDeadzone) {
        m_scrolling = false;
    } else {
        const qint64 dt = m_scrolling ? nowMs - m_lastScrollMs : kScrollTickMs;
        if (dt >= kScrollTickMs) {
            const double t = std::min(1.0, (ar - kScrollDeadzone) / (1.0 - kScrollDeadzone));
            const double px = t * t * kScrollPxPerSec * double(std::min<qint64>(dt, 50)) / 1000.0;
            out.append(GamepadAction::make(GamepadAction::Scroll, 0, 0, s.ry > 0 ? -px : px));  // stick up reads upward
            m_lastScrollMs = nowMs;
            m_scrolling = true;
        }
    }

    // Right stick horizontal: discrete detail inspect / zoom or horizontal step.
    const int rdx = (std::abs(s.rx) > kStickDeadzone && std::abs(s.rx) > std::abs(s.ry))
                        ? (s.rx > 0 ? 1 : -1)
                        : 0;
    if (rdx == 0) {
        m_rNavDx = 0;
    } else if (rdx != m_rNavDx) {
        m_rNavDx = rdx;
        m_nextRNavMs = nowMs + kRepeatDelayMs;
        out.append(GamepadAction::make(GamepadAction::HorizontalInspect, rdx));
    } else if (nowMs >= m_nextRNavMs) {
        m_nextRNavMs += kRepeatIntervalMs;
        if (m_nextRNavMs <= nowMs)
            m_nextRNavMs = nowMs + kRepeatIntervalMs;
        out.append(GamepadAction::make(GamepadAction::HorizontalInspect, rdx));
    }

    m_prev = s;
    return out;
}

void Gamepad::setCaptureMode(bool on)
{
    if (m_state.captureMode() == on)
        return;
    m_state.setCaptureMode(on);
    emit captureModeChanged();
}

void Gamepad::setBindingsMap(const QHash<QString, QString> &map)
{
    m_state.setBindings(map);
    emit bindingsChanged();
}

QString Gamepad::actionFor(const QString &inputId) const
{
    return m_state.bindings().value(inputId, QStringLiteral("none"));
}

QString Gamepad::actionLabel(const QString &actionId) const
{
    for (const ActionMeta &m : kActions)
        if (QLatin1String(m.id) == actionId)
            return QString::fromUtf8(m.label);
    return actionId;
}

QString Gamepad::inputLabel(const QString &inputId) const
{
    for (const InputMeta &m : kInputs)
        if (QLatin1String(m.id) == inputId)
            return QString::fromUtf8(m.label);
    return inputId;
}

QString Gamepad::glyphFor(const QString &actionId) const
{
    const QHash<QString, QString> map = m_state.bindings();
    for (const InputMeta &m : kInputs)
        if (map.value(QString::fromLatin1(m.id)) == actionId)
            return QString::fromUtf8(m.glyph);
    return QString();
}

QVariantList Gamepad::bindingSlots() const
{
    QVariantList out;
    const QHash<QString, QString> map = m_state.bindings();
    for (const InputMeta &m : kInputs) {
        const QString id = QString::fromLatin1(m.id);
        const QString act = map.value(id, QStringLiteral("none"));
        out << QVariantMap{
            {QStringLiteral("id"), id},
            {QStringLiteral("label"), QString::fromUtf8(m.label)},
            {QStringLiteral("group"), QString::fromUtf8(m.group)},
            {QStringLiteral("action"), act},
            {QStringLiteral("actionLabel"), actionLabel(act)},
            {QStringLiteral("glyph"), QString::fromUtf8(m.glyph)},
        };
    }
    return out;
}

QVariantList Gamepad::availableActions() const
{
    QVariantList out;
    for (const ActionMeta &m : kActions) {
        out << QVariantMap{
            {QStringLiteral("id"), QString::fromLatin1(m.id)},
            {QStringLiteral("label"), QString::fromUtf8(m.label)},
            {QStringLiteral("desc"), QString::fromUtf8(m.desc)},
        };
    }
    return out;
}

void Gamepad::processSnapshot(const Snapshot &s, qint64 nowMs)
{
    for (const GamepadAction &a : m_state.process(s, nowMs)) {
        switch (a.kind) {
        case GamepadAction::Nav: emit nav(a.dx, a.dy); break;
        case GamepadAction::Confirm: emit confirm(); break;
        case GamepadAction::Back: emit back(); break;
        case GamepadAction::BackHeld: emit backHeld(a.value > 0.5); break;
        case GamepadAction::AcceptPrediction: emit acceptPrediction(); break;
        case GamepadAction::Secondary: emit secondary(); break;
        case GamepadAction::Step: emit step(a.dx); break;
        case GamepadAction::Zoom: emit zoomStep(a.value); break;
        case GamepadAction::ZoomReset: emit zoomReset(); break;
        case GamepadAction::ToggleSettings: emit toggleSettings(); break;
        case GamepadAction::ToggleNewSession: emit toggleNewSession(); break;
        case GamepadAction::ToggleBriefing: emit toggleBriefing(); break;
        case GamepadAction::Scroll: emit scrollDelta(a.value); break;
        case GamepadAction::TouchpadPan: emit touchpadPan(double(a.dx), double(a.dy)); break;
        case GamepadAction::HorizontalInspect: emit horizontalInspect(a.dx); break;
        case GamepadAction::Send: emit sendPressed(); break;
        case GamepadAction::VoiceHold: {
            const bool active = a.value > 0.5;
            if (m_voiceActive != active) {
                m_voiceActive = active;
                if (m_enabled)
                    triggerVoiceKey(active);
                emit voiceActiveChanged();
                emit voiceHold(active);
            }
            break;
        }
        case GamepadAction::RawCaptured:
            setCaptureMode(false);
            emit rawInputCaptured(a.inputId);
            break;
        }
    }
}

void Gamepad::setConnected(bool on, const QString &name)
{
    if (on == m_connected && name == m_name)
        return;
    if (!on && m_voiceActive) {
        m_voiceActive = false;
        if (m_enabled)
            triggerVoiceKey(false);
        emit voiceActiveChanged();
        emit voiceHold(false);
    }
    m_connected = on;
    m_name = on ? name : QString();
    m_state.reset();
    emit connectedChanged();
}

}  // namespace studio
