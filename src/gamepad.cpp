// Pure C++ half of the gamepad bridge: the debouncer and the QObject's signal fan-out.
// No GameController calls here (see gamepad_mac.mm).
#include "gamepad.hpp"

#include <algorithm>
#include <cmath>

namespace studio {

void GamepadState::reset()
{
    *this = GamepadState();
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
        out.append({GamepadAction::Nav, dx, dy, 0});
    } else if (nowMs >= m_nextRepeatMs) {
        // On the nominal 110 ms grid; a stalled poll fires once and re-bases instead of bursting.
        m_nextRepeatMs += kRepeatIntervalMs;
        if (m_nextRepeatMs <= nowMs)
            m_nextRepeatMs = nowMs + kRepeatIntervalMs;
        out.append({GamepadAction::Nav, dx, dy, 0});
    }

    // Face, shoulder and system buttons: edge-triggered on press.
    auto pressed = [](bool now, bool before) { return now && !before; };
    if (pressed(s.a, p.a)) out.append({GamepadAction::Confirm});
    if (pressed(s.b, p.b)) out.append({GamepadAction::Back});
    if (pressed(s.y, p.y)) out.append({GamepadAction::AcceptPrediction});
    if (pressed(s.x, p.x) || pressed(s.l3, p.l3) || pressed(s.r3, p.r3)) out.append({GamepadAction::Secondary});
    if (pressed(s.l1, p.l1)) out.append({GamepadAction::Step, -1, 0, 0});
    if (pressed(s.r1, p.r1)) out.append({GamepadAction::Step, 1, 0, 0});
    if (pressed(s.menu, p.menu)) out.append({GamepadAction::ToggleSettings});
    if ((pressed(s.options, p.options) || pressed(s.touchpad, p.touchpad)) && !p.options && !p.touchpad)
        out.append({GamepadAction::ToggleNewSession});

    // Analog triggers: press past 0.55, re-arm below 0.45.
    if (!m_l2 && s.l2 > kTriggerPress) {
        m_l2 = true;
        out.append({GamepadAction::Zoom, 0, 0, -kZoomStep});
    } else if (m_l2 && s.l2 < kTriggerRelease) {
        m_l2 = false;
    }
    if (!m_r2 && s.r2 > kTriggerPress) {
        m_r2 = true;
        out.append({GamepadAction::Zoom, 0, 0, kZoomStep});
    } else if (m_r2 && s.r2 < kTriggerRelease) {
        m_r2 = false;
    }

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
            out.append({GamepadAction::Scroll, 0, 0, s.ry > 0 ? -px : px});  // stick up reads upward
            m_lastScrollMs = nowMs;
            m_scrolling = true;
        }
    }

    m_prev = s;
    return out;
}

void Gamepad::processSnapshot(const Snapshot &s, qint64 nowMs)
{
    for (const GamepadAction &a : m_state.process(s, nowMs)) {
        switch (a.kind) {
        case GamepadAction::Nav: emit nav(a.dx, a.dy); break;
        case GamepadAction::Confirm: emit confirm(); break;
        case GamepadAction::Back: emit back(); break;
        case GamepadAction::AcceptPrediction: emit acceptPrediction(); break;
        case GamepadAction::Secondary: emit secondary(); break;
        case GamepadAction::Step: emit step(a.dx); break;
        case GamepadAction::Zoom: emit zoomStep(a.value); break;
        case GamepadAction::ToggleSettings: emit toggleSettings(); break;
        case GamepadAction::ToggleNewSession: emit toggleNewSession(); break;
        case GamepadAction::Scroll: emit scrollDelta(a.value); break;
        }
    }
}

void Gamepad::setConnected(bool on, const QString &name)
{
    if (on == m_connected && name == m_name)
        return;
    m_connected = on;
    m_name = on ? name : QString();
    m_state.reset();
    emit connectedChanged();
}

}  // namespace studio
