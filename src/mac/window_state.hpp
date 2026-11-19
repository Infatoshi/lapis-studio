// Live-window placement persistence: restores the saved geometry / fullscreen / maximized state
// on the right display, and records changes as they settle. Interactive window only; --test,
// --bench-fps and --snap-* never construct one.
#pragma once

#include "core/window_geometry.hpp"

#include <QObject>
#include <QPointer>
#include <QQuickWindow>
#include <QTimer>

#include <functional>

class QScreen;

namespace studio {

// Stable per-display id: the CoreGraphics display UUID on macOS, else the screen name.
QString displayUuidForScreen(QScreen *screen);
QScreen *screenForDisplayUuid(const QString &uuid);  // nullptr when no attached screen matches

// Place the (hidden) window from `state` and show it: matching display or the primary screen,
// rect clamped into its availableGeometry, then show() / showFullScreen() / showMaximized().
// No geometry: `fallbackSize` centred on the primary screen.
void restoreWindowState(QQuickWindow *window, const WindowState &state, const QSize &fallbackSize);

// Watches x/y/width/height/visibility; 250 ms after the last change it records the state and
// calls `persist` (which should write to disk at once, so a pkill never loses it). Normal
// geometry is only taken while the window is Windowed, so the frames macOS reports on the way
// into fullscreen never clobber it.
class WindowStateTracker : public QObject {
    Q_OBJECT
public:
    WindowStateTracker(QQuickWindow *window, const WindowState &initial,
                       std::function<void(const WindowState &)> persist, QObject *parent = nullptr);
    WindowState state() const { return m_state; }
    void flush();  // apply a pending change now

private:
    void capture();
    QPointer<QQuickWindow> m_window;
    WindowState m_state;
    std::function<void(const WindowState &)> m_persist;
    QTimer m_debounce;
};

}  // namespace studio
