#include "mac/window_state.hpp"

#include <QGuiApplication>
#include <QQuickWindow>
#include <QScreen>

#ifdef Q_OS_MACOS
#include <ApplicationServices/ApplicationServices.h>
#endif

namespace studio {

QString displayUuidForScreen(QScreen *screen)
{
    if (!screen)
        return {};
#ifdef Q_OS_MACOS
    // Qt's screen geometry and CGDisplayBounds share one global, top-left-origin point space.
    CGDirectDisplayID ids[32];
    uint32_t count = 0;
    if (CGGetActiveDisplayList(32, ids, &count) == kCGErrorSuccess) {
        const QRect g = screen->geometry();
        for (uint32_t i = 0; i < count; ++i) {
            const CGRect b = CGDisplayBounds(ids[i]);
            if (int(b.origin.x) != g.x() || int(b.origin.y) != g.y() || int(b.size.width) != g.width()
                || int(b.size.height) != g.height())
                continue;
            CFUUIDRef uuid = CGDisplayCreateUUIDFromDisplayID(ids[i]);
            if (!uuid)
                break;
            CFStringRef str = CFUUIDCreateString(kCFAllocatorDefault, uuid);
            CFRelease(uuid);
            const QString out = QString::fromCFString(str);
            CFRelease(str);
            if (!out.isEmpty())
                return out;
            break;
        }
    }
#endif
    return screen->name();
}

QScreen *screenForDisplayUuid(const QString &uuid)
{
    if (uuid.isEmpty())
        return nullptr;
    for (QScreen *s : QGuiApplication::screens())
        if (displayUuidForScreen(s) == uuid)
            return s;
    return nullptr;
}

void restoreWindowState(QQuickWindow *window, const WindowState &state, const QSize &fallbackSize)
{
    if (!window)
        return;
    QScreen *screen = screenForDisplayUuid(state.displayUuid);
    if (!screen)
        screen = QGuiApplication::primaryScreen();
    const QRect avail = screen ? screen->availableGeometry() : QRect();
    const QSize minSize = window->minimumSize().isValid() ? window->minimumSize() : QSize(1100, 700);
    QRect want;
    if (state.hasGeometry()) {
        want = QRect(avail.topLeft() + QPoint(state.relX, state.relY), QSize(state.width, state.height));
    } else {
        want = QRect(QPoint(), fallbackSize);
        if (avail.isValid())
            want.moveCenter(avail.center());
    }
    if (screen)
        window->setScreen(screen);
    window->setGeometry(clampWindowRect(want, avail, minSize));
    if (state.fullscreen)
        window->showFullScreen();
    else if (state.maximized)
        window->showMaximized();
    else
        window->show();
}

WindowStateTracker::WindowStateTracker(QQuickWindow *window, const WindowState &initial,
                                       std::function<void(const WindowState &)> persist, QObject *parent)
    : QObject(parent)
    , m_window(window)
    , m_state(initial)
    , m_persist(std::move(persist))
{
    m_debounce.setSingleShot(true);
    m_debounce.setInterval(250);
    connect(&m_debounce, &QTimer::timeout, this, &WindowStateTracker::capture);
    if (!window)
        return;
    auto poke = [this] { m_debounce.start(); };
    connect(window, &QWindow::xChanged, this, poke);
    connect(window, &QWindow::yChanged, this, poke);
    connect(window, &QWindow::widthChanged, this, poke);
    connect(window, &QWindow::heightChanged, this, poke);
    connect(window, &QWindow::visibilityChanged, this, poke);
}

void WindowStateTracker::flush()
{
    if (m_debounce.isActive()) {
        m_debounce.stop();
        capture();
    }
}

void WindowStateTracker::capture()
{
    if (!m_window)
        return;
    const QWindow::Visibility vis = m_window->visibility();
    if (vis == QWindow::Hidden || vis == QWindow::Minimized)
        return;  // closing or in the Dock: keep the last real state
    WindowState next = m_state;
    next.fullscreen = vis == QWindow::FullScreen;
    next.maximized = vis == QWindow::Maximized;
    if (vis == QWindow::Windowed) {
        QScreen *screen = m_window->screen();
        const QRect g = m_window->geometry();
        const QRect avail = screen ? screen->availableGeometry() : QRect();
        next.displayUuid = displayUuidForScreen(screen);
        next.relX = g.x() - avail.x();
        next.relY = g.y() - avail.y();
        next.width = g.width();
        next.height = g.height();
    }
    if (next == m_state)
        return;
    m_state = next;
    if (m_persist)
        m_persist(m_state);
}

}  // namespace studio
