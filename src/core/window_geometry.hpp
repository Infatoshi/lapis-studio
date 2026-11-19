// Window placement as persisted in studio_prefs.json ("windowState"), and the clamp that keeps
// a restored window visibly on its screen. QtCore only, so test_core_ingest covers it; the
// macOS tracker that feeds it lives in src/mac/window_state.hpp.
#pragma once

#include <QJsonObject>
#include <QRect>
#include <QSize>
#include <QString>

namespace studio {

struct WindowState {
    QString displayUuid;  // CGDisplay UUID of the screen the normal geometry sits on
    int relX = 0;         // normal (windowed) geometry, relative to that screen's availableGeometry
    int relY = 0;
    int width = 0;
    int height = 0;
    bool fullscreen = false;
    bool maximized = false;

    bool hasGeometry() const { return width > 0 && height > 0; }
    QJsonObject toJson() const;
    // Missing or malformed fields read as defaults; a non-positive size means "no geometry".
    static WindowState fromJson(const QJsonObject &o);
    friend bool operator==(const WindowState &a, const WindowState &b)
    {
        return a.displayUuid == b.displayUuid && a.relX == b.relX && a.relY == b.relY && a.width == b.width
            && a.height == b.height && a.fullscreen == b.fullscreen && a.maximized == b.maximized;
    }
    friend bool operator!=(const WindowState &a, const WindowState &b) { return !(a == b); }
};

// `rect` moved (and shrunk only if it is larger) so it lies wholly inside `available`, never
// smaller than `minSize` unless the screen itself is smaller.
QRect clampWindowRect(const QRect &rect, const QRect &available, const QSize &minSize = QSize(1100, 700));

}  // namespace studio
