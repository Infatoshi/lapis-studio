#include "core/window_geometry.hpp"

#include <algorithm>

namespace studio {

QJsonObject WindowState::toJson() const
{
    return QJsonObject{{QStringLiteral("displayUuid"), displayUuid},
                       {QStringLiteral("relX"), relX},
                       {QStringLiteral("relY"), relY},
                       {QStringLiteral("width"), width},
                       {QStringLiteral("height"), height},
                       {QStringLiteral("fullscreen"), fullscreen},
                       {QStringLiteral("maximized"), maximized}};
}

WindowState WindowState::fromJson(const QJsonObject &o)
{
    WindowState s;
    s.displayUuid = o.value(QLatin1String("displayUuid")).toString();
    s.relX = o.value(QLatin1String("relX")).toInt(0);
    s.relY = o.value(QLatin1String("relY")).toInt(0);
    s.width = o.value(QLatin1String("width")).toInt(0);
    s.height = o.value(QLatin1String("height")).toInt(0);
    if (s.width <= 0 || s.height <= 0)
        s.width = s.height = 0;
    s.fullscreen = o.value(QLatin1String("fullscreen")).toBool(false);
    s.maximized = o.value(QLatin1String("maximized")).toBool(false);
    return s;
}

QRect clampWindowRect(const QRect &rect, const QRect &available, const QSize &minSize)
{
    if (!available.isValid())
        return rect;
    const int w = std::min(std::max(rect.width(), minSize.width()), available.width());
    const int h = std::min(std::max(rect.height(), minSize.height()), available.height());
    const int x = std::clamp(rect.x(), available.left(), available.left() + available.width() - w);
    const int y = std::clamp(rect.y(), available.top(), available.top() + available.height() - h);
    return QRect(x, y, w, h);
}

}  // namespace studio
