#pragma once
#include <QPointF>
#include <QString>
#include <cstdint>
#include <memory>

namespace studio::cua {
// All methods, construction, and destruction must run on the AppKit main thread.
// Pass windowNumber() to captureDisplayJpeg for explicit ScreenCaptureKit exclusion.
class CuaReticle {
public:
    CuaReticle();
    ~CuaReticle();
    CuaReticle(const CuaReticle &) = delete;
    CuaReticle &operator=(const CuaReticle &) = delete;
    void moveTo(QPointF globalPt, const QString &label = {});
    void hide();
    uint32_t windowNumber() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace studio::cua
