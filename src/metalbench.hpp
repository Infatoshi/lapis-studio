// Offscreen Metal render target for --bench-fps --metal (QQuickRenderControl, no window).
#pragma once

#include <QImage>
#include <QSize>
#include <QString>

class QQuickGraphicsDevice;
class QQuickRenderTarget;

namespace studio {

struct MetalTarget {
    void *device = nullptr;   // id<MTLDevice>
    void *queue = nullptr;    // id<MTLCommandQueue>, shared with Qt's QRhi
    void *texture = nullptr;  // id<MTLTexture>, RGBA8, the frame Qt renders into
    QSize size;
};

bool createMetalTarget(MetalTarget *t, QSize pixelSize);
void destroyMetalTarget(MetalTarget *t);
QString metalDeviceName(const MetalTarget &t);
QQuickGraphicsDevice metalGraphicsDevice(const MetalTarget &t);
QQuickRenderTarget metalRenderTarget(const MetalTarget &t, qreal dpr);
void metalFinish(const MetalTarget &t);  // block until the GPU drained the queue
QImage metalReadback(const MetalTarget &t);

}  // namespace studio
