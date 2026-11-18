// --bench-fps --metal: an offscreen Metal texture for QQuickRenderControl (no window, no layer).
#include "metalbench.hpp"

#include <QQuickGraphicsDevice>
#include <QQuickRenderTarget>

#import <Metal/Metal.h>

namespace studio {

bool createMetalTarget(MetalTarget *t, QSize pixelSize)
{
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev)
        return false;
    id<MTLCommandQueue> queue = [dev newCommandQueue];
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                 width:NSUInteger(pixelSize.width())
                                                                                height:NSUInteger(pixelSize.height())
                                                                             mipmapped:NO];
    d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    d.storageMode = MTLStorageModePrivate;
    id<MTLTexture> tex = [dev newTextureWithDescriptor:d];
    if (!queue || !tex)
        return false;
    t->device = (void *)dev;     // +1 from MTLCreateSystemDefaultDevice
    t->queue = (void *)queue;    // +1 from new...
    t->texture = (void *)tex;    // +1 from new...
    t->size = pixelSize;
    return true;
}

void destroyMetalTarget(MetalTarget *t)
{
    if (t->texture)
        [(id<MTLTexture>)t->texture release];
    if (t->queue)
        [(id<MTLCommandQueue>)t->queue release];
    if (t->device)
        [(id<MTLDevice>)t->device release];
    *t = MetalTarget{};
}

QString metalDeviceName(const MetalTarget &t)
{
    return t.device ? QString::fromNSString([(id<MTLDevice>)t.device name]) : QString();
}

QQuickGraphicsDevice metalGraphicsDevice(const MetalTarget &t)
{
    return QQuickGraphicsDevice::fromDeviceAndCommandQueue((MTLDevice *)t.device, (MTLCommandQueue *)t.queue);
}

QQuickRenderTarget metalRenderTarget(const MetalTarget &t, qreal dpr)
{
    QQuickRenderTarget rt = QQuickRenderTarget::fromMetalTexture((MTLTexture *)t.texture, MTLPixelFormatRGBA8Unorm, t.size);
    rt.setDevicePixelRatio(dpr);
    return rt;
}

void metalFinish(const MetalTarget &t)
{
    // The queue is in order: an empty command buffer completes only after every earlier one,
    // so this returns once the GPU has finished the frame Qt just submitted on this queue.
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)t.queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

QImage metalReadback(const MetalTarget &t)
{
    @autoreleasepool {
        id<MTLDevice> dev = (id<MTLDevice>)t.device;
        const NSUInteger w = NSUInteger(t.size.width()), h = NSUInteger(t.size.height());
        const NSUInteger bpr = w * 4;
        id<MTLBuffer> buf = [dev newBufferWithLength:bpr * h options:MTLResourceStorageModeShared];
        id<MTLCommandBuffer> cb = [(id<MTLCommandQueue>)t.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit copyFromTexture:(id<MTLTexture>)t.texture
                  sourceSlice:0
                  sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(w, h, 1)
                     toBuffer:buf
            destinationOffset:0
       destinationBytesPerRow:bpr
     destinationBytesPerImage:bpr * h];
        [blit endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        QImage img(static_cast<const uchar *>([buf contents]), int(w), int(h), int(bpr), QImage::Format_RGBA8888);
        QImage out = img.copy();
        [buf release];
        return out;
    }
}

}  // namespace studio
