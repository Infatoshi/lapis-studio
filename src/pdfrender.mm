#include "pdfrender.hpp"

#import <AppKit/AppKit.h>
#import <PDFKit/PDFKit.h>
#include <CoreGraphics/CoreGraphics.h>

#include <algorithm>

namespace studio {

namespace {

PDFDocument *openDoc(const QString &path)
{
    NSString *p = path.toNSString();
    NSURL *url = [NSURL fileURLWithPath:p];
    return [[PDFDocument alloc] initWithURL:url];
}

}  // namespace

int pdfPageCount(const QString &path)
{
    @autoreleasepool {
        PDFDocument *doc = openDoc(path);
        return doc ? int(doc.pageCount) : 0;
    }
}

QImage renderPdfPage(const QString &path, int page, int targetWidth, bool ink)
{
    @autoreleasepool {
        PDFDocument *doc = openDoc(path);
        if (!doc || page < 0 || page >= int(doc.pageCount))
            return {};
        PDFPage *pg = [doc pageAtIndex:NSUInteger(page)];
        if (!pg)
            return {};
        const NSRect box = [pg boundsForBox:kPDFDisplayBoxMediaBox];
        if (box.size.width <= 0 || box.size.height <= 0)
            return {};
        const int w = std::clamp(targetWidth, 16, 6000);
        const double scale = double(w) / box.size.width;
        const int h = std::max(16, int(box.size.height * scale + 0.5));

        QImage img(w, h, QImage::Format_ARGB32_Premultiplied);
        img.fill(Qt::white);
        CGColorSpaceRef cs = CGColorSpaceCreateDeviceRGB();
        CGContextRef ctx = CGBitmapContextCreate(img.bits(), size_t(w), size_t(h), 8, size_t(img.bytesPerLine()), cs,
                                                 uint32_t(kCGImageAlphaPremultipliedFirst) | uint32_t(kCGBitmapByteOrder32Little));
        CGColorSpaceRelease(cs);
        if (!ctx)
            return {};
        CGContextSetInterpolationQuality(ctx, kCGInterpolationHigh);
        CGContextSetShouldAntialias(ctx, true);
        CGContextSetShouldSmoothFonts(ctx, true);
        CGContextScaleCTM(ctx, scale, scale);
        CGContextTranslateCTM(ctx, -box.origin.x, -box.origin.y);
        [pg drawWithBox:kPDFDisplayBoxMediaBox toContext:ctx];
        CGContextRelease(ctx);

        if (ink) {
            // card #0d0c0b .. parchment ink #e8d5b5
            const int c0[3] = {0x0d, 0x0c, 0x0b};
            const int c1[3] = {0xe8, 0xd5, 0xb5};
            for (int y = 0; y < h; ++y) {
                auto *row = reinterpret_cast<QRgb *>(img.scanLine(y));
                for (int x = 0; x < w; ++x) {
                    const QRgb px = row[x];
                    const int r = qRed(px), g = qGreen(px), b = qBlue(px);
                    const int mx = std::max({r, g, b}), mn = std::min({r, g, b});
                    if (mx - mn > 48)
                        continue;  // a coloured figure: keep it
                    const int lum = (r * 299 + g * 587 + b * 114) / 1000;
                    const int t = 255 - lum;  // ink amount
                    row[x] = qRgb(c0[0] + (c1[0] - c0[0]) * t / 255, c0[1] + (c1[1] - c0[1]) * t / 255,
                                  c0[2] + (c1[2] - c0[2]) * t / 255);
                }
            }
        }
        return img;
    }
}

}  // namespace studio
