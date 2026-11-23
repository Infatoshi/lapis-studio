#include "mac/cua_pad_icon.hpp"

#import <CoreText/CoreText.h>
#import <Foundation/Foundation.h>

#include <algorithm>
#include <cmath>

namespace studio::cua {
namespace {

PadRgba rgb(uint32_t hex, CGFloat a = 1)
{
    return {((hex >> 16) & 0xFF) / 255.0, ((hex >> 8) & 0xFF) / 255.0, (hex & 0xFF) / 255.0, a};
}

PadRgba mix(const PadRgba &a, const PadRgba &b, CGFloat t)
{
    return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t};
}

PadRgba withAlpha(PadRgba c, CGFloat a)
{
    c.a *= a;
    return c;
}

// sRGB colour setters (CGContextSetRGB* would pick DeviceRGB).
void setFill(CGContextRef ctx, const PadRgba &c)
{
    CGColorRef col = CGColorCreateSRGB(c.r, c.g, c.b, c.a);
    CGContextSetFillColorWithColor(ctx, col);
    CGColorRelease(col);
}

void setStroke(CGContextRef ctx, const PadRgba &c)
{
    CGColorRef col = CGColorCreateSRGB(c.r, c.g, c.b, c.a);
    CGContextSetStrokeColorWithColor(ctx, col);
    CGColorRelease(col);
}

// Fills `path` with the body, then strokes it with the rim.
void keycap(CGContextRef ctx, CGPathRef path, const PadIconColors &c, CGFloat rimW)
{
    CGContextAddPath(ctx, path);
    setFill(ctx, c.fill);
    CGContextFillPath(ctx);
    CGContextAddPath(ctx, path);
    CGContextSetLineWidth(ctx, rimW);
    setStroke(ctx, c.rim);
    CGContextStrokePath(ctx);
}

// A soft highlight just inside the top of `path`: the path stroked, clipped to its upper band,
// in ink at low alpha (reads as a bevel catching light, not a second outline).
void topHighlight(CGContextRef ctx, CGPathRef path, CGRect bounds, CGFloat band, const PadIconColors &c)
{
    CGContextSaveGState(ctx);
    CGContextAddPath(ctx, path);
    CGContextClip(ctx);
    CGContextClipToRect(ctx, CGRectMake(bounds.origin.x, bounds.origin.y, bounds.size.width, band));
    CGContextAddPath(ctx, path);
    CGContextSetLineWidth(ctx, 2.4);
    setStroke(ctx, withAlpha(c.ink, 0.16));
    CGContextStrokePath(ctx);
    CGContextRestoreGState(ctx);
}

// A Menlo label centred on (cx, cy) by its cap height, in a y-down context.
void label(CGContextRef ctx, NSString *text, CGFloat size, bool bold, CGFloat cx, CGFloat cy, const PadRgba &ink)
{
    CTFontRef font = CTFontCreateWithName(bold ? CFSTR("Menlo-Bold") : CFSTR("Menlo-Regular"), size, nullptr);
    if (!font)
        return;
    CGColorRef col = CGColorCreateSRGB(ink.r, ink.g, ink.b, ink.a);
    NSDictionary *attrs = @{(id)kCTFontAttributeName: (id)font, (id)kCTForegroundColorAttributeName: (id)col};
    CFAttributedStringRef as = CFAttributedStringCreate(kCFAllocatorDefault, (CFStringRef)text, (CFDictionaryRef)attrs);
    CTLineRef line = CTLineCreateWithAttributedString(as);
    const double w = CTLineGetTypographicBounds(line, nullptr, nullptr, nullptr);
    const CGFloat cap = CTFontGetCapHeight(font);
    CGContextSaveGState(ctx);
    CGContextSetTextMatrix(ctx, CGAffineTransformMakeScale(1, -1));
    CGContextSetTextPosition(ctx, cx - w / 2, cy + cap / 2);
    CTLineDraw(line, ctx);
    CGContextRestoreGState(ctx);
    CFRelease(line);
    CFRelease(as);
    CGColorRelease(col);
    CFRelease(font);
}

void faceButton(CGContextRef ctx, PadButton b, CGRect r, const PadIconColors &c)
{
    const CGFloat h = std::min(r.size.width, r.size.height);
    const CGFloat rimW = std::clamp(h * 0.075, 1.0, 1.5);
    const CGFloat cx = CGRectGetMidX(r), cy = CGRectGetMidY(r), R = h / 2 - rimW / 2 - 0.25;
    CGPathRef body = CGPathCreateWithEllipseInRect(CGRectMake(cx - R, cy - R, 2 * R, 2 * R), nullptr);
    keycap(ctx, body, c, rimW);
    topHighlight(ctx, body, CGRectMake(cx - R, cy - R, 2 * R, 2 * R), R * 0.55, c);
    CGPathRelease(body);

    const CGFloat lw = std::max(1.15, h * 0.085);
    CGContextSaveGState(ctx);
    CGContextSetLineWidth(ctx, lw);
    CGContextSetLineCap(ctx, kCGLineCapRound);
    CGContextSetLineJoin(ctx, kCGLineJoinRound);
    setStroke(ctx, c.ink);
    switch (b) {
    case PadButton::Cross: {
        const CGFloat s = R * 0.36;
        CGContextMoveToPoint(ctx, cx - s, cy - s);
        CGContextAddLineToPoint(ctx, cx + s, cy + s);
        CGContextMoveToPoint(ctx, cx + s, cy - s);
        CGContextAddLineToPoint(ctx, cx - s, cy + s);
        CGContextStrokePath(ctx);
        break;
    }
    case PadButton::Circle: {
        const CGFloat s = R * 0.42;
        CGContextStrokeEllipseInRect(ctx, CGRectMake(cx - s, cy - s, 2 * s, 2 * s));
        break;
    }
    case PadButton::Square: {
        const CGFloat s = R * 0.74, k = s * 0.16;
        CGPathRef sq = CGPathCreateWithRoundedRect(CGRectMake(cx - s / 2, cy - s / 2, s, s), k, k, nullptr);
        CGContextAddPath(ctx, sq);
        CGContextStrokePath(ctx);
        CGPathRelease(sq);
        break;
    }
    case PadButton::Triangle: {
        // Equilateral, circumradius 0.5 R; the centroid sits a hair below centre so the glyph
        // reads optically centred (a triangle's mass is in its base).
        const CGFloat cr = R * 0.50, oy = cy + R * 0.05;
        const CGFloat dx = cr * std::sqrt(3.0) / 2;
        CGContextMoveToPoint(ctx, cx, oy - cr);
        CGContextAddLineToPoint(ctx, cx + dx, oy + cr / 2);
        CGContextAddLineToPoint(ctx, cx - dx, oy + cr / 2);
        CGContextClosePath(ctx);
        CGContextStrokePath(ctx);
        break;
    }
    default:
        break;
    }
    CGContextRestoreGState(ctx);
}

void bumper(CGContextRef ctx, PadButton b, CGRect r, const PadIconColors &c)
{
    const CGFloat h = r.size.height, w = r.size.width;
    const CGFloat rimW = std::clamp(h * 0.075, 1.0, 1.5), in = rimW / 2 + 0.25;
    const CGFloat x0 = r.origin.x + in, x1 = CGRectGetMaxX(r) - in;
    const CGFloat yt = r.origin.y + in + h * 0.04;  // arch apex
    const CGFloat ys = yt + h * 0.20;               // where the arch meets the sides
    const CGFloat yb = CGRectGetMaxY(r) - in;
    const CGFloat rb = h * 0.24, rs = h * 0.14;
    CGMutablePathRef p = CGPathCreateMutable();
    CGPathMoveToPoint(p, nullptr, x0, ys + rs);
    CGPathAddLineToPoint(p, nullptr, x0, yb - rb);
    CGPathAddArcToPoint(p, nullptr, x0, yb, x0 + rb, yb, rb);
    CGPathAddLineToPoint(p, nullptr, x1 - rb, yb);
    CGPathAddArcToPoint(p, nullptr, x1, yb, x1, yb - rb, rb);
    CGPathAddLineToPoint(p, nullptr, x1, ys + rs);
    CGPathAddQuadCurveToPoint(p, nullptr, x1, ys, x1 - rs, ys - rs * 0.25);
    CGPathAddCurveToPoint(p, nullptr, x1 - w * 0.28, yt, x0 + w * 0.28, yt, x0 + rs, ys - rs * 0.25);
    CGPathAddQuadCurveToPoint(p, nullptr, x0, ys, x0, ys + rs);
    CGPathCloseSubpath(p);
    keycap(ctx, p, c, rimW);
    topHighlight(ctx, p, CGRectMake(x0, yt, x1 - x0, yb - yt), h * 0.32, c);
    CGPathRelease(p);
    label(ctx, b == PadButton::L1 ? @"L1" : @"R1", h * 0.42, true, CGRectGetMidX(r), (ys + yb) / 2 - h * 0.02, c.ink);
}

void trigger(CGContextRef ctx, PadButton b, CGRect r, const PadIconColors &c)
{
    const CGFloat h = r.size.height;
    const CGFloat rimW = std::clamp(h * 0.075, 1.0, 1.5), in = rimW / 2 + 0.25;
    const CGFloat x0 = r.origin.x + in, x1 = CGRectGetMaxX(r) - in, cx = CGRectGetMidX(r);
    const CGFloat yt = r.origin.y + in, yb = CGRectGetMaxY(r) - in;
    const CGFloat ih = yb - yt, iw = x1 - x0;
    // Narrow curved top lip (72% wide), flanks flaring out to the full width, a concave bottom lip.
    const CGFloat tl = cx - iw * 0.36, tr = cx + iw * 0.36, shoulder = yt + ih * 0.34;
    const CGFloat corner = ih * 0.14, sag = ih * 0.16;
    CGMutablePathRef p = CGPathCreateMutable();
    CGPathMoveToPoint(p, nullptr, tl, shoulder);
    CGPathAddCurveToPoint(p, nullptr, tl, yt + ih * 0.08, cx - iw * 0.22, yt, cx, yt);
    CGPathAddCurveToPoint(p, nullptr, cx + iw * 0.22, yt, tr, yt + ih * 0.08, tr, shoulder);
    CGPathAddCurveToPoint(p, nullptr, tr + iw * 0.02, yt + ih * 0.62, x1, yb - ih * 0.36, x1, yb - corner);
    CGPathAddQuadCurveToPoint(p, nullptr, x1, yb, x1 - corner * 1.2, yb);
    CGPathAddCurveToPoint(p, nullptr, cx + iw * 0.20, yb - sag, cx - iw * 0.20, yb - sag, x0 + corner * 1.2, yb);
    CGPathAddQuadCurveToPoint(p, nullptr, x0, yb, x0, yb - corner);
    CGPathAddCurveToPoint(p, nullptr, x0, yb - ih * 0.36, tl - iw * 0.02, yt + ih * 0.62, tl, shoulder);
    CGPathCloseSubpath(p);
    keycap(ctx, p, c, rimW);
    topHighlight(ctx, p, CGRectMake(x0, yt, iw, ih), ih * 0.30, c);
    CGPathRelease(p);
    label(ctx, b == PadButton::L2 ? @"L2" : @"R2", h * 0.38, true, cx, yt + ih * 0.50, c.ink);
}

void stick(CGContextRef ctx, PadButton b, CGRect r, const PadIconColors &c)
{
    const CGFloat h = std::min(r.size.width, r.size.height);
    const CGFloat rimW = std::clamp(h * 0.07, 1.0, 1.4);
    const CGFloat cx = CGRectGetMidX(r), cy = CGRectGetMidY(r), R = h / 2 - rimW / 2 - 0.25;
    // Outer base ring: the well the stick sits in.
    CGPathRef base = CGPathCreateWithEllipseInRect(CGRectMake(cx - R, cy - R, 2 * R, 2 * R), nullptr);
    PadIconColors well = c;
    well.fill = mix(c.fill, rgb(0x000000), 0.35);
    keycap(ctx, base, well, rimW);
    CGPathRelease(base);
    // The domed cap, lighter, with its own thinner rim and a top highlight.
    const CGFloat k = R * 0.70;
    CGPathRef cap = CGPathCreateWithEllipseInRect(CGRectMake(cx - k, cy - k, 2 * k, 2 * k), nullptr);
    PadIconColors dome = c;
    dome.fill = mix(c.fill, c.ink, 0.07);
    keycap(ctx, cap, dome, std::max(0.9, rimW * 0.8));
    topHighlight(ctx, cap, CGRectMake(cx - k, cy - k, 2 * k, 2 * k), k * 0.7, c);
    CGPathRelease(cap);
    label(ctx, b == PadButton::L3 ? @"L" : @"R", h * 0.30, true, cx, cy - k * 0.26, c.ink);
    // Press chevron under the letter.
    const CGFloat chw = k * 0.24, chy = cy + k * 0.56;
    CGContextSaveGState(ctx);
    CGContextSetLineWidth(ctx, std::max(0.9, h * 0.06));
    CGContextSetLineCap(ctx, kCGLineCapRound);
    CGContextSetLineJoin(ctx, kCGLineJoinRound);
    setStroke(ctx, withAlpha(c.ink, 0.85));
    CGContextMoveToPoint(ctx, cx - chw, chy - chw * 0.45);
    CGContextAddLineToPoint(ctx, cx, chy + chw * 0.15);
    CGContextAddLineToPoint(ctx, cx + chw, chy - chw * 0.45);
    CGContextStrokePath(ctx);
    CGContextRestoreGState(ctx);
}

void options(CGContextRef ctx, CGRect r, const PadIconColors &c)
{
    const CGFloat h = r.size.height;
    const CGFloat rimW = std::clamp(h * 0.075, 1.0, 1.5), in = rimW / 2 + 0.25;
    const CGRect body = CGRectInset(r, in, in);
    const CGFloat rad = body.size.width / 2;
    CGPathRef p = CGPathCreateWithRoundedRect(body, rad, rad, nullptr);
    keycap(ctx, p, c, rimW);
    topHighlight(ctx, p, body, body.size.height * 0.3, c);
    CGPathRelease(p);
    const CGFloat cx = CGRectGetMidX(body), cy = CGRectGetMidY(body), bw = body.size.width * 0.26;
    const CGFloat gap = h * 0.15;
    CGContextSaveGState(ctx);
    CGContextSetLineWidth(ctx, std::max(1.0, h * 0.075));
    CGContextSetLineCap(ctx, kCGLineCapRound);
    setStroke(ctx, c.ink);
    for (int i = -1; i <= 1; ++i) {
        CGContextMoveToPoint(ctx, cx - bw, cy + i * gap);
        CGContextAddLineToPoint(ctx, cx + bw, cy + i * gap);
    }
    CGContextStrokePath(ctx);
    CGContextRestoreGState(ctx);
}

void touchpad(CGContextRef ctx, CGRect r, const PadIconColors &c)
{
    const CGFloat h = r.size.height;
    const CGFloat rimW = std::clamp(h * 0.075, 1.0, 1.5), in = rimW / 2 + 0.25;
    const CGFloat x0 = r.origin.x + in, x1 = CGRectGetMaxX(r) - in;
    const CGFloat yt = r.origin.y + in, yb = CGRectGetMaxY(r) - in;
    const CGFloat taper = (x1 - x0) * 0.07, k = h * 0.18;
    // Slightly trapezoidal: full width on top, 86% at the bottom, rounded corners.
    const CGPoint tl{x0, yt}, tr{x1, yt}, br{x1 - taper, yb}, bl{x0 + taper, yb};
    CGMutablePathRef p = CGPathCreateMutable();
    CGPathMoveToPoint(p, nullptr, (tl.x + tr.x) / 2, yt);
    CGPathAddArcToPoint(p, nullptr, tr.x, tr.y, br.x, br.y, k);
    CGPathAddArcToPoint(p, nullptr, br.x, br.y, bl.x, bl.y, k);
    CGPathAddArcToPoint(p, nullptr, bl.x, bl.y, tl.x, tl.y, k);
    CGPathAddArcToPoint(p, nullptr, tl.x, tl.y, tr.x, tr.y, k);
    CGPathCloseSubpath(p);
    keycap(ctx, p, c, rimW);
    topHighlight(ctx, p, CGRectMake(x0, yt, x1 - x0, yb - yt), h * 0.3, c);
    CGPathRelease(p);
    const CGFloat cx = (x0 + x1) / 2, cy = (yt + yb) / 2 + h * 0.02, dr = std::max(0.75, h * 0.05);
    const CGFloat step = (x1 - x0) * 0.13;
    setFill(ctx, withAlpha(c.ink, 0.85));
    for (int i = -2; i <= 2; ++i)
        CGContextFillEllipseInRect(ctx, CGRectMake(cx + i * step - dr, cy - dr, 2 * dr, 2 * dr));
}

}  // namespace

PadIconColors padIconColors(bool active, bool lapisBlue)
{
    PadIconColors c;
    if (!active) {
        c.fill = rgb(0x14120F);
        c.rim = rgb(0x6E665E);
        c.ink = rgb(0xE8D5B5);
        return c;
    }
    const PadRgba accent = lapisBlue ? rgb(0x4F8CFF) : rgb(0xE09F3E);
    c.fill = lapisBlue ? rgb(0x0B1324) : rgb(0x1A1309);
    c.rim = accent;
    c.ink = accent;
    return c;
}

QString padButtonName(PadButton b)
{
    switch (b) {
    case PadButton::Cross: return QStringLiteral("Cross");
    case PadButton::Circle: return QStringLiteral("Circle");
    case PadButton::Square: return QStringLiteral("Square");
    case PadButton::Triangle: return QStringLiteral("Triangle");
    case PadButton::L1: return QStringLiteral("L1");
    case PadButton::R1: return QStringLiteral("R1");
    case PadButton::L2: return QStringLiteral("L2");
    case PadButton::R2: return QStringLiteral("R2");
    case PadButton::L3: return QStringLiteral("L3");
    case PadButton::R3: return QStringLiteral("R3");
    case PadButton::Options: return QStringLiteral("Options");
    case PadButton::Touchpad: return QStringLiteral("Touchpad");
    }
    return {};
}

CGSize padIconSize(PadButton b, CGFloat height)
{
    switch (b) {
    case PadButton::L1:
    case PadButton::R1:
    case PadButton::L2:
    case PadButton::R2:
    case PadButton::Touchpad:
        return CGSizeMake(std::round(height * 1.45 * 2) / 2, height);
    case PadButton::Options:
        return CGSizeMake(std::round(height * 0.66 * 2) / 2, height);
    default:
        return CGSizeMake(height, height);
    }
}

void drawPadIcon(CGContextRef ctx, PadButton b, CGRect rect, const PadIconColors &colors)
{
    if (!ctx || CGRectIsEmpty(rect) || !std::isfinite(rect.size.width) || !std::isfinite(rect.size.height))
        return;
    CGContextSaveGState(ctx);
    CGContextSetShouldAntialias(ctx, true);
    switch (b) {
    case PadButton::Cross:
    case PadButton::Circle:
    case PadButton::Square:
    case PadButton::Triangle: faceButton(ctx, b, rect, colors); break;
    case PadButton::L1:
    case PadButton::R1: bumper(ctx, b, rect, colors); break;
    case PadButton::L2:
    case PadButton::R2: trigger(ctx, b, rect, colors); break;
    case PadButton::L3:
    case PadButton::R3: stick(ctx, b, rect, colors); break;
    case PadButton::Options: options(ctx, rect, colors); break;
    case PadButton::Touchpad: touchpad(ctx, rect, colors); break;
    }
    CGContextRestoreGState(ctx);
}

QImage renderPadIcon(PadButton b, double height, double scale, bool active, bool lapisBlue)
{
    if (!std::isfinite(height) || !std::isfinite(scale) || height <= 0 || scale <= 0)
        return {};
    const CGSize sz = padIconSize(b, height);
    const int w = int(std::ceil(sz.width * scale)), h = int(std::ceil(sz.height * scale));
    if (w <= 0 || h <= 0 || w > 8192 || h > 8192)
        return {};
    QImage img(w, h, QImage::Format_RGBA8888_Premultiplied);
    img.fill(Qt::transparent);
    @autoreleasepool {
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = CGBitmapContextCreate(img.bits(), size_t(w), size_t(h), 8, size_t(img.bytesPerLine()), cs,
                                                 uint32_t(kCGImageAlphaPremultipliedLast) | uint32_t(kCGBitmapByteOrder32Big));
        CGColorSpaceRelease(cs);
        if (!ctx)
            return {};
        CGContextTranslateCTM(ctx, 0, h);
        CGContextScaleCTM(ctx, scale, -scale);
        drawPadIcon(ctx, b, CGRectMake(0, 0, sz.width, sz.height), padIconColors(active, lapisBlue));
        CGContextRelease(ctx);
    }
    img.setDevicePixelRatio(scale);
    return img;
}

}  // namespace studio::cua
