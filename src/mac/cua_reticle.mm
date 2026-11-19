#include "mac/cua_reticle.hpp"
#include "mac/cua.hpp"
#import <AppKit/AppKit.h>
#include <QDir>
#include <QFileInfo>
#include <QPainter>
#include <algorithm>
#include <cmath>
#include <cstdio>

// ---- Pure geometry -----------------------------------------------------------------------------

namespace studio::cua {

bool reticleRectValid(const QRectF &r)
{
    const double v[] = {r.x(), r.y(), r.width(), r.height()};
    for (double x : v)
        if (!std::isfinite(x) || std::abs(x) > 1e6)
            return false;
    return r.width() > 0 && r.height() > 0 && r.width() <= 16384 && r.height() <= 16384;
}

QPointF glideControlPoint(QPointF from, QPointF to)
{
    const QPointF d = to - from;
    const double len = std::hypot(d.x(), d.y());
    const QPointF mid = (from + to) / 2;
    if (!(len > 1e-9) || !std::isfinite(len))
        return mid;
    QPointF n(-d.y() / len, d.x() / len);
    // Bow upward on screen (y grows downward); a purely vertical move bows left.
    if (n.y() > 1e-9 || (std::abs(n.y()) <= 1e-9 && n.x() > 0))
        n = -n;
    return mid + n * std::min(0.18 * len, 64.0);
}

QPointF quadBezier(QPointF a, QPointF c, QPointF b, double t)
{
    const double u = 1 - t;
    return a * (u * u) + c * (2 * u * t) + b * (t * t);
}

QRectF reticlePillRect(const QRectF &anchor, QSizeF pill, const QRectF &display)
{
    const double gap = 10, margin = 8, w = pill.width(), h = pill.height();
    const double minY = display.top() + margin, maxY = display.bottom() - margin - h;
    double y = anchor.bottom() + gap;
    if (y > maxY) {
        y = anchor.top() - gap - h;
        if (y < minY)
            y = std::max(minY, std::min(maxY, anchor.bottom() - h - margin));
    }
    const double maxX = display.right() - margin - w;
    const double x = std::max(display.left() + margin, std::min(maxX, anchor.left()));
    return QRectF(x, y, w, h);
}

} // namespace studio::cua

// ---- Scene: everything the overlay draws, as a pure function of time ---------------------------

namespace studio::cua::detail {

constexpr double kGlide = 0.14, kTailDecay = 0.20, kRipple = 0.28, kRippleStagger = 0.07;
constexpr double kFocusIn = 0.18, kIdle = 1.8, kFade = 0.45;
constexpr double kNever = -1e9;

double clamp01(double t) { return std::max(0.0, std::min(1.0, t)); }
double easeInOutCubic(double t) { return t < 0.5 ? 4 * t * t * t : 1 - std::pow(-2 * t + 2, 3) / 2; }
double easeOutCubic(double t) { return 1 - std::pow(1 - t, 3); }

struct Scene {
    bool offscreen = false;
    double manualNow = 1000;
    QRectF display;  // global points the overlay covers
    bool visible = false;
    bool hasTip = false;
    QPointF from, ctrl, to;
    double moveStart = kNever;
    QRectF focus;
    double focusStart = kNever;
    QPointF rippleAt;
    double rippleStart = kNever;
    QString label;
    double lastActivity = kNever;

    double now() const { return offscreen ? manualNow : [[NSProcessInfo processInfo] systemUptime]; }
    double glideP(double t) const { return easeInOutCubic(clamp01((t - moveStart) / kGlide)); }
    QPointF tip(double t) const { return quadBezier(from, ctrl, to, glideP(t)); }
    bool gliding(double t) const { return t - moveStart < kGlide; }
    double tailAmount(double t) const {
        if (std::hypot(to.x() - from.x(), to.y() - from.y()) < 3)
            return 0;
        const double since = t - moveStart;
        return since < kGlide ? 1 : clamp01(1 - (since - kGlide) / kTailDecay);
    }
    bool ripplePending(double t) const { return t - rippleStart < kRipple + kRippleStagger; }
    double chromeAlpha(double t) const { return clamp01(1 - (t - lastActivity - kIdle) / kFade); }
    bool focusShown(double t) const { return !focus.isEmpty() && chromeAlpha(t) > 0; }
    // Moving pixels this frame (the overlay is idle between landing and the idle fade).
    bool dynamic(double t) const {
        const double idle = t - lastActivity;
        return gliding(t) || tailAmount(t) > 0 || ripplePending(t) || t - focusStart < kFocusIn
               || (idle >= kIdle && idle <= kIdle + kFade);
    }
    bool animating(double t) const { return visible && (dynamic(t) || t - lastActivity < kIdle + kFade); }

    static NSDictionary *labelAttrs() {
        NSMutableParagraphStyle *style = [[[NSMutableParagraphStyle alloc] init] autorelease];
        style.lineBreakMode = NSLineBreakByTruncatingTail;
        return @{NSFontAttributeName: [NSFont systemFontOfSize:12 weight:NSFontWeightMedium],
                 NSForegroundColorAttributeName: [NSColor colorWithWhite:1 alpha:0.92],
                 NSParagraphStyleAttributeName: style};
    }
    QRectF pillRect(double t) const {
        if (label.isEmpty() || chromeAlpha(t) <= 0)
            return {};
        const double textW = std::ceil([label.toNSString() sizeWithAttributes:labelAttrs()].width);
        const QSizeF size(std::min(23 + textW + 12, 340.0), 24);
        QRectF anchor;
        if (focusShown(t))
            anchor = focus.adjusted(-4, -4, 4, 4);
        else if (hasTip) {
            const QPointF p = tip(t);
            anchor = QRectF(p.x() - 2, p.y() - 2, 18, 26);
        } else {
            return {};
        }
        return reticlePillRect(anchor, size, display);
    }
    // Everything that may have pixels this frame, global points.
    QRectF bounds(double t) const {
        if (!visible)
            return {};
        QRectF b;
        if (hasTip) {
            const QPointF p = tip(t);
            b |= QRectF(p.x() - 24, p.y() - 24, 64, 70);
            if (tailAmount(t) > 0) {
                const double l = std::min({from.x(), ctrl.x(), to.x()}), r = std::max({from.x(), ctrl.x(), to.x()});
                const double u = std::min({from.y(), ctrl.y(), to.y()}), d = std::max({from.y(), ctrl.y(), to.y()});
                b |= QRectF(l, u, r - l, d - u).adjusted(-14, -14, 14, 14);
            }
        }
        if (focusShown(t))
            b |= focus.adjusted(-26, -26, 26, 26);
        if (ripplePending(t))
            b |= QRectF(rippleAt.x() - 58, rippleAt.y() - 58, 116, 116);
        const QRectF pill = pillRect(t);
        if (!pill.isEmpty())
            b |= pill.adjusted(-18, -18, 18, 18);
        return b;
    }
};

} // namespace studio::cua::detail

using studio::cua::detail::Scene;

// ---- AppKit ------------------------------------------------------------------------------------

@interface LapisCuaPanel : NSPanel
@end
@implementation LapisCuaPanel
- (BOOL)canBecomeKeyWindow { return NO; }
- (BOOL)canBecomeMainWindow { return NO; }
@end

namespace {
CGColorRef rgba(int r, int g, int b, double a)
{
    return [NSColor colorWithSRGBRed:r / 255.0 green:g / 255.0 blue:b / 255.0 alpha:a].CGColor;
}
CGColorRef sapphire(double a) { return rgba(0x78, 0xB2, 0xFF, a); }
}

@interface LapisCuaReticleView : NSView {
@public
    const Scene *_scene;
    double _shadowScale;  // 1 in a window; the pixel scale in an offscreen bitmap
}
- (void)paintScene:(CGContextRef)ctx now:(double)now;
@end

@implementation LapisCuaReticleView
- (BOOL)isFlipped { return YES; }
- (BOOL)isOpaque { return NO; }
- (NSView *)hitTest:(NSPoint)point { (void)point; return nil; }

// Shadow offsets and blur live in device space: scale them so a 2x snapshot matches the screen.
- (void)shadow:(CGContextRef)ctx color:(CGColorRef)color blur:(double)blur dy:(double)dy
{
    CGContextSetShadowWithColor(ctx, CGSizeMake(0, dy * _shadowScale), blur * _shadowScale, color);
}

- (void)drawRect:(NSRect)dirtyRect
{
    CGContextRef ctx = [NSGraphicsContext currentContext].CGContext;
    CGContextClearRect(ctx, dirtyRect);
    if (_scene)
        [self paintScene:ctx now:_scene->now()];
}

static CGPathRef arrowPath()
{
    // The macOS arrow silhouette, tip at the origin, y down, ~14 x 22 pt.
    static const CGPoint pts[] = {{0, 0}, {0, 17.2}, {4.1, 13.3}, {6.9, 19.6}, {9.5, 18.5}, {6.8, 12.3}, {12.3, 12.3}};
    CGMutablePathRef p = CGPathCreateMutable();
    const CGAffineTransform s = CGAffineTransformMakeScale(1.12, 1.12);
    CGPathAddLines(p, &s, pts, sizeof(pts) / sizeof(pts[0]));
    CGPathCloseSubpath(p);
    return p;
}

- (void)paintFocus:(CGContextRef)ctx now:(double)now
{
    using namespace studio::cua::detail;
    const Scene &s = *_scene;
    if (!s.focusShown(now))
        return;
    const double in = easeOutCubic(clamp01((now - s.focusStart) / kFocusIn));
    const double grow = 4 + 8 * (1 - in);
    const QRectF r = s.focus.adjusted(-grow, -grow, grow, grow);
    CGContextSaveGState(ctx);
    CGContextSetAlpha(ctx, in * s.chromeAlpha(now));
    CGContextBeginTransparencyLayer(ctx, nullptr);
    const double corner = std::min({8.0, r.width() / 2, r.height() / 2});  // CG rejects radius > half a side
    CGPathRef path = CGPathCreateWithRoundedRect(r.toCGRect(), corner, corner, nullptr);
    CGContextAddPath(ctx, path);
    CGContextSetFillColorWithColor(ctx, sapphire(0.05));
    CGContextFillPath(ctx);
    [self shadow:ctx color:sapphire(0.6) blur:5 dy:0];
    CGContextAddPath(ctx, path);
    CGContextSetLineWidth(ctx, 1.5);
    CGContextSetStrokeColorWithColor(ctx, sapphire(0.95));
    CGContextStrokePath(ctx);
    CGPathRelease(path);
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);
}

- (void)paintTail:(CGContextRef)ctx now:(double)now
{
    using namespace studio::cua::detail;
    const Scene &s = *_scene;
    const double amount = s.tailAmount(now);
    if (!s.hasTip || amount <= 0)
        return;
    const double head = s.glideP(now), u0 = std::max(0.0, head - 0.42 * amount);
    if (head - u0 < 1e-4)
        return;
    constexpr int kSegments = 7;
    CGContextSaveGState(ctx);
    // Butt caps: round caps double the alpha where segments meet and read as beads.
    CGContextSetLineCap(ctx, kCGLineCapButt);
    [self shadow:ctx color:sapphire(0.45 * amount) blur:6 dy:0];
    for (int i = 0; i < kSegments; ++i) {
        const double a = u0 + (head - u0) * i / kSegments, b = u0 + (head - u0) * (i + 1) / kSegments;
        const double f = double(i + 1) / kSegments;
        const QPointF pa = studio::cua::quadBezier(s.from, s.ctrl, s.to, a);
        const QPointF pb = studio::cua::quadBezier(s.from, s.ctrl, s.to, b);
        CGContextSetLineWidth(ctx, 0.6 + 2.8 * f);
        CGContextSetStrokeColorWithColor(ctx, sapphire((0.03 + 0.34 * std::pow(f, 1.5)) * amount));
        CGContextMoveToPoint(ctx, pa.x(), pa.y());
        CGContextAddLineToPoint(ctx, pb.x(), pb.y());
        CGContextStrokePath(ctx);
    }
    CGContextRestoreGState(ctx);
}

- (void)paintRipple:(CGContextRef)ctx now:(double)now
{
    using namespace studio::cua::detail;
    const Scene &s = *_scene;
    const double since = now - s.rippleStart;
    if (since < 0 || !s.ripplePending(now))
        return;
    CGContextSaveGState(ctx);
    for (int k = 0; k < 2; ++k) {
        const double t = (since - k * kRippleStagger) / kRipple;
        if (t < 0 || t > 1)
            continue;
        const double radius = 8 + 36 * easeOutCubic(t);
        const double alpha = (k == 0 ? 0.8 : 0.5) * std::pow(1 - t, 1.6);
        [self shadow:ctx color:sapphire(0.5 * alpha) blur:6 dy:0];
        CGContextSetLineWidth(ctx, 0.6 + 1.2 * (1 - t));
        CGContextSetStrokeColorWithColor(ctx, sapphire(alpha));
        CGContextStrokeEllipseInRect(ctx, CGRectMake(s.rippleAt.x() - radius, s.rippleAt.y() - radius, 2 * radius, 2 * radius));
    }
    const double flash = clamp01(1 - since / (0.6 * kRipple));
    if (flash > 0) {
        CGContextSetShadowWithColor(ctx, CGSizeZero, 0, nullptr);
        CGContextSetFillColorWithColor(ctx, sapphire(0.35 * flash));
        CGContextFillEllipseInRect(ctx, CGRectMake(s.rippleAt.x() - 3.5, s.rippleAt.y() - 3.5, 7, 7));
    }
    CGContextRestoreGState(ctx);
}

- (void)paintPill:(CGContextRef)ctx now:(double)now
{
    const Scene &s = *_scene;
    const QRectF pr = s.pillRect(now);
    if (pr.isEmpty())
        return;
    const CGRect r = pr.toCGRect();
    CGContextSaveGState(ctx);
    CGContextSetAlpha(ctx, s.chromeAlpha(now));
    CGContextBeginTransparencyLayer(ctx, nullptr);
    CGPathRef body = CGPathCreateWithRoundedRect(r, r.size.height / 2, r.size.height / 2, nullptr);
    // Frosted dark glass: soft drop shadow, body, a faint top sheen, and a 1 pt white-15% edge.
    CGContextSaveGState(ctx);
    [self shadow:ctx color:rgba(0, 0, 0, 0.45) blur:10 dy:-3];
    CGContextAddPath(ctx, body);
    CGContextSetFillColorWithColor(ctx, rgba(0x18, 0x1A, 0x20, 0.86));
    CGContextFillPath(ctx);
    CGContextRestoreGState(ctx);
    CGContextSaveGState(ctx);
    CGContextAddPath(ctx, body);
    CGContextClip(ctx);
    NSGradient *sheen = [[[NSGradient alloc] initWithStartingColor:[NSColor colorWithWhite:1 alpha:0.07]
                                                       endingColor:[NSColor colorWithWhite:1 alpha:0]] autorelease];
    [sheen drawFromPoint:NSMakePoint(r.origin.x, r.origin.y) toPoint:NSMakePoint(r.origin.x, r.origin.y + r.size.height * 0.6)
                 options:0];
    CGContextRestoreGState(ctx);
    CGPathRef edge = CGPathCreateWithRoundedRect(CGRectInset(r, 0.5, 0.5), r.size.height / 2 - 0.5, r.size.height / 2 - 0.5, nullptr);
    CGContextAddPath(ctx, edge);
    CGContextSetLineWidth(ctx, 1);
    CGContextSetStrokeColorWithColor(ctx, rgba(255, 255, 255, 0.15));
    CGContextStrokePath(ctx);
    CGPathRelease(edge);
    CGPathRelease(body);
    // Glowing sapphire dot, then the label in SF Pro 12 pt.
    CGContextSaveGState(ctx);
    [self shadow:ctx color:sapphire(0.85) blur:6 dy:0];
    CGContextSetFillColorWithColor(ctx, sapphire(1));
    CGContextFillEllipseInRect(ctx, CGRectMake(r.origin.x + 9, r.origin.y + r.size.height / 2 - 3, 6, 6));
    CGContextRestoreGState(ctx);
    NSDictionary *attrs = Scene::labelAttrs();
    const double lineH = std::ceil([NSFont systemFontOfSize:12 weight:NSFontWeightMedium].boundingRectForFont.size.height);
    const double textH = std::min(lineH, 16.0);
    [s.label.toNSString() drawInRect:NSMakeRect(r.origin.x + 23, r.origin.y + (r.size.height - textH) / 2,
                                                r.size.width - 23 - 11, textH)
                      withAttributes:attrs];
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);
}

- (void)paintPointer:(CGContextRef)ctx now:(double)now
{
    const Scene &s = *_scene;
    if (!s.hasTip)
        return;
    const QPointF tip = s.tip(now);
    CGPathRef arrow = arrowPath();
    CGContextSaveGState(ctx);
    CGContextTranslateCTM(ctx, tip.x(), tip.y());
    CGContextSetLineJoin(ctx, kCGLineJoinRound);
    CGColorRef obsidian = rgba(0x12, 0x14, 0x1A, 0.96);
    // Faint sapphire ambient glow, then the soft Gaussian drop shadow (black 60%, 4 pt, -2 pt Y).
    CGContextSaveGState(ctx);
    [self shadow:ctx color:sapphire(0.32) blur:9 dy:0];
    CGContextAddPath(ctx, arrow);
    CGContextSetFillColorWithColor(ctx, obsidian);
    CGContextFillPath(ctx);
    CGContextRestoreGState(ctx);
    CGContextSaveGState(ctx);
    [self shadow:ctx color:rgba(0, 0, 0, 0.6) blur:4 dy:-2];
    CGContextAddPath(ctx, arrow);
    CGContextSetFillColorWithColor(ctx, obsidian);
    CGContextFillPath(ctx);
    CGContextRestoreGState(ctx);
    // Matte depth: a barely-there vertical gradient inside the silhouette.
    CGContextSaveGState(ctx);
    CGContextAddPath(ctx, arrow);
    CGContextClip(ctx);
    NSGradient *depth = [[[NSGradient alloc] initWithStartingColor:[NSColor colorWithSRGBRed:0x1D / 255.0 green:0x21 / 255.0 blue:0x2B / 255.0 alpha:0.96]
                                                       endingColor:[NSColor colorWithSRGBRed:0x0F / 255.0 green:0x11 / 255.0 blue:0x16 / 255.0 alpha:0.96]] autorelease];
    [depth drawFromPoint:NSMakePoint(0, 0) toPoint:NSMakePoint(0, 22) options:0];
    CGContextRestoreGState(ctx);
    // Razor-thin sapphire-ice rim.
    CGContextAddPath(ctx, arrow);
    CGContextSetLineWidth(ctx, 1.3);
    CGContextSetStrokeColorWithColor(ctx, sapphire(0.92));
    CGContextStrokePath(ctx);
    CGContextRestoreGState(ctx);
    CGPathRelease(arrow);
}

- (void)paintScene:(CGContextRef)ctx now:(double)now
{
    if (!_scene || !_scene->visible)
        return;
    CGContextSaveGState(ctx);
    CGContextTranslateCTM(ctx, -_scene->display.x(), -_scene->display.y());  // draw in global points
    [self paintFocus:ctx now:now];
    [self paintTail:ctx now:now];
    [self paintRipple:ctx now:now];
    [self paintPill:ctx now:now];
    [self paintPointer:ctx now:now];
    CGContextRestoreGState(ctx);
}
@end

// ---- CuaReticle --------------------------------------------------------------------------------

namespace studio::cua {
namespace {
void requireMainThread() { if (![NSThread isMainThread]) qFatal("CuaReticle requires the AppKit main thread"); }
bool finite(QPointF p) { return std::isfinite(p.x()) && std::isfinite(p.y()) && std::abs(p.x()) < 1e6 && std::abs(p.y()) < 1e6; }
}

struct CuaReticle::Impl {
    Scene scene;
    LapisCuaPanel *panel = nil;
    LapisCuaReticleView *view = nil;
    NSTimer *timer = nil;
    QRectF lastBounds;

    // The display (global points) under `p`, or an empty rect.
    QRectF displayAt(QPointF p) const {
        if (scene.offscreen)
            return scene.display.contains(p) ? scene.display : QRectF();
        for (const auto &d : listDisplays())
            if (d.globalPoints.contains(p))
                return d.globalPoints;
        return {};
    }
    void place(const QRectF &display) {
        if (display == scene.display)
            return;
        scene.display = display;
        lastBounds = {};
        if (scene.offscreen || !panel)
            return;
        QRectF primary;
        for (const auto &d : listDisplays())
            if (d.primary)
                primary = d.globalPoints;
        if (!primary.isValid())
            return;
        const QPointF bl = globalPointToAppKit(QPointF(display.left(), display.bottom()), primary);
        [panel setFrame:NSMakeRect(bl.x(), bl.y(), display.width(), display.height()) display:NO];
        view.frame = NSMakeRect(0, 0, display.width(), display.height());
        [view setNeedsDisplay:YES];
    }
    void startGlide(QPointF to, double now) {
        if (!scene.visible || !scene.hasTip) {
            scene.from = scene.ctrl = scene.to = to;
            scene.moveStart = kNeverStart;
        } else {
            const QPointF cur = scene.tip(now);
            scene.from = cur;
            scene.to = to;
            scene.ctrl = glideControlPoint(cur, to);
            scene.moveStart = std::hypot(to.x() - cur.x(), to.y() - cur.y()) < 2 ? kNeverStart : now;
            if (scene.moveStart == kNeverStart)
                scene.from = scene.ctrl = to;
        }
        scene.hasTip = true;
    }
    void show(double now) {
        scene.lastActivity = now;
        const bool wasVisible = scene.visible;
        scene.visible = true;
        if (scene.offscreen)
            return;
        if (!wasVisible)
            [panel orderFrontRegardless];
        invalidate(now, true);
        if (!timer) {
            Impl *owner = this;  // the timer is invalidated in ~CuaReticle, before Impl dies
            timer = [[NSTimer timerWithTimeInterval:1.0 / 120 repeats:YES block:^(NSTimer *t) { (void)t; owner->tick(); }] retain];
            timer.tolerance = 0;
            [[NSRunLoop mainRunLoop] addTimer:timer forMode:NSRunLoopCommonModes];
        }
    }
    void invalidate(double now, bool force) {
        if (!view)
            return;
        const bool dyn = scene.dynamic(now);
        const QRectF b = scene.bounds(now);
        if (force || dyn || b != lastBounds) {
            const QRectF dirty = (b | lastBounds).translated(-scene.display.topLeft());
            if (!dirty.isEmpty())
                [view setNeedsDisplayInRect:dirty.toCGRect()];
        }
        lastBounds = b;
    }
    void tick() {
        const double now = scene.now();
        invalidate(now, false);
        if (!scene.animating(now))
            stopTimer();
    }
    void stopTimer() {
        [timer invalidate];
        [timer release];
        timer = nil;
    }
    static constexpr double kNeverStart = detail::kNever;
};

CuaReticle::CuaReticle() : CuaReticle(Mode::Visible) {}

CuaReticle::CuaReticle(Mode mode, const QRectF &virtualDisplay) : impl(std::make_unique<Impl>())
{
    requireMainThread();
    impl->scene.offscreen = mode == Mode::Offscreen;
    @autoreleasepool {
        impl->view = [[LapisCuaReticleView alloc] initWithFrame:NSMakeRect(0, 0, 240, 72)];
        impl->view->_scene = &impl->scene;
        impl->view->_shadowScale = 1;
        if (impl->scene.offscreen) {
            impl->scene.display = virtualDisplay.isValid() ? virtualDisplay : QRectF(0, 0, 1440, 900);
            impl->view.frame = NSMakeRect(0, 0, impl->scene.display.width(), impl->scene.display.height());
            return;
        }
        impl->panel = [[LapisCuaPanel alloc] initWithContentRect:NSMakeRect(0, 0, 240, 72)
            styleMask:NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
            backing:NSBackingStoreBuffered defer:NO];
        impl->panel.opaque = NO;
        impl->panel.backgroundColor = [NSColor clearColor];
        impl->panel.hasShadow = NO;
        impl->panel.ignoresMouseEvents = YES;
        impl->panel.hidesOnDeactivate = NO;
        impl->panel.releasedWhenClosed = NO;
        impl->panel.level = NSStatusWindowLevel;
        impl->panel.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces |
            NSWindowCollectionBehaviorFullScreenAuxiliary | NSWindowCollectionBehaviorStationary |
            NSWindowCollectionBehaviorIgnoresCycle;
        impl->panel.sharingType = NSWindowSharingNone;
        impl->panel.contentView = impl->view;
        for (const auto &d : listDisplays())
            if (d.primary)
                impl->place(d.globalPoints);
    }
}

CuaReticle::~CuaReticle()
{
    requireMainThread();
    impl->stopTimer();
    impl->view->_scene = nullptr;
    if (impl->panel) {
        [impl->panel orderOut:nil];
        [impl->panel close];
        [impl->panel release];
    }
    [impl->view release];
}

void CuaReticle::moveTo(QPointF point, const QString &label)
{
    requireMainThread();
    if (!finite(point))
        return;
    @autoreleasepool {
        const QRectF display = impl->displayAt(point);
        if (display.isEmpty())
            return;
        const double now = impl->scene.now();
        if (!impl->scene.offscreen && impl->scene.visible)
            impl->invalidate(now, true);  // old position, before the panel may change display
        impl->place(display);
        impl->scene.label = label.left(120);
        impl->startGlide(point, now);
        impl->show(now);
    }
}

bool CuaReticle::highlightRect(QRectF rect, const QString &label)
{
    requireMainThread();
    if (!reticleRectValid(rect))
        return false;
    @autoreleasepool {
        const QRectF display = impl->displayAt(rect.center());
        if (display.isEmpty())
            return false;
        const double now = impl->scene.now();
        impl->place(display);
        impl->scene.focus = rect;
        impl->scene.focusStart = now;
        impl->scene.label = label.left(120);
        impl->show(now);
    }
    return true;
}

bool CuaReticle::pulseClick(QPointF point, const QString &label)
{
    requireMainThread();
    if (!finite(point))
        return false;
    @autoreleasepool {
        const QRectF display = impl->displayAt(point);
        if (display.isEmpty())
            return false;
        const double now = impl->scene.now();
        impl->place(display);
        if (!label.isEmpty())
            impl->scene.label = label.left(120);
        const QPointF at = impl->scene.to;
        if (!impl->scene.visible || !impl->scene.hasTip || std::hypot(at.x() - point.x(), at.y() - point.y()) > 1)
            impl->startGlide(point, now);
        impl->scene.rippleAt = point;
        impl->scene.rippleStart = std::max(now, impl->scene.moveStart + detail::kGlide);
        impl->show(now);
    }
    return true;
}

void CuaReticle::hide()
{
    requireMainThread();
    impl->scene.visible = false;
    impl->scene.focus = {};
    impl->scene.rippleStart = detail::kNever;
    impl->stopTimer();
    if (impl->panel) {
        [impl->panel orderOut:nil];
        [impl->view setNeedsDisplay:YES];
    }
}

uint32_t CuaReticle::windowNumber() const
{
    requireMainThread();
    return impl->panel ? uint32_t(impl->panel.windowNumber) : 0;
}

ReticleState CuaReticle::state() const
{
    requireMainThread();
    ReticleState st;
    @autoreleasepool {
        const Scene &s = impl->scene;
        const double now = s.now();
        st.visible = s.visible;
        st.tip = s.hasTip ? s.tip(now) : QPointF();
        st.target = s.to;
        st.gliding = s.visible && s.gliding(now);
        st.focusRect = s.visible && s.focusShown(now) ? s.focus : QRectF();
        st.rippleActive = s.visible && now >= s.rippleStart && s.ripplePending(now);
        st.label = s.label;
        st.pillRect = s.visible ? s.pillRect(now) : QRectF();
        st.overlayFrame = s.display;
    }
    return st;
}

void CuaReticle::advanceClock(double seconds)
{
    if (impl->scene.offscreen && std::isfinite(seconds) && seconds > 0)
        impl->scene.manualNow += seconds;
}

QImage CuaReticle::render(double scale, const QColor &background) const
{
    requireMainThread();
    const QRectF d = impl->scene.display;
    if (!std::isfinite(scale) || scale <= 0 || d.isEmpty())
        return {};
    const int w = int(std::ceil(d.width() * scale)), h = int(std::ceil(d.height() * scale));
    if (w <= 0 || h <= 0 || w > 16384 || h > 16384)
        return {};
    QImage img(w, h, QImage::Format_RGBA8888_Premultiplied);
    img.fill(background);
    @autoreleasepool {
        CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        CGContextRef ctx = CGBitmapContextCreate(img.bits(), size_t(w), size_t(h), 8, size_t(img.bytesPerLine()), cs,
                                                 uint32_t(kCGImageAlphaPremultipliedLast) | uint32_t(kCGBitmapByteOrder32Big));
        CGColorSpaceRelease(cs);
        if (!ctx)
            return {};
        // Row 0 of the bitmap is the top: flip to y-down points, like the flipped view.
        CGContextTranslateCTM(ctx, 0, h);
        CGContextScaleCTM(ctx, scale, -scale);
        NSGraphicsContext *previous = [[NSGraphicsContext currentContext] retain];
        [NSGraphicsContext setCurrentContext:[NSGraphicsContext graphicsContextWithCGContext:ctx flipped:YES]];
        impl->view->_shadowScale = scale;
        [impl->view paintScene:ctx now:impl->scene.now()];
        impl->view->_shadowScale = 1;
        [NSGraphicsContext setCurrentContext:previous];
        [previous release];
        CGContextRelease(ctx);
    }
    img.setDevicePixelRatio(scale);
    return img;
}

// ---- --snap-cursor -----------------------------------------------------------------------------

namespace {
// A plain dark desktop with one app window and a "Send" button: context for the overlay only.
void paintDesktop(QPainter &p, const QRectF &frame, const QRectF &button)
{
    p.fillRect(frame, QColor("#0B0D11"));
    const QRectF win = frame.adjusted(56, 48, -56, -48);
    p.setPen(QPen(QColor(255, 255, 255, 22), 1));
    p.setBrush(QColor("#15171D"));
    p.drawRoundedRect(win, 12, 12);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor("#1C1F26"));
    p.drawRoundedRect(QRectF(win.left(), win.top(), win.width(), 34), 12, 12);
    p.fillRect(QRectF(win.left(), win.top() + 22, win.width(), 12), QColor("#1C1F26"));
    for (int i = 0; i < 3; ++i) {
        p.setBrush(QColor(255, 255, 255, 40));
        p.drawEllipse(QPointF(win.left() + 18 + i * 16, win.top() + 17), 5, 5);
    }
    p.setBrush(QColor(255, 255, 255, 14));
    for (int i = 0; i < 4; ++i)
        p.drawRoundedRect(QRectF(win.left() + 28, win.top() + 62 + i * 30, (i % 2 ? 300 : 380), 12), 6, 6);
    p.setBrush(QColor("#2A2E38"));
    p.setPen(QPen(QColor(255, 255, 255, 30), 1));
    p.drawRoundedRect(button, 8, 8);
    QFont f(QStringLiteral(".AppleSystemUIFont"));
    f.setPixelSize(14);
    f.setWeight(QFont::Medium);
    p.setFont(f);
    p.setPen(QColor(255, 255, 255, 220));
    p.drawText(button, Qt::AlignCenter, QStringLiteral("Send"));
}
}

int snapCursor(const QString &outPath)
{
    requireMainThread();
    const QRectF frame(0, 0, 640, 400);
    const QRectF button(420, 268, 112, 36);
    const QPointF start(150, 120), click = button.center() + QPointF(-6, 2);
    const QString label = QStringLiteral("Clicking Send in Mail");
    QImage frames[2];
    {
        // Mid-glide: curved tail behind the pointer, ring and pill already on the target.
        CuaReticle r(CuaReticle::Mode::Offscreen, frame);
        r.moveTo(start);
        r.advanceClock(1.0);
        r.highlightRect(button, label);
        r.advanceClock(0.2);
        r.moveTo(click, label);
        r.advanceClock(0.085);
        frames[0] = r.render(2);
    }
    {
        // Click: pointer landed, ripple expanding, pill holding.
        CuaReticle r(CuaReticle::Mode::Offscreen, frame);
        r.moveTo(start);
        r.advanceClock(1.0);
        r.highlightRect(button, label);
        r.advanceClock(0.2);
        r.pulseClick(click, label);
        r.advanceClock(detail::kGlide + 0.11);
        frames[1] = r.render(2);
    }
    if (frames[0].isNull() || frames[1].isNull()) {
        std::fprintf(stderr, "snap-cursor: render failed\n");
        return 2;
    }
    const double scale = 2, gutter = 16;
    QImage out(int((frame.width() * 2 + gutter) * scale), int(frame.height() * scale), QImage::Format_ARGB32_Premultiplied);
    out.fill(QColor("#050608"));
    QPainter p(&out);
    p.setRenderHint(QPainter::Antialiasing);
    p.scale(scale, scale);
    for (int i = 0; i < 2; ++i) {
        const QRectF at = frame.translated(i * (frame.width() + gutter), 0);
        p.save();
        p.translate(at.topLeft());
        paintDesktop(p, frame, button);
        p.restore();
        frames[i].setDevicePixelRatio(scale);
        p.drawImage(at, frames[i]);
    }
    p.end();
    QDir().mkpath(QFileInfo(outPath).absolutePath());
    if (!out.save(outPath)) {
        std::fprintf(stderr, "snap-cursor: could not write %s\n", qPrintable(outPath));
        return 1;
    }
    std::printf("snap-cursor: %s (%dx%d)\n", qPrintable(outPath), out.width(), out.height());
    return 0;
}

} // namespace studio::cua
