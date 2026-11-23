#include "mac/cua_reticle.hpp"
#include "mac/cua.hpp"
#include "mac/cua_pad_icon.hpp"
#import <AppKit/AppKit.h>
#include <QDir>
#include <QFileInfo>
#include <QPainter>
#include <QPainterPath>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>

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

double glideDuration(QPointF from, QPointF to)
{
    const double len = std::hypot(to.x() - from.x(), to.y() - from.y());
    if (!std::isfinite(len))
        return kGlideMinSec;
    const double u = std::max(0.0, std::min(1.0, (len - 80) / 600));
    return kGlideMinSec + (kGlideMaxSec - kGlideMinSec) * u * u * (3 - 2 * u);
}

QRectF cuaHudRect(const QRectF &display)
{
    if (!display.isValid() || display.isEmpty())
        return {};
    return QRectF(std::round(display.center().x() - kHudWidth / 2), display.bottom() - kHudBottomGap - kHudHeight,
                  kHudWidth, kHudHeight);
}

} // namespace studio::cua

// ---- Scene: everything the overlay draws, as a pure function of time ---------------------------

namespace studio::cua::detail {

constexpr double kTailDecay = 0.22;   // the swoosh ribbon shrinks into the hand after landing
constexpr double kTrailLinger = 0.45; // trail stars finish twinkling out after landing
constexpr double kPuff = 0.32;        // launch "poof" clouds
constexpr double kLand = 0.18;        // landing squash wobble
constexpr double kPoke = 0.42;        // click poke and rebound
constexpr double kBurst = 0.46;       // click starburst, ticks, stars, puffs
constexpr double kFocusIn = 0.22, kIdle = 1.8, kFade = 0.45;
constexpr double kNever = -1e9, kForever = 1e18;
constexpr double kHudIn = 0.22, kHudOut = 0.40, kHudLinger = studio::cua::kHudLingerSec, kHudPad = 30;
constexpr double kRestAngle = -28;    // degrees; the hand's natural tilt (finger up-left)

double clamp01(double t) { return std::max(0.0, std::min(1.0, t)); }
double smooth(double t) { t = clamp01(t); return t * t * (3 - 2 * t); }
double easeInOutCubic(double t) { return t < 0.5 ? 4 * t * t * t : 1 - std::pow(-2 * t + 2, 3) / 2; }
double easeOutCubic(double t) { t = clamp01(t); return 1 - std::pow(1 - t, 3); }
double easeOutBack(double t) {
    t = clamp01(t);
    const double c1 = 1.9, c3 = c1 + 1;
    return 1 + c3 * std::pow(t - 1, 3) + c1 * std::pow(t - 1, 2);
}
double sq(double x) { return x * x; }
// Smoothstep between (time, value) keys; holds the end values outside the range.
template <size_t N>
double keyed(double t, const double (&k)[N][2])
{
    if (t <= k[0][0])
        return k[0][1];
    for (size_t i = 1; i < N; ++i)
        if (t <= k[i][0])
            return k[i - 1][1] + (k[i][1] - k[i - 1][1]) * smooth((t - k[i - 1][0]) / (k[i][0] - k[i - 1][0]));
    return k[N - 1][1];
}
// A deterministic pseudo-random value in [0, 1) from an integer seed (stable jitter, no state).
double hash01(int seed)
{
    uint32_t x = uint32_t(seed) * 0x9E3779B1u + 0x7F4A7C15u;
    x ^= x >> 15; x *= 0x2C1B3C6Du; x ^= x >> 12; x *= 0x297A2D39u; x ^= x >> 15;
    return (x & 0xFFFFFF) / double(0x1000000);
}

// The hand's transform on top of its tip position.
struct Pose {
    QPointF tip;
    double angle = kRestAngle;  // degrees, hand rotation about the fingertip
    double axis = 0;            // radians, motion direction (stretch axis)
    double along = 1, perp = 1; // stretch along / across the motion
    double sx = 1, sy = 1;      // hand-local squash (x across the finger, y along it)
    double lunge = 0;           // pt toward the fingertip
};

struct Scene {
    bool offscreen = false;
    double manualNow = 1000;
    QRectF display;  // global points the overlay covers
    bool visible = false;
    bool hasTip = false;
    bool gold = true;
    QPointF from, ctrl, to;
    double moveStart = kNever;
    double glideDur = studio::cua::kGlideMinSec;
    QRectF focus;
    double focusStart = kNever;
    QPointF rippleAt;
    double rippleStart = kNever;
    QString label;
    double lastActivity = kNever;
    // HUD bar.
    QString hudStatus;
    studio::cua::HudTone hudTone = studio::cua::HudTone::Working;
    QRectF hudDisplay;
    double hudShownAt = kNever;
    double hudHideAt = kForever;  // when the fade-out starts

    double now() const { return offscreen ? manualNow : [[NSProcessInfo processInfo] systemUptime]; }
    double glideLen() const { return std::hypot(to.x() - from.x(), to.y() - from.y()); }
    double landAt() const { return moveStart + glideDur; }
    // Path progress for normalized glide time u: a small back-up (anticipation), a fast eased
    // swoosh, and a springy overshoot past the target that settles exactly on it at u = 1.
    double progressAt(double u) const {
        if (u >= 1)
            return 1;
        if (u <= 0)
            return 0;
        const double len = glideLen();
        if (len < 1e-6)
            return 1;
        const double back = std::min(9.0, 0.03 * len) / len, over = std::min(8.0, 0.035 * len) / len;
        double p = easeInOutCubic(u);
        if (u < 0.2)
            p -= back * sq(std::sin(M_PI * u / 0.2));
        if (u > 0.68)
            p += over * sq(std::sin(M_PI * (u - 0.68) / 0.32));
        return p;
    }
    double glideU(double t) const { return clamp01((t - moveStart) / glideDur); }
    double glideP(double t) const { return progressAt(glideU(t)); }
    QPointF tip(double t) const { return quadBezier(from, ctrl, to, glideP(t)); }
    QPointF pathAt(double p) const { return quadBezier(from, ctrl, to, p); }
    QPointF pathDir(double p) const {
        const QPointF d = (ctrl - from) * (2 * (1 - p)) + (to - ctrl) * (2 * p);
        const double l = std::hypot(d.x(), d.y());
        return l > 1e-9 ? d / l : QPointF(1, 0);
    }
    bool gliding(double t) const { return t - moveStart < glideDur; }
    bool trailMove() const { return glideLen() >= 3; }
    double tailAmount(double t) const {
        if (!trailMove())
            return 0;
        const double since = t - moveStart;
        return since < glideDur ? 1 : clamp01(1 - (since - glideDur) / kTailDecay);
    }
    bool trailPending(double t) const { return trailMove() && t - moveStart < glideDur + kTrailLinger; }
    // When path progress first reaches `p` (the trail star born there), by a fixed 96-step scan.
    double crossTime(double p) const {
        for (int k = 0; k <= 96; ++k)
            if (progressAt(k / 96.0) >= p)
                return moveStart + glideDur * k / 96.0;
        return landAt();
    }
    double speedFactor(double t, QPointF *vel = nullptr) const {
        const double h = 1.0 / 480;
        const QPointF v = (tip(t + h) - tip(t - h)) / (2 * h);
        if (vel)
            *vel = v;
        const double len = glideLen();
        if (len < 3 || !gliding(t))
            return 0;
        const double peak = 1.5 * len / glideDur;
        return clamp01(std::hypot(v.x(), v.y()) / peak) * clamp01(len / 140);
    }
    Pose pose(double t) const {
        Pose ps;
        ps.tip = tip(t);
        QPointF v;
        const double sf = speedFactor(t, &v);
        if (sf > 1e-4) {
            ps.axis = std::atan2(v.y(), v.x());
            ps.along = 1 + 0.26 * sf;
            ps.perp = 1 / std::sqrt(ps.along);
            // Bank: the finger swings to lead the motion (cuff trailing into the swoosh), at most
            // 120° off rest, blended in by speed so it eases back to rest as it lands.
            double delta = ps.axis * 180 / M_PI + 90 - kRestAngle;
            delta = std::remainder(delta, 360.0);
            delta = std::max(-120.0, std::min(120.0, delta));
            ps.angle = kRestAngle + 0.85 * delta * smooth(sf * 1.8);
        }
        // Landing wobble: 1 -> 0.88 squash -> 1.07 rebound -> 1, along the finger.
        const double land = t - landAt();
        if (trailMove() && land >= 0 && land < kLand) {
            static const double k[][2] = {{0, 1.0}, {0.05, 0.88}, {0.11, 1.07}, {kLand, 1.0}};
            const double s = keyed(land, k);
            ps.sy *= s;
            ps.sx *= 1 / std::sqrt(s);
        }
        // Click poke: lunge and squash into the target, pop back with a flick, settle.
        const double c = t - rippleStart;
        if (c >= 0 && c < kPoke) {
            static const double kx[][2] = {{0, 1}, {0.09, 1.22}, {0.24, 0.88}, {kPoke, 1}};
            static const double ky[][2] = {{0, 1}, {0.09, 0.76}, {0.24, 1.18}, {kPoke, 1}};
            static const double kl[][2] = {{0, 0}, {0.09, 3}, {0.24, 0}, {kPoke, 0}};
            static const double ka[][2] = {{0, 0}, {0.09, 0}, {0.24, -8}, {kPoke, 0}};
            ps.sx *= keyed(c, kx);
            ps.sy *= keyed(c, ky);
            ps.lunge = keyed(c, kl);
            ps.angle += keyed(c, ka);
        }
        return ps;
    }
    bool burstPending(double t) const { return t - rippleStart < kBurst; }
    bool pokePending(double t) const { return t - rippleStart < kPoke; }
    double chromeAlpha(double t) const { return clamp01(1 - (t - lastActivity - kIdle) / kFade); }
    bool focusShown(double t) const { return !focus.isEmpty() && chromeAlpha(t) > 0; }
    // Moving pixels this frame (the overlay is still once the chrome has faded and the hand rests).
    bool dynamic(double t) const {
        const double idle = t - lastActivity;
        return gliding(t) || trailPending(t) || (trailMove() && t - landAt() < kLand) || burstPending(t)
               || pokePending(t) || (idle < kIdle + kFade && (focusShown(t) || !label.isEmpty()));
    }
    bool animating(double t) const { return visible && (dynamic(t) || t - lastActivity < kIdle + kFade); }

    double hudIn(double t) const { return easeOutCubic(clamp01((t - hudShownAt) / kHudIn)); }
    bool hudGone(double t) const { return hudShownAt == kNever || t >= hudHideAt + kHudOut; }
    double hudAlpha(double t) const {
        if (hudGone(t))
            return 0;
        return hudIn(t) * (t < hudHideAt ? 1 : clamp01(1 - (t - hudHideAt) / kHudOut));
    }
    bool hudPulsing() const {
        return hudTone == studio::cua::HudTone::Listening || hudTone == studio::cua::HudTone::Working;
    }
    // HUD pixels change this frame (mark animation, fade in, fade out).
    bool hudDynamic(double t) const {
        return !hudGone(t) && (hudPulsing() || t - hudShownAt < kHudIn || t >= hudHideAt);
    }
    // The timer must keep running: animating, or waiting for a scheduled fade-out.
    bool hudAnimating(double t) const { return !hudGone(t) && (hudDynamic(t) || hudHideAt < kForever); }
    QRectF hudRect() const { return hudShownAt == kNever ? QRectF() : studio::cua::cuaHudRect(hudDisplay); }

    static NSFont *labelFont() {
        NSFont *f = [NSFont fontWithName:@"AvenirNext-DemiBold" size:12.5];
        return f ? f : [NSFont systemFontOfSize:12.5 weight:NSFontWeightSemibold];
    }
    static NSDictionary *labelAttrs() {
        NSMutableParagraphStyle *style = [[[NSMutableParagraphStyle alloc] init] autorelease];
        style.lineBreakMode = NSLineBreakByTruncatingTail;
        return @{NSFontAttributeName: labelFont(),
                 NSForegroundColorAttributeName: [NSColor colorWithSRGBRed:0xE8 / 255.0 green:0xD5 / 255.0 blue:0xB5 / 255.0 alpha:1],
                 NSParagraphStyleAttributeName: style};
    }
    // What the speech bubble hangs off: the target frame, else the hand (which extends down-right
    // of the fingertip at rest).
    QRectF pillAnchor(double t) const {
        if (focusShown(t))
            return focus.adjusted(-6, -6, 6, 6);
        if (hasTip) {
            const QPointF p = tip(t);
            return QRectF(p.x() - 8, p.y() - 2, 34, 42);
        }
        return {};
    }
    QRectF pillRect(double t) const {
        if (label.isEmpty() || chromeAlpha(t) <= 0)
            return {};
        const QRectF anchor = pillAnchor(t);
        if (anchor.isEmpty())
            return {};
        const double textW = std::ceil([label.toNSString() sizeWithAttributes:labelAttrs()].width);
        const QSizeF size(std::min(30 + textW + 14, 340.0), 26);
        return reticlePillRect(anchor, size, display);
    }
    // Everything that may have pixels this frame, global points.
    QRectF bounds(double t) const {
        if (!visible)
            return {};
        QRectF b;
        if (hasTip) {
            const QPointF p = tip(t);
            b |= QRectF(p.x() - 64, p.y() - 64, 128, 128);
            if (trailPending(t)) {
                const double l = std::min({from.x(), ctrl.x(), to.x()}), r = std::max({from.x(), ctrl.x(), to.x()});
                const double u = std::min({from.y(), ctrl.y(), to.y()}), d = std::max({from.y(), ctrl.y(), to.y()});
                b |= QRectF(l, u, r - l, d - u).adjusted(-40, -40, 40, 40);
            }
        }
        if (focusShown(t)) {
            const double gx = focus.width() * 0.06 + 30, gy = focus.height() * 0.06 + 30;
            b |= focus.adjusted(-gx, -gy, gx, gy);
        }
        if (burstPending(t))
            b |= QRectF(rippleAt.x() - 80, rippleAt.y() - 80, 160, 170);
        const QRectF pill = pillRect(t);
        if (!pill.isEmpty())
            b |= pill.adjusted(-24, -24, 24, 24);
        return b;
    }
};

} // namespace studio::cua::detail

using studio::cua::detail::Scene;
using studio::cua::detail::Pose;

namespace {
// Hand-local points (fingertip at the origin) to global points: stretch along the motion, bank,
// lunge toward the fingertip, squash.
CGAffineTransform handTransform(const Pose &ps)
{
    CGAffineTransform t = CGAffineTransformMakeTranslation(ps.tip.x(), ps.tip.y());
    t = CGAffineTransformRotate(t, ps.axis);
    t = CGAffineTransformScale(t, ps.along, ps.perp);
    t = CGAffineTransformRotate(t, -ps.axis);
    t = CGAffineTransformRotate(t, ps.angle * M_PI / 180);
    t = CGAffineTransformTranslate(t, 0, -ps.lunge);
    return CGAffineTransformScale(t, ps.sx, ps.sy);
}
}

// ---- Palette and drawing helpers ---------------------------------------------------------------

namespace {
struct Rgb { int r, g, b; };
constexpr Rgb kInk{0x0A, 0x08, 0x06};        // espresso ink outline
constexpr Rgb kCream{0xF7, 0xEB, 0xD2};      // glove
constexpr Rgb kCel{0xDF, 0xC7, 0x9A};        // glove cel shadow
constexpr Rgb kCelDeep{0xCF, 0xA9, 0x6E};    // deeper shadow under curled fingers
constexpr Rgb kDart{0x1D, 0x17, 0x10};       // back-of-hand dart lines
constexpr Rgb kShine{0xFF, 0xFA, 0xF0};      // highlight bean
constexpr Rgb kStud{0xF5, 0xE2, 0xBA};       // cuff stud
constexpr Rgb kObsidian{0x0D, 0x0C, 0x0B};   // bubble and HUD body
constexpr Rgb kParchment{0xE8, 0xD5, 0xB5};  // label text
constexpr Rgb kCoral{0xD9, 0x6B, 0x43};      // error

struct Palette { Rgb accent, shadow; };
Palette palette(bool gold)
{
    return gold ? Palette{{0xE0, 0x9F, 0x3E}, {0xB5, 0x65, 0x3B}} : Palette{{0x4F, 0x8C, 0xFF}, {0x2D, 0x5A, 0xA8}};
}

CGColorRef rgba(int r, int g, int b, double a)
{
    return [NSColor colorWithSRGBRed:r / 255.0 green:g / 255.0 blue:b / 255.0 alpha:std::max(0.0, std::min(1.0, a))].CGColor;
}
CGColorRef col(Rgb c, double a = 1) { return rgba(c.r, c.g, c.b, a); }

void fillPath(CGContextRef ctx, CGPathRef p, CGColorRef c)
{
    CGContextAddPath(ctx, p);
    CGContextSetFillColorWithColor(ctx, c);
    CGContextFillPath(ctx);
}
void strokePath(CGContextRef ctx, CGPathRef p, CGColorRef c, double w)
{
    CGContextAddPath(ctx, p);
    CGContextSetStrokeColorWithColor(ctx, c);
    CGContextSetLineWidth(ctx, w);
    CGContextStrokePath(ctx);
}
void roundJoins(CGContextRef ctx)
{
    CGContextSetLineCap(ctx, kCGLineCapRound);
    CGContextSetLineJoin(ctx, kCGLineJoinRound);
}

// A concave 4-point cartoon twinkle star, radius R, centred at c, rotated `rot` radians.
CGPathRef starPath(CGPoint c, double R, double rot)
{
    CGMutablePathRef p = CGPathCreateMutable();
    const double inner = 0.2 * R;
    for (int i = 0; i < 4; ++i) {
        const double a = rot + i * M_PI / 2, m = a + M_PI / 4, b = a + M_PI / 2;
        if (i == 0)
            CGPathMoveToPoint(p, nullptr, c.x + R * std::cos(a), c.y + R * std::sin(a));
        CGPathAddQuadCurveToPoint(p, nullptr, c.x + inner * std::cos(m), c.y + inner * std::sin(m),
                                  c.x + R * std::cos(b), c.y + R * std::sin(b));
    }
    CGPathCloseSubpath(p);
    return p;
}
void drawStar(CGContextRef ctx, CGPoint c, double R, double rot, CGColorRef fill, double ink, double alpha)
{
    if (R <= 0.2 || alpha <= 0)
        return;
    CGPathRef s = starPath(c, R, rot);
    CGContextSaveGState(ctx);
    CGContextSetAlpha(ctx, alpha);
    roundJoins(ctx);
    if (ink > 0)
        strokePath(ctx, s, col(kInk, 0.95), ink);
    fillPath(ctx, s, fill);
    // A cream sparkle core.
    CGContextSetFillColorWithColor(ctx, col(kShine, 0.9));
    const double k = 0.22 * R;
    CGContextFillEllipseInRect(ctx, CGRectMake(c.x - k, c.y - k, 2 * k, 2 * k));
    CGContextRestoreGState(ctx);
    CGPathRelease(s);
}

// A puffy 3-lobe cartoon cloud: ink stroked under, cream filled over, so the outline is the union.
void drawPuff(CGContextRef ctx, CGPoint c, double r, double rot, double alpha)
{
    if (r <= 0.3 || alpha <= 0)
        return;
    const double lobes[3][3] = {{-0.55, 0.15, 0.72}, {0.5, 0.2, 0.62}, {0.0, -0.35, 0.85}};
    CGMutablePathRef p = CGPathCreateMutable();
    const double cs = std::cos(rot), sn = std::sin(rot);
    for (const auto &l : lobes) {
        const double x = l[0] * r, y = l[1] * r, lr = l[2] * r;
        const CGPoint o = CGPointMake(c.x + x * cs - y * sn, c.y + x * sn + y * cs);
        CGPathAddEllipseInRect(p, nullptr, CGRectMake(o.x - lr, o.y - lr, 2 * lr, 2 * lr));
    }
    CGContextSaveGState(ctx);
    CGContextSetAlpha(ctx, alpha);
    CGContextBeginTransparencyLayer(ctx, nullptr);
    strokePath(ctx, p, col(kInk, 0.9), std::min(3.2, 0.5 * r + 1));
    // Each lobe filled on its own: CG ellipse winding would cancel overlaps under a single fill.
    for (const auto &l : lobes) {
        const double x = l[0] * r, y = l[1] * r, lr = l[2] * r;
        const CGPoint o = CGPointMake(c.x + x * cs - y * sn, c.y + x * sn + y * cs);
        CGContextSetFillColorWithColor(ctx, col(kCream));
        CGContextFillEllipseInRect(ctx, CGRectMake(o.x - lr, o.y - lr, 2 * lr, 2 * lr));
    }
    // Cel shadow along the lower lobes.
    CGContextSetBlendMode(ctx, kCGBlendModeSourceAtop);
    CGContextSetFillColorWithColor(ctx, col(kCel));
    CGContextFillEllipseInRect(ctx, CGRectMake(c.x - 1.1 * r, c.y + 0.5 * r, 2.2 * r, 1.4 * r));
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);
    CGPathRelease(p);
}

// A rounded-rect speech bubble with a triangular tail at `tailX` on the top (tailUp) or bottom edge.
CGPathRef bubblePath(CGRect r, double rad, double tailX, int tail /* -1 up, 1 down, 0 none */, double tailH)
{
    const double x0 = r.origin.x, y0 = r.origin.y, x1 = x0 + r.size.width, y1 = y0 + r.size.height, tw = 9;
    CGMutablePathRef p = CGPathCreateMutable();
    CGPathMoveToPoint(p, nullptr, x0 + rad, y0);
    if (tail < 0) {
        CGPathAddLineToPoint(p, nullptr, tailX - tw / 2, y0);
        CGPathAddLineToPoint(p, nullptr, tailX - 2.5, y0 - tailH);
        CGPathAddLineToPoint(p, nullptr, tailX + tw / 2, y0);
    }
    CGPathAddArcToPoint(p, nullptr, x1, y0, x1, y0 + rad, rad);
    CGPathAddArcToPoint(p, nullptr, x1, y1, x1 - rad, y1, rad);
    if (tail > 0) {
        CGPathAddLineToPoint(p, nullptr, tailX + tw / 2, y1);
        CGPathAddLineToPoint(p, nullptr, tailX - 2.5, y1 + tailH);
        CGPathAddLineToPoint(p, nullptr, tailX - tw / 2, y1);
    }
    CGPathAddArcToPoint(p, nullptr, x0, y1, x0, y1 - rad, rad);
    CGPathAddArcToPoint(p, nullptr, x0, y0, x0 + rad, y0, rad);
    CGPathCloseSubpath(p);
    return p;
}

// ---- The cartoon glove, in hand-local points: fingertip hotspot at (0, 0), finger pointing up
// (−y), palm and cuff below; ~22 pt wide, ~39 pt tall including the ink. ---------------------------

CGPathRef capsule(double x, double y, double w, double h)
{
    const double r = std::min(w, h) / 2;
    return CGPathCreateWithRoundedRect(CGRectMake(x, y, w, h), r, r, nullptr);
}
CGPathRef rotatedEllipse(double cx, double cy, double w, double h, double deg)
{
    CGAffineTransform t = CGAffineTransformMakeTranslation(cx, cy);
    t = CGAffineTransformRotate(t, deg * M_PI / 180);
    return CGPathCreateWithEllipseInRect(CGRectMake(-w / 2, -h / 2, w, h), &t);
}

struct Glove {
    CGPathRef index, palm, thumb, knuckle[3], cuff;
    Glove() {
        index = capsule(-3.9, 2.5, 7.8, 20);  // top at 2.5: the 2.8 pt ink lands on the hotspot
        palm = CGPathCreateWithRoundedRect(CGRectMake(-5.6, 16.0, 18.6, 13.6), 6.5, 6.5, nullptr);
        thumb = rotatedEllipse(-6.0, 22.6, 6.2, 9.6, 24);
        knuckle[0] = rotatedEllipse(6.2, 17.8, 7.0, 6.4, 0);
        knuckle[1] = rotatedEllipse(10.0, 19.4, 6.4, 6.0, 0);
        knuckle[2] = rotatedEllipse(12.7, 22.0, 5.4, 5.4, 0);
        cuff = CGPathCreateWithRoundedRect(CGRectMake(-6.8, 28.6, 20.8, 7.4), 3.4, 3.4, nullptr);
    }
    ~Glove() {
        for (CGPathRef p : {index, palm, thumb, knuckle[0], knuckle[1], knuckle[2], cuff})
            CGPathRelease(p);
    }
    Glove(const Glove &) = delete;
    Glove &operator=(const Glove &) = delete;
    // Fill order: what is behind first; the index finger last, in front.
    template <class F>
    void eachGlovePart(F f) const {
        f(thumb);
        for (CGPathRef k : knuckle)
            f(k);
        f(palm);
        f(index);
    }
};

void paintGlove(CGContextRef ctx, const Palette &pal, double shadowScale)
{
    static const Glove g;
    roundJoins(ctx);
    const auto all = [&](auto fn) { g.eachGlovePart(fn); fn(g.cuff); };

    // 1. Contrast halo (reads on OLED black) and a soft warm drop shadow (reads on white pages).
    CGContextSaveGState(ctx);
    CGContextSetShadowWithColor(ctx, CGSizeMake(0, -2 * shadowScale), 4 * shadowScale, rgba(0x1A, 0x10, 0x06, 0.55));
    CGContextBeginTransparencyLayer(ctx, nullptr);
    all([&](CGPathRef p) { strokePath(ctx, p, col(kStud, 0.16), 2 * (2.8 + 1.5)); });
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);

    // 2. The bold ink outline: every part stroked 2 × 2.8 pt, then filled over, leaves the union.
    all([&](CGPathRef p) { strokePath(ctx, p, col(kInk), 2 * 2.8); });

    // 3. Glove body with cel shading, in its own layer so shading lands only on the glove.
    CGContextSaveGState(ctx);
    CGContextBeginTransparencyLayer(ctx, nullptr);
    g.eachGlovePart([&](CGPathRef p) { fillPath(ctx, p, col(kCream)); });
    CGContextSetBlendMode(ctx, kCGBlendModeSourceAtop);
    // Crescent shadow: everything shadow-coloured, then the glove shifted toward the light (up-left)
    // repainted cream, leaving a crisp crescent along the lower-right edges.
    CGContextSetFillColorWithColor(ctx, col(kCel));
    CGContextFillRect(ctx, CGRectMake(-14, -6, 34, 44));
    {
        CGContextSaveGState(ctx);
        CGContextTranslateCTM(ctx, -2.0, -2.3);
        g.eachGlovePart([&](CGPathRef p) { fillPath(ctx, p, col(kCream)); });
        CGContextRestoreGState(ctx);
    }
    // Deeper shadow tucked under the curled fingers and along the palm's base.
    CGContextSetStrokeColorWithColor(ctx, col(kCelDeep));
    CGContextSetLineWidth(ctx, 2.4);
    CGContextMoveToPoint(ctx, 4.2, 22.0);
    CGContextAddQuadCurveToPoint(ctx, 9.5, 24.2, 14.0, 24.8);
    CGContextStrokePath(ctx);
    CGContextSetLineWidth(ctx, 2.2);
    CGContextMoveToPoint(ctx, -3.5, 28.8);
    CGContextAddLineToPoint(ctx, 11.5, 28.8);
    CGContextStrokePath(ctx);
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);

    // 4. Creases: the curled fingers apart from each other and from the index, the thumb's seam.
    CGContextSetStrokeColorWithColor(ctx, col(kInk));
    CGContextSetLineWidth(ctx, 1.4);
    CGContextMoveToPoint(ctx, 3.9, 15.6);   // index / first knuckle
    CGContextAddLineToPoint(ctx, 3.9, 19.8);
    CGContextMoveToPoint(ctx, 8.4, 17.8);   // knuckle 1 / 2
    CGContextAddQuadCurveToPoint(ctx, 8.6, 19.6, 8.0, 20.8);
    CGContextMoveToPoint(ctx, 11.9, 20.4);  // knuckle 2 / 3
    CGContextAddQuadCurveToPoint(ctx, 11.8, 21.8, 11.0, 22.8);
    CGContextMoveToPoint(ctx, -3.6, 20.4);  // thumb seam
    CGContextAddQuadCurveToPoint(ctx, -2.7, 24.0, -4.0, 27.2);
    CGContextStrokePath(ctx);
    // Back-of-hand darts.
    CGContextSetStrokeColorWithColor(ctx, col(kDart));
    CGContextSetLineWidth(ctx, 2.0);
    CGContextMoveToPoint(ctx, 1.4, 23.4);
    CGContextAddQuadCurveToPoint(ctx, 1.1, 25.0, 1.5, 26.6);
    CGContextMoveToPoint(ctx, 5.4, 24.4);
    CGContextAddQuadCurveToPoint(ctx, 5.2, 25.8, 5.6, 27.0);
    CGContextStrokePath(ctx);

    // 5. The rolled cuff: accent body, copper shadow band, a rolled-edge line and a brass stud.
    CGContextSaveGState(ctx);
    CGContextBeginTransparencyLayer(ctx, nullptr);
    fillPath(ctx, g.cuff, col(pal.accent));
    CGContextSetBlendMode(ctx, kCGBlendModeSourceAtop);
    CGContextSetFillColorWithColor(ctx, col(pal.shadow));
    CGContextFillRect(ctx, CGRectMake(-8, 33.8, 24, 4));
    CGContextSetFillColorWithColor(ctx, col(kShine, 0.55));
    CGContextFillRect(ctx, CGRectMake(-4.2, 30.0, 13.5, 1.2));
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);
    CGContextSetStrokeColorWithColor(ctx, col(kInk));
    CGContextSetLineWidth(ctx, 2.2);
    CGContextMoveToPoint(ctx, -6.0, 28.8);  // where the glove meets the cuff
    CGContextAddLineToPoint(ctx, 13.2, 28.8);
    CGContextStrokePath(ctx);
    CGContextSetLineWidth(ctx, 1.1);
    CGContextSetStrokeColorWithColor(ctx, col(kInk, 0.75));
    CGContextMoveToPoint(ctx, -5.4, 33.6);  // the roll
    CGContextAddLineToPoint(ctx, 12.6, 33.6);
    CGContextStrokePath(ctx);
    CGContextSetFillColorWithColor(ctx, col(kInk));
    CGContextFillEllipseInRect(ctx, CGRectMake(10.4, 30.2, 3.4, 3.4));
    CGContextSetFillColorWithColor(ctx, col(kStud));
    CGContextFillEllipseInRect(ctx, CGRectMake(10.9, 30.7, 2.4, 2.4));

    // 6. Glossy highlight beans on the fingertip and the first knuckle.
    CGPathRef bean = rotatedEllipse(-1.3, 8.0, 2.3, 5.4, 0);
    fillPath(ctx, bean, col(kShine));
    CGPathRelease(bean);
    CGContextSetFillColorWithColor(ctx, col(kShine));
    CGContextFillEllipseInRect(ctx, CGRectMake(-1.9, 3.9, 1.5, 1.5));
    CGPathRef knuck = rotatedEllipse(5.2, 16.4, 2.6, 1.6, -20);
    fillPath(ctx, knuck, col(kShine, 0.9));
    CGPathRelease(knuck);
}
}  // namespace

// ---- AppKit ------------------------------------------------------------------------------------

@interface LapisCuaPanel : NSPanel
@end
@implementation LapisCuaPanel
- (BOOL)canBecomeKeyWindow { return NO; }
- (BOOL)canBecomeMainWindow { return NO; }
@end

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

// Bouncy gold corner brackets around the target, with marching dashes and a faint spotlight.
- (void)paintFocus:(CGContextRef)ctx now:(double)now
{
    using namespace studio::cua::detail;
    const Scene &s = *_scene;
    if (!s.focusShown(now))
        return;
    const Palette pal = palette(s.gold);
    const double since = std::max(0.0, now - s.focusStart);
    static const double kScale[][2] = {{0, 1.10}, {0.11, 0.97}, {kFocusIn, 1.0}};
    const double scale = keyed(since, kScale);
    const double breathe = 1.3 * std::sin(2 * M_PI * since / 1.6);
    const QRectF base = s.focus.adjusted(-5 - breathe, -5 - breathe, 5 + breathe, 5 + breathe);
    const QPointF c = base.center();
    const QRectF r(c.x() - base.width() * scale / 2, c.y() - base.height() * scale / 2, base.width() * scale,
                   base.height() * scale);
    CGContextSaveGState(ctx);
    CGContextSetAlpha(ctx, clamp01(since / 0.07) * s.chromeAlpha(now));
    CGContextBeginTransparencyLayer(ctx, nullptr);
    roundJoins(ctx);
    const double corner = std::min({6.0, r.width() / 2, r.height() / 2});
    CGPathRef spot = CGPathCreateWithRoundedRect(r.toCGRect(), corner, corner, nullptr);
    fillPath(ctx, spot, col(pal.accent, 0.06));
    CGPathRelease(spot);

    const double L = std::max(7.0, std::min({16.0, r.width() * 0.3, r.height() * 0.3}));
    const double rad = std::min(4.5, L / 2);
    CGMutablePathRef brackets = CGPathCreateMutable();
    const double x0 = r.left(), y0 = r.top(), x1 = r.right(), y1 = r.bottom();
    const double pts[4][6] = {{x0, y0 + L, x0, y0, x0 + L, y0}, {x1 - L, y0, x1, y0, x1, y0 + L},
                              {x1, y1 - L, x1, y1, x1 - L, y1}, {x0 + L, y1, x0, y1, x0, y1 - L}};
    for (const auto &p : pts) {
        CGPathMoveToPoint(brackets, nullptr, p[0], p[1]);
        CGPathAddArcToPoint(brackets, nullptr, p[2], p[3], p[4], p[5], rad);
        CGPathAddLineToPoint(brackets, nullptr, p[4], p[5]);
    }
    // Marching dashes between the brackets.
    CGMutablePathRef dashes = CGPathCreateMutable();
    const double gap = 5;
    if (x1 - x0 > 2 * L + 2 * gap + 4) {
        CGPathMoveToPoint(dashes, nullptr, x0 + L + gap, y0);
        CGPathAddLineToPoint(dashes, nullptr, x1 - L - gap, y0);
        CGPathMoveToPoint(dashes, nullptr, x1 - L - gap, y1);
        CGPathAddLineToPoint(dashes, nullptr, x0 + L + gap, y1);
    }
    if (y1 - y0 > 2 * L + 2 * gap + 4) {
        CGPathMoveToPoint(dashes, nullptr, x1, y0 + L + gap);
        CGPathAddLineToPoint(dashes, nullptr, x1, y1 - L - gap);
        CGPathMoveToPoint(dashes, nullptr, x0, y1 - L - gap);
        CGPathAddLineToPoint(dashes, nullptr, x0, y0 + L + gap);
    }
    CGContextSaveGState(ctx);
    const CGFloat dash[] = {3.5, 5};
    CGContextSetLineDash(ctx, -since * 22, dash, 2);
    strokePath(ctx, dashes, col(kInk, 0.45), 3.0);
    strokePath(ctx, dashes, col(pal.accent, 0.7), 1.4);
    CGContextRestoreGState(ctx);
    CGPathRelease(dashes);

    CGContextSaveGState(ctx);
    [self shadow:ctx color:rgba(0, 0, 0, 0.45) blur:3 dy:-1.5];
    strokePath(ctx, brackets, col(kInk), 3.0 + 2 * 1.6);
    CGContextRestoreGState(ctx);
    strokePath(ctx, brackets, col(pal.accent), 3.0);
    // A cream glint on the top-left bracket.
    CGContextSetStrokeColorWithColor(ctx, col(kShine, 0.85));
    CGContextSetLineWidth(ctx, 1.0);
    CGContextMoveToPoint(ctx, x0 + rad + 1, y0 - 0.6);
    CGContextAddLineToPoint(ctx, x0 + L - 1.5, y0 - 0.6);
    CGContextStrokePath(ctx);
    CGPathRelease(brackets);
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);
}

// The swoosh: a tapered cel ribbon with a cream core, speed lines, launch puffs and trail stars.
- (void)paintTail:(CGContextRef)ctx now:(double)now
{
    using namespace studio::cua::detail;
    const Scene &s = *_scene;
    if (!s.hasTip || !s.trailPending(now))
        return;
    const Palette pal = palette(s.gold);
    const double len = s.glideLen();
    const double amount = s.tailAmount(now);
    const double reach = std::min(0.42, 150 / len);  // the ribbon trails ~150 pt at most
    // The ribbon's wide end feeds into the cuff, ~28 pt behind the fingertip, not past the hand.
    const double head = std::max(0.0, clamp01(s.glideP(now)) - std::min(0.5 * reach, 28 / len));
    const double u0 = std::max(0.0, head - reach * amount);
    const double size = clamp01(len / 90);
    CGContextSaveGState(ctx);
    roundJoins(ctx);

    // 1. The ribbon (and its core) as closed tapered polygons sampled at 18 points.
    if (amount > 0 && head - u0 > 1e-3) {
        constexpr int kN = 18;
        QPointF c[kN], n[kN];
        // Bend the last stretch of the ribbon onto the cuff wherever the banked hand put it.
        const CGPoint cuff = CGPointApplyAffineTransform(CGPointMake(3.6, 32.0), handTransform(s.pose(now)));
        const QPointF bend = QPointF(cuff.x, cuff.y) - s.pathAt(head);
        for (int i = 0; i < kN; ++i) {
            const double f = double(i) / (kN - 1), p = u0 + (head - u0) * f;
            c[i] = s.pathAt(p) + bend * (f * f * f);
        }
        for (int i = 0; i < kN; ++i) {
            const QPointF d = c[std::min(i + 1, kN - 1)] - c[std::max(i - 1, 0)];
            const double l = std::hypot(d.x(), d.y());
            n[i] = l > 1e-9 ? QPointF(-d.y() / l, d.x() / l) : QPointF(0, 1);
        }
        const auto ribbon = [&](double width) {
            CGMutablePathRef path = CGPathCreateMutable();
            for (int i = 0; i < kN; ++i) {
                const double f = double(i) / (kN - 1), w = width * std::pow(f, 0.85) / 2;
                const QPointF q = c[i] + n[i] * w;
                if (i == 0)
                    CGPathMoveToPoint(path, nullptr, q.x(), q.y());
                else
                    CGPathAddLineToPoint(path, nullptr, q.x(), q.y());
            }
            for (int i = kN - 1; i >= 0; --i) {
                const double f = double(i) / (kN - 1), w = width * std::pow(f, 0.85) / 2;
                const QPointF q = c[i] - n[i] * w;
                CGPathAddLineToPoint(path, nullptr, q.x(), q.y());
            }
            CGPathCloseSubpath(path);
            return path;
        };
        const double w = 14 * size * (0.55 + 0.45 * amount);
        CGPathRef outer = ribbon(w);
        CGPathRef core = ribbon(w * 0.42);
        CGContextSaveGState(ctx);
        CGContextSetAlpha(ctx, amount);
        CGContextBeginTransparencyLayer(ctx, nullptr);
        fillPath(ctx, outer, col(pal.accent, 0.45));
        fillPath(ctx, core, col(kCream, 0.55));
        strokePath(ctx, outer, col(pal.accent, 0.9), 1.3);
        CGContextEndTransparencyLayer(ctx);
        CGContextRestoreGState(ctx);
        CGPathRelease(outer);
        CGPathRelease(core);

        // 2. Speed lines flanking the ribbon: ink underlay, accent / cream streak.
        const double lines[3][4] = {{0.34, 0.12, 12, 0}, {0.30, 0.08, -12, 1}, {0.40, 0.24, 19, 0}};
        for (const auto &l : lines) {
            const double a = std::max(u0, head - l[0] * reach / 0.42), b = std::max(u0, head - l[1] * reach / 0.42);
            if (b - a < 0.01)
                continue;
            CGMutablePathRef sl = CGPathCreateMutable();
            for (int i = 0; i <= 6; ++i) {
                const double p = a + (b - a) * i / 6;
                const QPointF d = s.pathDir(p), q = s.pathAt(p) + QPointF(-d.y(), d.x()) * (l[2] * size);
                if (i == 0)
                    CGPathMoveToPoint(sl, nullptr, q.x(), q.y());
                else
                    CGPathAddLineToPoint(sl, nullptr, q.x(), q.y());
            }
            strokePath(ctx, sl, col(kInk, 0.55 * amount), 3.6);
            strokePath(ctx, sl, l[3] > 0 ? col(kCream, 0.9 * amount) : col(pal.accent, 0.95 * amount), 1.7);
            CGPathRelease(sl);
        }
    }

    // 3a. Launch puffs ("poof!") behind the start point.
    const double since = now - s.moveStart;
    if (len >= 24 && since >= 0 && since < kPuff) {
        const double f = since / kPuff;
        const QPointF d = s.pathDir(0), nn(-d.y(), d.x());
        const double alpha = std::pow(1 - f, 1.2), grow = easeOutCubic(f);
        const double puffs[3][4] = {{-7, 7, 1.0, 0.3}, {-5, -8, 0.8, 1.7}, {-13, 0, 0.6, 3.1}};
        for (const auto &p : puffs) {
            const QPointF at = s.from + d * (p[0] - 6 * grow) + nn * (p[1] * (1 + 0.6 * grow));
            drawPuff(ctx, at.toCGPoint(), (3 + 5 * grow) * p[2], p[3] + f * 1.2, alpha);
        }
    }

    // 3b. Spinning gold stars born along the arc, drifting off it and shrinking away.
    if (len >= 40) {
        const double births[5] = {0.16, 0.34, 0.52, 0.70, 0.86};
        const int count = len >= 160 ? 5 : 3;
        for (int i = 0; i < count; ++i) {
            const double p = births[i];
            const double born = s.crossTime(p), age = now - born, life = 0.40;
            if (age < 0 || age > life)
                continue;
            const double f = age / life;
            const QPointF d = s.pathDir(p), nn(-d.y(), d.x());
            const double side = (i % 2 ? -1 : 1) * (6 + 6 * hash01(i * 7 + 1));
            const QPointF at = s.pathAt(p) + nn * (side * (1 + 1.2 * f)) - d * (8 * f);
            const double R = (5.6 + 2.6 * hash01(i * 13 + 5)) * std::min(1.0, age / 0.06) * (1 - f);
            drawStar(ctx, at.toCGPoint(), R, age * 9.0 + i, col(i % 2 ? kCream : pal.accent), 1.1, 1);
        }
    }
    CGContextRestoreGState(ctx);
}

// The comic click burst: an 8-point POW starburst, radial action ticks, flying stars, smoke puffs.
- (void)paintRipple:(CGContextRef)ctx now:(double)now
{
    using namespace studio::cua::detail;
    const Scene &s = *_scene;
    const double t = now - s.rippleStart;
    if (t < 0 || !s.burstPending(now))
        return;
    const Palette pal = palette(s.gold);
    const QPointF c = s.rippleAt;
    CGContextSaveGState(ctx);
    roundJoins(ctx);

    // 1. Starburst (0 .. 0.22 s): springs out, holds, pops away.
    if (t < 0.22) {
        const double R = 9 + 22 * easeOutBack(t / 0.13);
        const double alpha = t < 0.15 ? 1 : 1 - (t - 0.15) / 0.07;
        const double rot = 0.15 + 0.6 * t;
        const auto burst = [&](double outer, double inner) {
            CGMutablePathRef p = CGPathCreateMutable();
            for (int k = 0; k < 16; ++k) {
                const double a = rot + k * M_PI / 8;
                const double r = k % 2 ? inner : outer * (k % 4 == 0 ? 1.0 : 0.8);
                const double jit = 1 + 0.12 * (hash01(k + 31) - 0.5);
                const double x = c.x() + r * jit * std::cos(a), y = c.y() + r * jit * std::sin(a);
                if (k == 0)
                    CGPathMoveToPoint(p, nullptr, x, y);
                else
                    CGPathAddLineToPoint(p, nullptr, x, y);
            }
            CGPathCloseSubpath(p);
            return p;
        };
        CGPathRef outer = burst(R, R * 0.5);
        CGPathRef inner = burst(R * 0.62, R * 0.32);
        CGContextSaveGState(ctx);
        CGContextSetAlpha(ctx, alpha);
        CGContextBeginTransparencyLayer(ctx, nullptr);
        strokePath(ctx, outer, col(kInk), 2 * 2.2);
        fillPath(ctx, outer, col(pal.accent));
        fillPath(ctx, inner, col(kCream));
        strokePath(ctx, inner, col(pal.shadow, 0.8), 1.0);
        CGContextEndTransparencyLayer(ctx);
        CGContextRestoreGState(ctx);
        CGPathRelease(outer);
        CGPathRelease(inner);
    }

    // 2. Eight radial action ticks (0.02 .. 0.30 s), tapered, staggered angles and lengths.
    for (int k = 0; k < 8; ++k) {
        const double start = 0.02 + 0.012 * (k % 3), f = (t - start) / 0.28;
        if (f <= 0 || f >= 1)
            continue;
        const double a = k * M_PI / 4 + M_PI / 8 + 0.18 * (hash01(k + 3) - 0.5);
        const double reach = 0.75 + 0.25 * hash01(k + 11);
        const double r1 = 10 + 26 * reach * easeOutCubic(f), r0 = 10 + 26 * reach * easeOutCubic(std::max(0.0, f - 0.35));
        if (r1 - r0 < 0.5)
            continue;
        const QPointF d(std::cos(a), std::sin(a)), n(-d.y(), d.x());
        const QPointF p0 = c + d * r0, p1 = c + d * r1;
        const double w1 = 3.2 * (1 - 0.5 * f), w0 = 0.8;
        CGMutablePathRef tick = CGPathCreateMutable();
        const QPointF q[4] = {p0 + n * (w0 / 2), p1 + n * (w1 / 2), p1 - n * (w1 / 2), p0 - n * (w0 / 2)};
        CGPathMoveToPoint(tick, nullptr, q[0].x(), q[0].y());
        CGPathAddLineToPoint(tick, nullptr, q[1].x(), q[1].y());
        CGPathAddArcToPoint(tick, nullptr, (p1 + d * w1).x(), (p1 + d * w1).y(), q[2].x(), q[2].y(), w1 / 2);
        CGPathAddLineToPoint(tick, nullptr, q[2].x(), q[2].y());
        CGPathAddLineToPoint(tick, nullptr, q[3].x(), q[3].y());
        CGPathCloseSubpath(tick);
        const double alpha = f < 0.7 ? 1 : 1 - (f - 0.7) / 0.3;
        CGContextSaveGState(ctx);
        CGContextSetAlpha(ctx, alpha);
        strokePath(ctx, tick, col(kInk), 2.4);
        fillPath(ctx, tick, k % 2 ? col(kCream) : col(pal.accent));
        CGContextRestoreGState(ctx);
        CGPathRelease(tick);
    }

    // 3. Four flying stars on falling arcs and three smoke puffs drifting off.
    for (int k = 0; k < 4; ++k) {
        const double start = 0.04 + 0.02 * k, f = (t - start) / (kBurst - start);
        if (f <= 0 || f >= 1)
            continue;
        const double a = -M_PI / 2 + (k - 1.5) * 0.95 + 0.2 * (hash01(k + 41) - 0.5);
        const double dist = 12 + (26 + 8 * hash01(k + 51)) * easeOutCubic(f);
        const QPointF at = c + QPointF(std::cos(a), std::sin(a)) * dist + QPointF(0, 34 * f * f);
        const double R = 5.5 * std::min(1.0, f / 0.12) * (1 - f);
        drawStar(ctx, at.toCGPoint(), R, f * 8 + k, col(k % 2 ? kCream : pal.accent), 1.1, 1);
    }
    for (int k = 0; k < 3; ++k) {
        const double start = 0.06, f = (t - start) / 0.34;
        if (f <= 0 || f >= 1)
            continue;
        const double a = M_PI * (0.85 + 0.65 * k / 2.0);
        const QPointF at = c + QPointF(std::cos(a), std::sin(a)) * (13 + 14 * easeOutCubic(f));
        drawPuff(ctx, at.toCGPoint(), 2.6 + 3.4 * easeOutCubic(f), k * 1.3 + f, std::pow(1 - f, 1.3));
    }
    CGContextRestoreGState(ctx);
}

// The status speech bubble: obsidian body, ink outline, accent inner rim, a tail toward the target.
- (void)paintPill:(CGContextRef)ctx now:(double)now
{
    using namespace studio::cua::detail;
    const Scene &s = *_scene;
    const QRectF pr = s.pillRect(now);
    if (pr.isEmpty())
        return;
    const Palette pal = palette(s.gold);
    const QRectF anchor = s.pillAnchor(now);
    const int tail = pr.top() >= anchor.bottom() - 0.5 ? -1 : (pr.bottom() <= anchor.top() + 0.5 ? 1 : 0);
    const CGRect r = pr.toCGRect();
    const double rad = r.size.height / 2;
    const double aimX = s.focusShown(now) ? s.focus.left() + std::min(26.0, s.focus.width() / 2)
                                          : (s.hasTip ? s.tip(now).x() + 4 : pr.left());
    const double tailX = std::max(r.origin.x + rad + 5, std::min(r.origin.x + r.size.width - rad - 5, aimX));
    CGPathRef body = bubblePath(r, rad, tailX, tail, 7);
    CGContextSaveGState(ctx);
    CGContextSetAlpha(ctx, s.chromeAlpha(now));
    CGContextBeginTransparencyLayer(ctx, nullptr);
    roundJoins(ctx);
    CGContextSaveGState(ctx);
    [self shadow:ctx color:rgba(0, 0, 0, 0.5) blur:7 dy:-2.5];
    strokePath(ctx, body, col(kInk), 2 * 2.2);
    CGContextRestoreGState(ctx);
    fillPath(ctx, body, col(kObsidian));
    CGContextSaveGState(ctx);
    CGContextAddPath(ctx, body);
    CGContextClip(ctx);
    strokePath(ctx, body, col(pal.accent), 2 * 1.4);
    CGContextRestoreGState(ctx);
    // A bouncing, spinning star icon on the left.
    const double since = std::max(0.0, now - s.lastActivity);
    const double bob = 1.4 * std::abs(std::sin(since * 2 * M_PI / 0.9));
    drawStar(ctx, CGPointMake(r.origin.x + 15, r.origin.y + rad - bob + 0.5), 6.2, since * 2.2, col(pal.accent), 1.1, 1);
    // The label: Avenir Next DemiBold 12.5 pt parchment.
    NSFont *font = Scene::labelFont();
    const double textH = std::ceil(font.ascender - font.descender);
    [s.label.toNSString() drawInRect:NSMakeRect(r.origin.x + 28, r.origin.y + (r.size.height - textH) / 2,
                                                r.size.width - 28 - 12, textH)
                      withAttributes:Scene::labelAttrs()];
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);
    CGPathRelease(body);
}

// The cartoon glove at the fingertip, banked, stretched along the motion, squashed by pokes.
- (void)paintPointer:(CGContextRef)ctx now:(double)now
{
    const Scene &s = *_scene;
    if (!s.hasTip)
        return;
    CGContextSaveGState(ctx);
    CGContextConcatCTM(ctx, handTransform(s.pose(now)));
    paintGlove(ctx, palette(s.gold), _shadowScale);
    CGContextRestoreGState(ctx);
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

// ---- HUD bar -----------------------------------------------------------------------------------

namespace {
NSFont *hudFont()
{
    NSFont *f = [NSFont fontWithName:@"AvenirNext-Medium" size:13];
    return f ? f : [NSFont systemFontOfSize:13 weight:NSFontWeightMedium];
}

// The capsule in global points (the caller's CTM maps global points to its surface).
void paintHud(CGContextRef ctx, const Scene &s, double now, double shadowScale)
{
    using namespace studio::cua::detail;
    using studio::cua::HudTone;
    const double alpha = s.hudAlpha(now);
    if (alpha <= 0 || s.hudShownAt == kNever)
        return;
    const Palette pal = palette(s.gold);
    const auto shadow = [&](CGColorRef color, double blur, double dy) {
        CGContextSetShadowWithColor(ctx, CGSizeMake(0, dy * shadowScale), blur * shadowScale, color);
    };
    const CGRect r = s.hudRect().translated(0, 6 * (1 - s.hudIn(now))).toCGRect();  // rises 6 pt as it fades in
    const double radius = r.size.height / 2;
    CGContextSaveGState(ctx);
    CGContextSetAlpha(ctx, alpha);
    CGContextBeginTransparencyLayer(ctx, nullptr);
    roundJoins(ctx);
    CGPathRef body = CGPathCreateWithRoundedRect(r, radius, radius, nullptr);
    // Espresso-obsidian body over a soft shadow, then the warm accent rim.
    CGContextSaveGState(ctx);
    shadow(rgba(0, 0, 0, 0.55), 18, -6);
    fillPath(ctx, body, col(kObsidian, 0.95));
    CGContextRestoreGState(ctx);
    CGPathRef rim = CGPathCreateWithRoundedRect(CGRectInset(r, 0.75, 0.75), radius - 0.75, radius - 0.75, nullptr);
    strokePath(ctx, rim, col(pal.accent, 0.9), 1.5);
    CGPathRelease(rim);
    CGPathRelease(body);

    // The status mark.
    const CGPoint dot = CGPointMake(r.origin.x + 20, r.origin.y + r.size.height / 2);
    const double since = std::max(0.0, now - s.hudShownAt);
    CGContextSaveGState(ctx);
    switch (s.hudTone) {
    case HudTone::Done:
        shadow(col(pal.accent, 0.6), 5, 0);
        CGContextSetLineWidth(ctx, 2.4);
        CGContextSetStrokeColorWithColor(ctx, col(pal.accent));
        CGContextMoveToPoint(ctx, dot.x - 5, dot.y + 0.2);
        CGContextAddLineToPoint(ctx, dot.x - 1.6, dot.y + 3.8);
        CGContextAddLineToPoint(ctx, dot.x + 5.2, dot.y - 4);
        CGContextStrokePath(ctx);
        break;
    case HudTone::Error:
        shadow(col(kCoral, 0.5), 5, 0);
        CGContextSetFillColorWithColor(ctx, col(kCoral));
        CGContextFillEllipseInRect(ctx, CGRectMake(dot.x - 6, dot.y - 6, 12, 12));
        shadow(nullptr, 0, 0);
        CGContextSetStrokeColorWithColor(ctx, col(kObsidian));
        CGContextSetLineWidth(ctx, 1.9);
        CGContextMoveToPoint(ctx, dot.x, dot.y - 3.2);
        CGContextAddLineToPoint(ctx, dot.x, dot.y + 0.6);
        CGContextStrokePath(ctx);
        CGContextSetFillColorWithColor(ctx, col(kObsidian));
        CGContextFillEllipseInRect(ctx, CGRectMake(dot.x - 1.1, dot.y + 2.2, 2.2, 2.2));
        break;
    case HudTone::Listening: {
        const double phase = std::fmod(since, 1.2) / 1.2, breathe = 0.5 + 0.5 * std::cos(2 * M_PI * phase);
        const double hr = 4.5 + 7 * easeOutCubic(phase);
        CGContextSetLineWidth(ctx, 1.3);
        CGContextSetStrokeColorWithColor(ctx, col(pal.accent, 0.55 * (1 - phase)));
        CGContextStrokeEllipseInRect(ctx, CGRectMake(dot.x - hr, dot.y - hr, 2 * hr, 2 * hr));
        shadow(col(pal.accent, 0.4 + 0.4 * breathe), 3 + 4 * breathe, 0);
        const double dr = 4 + 0.6 * breathe;
        CGContextSetFillColorWithColor(ctx, col(pal.accent));
        CGContextFillEllipseInRect(ctx, CGRectMake(dot.x - dr, dot.y - dr, 2 * dr, 2 * dr));
        break;
    }
    case HudTone::Working: {
        const double beat = 0.5 + 0.5 * std::cos(2 * M_PI * std::fmod(since, 0.9) / 0.9);
        shadow(col(pal.accent, 0.35 + 0.35 * beat), 3 + 3 * beat, 0);
        drawStar(ctx, dot, 6.4 + 0.8 * beat, since * 2.6, col(pal.accent), 0, 1);
        break;
    }
    }
    CGContextRestoreGState(ctx);

    // A trailing "L3+R3 Run" becomes vector badges on the right: [L3] + [R3] Run.
    QString text = s.hudStatus;
    bool runBadge = false;
    static const QString kRunSuffix = QStringLiteral("L3+R3 Run");
    if (text.endsWith(kRunSuffix)) {
        runBadge = true;
        text = text.left(text.size() - kRunSuffix.size()).trimmed();
        while (!text.isEmpty() && (text.endsWith(QChar(0x00B7)) || text.endsWith(QLatin1Char('-'))
                                   || text.endsWith(QChar(0x2014)) || text.endsWith(QLatin1Char('|'))
                                   || text.endsWith(QLatin1Char(':'))))
            text = text.left(text.size() - 1).trimmed();
    }
    double textRight = r.origin.x + r.size.width - 18;
    if (runBadge) {
        NSFont *mono = [NSFont fontWithName:@"Menlo-Regular" size:11] ?: [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular];
        NSFont *monoBold = [NSFont fontWithName:@"Menlo-Bold" size:11] ?: [NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightBold];
        NSColor *dim = [NSColor colorWithSRGBRed:0x6E / 255.0 green:0x66 / 255.0 blue:0x5E / 255.0 alpha:1];
        NSColor *acc = [NSColor colorWithSRGBRed:pal.accent.r / 255.0 green:pal.accent.g / 255.0 blue:pal.accent.b / 255.0 alpha:1];
        NSDictionary *plusAttrs = @{NSFontAttributeName: mono, NSForegroundColorAttributeName: dim};
        NSDictionary *runAttrs = @{NSFontAttributeName: monoBold, NSForegroundColorAttributeName: acc};
        const double icon = 20, cy = r.origin.y + r.size.height / 2;
        const NSSize runSz = [@"Run" sizeWithAttributes:runAttrs], plusSz = [@"+" sizeWithAttributes:plusAttrs];
        double x = r.origin.x + r.size.width - 16 - std::ceil(runSz.width);
        [@"Run" drawAtPoint:NSMakePoint(x, cy - runSz.height / 2) withAttributes:runAttrs];
        x -= 6 + icon;
        const studio::cua::PadIconColors colors = studio::cua::padIconColors(false, !s.gold);
        studio::cua::drawPadIcon(ctx, studio::cua::PadButton::R3, CGRectMake(x, cy - icon / 2, icon, icon), colors);
        x -= 3 + std::ceil(plusSz.width);
        [@"+" drawAtPoint:NSMakePoint(x, cy - plusSz.height / 2) withAttributes:plusAttrs];
        x -= 3 + icon;
        studio::cua::drawPadIcon(ctx, studio::cua::PadButton::L3, CGRectMake(x, cy - icon / 2, icon, icon), colors);
        textRight = x - 10;
    }

    // Status text, Avenir Next Medium 13 pt parchment, truncated at the right padding.
    NSMutableParagraphStyle *style = [[[NSMutableParagraphStyle alloc] init] autorelease];
    style.lineBreakMode = NSLineBreakByTruncatingTail;
    NSFont *font = hudFont();
    NSDictionary *attrs = @{NSFontAttributeName: font,
                            NSForegroundColorAttributeName: [NSColor colorWithSRGBRed:kParchment.r / 255.0 green:kParchment.g / 255.0
                                                                                 blue:kParchment.b / 255.0 alpha:1],
                            NSParagraphStyleAttributeName: style};
    const double textH = std::ceil(font.ascender - font.descender);
    const double textX = r.origin.x + 36;
    if (textRight > textX)
        [text.toNSString() drawInRect:NSMakeRect(textX, r.origin.y + (r.size.height - textH) / 2,
                                                 textRight - textX, textH)
                       withAttributes:attrs];
    CGContextEndTransparencyLayer(ctx);
    CGContextRestoreGState(ctx);
}
}

@interface LapisCuaHudView : NSView {
@public
    const Scene *_scene;
    QPointF _origin;  // global point at the view's top left
}
@end

@implementation LapisCuaHudView
- (BOOL)isFlipped { return YES; }
- (BOOL)isOpaque { return NO; }
- (NSView *)hitTest:(NSPoint)point { (void)point; return nil; }
- (void)drawRect:(NSRect)dirtyRect
{
    CGContextRef ctx = [NSGraphicsContext currentContext].CGContext;
    CGContextClearRect(ctx, dirtyRect);
    if (!_scene)
        return;
    CGContextSaveGState(ctx);
    CGContextTranslateCTM(ctx, -_origin.x(), -_origin.y());
    paintHud(ctx, *_scene, _scene->now(), 1);
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
    LapisCuaPanel *hudPanel = nil;
    LapisCuaHudView *hudView = nil;
    NSTimer *timer = nil;
    QRectF lastBounds;

    static QRectF primaryDisplay() {
        for (const auto &d : listDisplays())
            if (d.primary)
                return d.globalPoints;
        return {};
    }
    // The display the HUD pins to when it first appears: the cursor overlay's display while it is
    // up, else the one under the mouse, else the primary.
    QRectF hudHomeDisplay() const {
        if (scene.offscreen || scene.visible)
            return scene.display;
        const QRectF primary = primaryDisplay();
        if (primary.isValid()) {
            const NSPoint m = [NSEvent mouseLocation];  // AppKit: y up from the primary's bottom
            const QPointF global(m.x, primary.y() + primary.height() - m.y);
            for (const auto &d : listDisplays())
                if (d.globalPoints.adjusted(0, 0, 1, 1).contains(global))
                    return d.globalPoints;
        }
        return primary;
    }
    void placeHud() {
        if (!hudPanel)
            return;
        const QRectF primary = primaryDisplay();
        const QRectF frame = scene.hudRect().adjusted(-detail::kHudPad, -detail::kHudPad, detail::kHudPad, detail::kHudPad);
        if (!primary.isValid() || frame.isEmpty())
            return;
        const QPointF bl = globalPointToAppKit(QPointF(frame.left(), frame.bottom()), primary);
        [hudPanel setFrame:NSMakeRect(bl.x(), bl.y(), frame.width(), frame.height()) display:NO];
        hudView.frame = NSMakeRect(0, 0, frame.width(), frame.height());
        hudView->_origin = frame.topLeft();
        [hudView setNeedsDisplay:YES];
    }
    void ensureTimer() {
        if (scene.offscreen || timer)
            return;
        Impl *owner = this;  // the timer is invalidated in ~CuaReticle, before Impl dies
        timer = [[NSTimer timerWithTimeInterval:1.0 / 120 repeats:YES block:^(NSTimer *t) { (void)t; owner->tick(); }] retain];
        timer.tolerance = 0;
        [[NSRunLoop mainRunLoop] addTimer:timer forMode:NSRunLoopCommonModes];
    }
    void hudTick(double now) {
        if (!hudPanel)
            return;
        if (scene.hudGone(now)) {
            if (hudPanel.visible)
                [hudPanel orderOut:nil];
            return;
        }
        if (scene.hudDynamic(now))
            [hudView setNeedsDisplay:YES];
    }

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
        const QRectF primary = primaryDisplay();
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
            scene.moveStart = detail::kNever;
        } else {
            const QPointF cur = scene.tip(now);
            scene.from = cur;
            scene.to = to;
            scene.ctrl = glideControlPoint(cur, to);
            scene.glideDur = glideDuration(cur, to);
            scene.moveStart = std::hypot(to.x() - cur.x(), to.y() - cur.y()) < 2 ? detail::kNever : now;
            if (scene.moveStart == detail::kNever)
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
        ensureTimer();
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
        if (scene.visible)
            invalidate(now, false);
        hudTick(now);
        if (!scene.animating(now) && !scene.hudAnimating(now))
            stopTimer();
    }
    void stopTimer() {
        [timer invalidate];
        [timer release];
        timer = nil;
    }
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
        const auto makePanel = [](NSRect frame) {
            LapisCuaPanel *p = [[LapisCuaPanel alloc] initWithContentRect:frame
                styleMask:NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel
                backing:NSBackingStoreBuffered defer:NO];
            p.opaque = NO;
            p.backgroundColor = [NSColor clearColor];
            p.hasShadow = NO;  // drawn soft in the view
            p.ignoresMouseEvents = YES;  // click-through: never blocks the desktop under it
            p.hidesOnDeactivate = NO;
            p.releasedWhenClosed = NO;
            p.level = NSStatusWindowLevel;
            p.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces |
                NSWindowCollectionBehaviorFullScreenAuxiliary | NSWindowCollectionBehaviorStationary |
                NSWindowCollectionBehaviorIgnoresCycle;
            p.sharingType = NSWindowSharingNone;  // and excluded from captures by window id
            return p;
        };
        impl->panel = makePanel(NSMakeRect(0, 0, 240, 72));
        impl->panel.contentView = impl->view;
        const QRectF primary = Impl::primaryDisplay();
        if (primary.isValid())
            impl->place(primary);

        // The HUD bar: its own small panel, so it can stay pinned while the cursor overlay
        // follows the target onto another display.
        impl->hudView = [[LapisCuaHudView alloc] initWithFrame:NSMakeRect(0, 0, kHudWidth, kHudHeight)];
        impl->hudView->_scene = &impl->scene;
        impl->hudPanel = makePanel(NSMakeRect(0, 0, kHudWidth, kHudHeight));
        impl->hudPanel.contentView = impl->hudView;
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
    if (impl->hudView)
        impl->hudView->_scene = nullptr;
    if (impl->hudPanel) {
        [impl->hudPanel orderOut:nil];
        [impl->hudPanel close];
        [impl->hudPanel release];
    }
    [impl->hudView release];
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
        impl->scene.rippleStart = std::max(now, impl->scene.landAt());
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
    if (!impl->scene.hudAnimating(impl->scene.now()))
        impl->stopTimer();  // the HUD keeps its own animation and fade going
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

void CuaReticle::setGoldTheme(bool gold)
{
    requireMainThread();
    if (impl->scene.gold == gold)
        return;
    impl->scene.gold = gold;
    if (impl->view)
        [impl->view setNeedsDisplay:YES];
    if (impl->hudView)
        [impl->hudView setNeedsDisplay:YES];
}

bool CuaReticle::goldTheme() const
{
    requireMainThread();
    return impl->scene.gold;
}

void CuaReticle::showHud(const QString &status, HudTone tone)
{
    requireMainThread();
    @autoreleasepool {
        Scene &s = impl->scene;
        const double now = s.now();
        const bool wasUp = !s.hudGone(now);
        if (!wasUp) {
            s.hudDisplay = impl->hudHomeDisplay();
            s.hudShownAt = now;
        } else if (now >= s.hudHideAt) {
            // Caught mid fade-out: resume from the current alpha instead of popping.
            s.hudShownAt = now - detail::kHudIn * s.hudAlpha(now);
        }
        if (s.hudDisplay.isEmpty())
            return;
        s.hudStatus = status.simplified().left(160);
        s.hudTone = tone;
        s.hudHideAt = tone == HudTone::Done || tone == HudTone::Error ? now + detail::kHudLinger : detail::kForever;
        if (impl->hudPanel) {
            if (!wasUp)
                impl->placeHud();  // pinned: the frame only moves when the HUD reappears
            [impl->hudView setNeedsDisplay:YES];
            if (!impl->hudPanel.visible)
                [impl->hudPanel orderFrontRegardless];
            impl->ensureTimer();
        }
    }
}

void CuaReticle::hideHud(double afterSeconds)
{
    requireMainThread();
    Scene &s = impl->scene;
    const double now = s.now();
    if (s.hudGone(now))
        return;
    const double at = now + (std::isfinite(afterSeconds) ? std::max(0.0, afterSeconds) : 0.0);
    s.hudHideAt = std::min(s.hudHideAt, at);
    impl->ensureTimer();
}

uint32_t CuaReticle::hudWindowNumber() const
{
    requireMainThread();
    return impl->hudPanel ? uint32_t(impl->hudPanel.windowNumber) : 0;
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
        st.rippleActive = s.visible && now >= s.rippleStart && s.burstPending(now);
        st.label = s.label;
        st.pillRect = s.visible ? s.pillRect(now) : QRectF();
        st.overlayFrame = s.display;
        if (s.visible && s.hasTip) {
            const Pose ps = s.pose(now);
            st.handAngleDeg = ps.angle;
            st.stretch = ps.along;
        }
        st.hudVisible = !s.hudGone(now);
        st.hudStatus = st.hudVisible ? s.hudStatus : QString();
        st.hudTone = s.hudTone;
        st.hudRect = st.hudVisible ? s.hudRect() : QRectF();
        st.hudAlpha = s.hudAlpha(now);
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
        if (impl->scene.hudDisplay == d) {
            CGContextSaveGState(ctx);
            CGContextTranslateCTM(ctx, -d.x(), -d.y());
            paintHud(ctx, impl->scene, impl->scene.now(), scale);
            CGContextRestoreGState(ctx);
        }
        [NSGraphicsContext setCurrentContext:previous];
        [previous release];
        CGContextRelease(ctx);
    }
    img.setDevicePixelRatio(scale);
    return img;
}

// ---- --snap-cursor -----------------------------------------------------------------------------

namespace {
// A warm, dark Lapis-toned desktop with one app window and a "Send" button: context only.
void paintDesktop(QPainter &p, const QRectF &frame, const QRectF &button)
{
    p.fillRect(frame, QColor("#0d0c0b"));
    const QRectF win = frame.adjusted(56, 48, -56, -48);
    p.setPen(QPen(QColor("#1d1915"), 1));
    p.setBrush(QColor("#14120f"));
    p.drawRoundedRect(win, 12, 12);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor("#1a1714"));
    p.drawRoundedRect(QRectF(win.left(), win.top(), win.width(), 34), 12, 12);
    p.fillRect(QRectF(win.left(), win.top() + 22, win.width(), 12), QColor("#1a1714"));
    for (int i = 0; i < 3; ++i) {
        p.setBrush(QColor(0xe8, 0xd5, 0xb5, 40));
        p.drawEllipse(QPointF(win.left() + 18 + i * 16, win.top() + 17), 5, 5);
    }
    p.setBrush(QColor(0xe8, 0xd5, 0xb5, 16));
    for (int i = 0; i < 4; ++i)
        p.drawRoundedRect(QRectF(win.left() + 28, win.top() + 62 + i * 30, (i % 2 ? 300 : 380), 12), 6, 6);
    p.setBrush(QColor("#241f1a"));
    p.setPen(QPen(QColor(0xe8, 0xd5, 0xb5, 40), 1));
    p.drawRoundedRect(button, 8, 8);
    QFont f(QStringLiteral("Avenir Next"));
    f.setPixelSize(14);
    f.setWeight(QFont::DemiBold);
    p.setFont(f);
    p.setPen(QColor("#e8d5b5"));
    p.drawText(button, Qt::AlignCenter, QStringLiteral("Send"));
}

// A comic caption box in the panel's top-left corner ("SWOOSH!", "POW!").
void paintCaption(QPainter &p, const QRectF &panel, const QString &text)
{
    QFont f(QStringLiteral("Avenir Next"));
    f.setPixelSize(17);
    f.setWeight(QFont::Black);
    f.setItalic(true);
    f.setLetterSpacing(QFont::AbsoluteSpacing, 1);
    p.setFont(f);
    const double w = QFontMetricsF(f).horizontalAdvance(text) + 22;
    const QRectF box(panel.left() + 14, panel.top() + 12, w, 28);
    QPainterPath path;
    path.addRoundedRect(box, 5, 5);
    p.save();
    p.translate(box.center());
    p.rotate(-3);
    p.translate(-box.center());
    p.setPen(QPen(QColor("#0a0806"), 4.4, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setBrush(QColor("#e09f3e"));
    p.drawPath(path);
    p.setPen(QColor("#0a0806"));
    p.drawText(box, Qt::AlignCenter, text);
    p.restore();
}
}

int snapCursor(const QString &outPath)
{
    requireMainThread();
    const QRectF frame(0, 0, 640, 400);
    const QRectF button(420, 220, 112, 36);
    const QPointF start(150, 110), click = button.center() + QPointF(-6, 2);
    const QString label = QStringLiteral("Clicking Send in Mail");
    QImage frames[2];
    {
        // Mid-swoosh at peak speed: the hand stretched and banked, the ribbon, speed lines, launch
        // puffs and trail stars behind it, the bracket frame and bubble already on the target.
        CuaReticle r(CuaReticle::Mode::Offscreen, frame);
        r.showHud(QStringLiteral("Mail · Clicking Send"), HudTone::Working);
        r.moveTo(start);
        r.advanceClock(1.0);
        r.highlightRect(button, label);
        r.advanceClock(0.2);
        r.moveTo(click, label);
        r.advanceClock(0.47 * glideDuration(start, click));
        frames[0] = r.render(2);
    }
    {
        // The click: hand squashed into the button, the POW starburst, ticks and stars flying.
        CuaReticle r(CuaReticle::Mode::Offscreen, frame);
        r.moveTo(start);
        r.advanceClock(1.0);
        r.highlightRect(button, label);
        r.showHud(QStringLiteral("Done · Send in Mail"), HudTone::Done);
        r.advanceClock(0.2);
        r.pulseClick(click, label);
        r.advanceClock(glideDuration(start, click) + 0.085);
        frames[1] = r.render(2);
    }
    // The other two HUD tones on their own strip.
    const QRectF stripFrame(0, 0, 640, 110);
    QImage strip[2];
    const std::pair<QString, HudTone> tones[2] = {{QStringLiteral("Listening…"), HudTone::Listening},
                                                  {QStringLiteral("Couldn't find Send in Mail"), HudTone::Error}};
    for (int i = 0; i < 2; ++i) {
        CuaReticle r(CuaReticle::Mode::Offscreen, stripFrame);
        r.showHud(tones[i].first, tones[i].second);
        r.advanceClock(0.5);
        strip[i] = r.render(2);
    }
    if (frames[0].isNull() || frames[1].isNull() || strip[0].isNull() || strip[1].isNull()) {
        std::fprintf(stderr, "snap-cursor: render failed\n");
        return 2;
    }
    const double scale = 2, gutter = 16;
    QImage out(int((frame.width() * 2 + gutter) * scale), int((frame.height() + stripFrame.height()) * scale),
               QImage::Format_ARGB32_Premultiplied);
    out.fill(QColor("#000000"));
    QPainter p(&out);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::TextAntialiasing);
    p.scale(scale, scale);
    const QString captions[2] = {QStringLiteral("SWOOSH!"), QStringLiteral("POW!")};
    for (int i = 0; i < 2; ++i) {
        const QRectF at = frame.translated(i * (frame.width() + gutter), 0);
        p.save();
        p.translate(at.topLeft());
        paintDesktop(p, frame, button);
        p.restore();
        frames[i].setDevicePixelRatio(scale);
        p.drawImage(at, frames[i]);
        paintCaption(p, at, captions[i]);
        const QRectF stripAt = stripFrame.translated(i * (frame.width() + gutter), frame.height());
        p.fillRect(stripAt, QColor("#0d0c0b"));
        strip[i].setDevicePixelRatio(scale);
        p.drawImage(stripAt, strip[i]);
        // Comic panel border.
        p.setPen(QPen(QColor("#e09f3e"), 1.2));
        p.setBrush(Qt::NoBrush);
        p.drawRect(at.adjusted(0.6, 0.6, -0.6, -0.6));
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
