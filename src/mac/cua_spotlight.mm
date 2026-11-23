#include "mac/cua_spotlight.hpp"
#include "mac/cua_pad_icon.hpp"

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

#include <QDir>
#include <QFileInfo>
#include <QPainter>
#include <QVector>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <iterator>
#include <unistd.h>

// ---- Pure helpers --------------------------------------------------------------------------------

namespace studio::cua {

QRectF spotlightFrame(const QRectF &screen, double height)
{
    const double h = std::clamp(height, kSpotlightBaseHeight, kSpotlightMaxHeight);
    const double x = screen.x() + std::round((screen.width() - kSpotlightWidth) / 2);
    const double top = screen.y() + screen.height() - std::round(screen.height() * 0.20);
    return QRectF(x, top - h, kSpotlightWidth, h);
}

double spotlightHeight(bool drawer, double answerHeight)
{
    if (!drawer)
        return kSpotlightBaseHeight;
    // capsule 56 + hairline 1 + gap 9 + question 18 + gap 6 + answer + bottom 12
    const double body = std::clamp(answerHeight, 0.0, kSpotlightAnswerMaxHeight);
    return std::min(kSpotlightMaxHeight, kSpotlightBaseHeight + kSpotlightDrawerChrome + body);
}

QString spotlightHint(SpotlightVoice voice, bool answerShown, bool hasText)
{
    switch (voice) {
    case SpotlightVoice::Held: return QStringLiteral("L2 release");
    case SpotlightVoice::Toggled: return QStringLiteral("Listening · tap L2 to send");
    case SpotlightVoice::Transcribing: return QStringLiteral("Transcribing…");
    case SpotlightVoice::Idle: break;
    }
    if (hasText)
        return QStringLiteral("R2 send");
    return answerShown ? QStringLiteral("○ close") : QStringLiteral("L2 voice");
}

QString spotlightPlaceholder(const QString &context, SpotlightVoice voice)
{
    switch (voice) {
    case SpotlightVoice::Held: return QStringLiteral("Speak or type…");
    case SpotlightVoice::Toggled: return QStringLiteral("Tap L2 to send…");
    case SpotlightVoice::Transcribing: return QStringLiteral("Transcribing…");
    case SpotlightVoice::Idle: break;
    }
    return context.isEmpty() ? QStringLiteral("Ask or control anything…") : QStringLiteral("Ask or control %1…").arg(context);
}

QString spotlightContextLabel(const QString &appName)
{
    QString n = appName.simplified();
    if (n.startsWith(QLatin1String("Google ")) && n.size() > 7)
        n = n.mid(7);
    return n;
}

bool studioAppIsActive()
{
    return NSApp && [NSApp isActive];
}

}  // namespace studio::cua

// ---- Lapis palette and the shared painter ---------------------------------------------------------

namespace {

std::atomic<bool> g_lapisBlue{false};

NSColor *hex(uint32_t rgb, double a = 1.0)
{
    return [NSColor colorWithSRGBRed:((rgb >> 16) & 0xFF) / 255.0 green:((rgb >> 8) & 0xFF) / 255.0
                                blue:(rgb & 0xFF) / 255.0 alpha:a];
}

NSString *ns(const QString &s)
{
    return s.toNSString();
}

// Lapis Gold (qml/Theme.qml); lapis-blue mode swaps only the accent.
constexpr uint32_t kCard = 0x0D0C0B, kCardHi = 0x14120F, kHairline = 0x1D1915;
constexpr uint32_t kInk = 0xE8D5B5, kBodyText = 0xD9CFC3, kDim = 0x6E665E, kMuted = 0x4C453D;
constexpr uint32_t kGold = 0xE09F3E, kBlue = 0x4F8CFF;
constexpr double kW = studio::cua::kSpotlightWidth;
constexpr double kBarH = studio::cua::kSpotlightBaseHeight;
// The capsule's row: the status mark centred at x 22, the pill from x 38, the badge 16 pt in
// from the right, everything centred on y 28.
constexpr double kMarkCx = 22, kRowCy = kBarH / 2, kPillX = 38, kPillH = 22, kBadgeIcon = 16, kBadgeH = 24;
constexpr double kPromptH = 26;

uint32_t accentRgb() { return g_lapisBlue.load() ? kBlue : kGold; }

NSFont *avenir(NSString *name, double size, NSFontWeight fallback)
{
    NSFont *f = [NSFont fontWithName:name size:size];
    return f ?: [NSFont systemFontOfSize:size weight:fallback];
}
NSFont *menlo(double size, bool bold = false)
{
    NSFont *f = [NSFont fontWithName:bold ? @"Menlo-Bold" : @"Menlo-Regular" size:size];
    return f ?: [NSFont monospacedSystemFontOfSize:size weight:bold ? NSFontWeightBold : NSFontWeightRegular];
}
NSFont *promptFont() { return avenir(@"AvenirNext-Regular", 17.5, NSFontWeightRegular); }

void fillRounded(CGContextRef ctx, CGRect r, double radius, NSColor *fill, NSColor *stroke = nil, double lw = 1)
{
    if (fill) {
        CGPathRef p = CGPathCreateWithRoundedRect(r, radius, radius, nullptr);
        CGContextAddPath(ctx, p);
        CGContextSetFillColorWithColor(ctx, fill.CGColor);
        CGContextFillPath(ctx);
        CGPathRelease(p);
    }
    if (stroke) {
        const CGRect e = CGRectInset(r, lw / 2, lw / 2);
        const double er = std::max(0.0, radius - lw / 2);
        CGPathRef p = CGPathCreateWithRoundedRect(e, er, er, nullptr);
        CGContextAddPath(ctx, p);
        CGContextSetLineWidth(ctx, lw);
        CGContextSetStrokeColorWithColor(ctx, stroke.CGColor);
        CGContextStrokePath(ctx);
        CGPathRelease(p);
    }
}

// ---- The right badge: one controller icon + a word ------------------------------------------------
// Held [L2] release, Toggled [L2] send, Transcribing "Transcribing…" (no icon), an answer [○] close,
// typed text [R2] send, else [L2] voice.
struct Badge {
    bool icon = true;
    studio::cua::PadButton button = studio::cua::PadButton::L2;
    NSString *label = @"voice";
    bool accent = false;  // the word in the accent (live states), else dim
};
Badge badgeFor(studio::cua::SpotlightVoice v, bool answerShown, bool hasText)
{
    using V = studio::cua::SpotlightVoice;
    if (v == V::Held)
        return {true, studio::cua::PadButton::L2, @"release", true};
    if (v == V::Toggled)
        return {true, studio::cua::PadButton::L2, @"send", true};
    if (v == V::Transcribing)
        return {false, studio::cua::PadButton::L2, @"Transcribing…", true};
    if (hasText)
        return {true, studio::cua::PadButton::R2, @"send", false};
    if (answerShown)
        return {true, studio::cua::PadButton::Circle, @"close", false};
    return {true, studio::cua::PadButton::L2, @"voice", false};
}
constexpr double kBadgeGap = 6;
NSDictionary *badgeAttrs(const Badge &b)
{
    return @{NSFontAttributeName: menlo(11), NSForegroundColorAttributeName: b.accent ? hex(accentRgb(), 0.95) : hex(kDim)};
}
double badgeWidth(const Badge &b)
{
    const double text = std::ceil([b.label sizeWithAttributes:badgeAttrs(b)].width);
    return b.icon ? studio::cua::padIconSize(b.button, kBadgeIcon).width + kBadgeGap + text : text;
}
// Draws the badge right-aligned in `r` (y-down, an NSGraphicsContext current).
void paintBadge(CGContextRef ctx, CGRect r, const Badge &b)
{
    NSDictionary *attrs = badgeAttrs(b);
    const double cy = CGRectGetMidY(r);
    const double lineH = std::ceil(menlo(11).ascender - menlo(11).descender);
    double x = CGRectGetMaxX(r) - badgeWidth(b);
    if (b.icon) {
        // The live states light the icon in the accent; at rest it is warm ink.
        const studio::cua::PadIconColors colors = studio::cua::padIconColors(b.accent, g_lapisBlue.load());
        const CGSize sz = studio::cua::padIconSize(b.button, kBadgeIcon);
        studio::cua::drawPadIcon(ctx, b.button, CGRectMake(x, cy - sz.height / 2, sz.width, sz.height), colors);
        x += sz.width + kBadgeGap;
    }
    [b.label drawAtPoint:NSMakePoint(x, cy - lineH / 2) withAttributes:attrs];
}

// ---- The left status mark -----------------------------------------------------------------------
// Listening: a 10 pt breathing accent orb with a ripple ring. Working (transcribing, thinking,
// acting): a spinning 4-point accent star. Idle: a steady dim accent dot. `phase` in [0, 1).
enum class Mark { Idle, Listening, Working };
void paintMark(CGContextRef ctx, CGPoint c, Mark m, double phase)
{
    CGContextSaveGState(ctx);
    if (m == Mark::Listening) {
        const double breathe = 0.5 + 0.5 * std::cos(2 * M_PI * phase);
        const double ring = 5 + 7 * (1 - std::pow(1 - phase, 3));
        CGContextSetLineWidth(ctx, 1.2);
        CGContextSetStrokeColorWithColor(ctx, hex(accentRgb(), 0.55 * (1 - phase)).CGColor);
        CGContextStrokeEllipseInRect(ctx, CGRectMake(c.x - ring, c.y - ring, 2 * ring, 2 * ring));
        const double r = 4.4 + 0.6 * breathe;
        CGContextSetShadowWithColor(ctx, CGSizeZero, 3 + 5 * breathe, hex(accentRgb(), 0.45 + 0.4 * breathe).CGColor);
        CGContextSetFillColorWithColor(ctx, hex(accentRgb(), 0.88 + 0.12 * breathe).CGColor);
        CGContextFillEllipseInRect(ctx, CGRectMake(c.x - r, c.y - r, 2 * r, 2 * r));
    } else if (m == Mark::Working) {
        // Four long points, four short notches, turning once every period.
        const double rot = 2 * M_PI * phase, R = 6.2, r = 1.9;
        CGMutablePathRef path = CGPathCreateMutable();
        for (int k = 0; k < 8; ++k) {
            const double a = rot + k * M_PI / 4 - M_PI / 2;
            const double rad = (k % 2 == 0) ? R : r;
            const CGPoint pt = CGPointMake(c.x + rad * std::cos(a), c.y + rad * std::sin(a));
            if (k == 0)
                CGPathMoveToPoint(path, nullptr, pt.x, pt.y);
            else
                CGPathAddLineToPoint(path, nullptr, pt.x, pt.y);
        }
        CGPathCloseSubpath(path);
        CGContextSetShadowWithColor(ctx, CGSizeZero, 4, hex(accentRgb(), 0.55).CGColor);
        CGContextAddPath(ctx, path);
        CGContextSetFillColorWithColor(ctx, hex(accentRgb()).CGColor);
        CGContextFillPath(ctx);
        CGPathRelease(path);
    } else {
        CGContextSetFillColorWithColor(ctx, hex(accentRgb(), 0.42).CGColor);
        CGContextFillEllipseInRect(ctx, CGRectMake(c.x - 4, c.y - 4, 8, 8));
    }
    CGContextRestoreGState(ctx);
}

studio::cua::CuaSpotlight *g_liveSpotlight = nullptr;  // the Visible instance (capture exclusion)
studio::cua::CuaSpotlight::Impl *g_liveImpl = nullptr;  // ... and its state, for theme refreshes

}  // namespace

@interface LapisSpotlightPanel : NSPanel
@property(nonatomic, assign) studio::cua::CuaSpotlight::Impl *owner;
@end

// The card itself: everything that is not a live control, drawn by the same painter render() uses.
@interface LapisSpotlightSurface : NSView
@property(nonatomic, assign) studio::cua::CuaSpotlight::Impl *owner;
@end

// The animated bits (the status mark and the badge), redrawn alone by the pulse timer.
@interface LapisSpotlightAnimView : NSView
@property(nonatomic, assign) studio::cua::CuaSpotlight::Impl *owner;
@property(nonatomic, assign) BOOL isMark;  // the status mark, else the badge
@property(nonatomic, assign) double phase;
@end

@interface LapisSpotlightFieldDelegate : NSObject <NSTextFieldDelegate>
@property(nonatomic, assign) studio::cua::CuaSpotlight::Impl *owner;
@end

namespace studio::cua {

struct CuaSpotlight::Impl {
    // Where everything sits for the current state (flipped card points).
    struct Layout {
        bool drawer = false;
        bool pill = false;
        CGRect pillRect = CGRectZero;
        CGRect badgeRect = CGRectZero;
        CGRect promptRect = CGRectZero;  // where the inline field sits
        double answerShown = 0;
        double height = kSpotlightBaseHeight;
    };

    CuaSpotlight *q = nullptr;
    bool offscreen = false;
    SpotlightState s;
    Layout lay;
    std::function<void(int *, QString *)> probe;
    QRectF virtualScreen{0, 0, 1440, 900};

    LapisSpotlightPanel *panel = nil;
    LapisSpotlightFieldDelegate *delegate = nil;
    LapisSpotlightSurface *surface = nil;
    LapisSpotlightAnimView *markView = nil;
    LapisSpotlightAnimView *badgeView = nil;
    NSTextField *input = nil;
    NSScrollView *answerScroll = nil;
    NSTextView *answerView = nil;
    NSTimer *pulseTimer = nil;         // redraws the mark and badge while they animate
    NSMutableArray *orderedOut = nil;  // Studio windows ordered out so activation stays on this Space
    id activeObserver = nil;
    id resignObserver = nil;
    id escapeMonitor = nil;            // global Esc while a turn acts (setEscapeWatch)

    // ---- state -------------------------------------------------------------------------------
    bool drawer() const { return !s.lastQuestion.isEmpty() || !s.answer.isEmpty() || !s.statusLine.isEmpty(); }
    QString bodyText() const
    {
        if (!s.answer.isEmpty())
            return s.answer;
        return s.statusLine;
    }
    Mark mark() const
    {
        if (s.voice == SpotlightVoice::Held || s.voice == SpotlightVoice::Toggled)
            return Mark::Listening;
        if (s.voice == SpotlightVoice::Transcribing || s.busy || s.actionRunning)
            return Mark::Working;
        return Mark::Idle;
    }
    Badge badge() const { return badgeFor(s.voice, !s.answer.isEmpty(), !fieldText().trimmed().isEmpty()); }
    bool animating() const { return mark() != Mark::Idle; }
    void syncDerived()
    {
        s.listening = s.voice == SpotlightVoice::Held || s.voice == SpotlightVoice::Toggled;
        s.toggled = s.voice == SpotlightVoice::Toggled;
        s.transcribing = s.voice == SpotlightVoice::Transcribing;
        s.hint = spotlightHint(s.voice, !s.answer.isEmpty(), !fieldText().trimmed().isEmpty());
        s.placeholder = spotlightPlaceholder(s.contextLabel, s.voice);
        s.drawerVisible = drawer();
        if (offscreen) {
            s.key = s.open;
            s.alpha = !s.open ? 0.0 : (s.actionRunning ? 0.0 : 1.0);
        }
    }

    // ---- text styles -------------------------------------------------------------------------
    // The answer: Avenir Next Regular 15 pt #d9cfc3 with 4 pt line spacing. A bare status line
    // ("Thinking…") is the accent while busy, else dim.
    NSAttributedString *bodyAttributed() const
    {
        NSMutableParagraphStyle *style = [[[NSMutableParagraphStyle alloc] init] autorelease];
        style.lineSpacing = 4;
        const bool status = s.answer.isEmpty();
        NSColor *color = status ? ((s.busy || s.actionRunning) ? hex(accentRgb(), 0.9) : hex(kDim)) : hex(kBodyText);
        return [[[NSAttributedString alloc]
            initWithString:ns(bodyText())
                attributes:@{NSForegroundColorAttributeName: color,
                             NSFontAttributeName: avenir(@"AvenirNext-Regular", 15, NSFontWeightRegular),
                             NSParagraphStyleAttributeName: style}] autorelease];
    }
    static double measure(NSAttributedString *as, double width)
    {
        if (as.length == 0)
            return 0;
        const NSRect r = [as boundingRectWithSize:NSMakeSize(width, 1e5)
                                          options:NSStringDrawingUsesLineFragmentOrigin | NSStringDrawingUsesFontLeading];
        return std::ceil(r.size.height);
    }
    // The pill: Menlo 11.5 parchment ("Chrome"); over Studio, "Lapis" in the accent and the
    // project dim ("Lapis · infinity").
    NSAttributedString *pillAttributed() const
    {
        NSFont *f = menlo(11.5);
        NSMutableAttributedString *as = [[[NSMutableAttributedString alloc] init] autorelease];
        const QString label = s.contextLabel;
        const qsizetype dot = s.overStudio ? label.indexOf(QStringLiteral(" · ")) : -1;
        if (dot > 0) {
            [as appendAttributedString:[[[NSAttributedString alloc]
                                           initWithString:ns(label.left(dot))
                                               attributes:@{NSFontAttributeName: f, NSForegroundColorAttributeName: hex(accentRgb())}] autorelease]];
            [as appendAttributedString:[[[NSAttributedString alloc]
                                           initWithString:ns(label.mid(dot))
                                               attributes:@{NSFontAttributeName: f, NSForegroundColorAttributeName: hex(kInk)}] autorelease]];
        } else {
            [as appendAttributedString:[[[NSAttributedString alloc]
                                           initWithString:ns(label)
                                               attributes:@{NSFontAttributeName: f,
                                                            NSForegroundColorAttributeName: s.overStudio ? hex(accentRgb()) : hex(kInk)}] autorelease]];
        }
        return as;
    }
    NSDictionary *questionAttrs() const
    {
        NSMutableParagraphStyle *style = [[[NSMutableParagraphStyle alloc] init] autorelease];
        style.lineBreakMode = NSLineBreakByTruncatingTail;
        return @{NSFontAttributeName: avenir(@"AvenirNext-Medium", 13.5, NSFontWeightMedium),
                 NSForegroundColorAttributeName: hex(accentRgb()), NSParagraphStyleAttributeName: style};
    }
    NSAttributedString *placeholderAttributed() const
    {
        const bool live = s.voice != SpotlightVoice::Idle;
        return [[[NSAttributedString alloc]
            initWithString:ns(s.placeholder)
                attributes:@{NSForegroundColorAttributeName: live ? hex(accentRgb(), 0.55) : hex(kMuted),
                             NSFontAttributeName: promptFont()}] autorelease];
    }

    // The layout for `answerH` points of answer body.
    Layout layoutFor(double answerH) const
    {
        Layout l;
        l.drawer = s.drawerVisible;
        const double bw = std::min(220.0, badgeWidth(badge()));
        l.badgeRect = CGRectMake(kW - 16 - bw, kRowCy - kBadgeH / 2, bw, kBadgeH);
        double promptX = kPillX + 2;
        l.pill = !s.contextLabel.isEmpty();
        if (l.pill) {
            const double textW = std::ceil([pillAttributed() size].width);
            const double tw = std::max(0.0, std::min(170.0, textW));
            l.pillRect = CGRectMake(kPillX, kRowCy - kPillH / 2, tw + 18, kPillH);
            promptX = CGRectGetMaxX(l.pillRect) + 12;
        }
        const double promptR = CGRectGetMinX(l.badgeRect) - 12;
        l.promptRect = CGRectMake(promptX, kRowCy - kPromptH / 2, std::max(40.0, promptR - promptX), kPromptH);
        if (l.drawer)
            l.answerShown = std::min(kSpotlightAnswerMaxHeight, std::max(18.0, answerH));
        l.height = spotlightHeight(l.drawer, l.drawer ? l.answerShown : 0);
        return l;
    }

    // The card and every non-control element, y-down card points; `withControls` also draws the
    // mark, badge, prompt and answer (offscreen render) that live views draw themselves on screen.
    void paint(CGContextRef ctx, const Layout &l, bool withControls, double now) const
    {
        @autoreleasepool {
            const CGRect card = CGRectMake(0, 0, kW, l.height);
            // One capsule: #0d0c0b, radius 16, a 1.2 pt accent rim (75% while listening, else 38%).
            const bool live = s.voice == SpotlightVoice::Held || s.voice == SpotlightVoice::Toggled;
            fillRounded(ctx, card, 16, hex(kCard, 0.97), hex(accentRgb(), live ? 0.75 : 0.38), 1.2);
            if (l.pill) {
                fillRounded(ctx, l.pillRect, kPillH / 2, hex(kCardHi), hex(kHairline));
                NSAttributedString *pt = pillAttributed();
                const double ph = std::ceil(menlo(11.5).ascender - menlo(11.5).descender);
                [pt drawWithRect:NSMakeRect(l.pillRect.origin.x + 9, CGRectGetMidY(l.pillRect) - ph / 2,
                                            l.pillRect.size.width - 18, ph)
                         options:NSStringDrawingTruncatesLastVisibleLine | NSStringDrawingUsesLineFragmentOrigin];
            }
            if (l.drawer) {
                CGContextSetFillColorWithColor(ctx, hex(kHairline).CGColor);
                CGContextFillRect(ctx, CGRectMake(1.2, kSpotlightBaseHeight, kW - 2.4, 1));
                if (!s.lastQuestion.isEmpty())
                    [ns(QStringLiteral("› ") + s.lastQuestion) drawInRect:NSMakeRect(18, kSpotlightBaseHeight + 10, kW - 36, 20)
                                                           withAttributes:questionAttrs()];
            }
            if (!withControls)
                return;
            const double period = mark() == Mark::Working ? 1.1 : 1.4;
            const double phase = std::fmod(std::max(0.0, now), period) / period;
            paintMark(ctx, CGPointMake(kMarkCx, kRowCy), mark(), phase);
            paintBadge(ctx, l.badgeRect, badge());
            // Prompt text (or placeholder) where the field draws it, with the caret when key.
            NSFont *pf = promptFont();
            const double lineH = std::ceil(pf.ascender - pf.descender);
            const CGRect box = l.promptRect;
            const NSPoint at = NSMakePoint(box.origin.x + 2, CGRectGetMidY(box) - lineH / 2);
            double caretX = at.x - 3;  // before the placeholder, not over its first glyph
            CGContextSaveGState(ctx);
            CGContextClipToRect(ctx, CGRectInset(box, -2, -4));
            if (!s.inputText.isEmpty()) {
                NSDictionary *a = @{NSFontAttributeName: pf, NSForegroundColorAttributeName: hex(kInk)};
                [ns(s.inputText) drawAtPoint:at withAttributes:a];
                caretX = at.x + std::ceil([ns(s.inputText) sizeWithAttributes:a].width) + 1;
            } else {
                [placeholderAttributed() drawAtPoint:at];
            }
            if (s.open && s.key && !s.actionRunning) {
                CGContextSetFillColorWithColor(ctx, hex(accentRgb()).CGColor);
                CGContextFillRect(ctx, CGRectMake(std::min(caretX, CGRectGetMaxX(box)), CGRectGetMidY(box) - 10, 2, 20));
            }
            CGContextRestoreGState(ctx);
            if (l.drawer) {
                const CGRect ar = CGRectMake(18, kSpotlightBaseHeight + 35, kW - 36, l.answerShown);
                CGContextSaveGState(ctx);
                CGContextClipToRect(ctx, ar);
                [bodyAttributed() drawWithRect:NSMakeRect(ar.origin.x, ar.origin.y, ar.size.width, 1e5)
                                       options:NSStringDrawingUsesLineFragmentOrigin | NSStringDrawingUsesFontLeading];
                CGContextRestoreGState(ctx);
            }
        }
    }
    void paintAnim(CGContextRef ctx, bool isMark, CGRect bounds, double phase) const
    {
        @autoreleasepool {
            if (isMark)
                paintMark(ctx, CGPointMake(CGRectGetMidX(bounds), CGRectGetMidY(bounds)), mark(), phase);
            else
                paintBadge(ctx, bounds, badge());
        }
    }

    QImage renderImage(double scale) const
    {
        if (!std::isfinite(scale) || scale <= 0 || scale > 8)
            return {};
        Impl copy;  // never touches the live views: a pure function of the state
        copy.s = s;
        copy.s.inputText = fieldText();
        copy.offscreen = true;
        copy.syncDerived();
        copy.s.key = s.open && (offscreen || (panel && panel.keyWindow));
        const Layout l = copy.layoutFor(copy.s.drawerVisible ? measure(copy.bodyAttributed(), kW - 36) : 0);
        const int w = int(std::ceil(kW * scale)), h = int(std::ceil(l.height * scale));
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
            NSGraphicsContext *previous = [[NSGraphicsContext currentContext] retain];
            [NSGraphicsContext setCurrentContext:[NSGraphicsContext graphicsContextWithCGContext:ctx flipped:YES]];
            copy.paint(ctx, l, true, 0.3);
            [NSGraphicsContext setCurrentContext:previous];
            [previous release];
            CGContextRelease(ctx);
        }
        img.setDevicePixelRatio(scale);
        return img;
    }

    void currentScreen(QRectF *out) const
    {
        if (offscreen) {
            *out = virtualScreen;
            return;
        }
        const NSPoint m = [NSEvent mouseLocation];
        NSScreen *pick = nil;
        for (NSScreen *sc in [NSScreen screens])
            if (NSPointInRect(m, sc.frame)) {
                pick = sc;
                break;
            }
        if (!pick)
            pick = [NSScreen mainScreen] ?: [NSScreen screens].firstObject;
        const NSRect f = pick ? pick.frame : NSMakeRect(0, 0, 1440, 900);
        *out = QRectF(f.origin.x, f.origin.y, f.size.width, f.size.height);
    }

    // ---- AppKit ------------------------------------------------------------------------------
    void build()
    {
        @autoreleasepool {
            panel = [[LapisSpotlightPanel alloc] initWithContentRect:NSMakeRect(0, 0, kW, kSpotlightBaseHeight)
                                                           styleMask:NSWindowStyleMaskBorderless
                                                             backing:NSBackingStoreBuffered
                                                               defer:NO];
            panel.owner = this;
            panel.opaque = NO;
            panel.backgroundColor = [NSColor clearColor];
            panel.hasShadow = YES;
            panel.hidesOnDeactivate = NO;
            panel.releasedWhenClosed = NO;
            panel.floatingPanel = YES;
            panel.becomesKeyOnlyIfNeeded = NO;
            panel.level = NSStatusWindowLevel;
            panel.sharingType = NSWindowSharingNone;  // never in ScreenCaptureKit or the shadow recorder
            panel.appearance = [NSAppearance appearanceNamed:NSAppearanceNameDarkAqua];
            // CanJoinAllSpaces and MoveToActiveSpace are mutually exclusive (AppKit raises on
            // both): all-Spaces wins, so the bar is already on whichever Space is active.
            panel.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces
                                       | NSWindowCollectionBehaviorFullScreenAuxiliary
                                       | NSWindowCollectionBehaviorIgnoresCycle;

            surface = [[[LapisSpotlightSurface alloc] initWithFrame:NSMakeRect(0, 0, kW, kSpotlightBaseHeight)] autorelease];
            surface.owner = this;
            surface.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
            panel.contentView = surface;

            Impl *me = this;
            markView = [[[LapisSpotlightAnimView alloc] initWithFrame:NSMakeRect(kMarkCx - 13, kRowCy - 13, 26, 26)] autorelease];
            markView.owner = me;
            markView.isMark = YES;
            [surface addSubview:markView];
            badgeView = [[[LapisSpotlightAnimView alloc] initWithFrame:NSMakeRect(kW - 16 - 120, kRowCy - kBadgeH / 2, 120, kBadgeH)] autorelease];
            badgeView.owner = me;
            badgeView.isMark = NO;
            [surface addSubview:badgeView];

            input = [[[NSTextField alloc] initWithFrame:NSMakeRect(kPillX + 2, kRowCy - kPromptH / 2, kW - 200, kPromptH)] autorelease];
            input.editable = YES;
            input.selectable = YES;
            input.bezeled = NO;
            input.bordered = NO;
            input.drawsBackground = NO;
            input.focusRingType = NSFocusRingTypeNone;
            input.font = promptFont();
            input.textColor = hex(kInk);
            input.usesSingleLineMode = YES;
            input.cell.scrollable = YES;
            input.cell.wraps = NO;
            delegate = [[LapisSpotlightFieldDelegate alloc] init];
            delegate.owner = this;
            input.delegate = delegate;
            [surface addSubview:input];

            answerScroll = [[[NSScrollView alloc] initWithFrame:NSMakeRect(18, kSpotlightBaseHeight + 35, kW - 36, 20)] autorelease];
            answerScroll.drawsBackground = NO;
            answerScroll.borderType = NSNoBorder;
            answerScroll.hasVerticalScroller = YES;
            answerScroll.autohidesScrollers = YES;
            answerScroll.scrollerStyle = NSScrollerStyleOverlay;
            answerScroll.scrollerKnobStyle = NSScrollerKnobStyleLight;
            answerView = [[[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, kW - 36, 20)] autorelease];
            answerView.editable = NO;
            answerView.selectable = YES;
            answerView.drawsBackground = NO;
            answerView.textContainerInset = NSZeroSize;
            answerView.textContainer.lineFragmentPadding = 0;
            answerView.textContainer.widthTracksTextView = YES;
            answerView.verticallyResizable = YES;
            answerView.horizontallyResizable = NO;
            answerView.autoresizingMask = NSViewWidthSizable;
            answerScroll.documentView = answerView;
            [surface addSubview:answerScroll];

            orderedOut = [[NSMutableArray alloc] init];
            Impl *owner = this;
            activeObserver = [[[NSNotificationCenter defaultCenter]
                addObserverForName:NSApplicationDidBecomeActiveNotification
                            object:nil
                             queue:[NSOperationQueue mainQueue]
                        usingBlock:^(NSNotification *) {
                            // Studio came forward for real (Dock click, Cmd+Tab): bring back what
                            // the bar ordered out. Its own activation (bar open) is not that.
                            if (!owner->s.open)
                                owner->restoreWindows(false);
                        }] retain];
            resignObserver = [[[NSNotificationCenter defaultCenter]
                addObserverForName:NSApplicationDidResignActiveNotification
                            object:nil
                             queue:[NSOperationQueue mainQueue]
                        usingBlock:^(NSNotification *) {
                            // Clicking into another app dismisses the bar cleanly (keeping the
                            // last exchange for the next L2), just like Raycast/Spotlight.
                            if (owner->s.open && !owner->s.actionRunning && owner->s.voice == SpotlightVoice::Idle)
                                owner->dismissFromUser();
                        }] retain];
        }
    }

    void stopPulse()
    {
        [pulseTimer invalidate];
        [pulseTimer release];
        pulseTimer = nil;
    }
    // The status mark breathes / spins: redraw only it (and the badge), ~30 fps, while it animates.
    void syncPulse()
    {
        const bool want = panel && s.open && animating();
        if (!want) {
            stopPulse();
            return;
        }
        if (pulseTimer)
            return;
        LapisSpotlightAnimView *m = markView;
        Impl *owner = this;
        pulseTimer = [[NSTimer timerWithTimeInterval:1.0 / 30 repeats:YES block:^(NSTimer *t) {
            (void)t;
            const double period = owner->mark() == Mark::Working ? 1.1 : 1.4;
            m.phase = std::fmod([[NSProcessInfo processInfo] systemUptime], period) / period;
            [m setNeedsDisplay:YES];
        }] retain];
        [[NSRunLoop mainRunLoop] addTimer:pulseTimer forMode:NSRunLoopCommonModes];
    }

    void setEscapeWatch(bool on)
    {
        s.escapeWatch = on;
        if (offscreen)
            return;
        if (!on) {
            if (escapeMonitor) {
                [NSEvent removeMonitor:escapeMonitor];
                [escapeMonitor release];
                escapeMonitor = nil;
            }
            return;
        }
        if (escapeMonitor)
            return;
        Impl *owner = this;  // removed in teardown, before Impl dies
        const pid_t myPid = getpid();
        escapeMonitor = [[NSEvent addGlobalMonitorForEventsMatchingMask:NSEventMaskKeyDown
                                                                handler:^(NSEvent *e) {
                                                                    if (e.keyCode != 53 || !owner->s.escapeWatch)
                                                                        return;
                                                                    // Our own synthetic keys (a cua_hotkey "esc") never cancel.
                                                                    CGEventRef cg = e.CGEvent;
                                                                    if (cg && CGEventGetIntegerValueField(cg, kCGEventSourceUnixProcessID) == myPid)
                                                                        return;
                                                                    if (owner->q && owner->q->onGlobalEscape)
                                                                        owner->q->onGlobalEscape();
                                                                }] retain];
    }

    void teardown()
    {
        stopPulse();
        setEscapeWatch(false);
        if (activeObserver) {
            [[NSNotificationCenter defaultCenter] removeObserver:activeObserver];
            [activeObserver release];
            activeObserver = nil;
        }
        if (resignObserver) {
            [[NSNotificationCenter defaultCenter] removeObserver:resignObserver];
            [resignObserver release];
            resignObserver = nil;
        }
        if (panel) {
            input.delegate = nil;
            surface.owner = nullptr;
            markView.owner = nullptr;
            badgeView.owner = nullptr;
            panel.owner = nullptr;
            [panel orderOut:nil];
            [panel close];
            [panel release];
            panel = nil;
        }
        if (delegate) {
            delegate.owner = nullptr;
            [delegate release];
            delegate = nil;
        }
        [orderedOut release];
        orderedOut = nil;
    }

    QString fieldText() const
    {
        return input ? QString::fromNSString(input.stringValue ?: @"") : s.inputText;
    }

    // Re-renders texts, colours and the frame (top edge kept) from s.
    void render(bool reposition)
    {
        syncDerived();
        double answerH = 0;
        if (offscreen || !panel) {
            if (s.drawerVisible)
                answerH = measure(bodyAttributed(), kW - 36);
        } else {
            @autoreleasepool {
                input.placeholderAttributedString = placeholderAttributed();
                input.textColor = hex(kInk);
                const bool d = s.drawerVisible;
                answerScroll.hidden = !d;
                if (d) {
                    [answerView.textStorage setAttributedString:bodyAttributed()];
                    answerView.selectedTextAttributes = @{NSBackgroundColorAttributeName: hex(accentRgb(), 0.28)};
                    answerView.frame = NSMakeRect(0, 0, kW - 36, answerView.frame.size.height);
                    [answerView.layoutManager ensureLayoutForTextContainer:answerView.textContainer];
                    answerH = std::ceil([answerView.layoutManager usedRectForTextContainer:answerView.textContainer].size.height);
                }
            }
        }
        lay = layoutFor(answerH);
        if (panel) {
            @autoreleasepool {
                badgeView.frame = NSRectFromCGRect(lay.badgeRect);
                [badgeView setNeedsDisplay:YES];
                [markView setNeedsDisplay:YES];
                input.frame = NSRectFromCGRect(lay.promptRect);
                if (lay.drawer) {
                    answerScroll.frame = NSMakeRect(18, kSpotlightBaseHeight + 35, kW - 36, lay.answerShown);
                    if (s.busy)
                        [answerView scrollRangeToVisible:NSMakeRange(answerView.string.length, 0)];
                }
                [surface setNeedsDisplay:YES];
            }
            syncPulse();
        }
        const double h = lay.height;
        QRectF f;
        if (reposition || !s.frame.isValid()) {
            QRectF screen;
            currentScreen(&screen);
            f = spotlightFrame(screen, h);
        } else {
            const double top = s.frame.y() + s.frame.height();
            f = QRectF(s.frame.x(), top - h, s.frame.width(), h);
        }
        s.frame = f;
        if (panel) {
            [panel setFrame:NSMakeRect(f.x(), f.y(), f.width(), f.height()) display:YES animate:NO];
            [panel invalidateShadow];
        }
    }

    void focusInput()
    {
        if (!panel)
            return;
        [panel makeKeyWindow];
        [panel makeFirstResponder:input];
        if (NSText *ed = [input currentEditor]) {
            [ed setSelectedRange:NSMakeRange(input.stringValue.length, 0)];
            if ([ed isKindOfClass:[NSTextView class]]) {
                NSTextView *tv = (NSTextView *)ed;
                tv.insertionPointColor = hex(accentRgb());
                tv.selectedTextAttributes = @{NSBackgroundColorAttributeName: hex(accentRgb(), 0.28)};
            }
        }
    }

    void snapshotTarget()
    {
        int pid = 0;
        QString name;
        if (probe) {
            probe(&pid, &name);
        } else if (!offscreen) {
            NSRunningApplication *ext = NSWorkspace.sharedWorkspace.frontmostApplication;
            if (ext && ext.processIdentifier != getpid()) {
                pid = ext.processIdentifier;
                name = QString::fromNSString(ext.localizedName ?: @"");
            }
        }
        if (pid > 0) {
            s.targetPid = pid;
            s.targetApp = name;
        }
    }

    // Studio windows on another Space (fullscreen Studio, usually) would make activation slide
    // there: order them out until Studio is brought forward for real.
    void orderOutOffSpaceWindows(bool wasHidden)
    {
        for (NSWindow *w in [NSApp windows]) {
            if (w == panel || [w isKindOfClass:[NSPanel class]] || !w.visible)
                continue;
            if (wasHidden || ![w isOnActiveSpace]) {
                if (![orderedOut containsObject:w])
                    [orderedOut addObject:w];
                [w orderOut:nil];
            }
        }
    }

    void restoreWindows(bool makeKey)
    {
        if (!orderedOut || orderedOut.count == 0) {
            if (makeKey)
                for (NSWindow *w in [NSApp windows])
                    if (w != panel && ![w isKindOfClass:[NSPanel class]] && w.visible && w.canBecomeKeyWindow) {
                        [w makeKeyAndOrderFront:nil];
                        break;
                    }
            return;
        }
        NSArray *wins = [[orderedOut copy] autorelease];
        [orderedOut removeAllObjects];
        bool keyed = false;
        for (NSWindow *w in wins) {
            if (makeKey && !keyed && w.canBecomeKeyWindow) {
                [w makeKeyAndOrderFront:nil];
                keyed = true;
            } else {
                [w orderFront:nil];
            }
        }
    }

    bool openOrFocus()
    {
        const bool active = offscreen ? false : studioAppIsActive();
        const bool wasKey = s.open && !s.actionRunning && (offscreen ? true : (active && panel && panel.keyWindow));
        if (wasKey) {
            if (!offscreen)
                focusInput();
            return true;
        }
        if (offscreen || !active)
            snapshotTarget();
        s.overStudio = false;
        s.contextLabel = spotlightContextLabel(s.targetApp);
        return show(active, reposition0());
    }
    bool reposition0() const { return !s.open; }

    // Over Studio itself: no target app, Studio's windows stay where they are.
    bool openOverStudio(const QString &label)
    {
        const bool active = offscreen ? true : studioAppIsActive();
        const bool wasKey = s.open && !s.actionRunning && s.overStudio && s.contextLabel == label
                            && (offscreen ? true : (active && panel && panel.keyWindow));
        if (wasKey) {
            if (!offscreen)
                focusInput();
            return true;
        }
        s.overStudio = true;
        s.targetPid = 0;
        s.targetApp.clear();
        s.contextLabel = label;
        const bool key = show(true, reposition0());
        if (!offscreen && !active)
            [NSApp activateIgnoringOtherApps:YES];  // never orders Studio's windows out
        return key;
    }

    bool show(bool active, bool reposition)
    {
        s.open = true;
        s.actionRunning = false;
        ++s.opens;
        if (offscreen) {
            render(reposition);
            return false;
        }
        if (!panel)
            build();
        @autoreleasepool {
            const bool hidden = [NSApp isHidden];
            if (!active) {
                if (hidden)
                    [NSApp unhideWithoutActivation];
                orderOutOffSpaceWindows(hidden);
            }
            panel.alphaValue = 1.0;
            panel.ignoresMouseEvents = NO;
            render(reposition);
            [panel orderFrontRegardless];
            if (!active)
                [NSApp activateIgnoringOtherApps:YES];
            focusInput();
            s.key = true;
            s.alpha = 1.0;
        }
        return false;
    }

    void close(bool restoreTarget)
    {
        const bool was = s.open;
        s.open = false;
        s.voice = SpotlightVoice::Idle;
        s.actionRunning = false;
        s.key = false;
        s.alpha = 0;
        if (was)
            ++s.dismissals;
        syncDerived();
        stopPulse();
        if (panel) {
            [panel orderOut:nil];
            panel.alphaValue = 1.0;
            panel.ignoresMouseEvents = NO;
        }
        // The app it was opened over is gone: never leave Studio active with its windows hidden.
        if (restoreTarget && !reactivate() && !offscreen && orderedOut.count > 0)
            restoreWindows(true);
    }

    bool reactivate()
    {
        if (offscreen || s.targetPid <= 0)
            return false;
        @autoreleasepool {
            NSRunningApplication *app = [NSRunningApplication runningApplicationWithProcessIdentifier:s.targetPid];
            if (!app || app.terminated)
                return false;
            if (@available(macOS 14.0, *)) {
                if (NSApp)
                    [NSApp yieldActivationToApplication:app];
                return [app activateFromApplication:NSRunningApplication.currentApplication options:0]
                       || [app activateWithOptions:0];
            }
            return [app activateWithOptions:0];
        }
    }

    void inputEdited()
    {
        render(false);  // the badge follows the text (R2 send)
        if (q && q->onInputEdited)
            q->onInputEdited();
    }
    void submitFromField()
    {
        if (q && q->onSubmit)
            q->onSubmit(fieldText());
    }
    void dismissFromUser()
    {
        if (q && q->onDismiss)
            q->onDismiss();
        else
            close(true);
    }
};

}  // namespace studio::cua

@implementation LapisSpotlightPanel
- (BOOL)canBecomeKeyWindow { return YES; }
- (BOOL)canBecomeMainWindow { return NO; }
// lapis_studio is a bare executable (no bundle, no Edit menu): the panel answers the editing
// shortcuts itself, above all Cmd+V, which is how Phonon delivers dictation.
- (BOOL)performKeyEquivalent:(NSEvent *)event
{
    if (event.type == NSEventTypeKeyDown) {
        if (event.keyCode == 53) {  // Esc
            if (self.owner)
                self.owner->dismissFromUser();
            return YES;
        }
        const NSEventModifierFlags mods = event.modifierFlags & NSEventModifierFlagDeviceIndependentFlagsMask
                                          & (NSEventModifierFlagCommand | NSEventModifierFlagShift
                                             | NSEventModifierFlagOption | NSEventModifierFlagControl);
        if (mods == NSEventModifierFlagCommand) {
            NSString *k = event.charactersIgnoringModifiers.lowercaseString;
            SEL action = nullptr;
            if ([k isEqualToString:@"v"])
                action = @selector(paste:);
            else if ([k isEqualToString:@"a"])
                action = @selector(selectAll:);
            else if ([k isEqualToString:@"c"])
                action = @selector(copy:);
            else if ([k isEqualToString:@"x"])
                action = @selector(cut:);
            else if ([k isEqualToString:@"z"])
                action = @selector(undo:);
            if (action && [[self firstResponder] tryToPerform:action with:self])
                return YES;
        }
    }
    return [super performKeyEquivalent:event];
}
- (void)cancelOperation:(id)sender
{
    (void)sender;
    if (self.owner)
        self.owner->dismissFromUser();
}
@end

@implementation LapisSpotlightSurface
- (BOOL)isFlipped { return YES; }
- (BOOL)isOpaque { return NO; }
- (void)drawRect:(NSRect)dirtyRect
{
    CGContextRef ctx = [NSGraphicsContext currentContext].CGContext;
    CGContextClearRect(ctx, dirtyRect);
    if (self.owner)
        self.owner->paint(ctx, self.owner->lay, false, 0);
}
@end

@implementation LapisSpotlightAnimView
- (BOOL)isFlipped { return YES; }
- (BOOL)isOpaque { return NO; }
- (NSView *)hitTest:(NSPoint)point { (void)point; return nil; }
- (void)drawRect:(NSRect)dirtyRect
{
    (void)dirtyRect;
    if (self.owner)
        self.owner->paintAnim([NSGraphicsContext currentContext].CGContext, self.isMark, NSRectToCGRect(self.bounds), self.phase);
}
@end

@implementation LapisSpotlightFieldDelegate
- (void)controlTextDidChange:(NSNotification *)note
{
    (void)note;
    if (self.owner)
        self.owner->inputEdited();  // Phonon's Cmd+V paste lands here
}
- (BOOL)control:(NSControl *)control textView:(NSTextView *)textView doCommandBySelector:(SEL)selector
{
    (void)control;
    (void)textView;
    if (!self.owner)
        return NO;
    if (selector == @selector(insertNewline:)) {
        self.owner->submitFromField();
        return YES;
    }
    if (selector == @selector(cancelOperation:)) {
        self.owner->dismissFromUser();
        return YES;
    }
    return NO;
}
@end

namespace studio::cua {

uint32_t spotlightWindowNumber()
{
    return g_liveSpotlight ? g_liveSpotlight->windowNumber() : 0;
}

void setSpotlightLapisBlue(bool lapisBlue) { g_lapisBlue.store(lapisBlue); }

bool spotlightLapisBlue() { return g_lapisBlue.load(); }

void refreshSpotlightTheme()
{
    if (g_liveImpl && g_liveImpl->panel)
        g_liveImpl->render(false);
}

CuaSpotlight::CuaSpotlight(Mode mode) : impl(std::make_unique<Impl>())
{
    impl->q = this;
    impl->offscreen = mode == Mode::Offscreen;
    impl->syncDerived();
    if (!impl->offscreen) {
        g_liveSpotlight = this;
        g_liveImpl = impl.get();
    }
}

CuaSpotlight::~CuaSpotlight()
{
    if (g_liveSpotlight == this) {
        g_liveSpotlight = nullptr;
        g_liveImpl = nullptr;
    }
    impl->teardown();
}

bool CuaSpotlight::openOrFocus() { return impl->openOrFocus(); }

bool CuaSpotlight::openOrFocusOverStudio(const QString &contextLabel)
{
    return impl->openOverStudio(contextLabel.simplified().isEmpty() ? QStringLiteral("Lapis") : contextLabel.simplified());
}

void CuaSpotlight::dismiss(bool restoreTarget) { impl->close(restoreTarget); }

bool CuaSpotlight::isOpen() const { return impl->s.open; }

void CuaSpotlight::setListening(bool on)
{
    setVoice(on ? SpotlightVoice::Held : SpotlightVoice::Idle);
}

void CuaSpotlight::setVoice(SpotlightVoice voice)
{
    if (impl->s.voice == voice)
        return;
    impl->s.voice = voice;
    impl->render(false);
}

void CuaSpotlight::notifyInputEdited() { impl->inputEdited(); }

void CuaSpotlight::setEscapeWatch(bool on) { impl->setEscapeWatch(on); }

void CuaSpotlight::setInputText(const QString &text)
{
    impl->s.inputText = text;
    if (impl->input) {
        impl->input.stringValue = ns(text);
        if (NSText *ed = [impl->input currentEditor])
            [ed setSelectedRange:NSMakeRange(impl->input.stringValue.length, 0)];
    }
    impl->render(false);  // the badge follows the text
}

QString CuaSpotlight::inputText() const { return impl->fieldText(); }

QString CuaSpotlight::takeInputText()
{
    const QString t = impl->fieldText();
    setInputText(QString());
    return t;
}

void CuaSpotlight::beginTurn(const QString &question)
{
    impl->s.lastQuestion = question.simplified();
    impl->s.answer.clear();
    impl->s.statusLine = QStringLiteral("Thinking…");
    impl->s.busy = true;
    impl->render(false);
}

void CuaSpotlight::setAnswer(const QString &displayText)
{
    impl->s.answer = displayText.trimmed();
    impl->render(false);
}

void CuaSpotlight::setStatusLine(const QString &status)
{
    impl->s.statusLine = status;
    impl->render(false);
}

void CuaSpotlight::setBusy(bool busy)
{
    impl->s.busy = busy;
    if (!busy && impl->s.answer.isEmpty() && impl->s.statusLine == QStringLiteral("Thinking…"))
        impl->s.statusLine.clear();
    impl->render(false);
    if (!busy && impl->s.open && !impl->s.actionRunning && impl->panel && studioAppIsActive())
        impl->focusInput();
}

void CuaSpotlight::setActionRunning(bool running, const QString &status)
{
    if (!status.isEmpty())
        impl->s.statusLine = status;
    if (running && !impl->s.actionRunning)
        ++impl->s.actionRuns;
    const bool was = impl->s.actionRunning;
    impl->s.actionRunning = running && impl->s.open;
    if (impl->panel && impl->s.open) {
        impl->panel.alphaValue = running ? 0.0 : 1.0;
        impl->panel.ignoresMouseEvents = running ? YES : NO;
        impl->s.alpha = running ? 0.0 : 1.0;
        // The action types and clicks into the app behind: it has to be frontmost, not us.
        if (running && !was && studioAppIsActive())
            impl->reactivate();
    }
    impl->render(false);
}

void CuaSpotlight::finishAfterAction(const QString &summary)
{
    impl->s.busy = false;
    if (!summary.isEmpty())
        impl->s.statusLine = summary;
    impl->close(false);  // the target app is already frontmost (the actions ran there)
    impl->render(false);
}

bool CuaSpotlight::reactivateTarget() { return impl->reactivate(); }

void CuaSpotlight::restoreMainWindows()
{
    if (!impl->offscreen)
        impl->restoreWindows(false);
}

void CuaSpotlight::summonStudio()
{
    impl->close(false);
    if (impl->offscreen || !NSApp)
        return;
    [NSApp unhide:nil];
    [NSApp activateIgnoringOtherApps:YES];
    impl->restoreWindows(true);
}

QString CuaSpotlight::targetApp() const { return impl->s.targetApp; }

int CuaSpotlight::targetPid() const { return impl->s.targetPid; }

void CuaSpotlight::setFrontmostProbe(std::function<void(int *, QString *)> probe) { impl->probe = std::move(probe); }

uint32_t CuaSpotlight::windowNumber() const
{
    if (!impl->panel)
        return 0;
    const NSInteger n = impl->panel.windowNumber;
    return n > 0 ? uint32_t(n) : 0;
}

SpotlightState CuaSpotlight::state() const
{
    SpotlightState out = impl->s;
    out.inputText = impl->fieldText();
    if (impl->panel) {
        out.key = impl->s.open && impl->panel.keyWindow;
        out.alpha = impl->s.open ? impl->panel.alphaValue : 0.0;
    }
    return out;
}

QImage CuaSpotlight::render(double scale) const { return impl->renderImage(scale); }

// ---- --snap-spotlight ----------------------------------------------------------------------------

int snapSpotlight(const QString &outPath)
{
    const auto chrome = [](int *pid, QString *name) { *pid = 4242; *name = QStringLiteral("Google Chrome"); };
    const QString answer = QStringLiteral(
        "That error means the page asked for a cookie the browser has blocked, so the sign-in "
        "form never gets its session token and quietly reloads.\n\n"
        "Open Settings, Privacy, Third-party cookies, and allow them for accounts.example.com, then "
        "reload the tab. If it still loops, clear that site's data and sign in again.");
    QVector<QImage> shots;
    const bool wasBlue = spotlightLapisBlue();
    for (int i = 0; i < 6; ++i) {
        setSpotlightLapisBlue(i == 5);
        CuaSpotlight sp(CuaSpotlight::Mode::Offscreen);
        sp.setFrontmostProbe(chrome);
        if (i == 2)
            sp.openOrFocusOverStudio(QStringLiteral("Lapis · infinity"));
        else
            sp.openOrFocus();
        if (i == 1) {
            sp.setListening(true);
            sp.setInputText(QStringLiteral("summarize this thread and draft a reply"));
        } else if (i == 2) {
            sp.setVoice(SpotlightVoice::Toggled);
        } else if (i == 3) {
            sp.setInputText(QStringLiteral("open the latest run in a new tab"));
            sp.setVoice(SpotlightVoice::Transcribing);
        } else if (i >= 4) {
            sp.beginTurn(QStringLiteral("why does the sign in page keep reloading?"));
            sp.setAnswer(answer);
            sp.setBusy(false);
        }
        shots.push_back(sp.render(2));
    }
    setSpotlightLapisBlue(wasBlue);
    double h = 32;
    for (const QImage &s : shots) {
        if (s.isNull()) {
            std::fprintf(stderr, "snap-spotlight: render failed\n");
            return 2;
        }
        h += s.height() / 2.0 + 32;
    }
    const double scale = 2, margin = 32;
    QImage out(int((kSpotlightWidth + 2 * margin) * scale), int(h * scale), QImage::Format_RGBA8888_Premultiplied);
    out.fill(QColor(0, 0, 0));
    QPainter p(&out);
    p.setRenderHint(QPainter::SmoothPixmapTransform);
    p.scale(scale, scale);
    double y = margin;
    for (QImage &s : shots) {
        s.setDevicePixelRatio(scale);
        p.drawImage(QPointF(margin, y), s);
        y += s.height() / scale + 32;
    }
    p.end();
    QDir().mkpath(QFileInfo(outPath).absolutePath());
    if (!out.save(outPath)) {
        std::fprintf(stderr, "snap-spotlight: could not write %s\n", qPrintable(outPath));
        return 1;
    }
    std::printf("snap-spotlight: %s (%dx%d)\n", qPrintable(outPath), out.width(), out.height());
    return 0;
}

}  // namespace studio::cua
