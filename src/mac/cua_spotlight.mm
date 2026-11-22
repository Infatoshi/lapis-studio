#include "mac/cua_spotlight.hpp"

#import <AppKit/AppKit.h>
#import <QuartzCore/QuartzCore.h>

#include <QVector>
#include <algorithm>
#include <cmath>
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
    // base 96 + divider gap 10 + 1 + gap 10 + question 18 + gap 6 + answer + bottom 12 (- 1 overlap)
    const double body = std::clamp(answerHeight, 0.0, kSpotlightAnswerMaxHeight);
    return std::min(kSpotlightMaxHeight, kSpotlightBaseHeight + 46 + body);
}

QString spotlightHint(bool listening)
{
    return listening ? QStringLiteral("● Listening…") : QStringLiteral("L2 voice · R2 send · ○ close");
}

QString spotlightPlaceholder(const QString &targetApp)
{
    return targetApp.isEmpty() ? QStringLiteral("Command or ask anything…")
                               : QStringLiteral("Command or ask about %1…").arg(targetApp);
}

bool studioAppIsActive()
{
    return NSApp && [NSApp isActive];
}

}  // namespace studio::cua

namespace {

NSColor *hex(uint32_t rgb, double a = 1.0)
{
    return [NSColor colorWithSRGBRed:((rgb >> 16) & 0xFF) / 255.0 green:((rgb >> 8) & 0xFF) / 255.0
                                blue:(rgb & 0xFF) / 255.0 alpha:a];
}

NSString *ns(const QString &s)
{
    return s.toNSString();
}

constexpr uint32_t kGold = 0xF5C451, kSapphire = 0x4DA3FF, kDimGold = 0xC5A049, kBorderGold = 0xD4AF37;
constexpr double kW = studio::cua::kSpotlightWidth;

studio::cua::CuaSpotlight *g_liveSpotlight = nullptr;  // the Visible instance (capture exclusion)

}  // namespace

@interface LapisSpotlightPanel : NSPanel
@property(nonatomic, assign) studio::cua::CuaSpotlight::Impl *owner;
@end

@interface LapisSpotlightFlipped : NSView
@end
@implementation LapisSpotlightFlipped
- (BOOL)isFlipped { return YES; }
@end

@interface LapisSpotlightFieldDelegate : NSObject <NSTextFieldDelegate>
@property(nonatomic, assign) studio::cua::CuaSpotlight::Impl *owner;
@end

namespace studio::cua {

struct CuaSpotlight::Impl {
    CuaSpotlight *q = nullptr;
    bool offscreen = false;
    SpotlightState s;
    std::function<void(int *, QString *)> probe;
    QRectF virtualScreen{0, 0, 1440, 900};

    LapisSpotlightPanel *panel = nil;
    LapisSpotlightFieldDelegate *delegate = nil;
    NSView *card = nil;            // tinted overlay carrying the border
    NSView *dot = nil;
    NSTextField *title = nil;
    NSView *pill = nil;
    NSTextField *pillText = nil;
    NSTextField *hint = nil;
    NSView *inputBox = nil;
    NSTextField *input = nil;
    NSView *divider = nil;
    NSTextField *question = nil;
    NSScrollView *answerScroll = nil;
    NSTextView *answerView = nil;
    NSMutableArray *orderedOut = nil;  // Studio windows ordered out so activation stays on this Space
    id activeObserver = nil;
    id resignObserver = nil;

    // ---- state -------------------------------------------------------------------------------
    bool drawer() const { return !s.lastQuestion.isEmpty() || !s.answer.isEmpty() || !s.statusLine.isEmpty(); }
    QString bodyText() const
    {
        if (!s.answer.isEmpty())
            return s.answer;
        return s.statusLine;
    }
    void syncDerived()
    {
        s.hint = spotlightHint(s.listening);
        s.placeholder = spotlightPlaceholder(s.targetApp);
        s.drawerVisible = drawer();
        if (offscreen) {
            s.key = s.open;
            s.alpha = !s.open ? 0.0 : (s.actionRunning ? 0.0 : 1.0);
        }
    }
    // Offscreen: a rough line estimate (14 pt text, ~7.4 pt per glyph, 18 pt lines).
    double estimateAnswerHeight() const
    {
        const QString t = bodyText();
        if (t.isEmpty())
            return 0;
        const double perLine = std::max(1.0, std::floor((kW - 32) / 7.4));
        double lines = 0;
        for (const QString &para : t.split(QLatin1Char('\n')))
            lines += std::max(1.0, std::ceil(para.size() / perLine));
        return lines * 18;
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
    NSTextField *makeLabel(NSFont *font, NSColor *color)
    {
        NSTextField *l = [NSTextField labelWithString:@""];
        l.font = font;
        l.textColor = color;
        l.drawsBackground = NO;
        l.bezeled = NO;
        l.editable = NO;
        l.selectable = NO;
        l.lineBreakMode = NSLineBreakByTruncatingTail;
        l.usesSingleLineMode = YES;
        return l;
    }
    NSView *makeLayerView(Class cls, NSRect frame, NSColor *bg, double radius)
    {
        NSView *v = [[[cls alloc] initWithFrame:frame] autorelease];
        v.wantsLayer = YES;
        v.layer.backgroundColor = bg.CGColor;
        v.layer.cornerRadius = radius;
        return v;
    }

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
            // CanJoinAllSpaces and MoveToActiveSpace are mutually exclusive (AppKit raises on
            // both): all-Spaces wins, so the bar is already on whichever Space is active.
            panel.collectionBehavior = NSWindowCollectionBehaviorCanJoinAllSpaces
                                       | NSWindowCollectionBehaviorFullScreenAuxiliary
                                       | NSWindowCollectionBehaviorIgnoresCycle;

            NSVisualEffectView *fx = [[[NSVisualEffectView alloc] initWithFrame:NSMakeRect(0, 0, kW, kSpotlightBaseHeight)] autorelease];
            fx.material = NSVisualEffectMaterialHUDWindow;
            fx.blendingMode = NSVisualEffectBlendingModeBehindWindow;
            fx.state = NSVisualEffectStateActive;
            fx.appearance = [NSAppearance appearanceNamed:NSAppearanceNameVibrantDark];
            fx.wantsLayer = YES;
            fx.layer.cornerRadius = 16;
            fx.layer.masksToBounds = YES;
            fx.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
            panel.contentView = fx;

            card = makeLayerView([LapisSpotlightFlipped class], fx.bounds, hex(0x12141A, 0.90), 16);
            card.layer.borderWidth = 1;
            card.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
            [fx addSubview:card];

            dot = makeLayerView([NSView class], NSMakeRect(16, 19, 8, 8), hex(kDimGold), 4);
            [card addSubview:dot];
            title = makeLabel([NSFont systemFontOfSize:13 weight:NSFontWeightSemibold], hex(0xF3EFE6));
            title.stringValue = @"Lapis";
            [card addSubview:title];
            pill = makeLayerView([LapisSpotlightFlipped class], NSZeroRect, hex(0x1D212B), 9);
            [card addSubview:pill];
            pillText = makeLabel([NSFont monospacedSystemFontOfSize:12 weight:NSFontWeightRegular], hex(0x9AA4B2));
            [pill addSubview:pillText];
            hint = makeLabel([NSFont monospacedSystemFontOfSize:11 weight:NSFontWeightRegular], hex(0x7D8590));
            hint.alignment = NSTextAlignmentRight;
            [card addSubview:hint];

            inputBox = makeLayerView([LapisSpotlightFlipped class], NSMakeRect(12, 44, kW - 24, 42), hex(0x0B0D12), 10);
            inputBox.layer.borderWidth = 1;
            inputBox.layer.borderColor = hex(0x262B36).CGColor;
            [card addSubview:inputBox];
            input = [[[NSTextField alloc] initWithFrame:NSMakeRect(14, 9, kW - 24 - 28, 24)] autorelease];
            input.editable = YES;
            input.selectable = YES;
            input.bezeled = NO;
            input.bordered = NO;
            input.drawsBackground = NO;
            input.focusRingType = NSFocusRingTypeNone;
            input.font = [NSFont systemFontOfSize:18];
            input.textColor = hex(0xF3EFE6);
            input.usesSingleLineMode = YES;
            input.cell.scrollable = YES;
            input.cell.wraps = NO;
            delegate = [[LapisSpotlightFieldDelegate alloc] init];
            delegate.owner = this;
            input.delegate = delegate;
            [inputBox addSubview:input];

            divider = makeLayerView([NSView class], NSZeroRect, hex(0x222630), 0);
            [card addSubview:divider];
            question = makeLabel([NSFont systemFontOfSize:12.5], hex(0x8B949E));
            [card addSubview:question];

            answerScroll = [[[NSScrollView alloc] initWithFrame:NSMakeRect(16, 130, kW - 32, 20)] autorelease];
            answerScroll.drawsBackground = NO;
            answerScroll.borderType = NSNoBorder;
            answerScroll.hasVerticalScroller = YES;
            answerScroll.autohidesScrollers = YES;
            answerScroll.scrollerStyle = NSScrollerStyleOverlay;
            answerView = [[[NSTextView alloc] initWithFrame:NSMakeRect(0, 0, kW - 32, 20)] autorelease];
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
            [card addSubview:answerScroll];

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
                            if (owner->s.open && !owner->s.actionRunning && !owner->s.listening)
                                owner->dismissFromUser();
                        }] retain];
        }
    }

    void teardown()
    {
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
        if (offscreen) {
            answerH = estimateAnswerHeight();
        } else if (panel) {
            @autoreleasepool {
                const uint32_t dotRgb = s.listening ? kGold : ((s.busy || s.actionRunning) ? kSapphire : kDimGold);
                dot.layer.backgroundColor = hex(dotRgb).CGColor;
                card.layer.borderColor = hex(kBorderGold, s.listening ? 0.65 : 0.35).CGColor;
                inputBox.layer.borderColor = (s.listening ? hex(kBorderGold, 0.65) : hex(0x262B36)).CGColor;
                // Top row: dot, "Lapis", the context pill, the hint on the right.
                title.frame = NSMakeRect(32, 14, 44, 18);
                const bool hasPill = !s.targetApp.isEmpty();
                pill.hidden = !hasPill;
                hint.stringValue = ns(s.hint);
                hint.textColor = s.listening ? hex(kGold) : hex(0x7D8590);
                [hint sizeToFit];
                const double hintW = std::min(260.0, double(hint.frame.size.width));
                hint.frame = NSMakeRect(kW - 16 - hintW, 15, hintW, 16);
                if (hasPill) {
                    pillText.stringValue = ns(QStringLiteral("in ") + s.targetApp);
                    [pillText sizeToFit];
                    const double maxPill = kW - 16 - hintW - 12 - 84;
                    const double tw = std::min(maxPill - 18, double(pillText.frame.size.width));
                    pill.frame = NSMakeRect(84, 13, tw + 18, 20);
                    pillText.frame = NSMakeRect(9, 2, tw, 16);
                }
                NSString *ph = ns(s.placeholder);
                input.placeholderAttributedString = [[[NSAttributedString alloc]
                    initWithString:ph
                        attributes:@{NSForegroundColorAttributeName: hex(0x5C6370),
                                     NSFontAttributeName: [NSFont systemFontOfSize:18]}] autorelease];
                // Drawer.
                const bool d = s.drawerVisible;
                divider.hidden = question.hidden = answerScroll.hidden = !d;
                if (d) {
                    divider.frame = NSMakeRect(0, kSpotlightBaseHeight, kW, 1);
                    question.stringValue = s.lastQuestion.isEmpty() ? @"" : ns(QStringLiteral("› ") + s.lastQuestion);
                    question.frame = NSMakeRect(16, kSpotlightBaseHeight + 11, kW - 32, 18);
                    const bool status = s.answer.isEmpty();
                    NSColor *color = status ? (s.busy || s.actionRunning ? hex(0x58A6FF) : hex(0x8B949E)) : hex(0xE6EDF3);
                    NSAttributedString *body = [[[NSAttributedString alloc]
                        initWithString:ns(bodyText())
                            attributes:@{NSForegroundColorAttributeName: color,
                                         NSFontAttributeName: [NSFont systemFontOfSize:14]}] autorelease];
                    [answerView.textStorage setAttributedString:body];
                    answerView.frame = NSMakeRect(0, 0, kW - 32, answerView.frame.size.height);
                    [answerView.layoutManager ensureLayoutForTextContainer:answerView.textContainer];
                    answerH = std::ceil([answerView.layoutManager usedRectForTextContainer:answerView.textContainer].size.height);
                    const double shown = std::min(kSpotlightAnswerMaxHeight, std::max(18.0, answerH));
                    answerScroll.frame = NSMakeRect(16, kSpotlightBaseHeight + 35, kW - 32, shown);
                    if (s.busy)
                        [answerView scrollRangeToVisible:NSMakeRange(answerView.string.length, 0)];
                    answerH = shown;
                }
            }
        }
        const double h = spotlightHeight(s.drawerVisible, s.drawerVisible ? std::max(18.0, answerH) : 0);
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
        if (NSText *ed = [input currentEditor])
            [ed setSelectedRange:NSMakeRange(input.stringValue.length, 0)];
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
        const bool reposition = !s.open;
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
        s.listening = false;
        s.actionRunning = false;
        s.key = false;
        s.alpha = 0;
        if (was)
            ++s.dismissals;
        syncDerived();
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

@implementation LapisSpotlightFieldDelegate
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

CuaSpotlight::CuaSpotlight(Mode mode) : impl(std::make_unique<Impl>())
{
    impl->q = this;
    impl->offscreen = mode == Mode::Offscreen;
    impl->syncDerived();
    if (!impl->offscreen)
        g_liveSpotlight = this;
}

CuaSpotlight::~CuaSpotlight()
{
    if (g_liveSpotlight == this)
        g_liveSpotlight = nullptr;
    impl->teardown();
}

bool CuaSpotlight::openOrFocus() { return impl->openOrFocus(); }

void CuaSpotlight::dismiss(bool restoreTarget) { impl->close(restoreTarget); }

bool CuaSpotlight::isOpen() const { return impl->s.open; }

void CuaSpotlight::setListening(bool on)
{
    if (impl->s.listening == on)
        return;
    impl->s.listening = on;
    impl->render(false);
}

void CuaSpotlight::setInputText(const QString &text)
{
    impl->s.inputText = text;
    if (impl->input) {
        impl->input.stringValue = ns(text);
        if (NSText *ed = [impl->input currentEditor])
            [ed setSelectedRange:NSMakeRange(impl->input.stringValue.length, 0)];
    }
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

}  // namespace studio::cua
