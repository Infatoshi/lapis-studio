#include "mac/cua_reticle.hpp"
#include "mac/cua.hpp"
#import <AppKit/AppKit.h>
#include <cmath>

@interface LapisCuaPanel : NSPanel
@end
@implementation LapisCuaPanel
- (BOOL)canBecomeKeyWindow { return NO; }
- (BOOL)canBecomeMainWindow { return NO; }
@end

@interface LapisCuaReticleView : NSView {
    NSString *_label;
}
- (void)setLabel:(NSString *)label;
@end
@implementation LapisCuaReticleView
- (void)dealloc { [_label release]; [super dealloc]; }
- (void)setLabel:(NSString *)label {
    [_label release]; _label = [label copy]; [self setNeedsDisplay:YES];
}
- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    [[NSColor colorWithSRGBRed:0.4 green:0.85 blue:1 alpha:0.95] setStroke];
    NSBezierPath *ring = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(8, 38, 24, 24)];
    ring.lineWidth = 2; [ring stroke];
    NSBezierPath *dot = [NSBezierPath bezierPathWithOvalInRect:NSMakeRect(18, 48, 4, 4)];
    [[NSColor whiteColor] setFill]; [dot fill];
    if (_label.length) {
        [[NSColor colorWithWhite:0.08 alpha:0.88] setFill];
        [[NSBezierPath bezierPathWithRoundedRect:NSMakeRect(6, 6, 228, 26) xRadius:7 yRadius:7] fill];
        NSMutableParagraphStyle *style = [[[NSMutableParagraphStyle alloc] init] autorelease];
        style.lineBreakMode = NSLineBreakByTruncatingTail;
        [_label drawInRect:NSMakeRect(13, 10, 214, 18) withAttributes:@{
            NSFontAttributeName: [NSFont systemFontOfSize:12 weight:NSFontWeightMedium],
            NSForegroundColorAttributeName: [NSColor colorWithWhite:1 alpha:0.9],
            NSParagraphStyleAttributeName: style}];
    }
}
@end

namespace studio::cua {
namespace {
void requireMainThread() { if (![NSThread isMainThread]) qFatal("CuaReticle requires the AppKit main thread"); }
}
struct CuaReticle::Impl {
    LapisCuaPanel *panel = nil;
    LapisCuaReticleView *view = nil;
};
CuaReticle::CuaReticle() : impl(std::make_unique<Impl>()) {
    requireMainThread();
    @autoreleasepool {
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
            NSWindowCollectionBehaviorFullScreenAuxiliary | NSWindowCollectionBehaviorIgnoresCycle;
        impl->panel.sharingType = NSWindowSharingNone;
        impl->view = [[LapisCuaReticleView alloc] initWithFrame:NSMakeRect(0, 0, 240, 72)];
        impl->panel.contentView = impl->view;
    }
}
CuaReticle::~CuaReticle() {
    requireMainThread();
    [impl->panel orderOut:nil]; [impl->panel close]; [impl->panel release]; [impl->view release];
}
void CuaReticle::moveTo(QPointF point, const QString &label) {
    requireMainThread();
    if (!std::isfinite(point.x()) || !std::isfinite(point.y())) return;
    @autoreleasepool {
        const auto displays = listDisplays();
        QRectF primary;
        for (const auto &d : displays) if (d.primary) primary = d.globalPoints;
        if (!primary.isValid()) return;
        const auto p = globalPointToAppKit(point, primary);
        [impl->view setLabel:label.left(120).toNSString()];
        NSRect frame = impl->panel.frame;
        frame.origin = NSMakePoint(p.x() - 20, p.y() - 50);
        if (impl->panel.visible) {
            [NSAnimationContext runAnimationGroup:^(NSAnimationContext *context) {
                context.duration = 0.10;
                [[impl->panel animator] setFrame:frame display:YES];
            } completionHandler:nil];
        } else {
            [impl->panel setFrame:frame display:YES];
            [impl->panel orderFrontRegardless];
        }
    }
}
void CuaReticle::hide() { requireMainThread(); [impl->panel orderOut:nil]; }
uint32_t CuaReticle::windowNumber() const { requireMainThread(); return uint32_t(impl->panel.windowNumber); }
} // namespace studio::cua
