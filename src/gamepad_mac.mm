// GameController.framework half of the gamepad bridge. Nothing here runs unless
// Gamepad::setEnabled(true) is called, which only the interactive live window does.
#include "gamepad.hpp"

#include <QElapsedTimer>
#include <QTimer>

#import <AppKit/AppKit.h>
#import <CoreGraphics/CoreGraphics.h>
#import <Foundation/Foundation.h>
#import <GameController/GameController.h>

#include <atomic>

namespace studio {

namespace {
std::atomic<int> g_platformCalls{0};

// The target builds without ARC (like pdfrender.mm): own what we keep explicitly.
template <typename T> T keep(T obj)
{
#if __has_feature(objc_arc)
    return obj;
#else
    return [obj retain];
#endif
}
void drop(id obj)
{
#if !__has_feature(objc_arc)
    [obj release];
#else
    (void)obj;
#endif
}
}  // namespace

int Gamepad::platformCalls()
{
    return g_platformCalls.load();
}

struct Gamepad::Platform {
    Gamepad *q = nullptr;
    id connectObserver = nil;
    id disconnectObserver = nil;
    GCController *pad = nil;  // the controller we read (first with an extended profile)
    QTimer poll;
    QElapsedTimer clock;
    bool bgDpadHeld = false;  // consumes one D-pad press to summon Lapis Studio when backgrounded

    explicit Platform(Gamepad *owner) : q(owner)
    {
        poll.setInterval(int(GamepadState::kScrollTickMs));  // 8 ms
        poll.setTimerType(Qt::PreciseTimer);
        QObject::connect(&poll, &QTimer::timeout, q, [this] { tick(); });
        clock.start();
    }

    ~Platform()
    {
        stop();
    }

    void start()
    {
        ++g_platformCalls;
        if (@available(macOS 11.3, *))
            GCController.shouldMonitorBackgroundEvents = YES;  // keep controller active even if another app takes focus
        NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];
        connectObserver = keep([nc addObserverForName:GCControllerDidConnectNotification object:nil
                                           queue:[NSOperationQueue mainQueue]
                                      usingBlock:^(NSNotification *) { rescan(); }]);
        disconnectObserver = keep([nc addObserverForName:GCControllerDidDisconnectNotification object:nil
                                              queue:[NSOperationQueue mainQueue]
                                         usingBlock:^(NSNotification *) { rescan(); }]);
        rescan();  // a pad paired before launch
    }

    void stop()
    {
        poll.stop();
        NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];
        if (connectObserver)
            [nc removeObserver:connectObserver];
        if (disconnectObserver)
            [nc removeObserver:disconnectObserver];
        drop(connectObserver);
        drop(disconnectObserver);
        drop(pad);
        connectObserver = nil;
        disconnectObserver = nil;
        pad = nil;
    }

    void rescan()
    {
        ++g_platformCalls;
        GCController *found = nil;
        for (GCController *c in [GCController controllers]) {
            if (c.extendedGamepad) {
                found = c;
                break;
            }
        }
        if (found != pad) {
            drop(pad);
            pad = found ? keep(found) : nil;
        }
        if (!found) {
            poll.stop();
            q->setConnected(false, QString());
            return;
        }
        NSString *label = nil;
        if (@available(macOS 10.15, *))
            label = found.productCategory;  // "DualShock 4", "DualSense", "Xbox One", ...
        if (label.length == 0)
            label = found.vendorName;
        q->setConnected(true, label.length > 0 ? QString::fromNSString(label) : QStringLiteral("Gamepad"));
        if (!poll.isActive())
            poll.start();
    }

    void tick()
    {
        GCExtendedGamepad *g = pad.extendedGamepad;
        if (!g)
            return;
        GamepadSnapshot s;
        s.lx = g.leftThumbstick.xAxis.value;
        s.ly = g.leftThumbstick.yAxis.value;
        s.rx = g.rightThumbstick.xAxis.value;
        s.ry = g.rightThumbstick.yAxis.value;
        s.dpadX = g.dpad.right.pressed ? 1 : (g.dpad.left.pressed ? -1 : 0);
        s.dpadY = g.dpad.up.pressed ? 1 : (g.dpad.down.pressed ? -1 : 0);
        const bool dpadDown = (s.dpadX != 0 || s.dpadY != 0);
        // Background (another app is frontmost, or only the Spotlight bar is up over it): a
        // d-pad press summons full Studio instead of navigating a window nobody is looking at.
        const bool overlay = q->m_overlayActive && q->m_overlayActive();
        if (NSApp && (![NSApp isActive] || overlay)) {
            if (dpadDown && !bgDpadHeld) {
                bgDpadHeld = true;
                if (q->m_summonStudio) {
                    q->m_summonStudio();
                } else {
                    [NSApp unhide:nil];
                    [NSApp activateIgnoringOtherApps:YES];
                }
            }
            if (dpadDown || bgDpadHeld) {
                s.dpadX = 0;
                s.dpadY = 0;
            }
        }
        if (!dpadDown)
            bgDpadHeld = false;
        s.a = g.buttonA.pressed;
        s.b = g.buttonB.pressed;
        s.x = g.buttonX.pressed;
        s.y = g.buttonY.pressed;
        s.l1 = g.leftShoulder.pressed;
        s.r1 = g.rightShoulder.pressed;
        s.l2 = g.leftTrigger.value;
        s.r2 = g.rightTrigger.value;
        s.l3 = g.leftThumbstickButton.pressed;
        s.r3 = g.rightThumbstickButton.pressed;
        s.menu = g.buttonMenu.pressed;
        s.options = g.buttonOptions.pressed;
        if ([g isKindOfClass:[GCDualShockGamepad class]]) {
            GCDualShockGamepad *ds4 = static_cast<GCDualShockGamepad *>(g);
            s.touchpad = ds4.touchpadButton.pressed;
            s.touchpadX = ds4.touchpadPrimary.xAxis.value;
            s.touchpadY = ds4.touchpadPrimary.yAxis.value;
        } else if (@available(macOS 11.3, *)) {
            if ([g isKindOfClass:[GCDualSenseGamepad class]]) {
                GCDualSenseGamepad *ds = static_cast<GCDualSenseGamepad *>(g);
                s.touchpad = ds.touchpadButton.pressed;
                s.touchpadX = ds.touchpadPrimary.xAxis.value;
                s.touchpadY = ds.touchpadPrimary.yAxis.value;
            }
        }
        q->processSnapshot(s, clock.elapsed());
    }
};

Gamepad::Gamepad(QObject *parent) : QObject(parent) {}

Gamepad::~Gamepad() = default;

void Gamepad::setEnabled(bool on)
{
    if (on == m_enabled)
        return;
    m_enabled = on;
    if (on) {
        m_platform = std::make_unique<Platform>(this);
        m_platform->start();
    } else {
        m_platform.reset();
        setConnected(false, QString());
    }
    emit enabledChanged();
}

void Gamepad::triggerVoiceKey(bool down)
{
    if (m_voiceKeySink) {
        m_voiceKeySink(down);
        return;
    }
    if (!m_enabled)
        return;
    ++g_platformCalls;
    // Phonon listens via CGEventTap for kCGEventFlagsChanged with kCGEventFlagMaskSecondaryFn (fn/Globe).
    CGEventRef ev = CGEventCreate(nullptr);
    if (!ev)
        return;
    CGEventSetType(ev, kCGEventFlagsChanged);
    CGEventSetIntegerValueField(ev, kCGKeyboardEventKeycode, 0x3F);
    CGEventSetFlags(ev, down ? kCGEventFlagMaskSecondaryFn : 0);
    CGEventPost(kCGHIDEventTap, ev);
    CFRelease(ev);
}

}  // namespace studio
