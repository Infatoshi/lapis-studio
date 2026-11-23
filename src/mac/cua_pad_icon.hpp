// Vector DualSense-style controller-button icons for the desktop overlays (Spotlight hint bar,
// HUD capsule). Pure CoreGraphics + CoreText: safe off the main thread and without an NSApp.
#pragma once

#include <CoreGraphics/CoreGraphics.h>
#include <QImage>
#include <QString>

namespace studio::cua {

enum class PadButton { Cross, Circle, Square, Triangle, L1, R1, L2, R2, L3, R3, Options, Touchpad };
constexpr int kPadButtonCount = 12;

struct PadRgba {
    CGFloat r = 0, g = 0, b = 0, a = 1;
};
struct PadIconColors {
    PadRgba ink;   // glyph / label
    PadRgba rim;   // outline
    PadRgba fill;  // keycap body
};

// Lapis palette. Idle: fill #14120f, rim #6e665e, ink #e8d5b5. Active: fill #1a1309 with rim and
// ink in the accent (Amber Gold #e09f3e, or #4f8cff in lapis-blue mode).
PadIconColors padIconColors(bool active, bool lapisBlue);
// "Cross", "L2", ... (tests, accessibility).
QString padButtonName(PadButton b);
// The icon's natural size at `height` points: face buttons and sticks 1:1, bumpers, triggers and
// the touchpad 1.45:1, Options a vertical stadium 0.66:1.
CGSize padIconSize(PadButton b, CGFloat height);
// Draws the icon filling `rect` in a y-down (flipped) user space, e.g. a flipped NSView or a
// bitmap context with a flipped CTM. Leaves the context's graphics state as it found it.
void drawPadIcon(CGContextRef ctx, PadButton b, CGRect rect, const PadIconColors &colors);
// One icon on a transparent ARGB32-premultiplied image, padIconSize(height) * scale pixels.
QImage renderPadIcon(PadButton b, double height, double scale = 2, bool active = false, bool lapisBlue = false);

}  // namespace studio::cua
