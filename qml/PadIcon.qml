import QtQuick

// A controller button drawn as a small vector keycap (DualSense silhouettes, Lapis palette only:
// obsidian fill, dim rim, parchment glyph; gold when active). `button` takes a binding slot id
// ("a", "r2", "l3", "menu", "dpad_lr", ...) or the on-screen glyph for it ("✕", "R2", "☰",
// "Share", "PAD", "↕", "◄►"). `size` is the height; the width follows the silhouette: round for
// face buttons, sticks and the d-pad, wider for bumpers, triggers and the touchpad.
Canvas {
    id: icon
    property string button: ""
    property real size: 20
    property bool active: false
    property color inkColor: active ? Theme.gold : Theme.ink
    property color rimColor: active ? Theme.gold : Theme.dim
    property color fillColor: active ? Theme.addBg : Theme.cardHi

    // cross circle square triangle l1 r1 l2 r2 l3 r3 lstick rstick options touchpad
    // dpad dpad_up dpad_down dpad_left dpad_right dpad_ud dpad_lr, or "" (unknown: draws nothing)
    readonly property string kind: kindOf(button)
    readonly property real aspect: {
        switch (kind) {
        case "l1": case "r1": case "l2": case "r2": case "touchpad": return 1.45;
        case "options": return 0.8;
        case "": return 0;
        default: return 1;
        }
    }
    implicitHeight: kind.length > 0 ? size : 0
    implicitWidth: Math.round(size * aspect)
    width: implicitWidth
    height: implicitHeight
    renderTarget: Canvas.Image
    antialiasing: true

    function kindOf(b) {
        const s = String(b).trim();
        const l = s.toLowerCase();
        switch (l) {
        case "a": case "✕": case "×": case "✖": case "cross": return "cross";
        case "b": case "○": case "◯": case "circle": return "circle";
        case "x": case "□": case "◻": case "square": return "square";
        case "y": case "△": case "triangle": return "triangle";
        case "l1": case "r1": case "l2": case "r2": case "l3": case "r3": case "lstick": case "rstick": return l;
        case "menu": case "☰": case "options ☰": return "options";
        case "options": case "share": case "pad": case "touchpad": return "touchpad";
        case "dpad": case "d-pad": case "dpad_up": case "dpad_down": case "dpad_left": case "dpad_right":
        case "dpad_ud": case "dpad_lr": return l === "d-pad" ? "dpad" : l;
        case "↕": case "▲▼": return "dpad_ud";
        case "◄►": case "↔": case "◀▶": case "←→": return "dpad_lr";
        case "↑": case "▲": return "dpad_up";
        case "↓": case "▼": return "dpad_down";
        case "←": case "◄": case "◀": return "dpad_left";
        case "→": case "►": case "▶": return "dpad_right";
        }
        return "";
    }

    onKindChanged: requestPaint()
    onActiveChanged: requestPaint()
    onInkColorChanged: requestPaint()
    onRimColorChanged: requestPaint()
    onFillColorChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    function tint(c, a) { return Qt.rgba(c.r, c.g, c.b, a); }

    // A closed polygon with every corner rounded by r (arcTo between edge midpoints).
    function roundedPoly(ctx, pts, r) {
        const n = pts.length;
        const mid = (p, q) => [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2];
        const m0 = mid(pts[n - 1], pts[0]);
        ctx.beginPath();
        ctx.moveTo(m0[0], m0[1]);
        for (let i = 0; i < n; ++i) {
            const p = pts[i], q = pts[(i + 1) % n];
            const m = mid(p, q);
            ctx.arcTo(p[0], p[1], m[0], m[1], r);
        }
        ctx.closePath();
    }

    function keycapCircle(ctx, cx, cy, r) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, 2 * Math.PI);
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.lineWidth = 1.25;
        ctx.strokeStyle = rimColor;
        ctx.stroke();
        // A faint top highlight: the cap catches light from above.
        ctx.beginPath();
        ctx.arc(cx, cy, r - 1.6, Math.PI * 1.18, Math.PI * 1.82);
        ctx.lineWidth = 1;
        ctx.strokeStyle = tint(Theme.ink, active ? 0.18 : 0.10);
        ctx.stroke();
    }

    function label(ctx, text, cx, cy, px) {
        ctx.fillStyle = inkColor;
        ctx.font = "bold " + Math.max(6, Math.round(px)) + "px Menlo";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, cx, cy);
    }

    onPaint: {
        const ctx = getContext("2d");
        ctx.reset();
        const w = width, h = height;
        if (w <= 0 || h <= 0 || kind.length === 0)
            return;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        const cx = w / 2, cy = h / 2;
        switch (kind) {
        case "cross": {
            const r = h / 2 - 0.9;
            keycapCircle(ctx, cx, cy, r);
            const d = r * 0.36;
            ctx.beginPath();
            ctx.moveTo(cx - d, cy - d); ctx.lineTo(cx + d, cy + d);
            ctx.moveTo(cx + d, cy - d); ctx.lineTo(cx - d, cy + d);
            ctx.lineWidth = 1.6;
            ctx.strokeStyle = inkColor;
            ctx.stroke();
            break;
        }
        case "circle": {
            keycapCircle(ctx, cx, cy, h / 2 - 0.9);
            ctx.beginPath();
            ctx.arc(cx, cy, h * 0.2, 0, 2 * Math.PI);
            ctx.lineWidth = 1.6;
            ctx.strokeStyle = inkColor;
            ctx.stroke();
            break;
        }
        case "square": {
            keycapCircle(ctx, cx, cy, h / 2 - 0.9);
            const s = h * 0.34;
            ctx.beginPath();
            ctx.roundedRect(cx - s / 2, cy - s / 2, s, s, 1.2, 1.2);
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = inkColor;
            ctx.stroke();
            break;
        }
        case "triangle": {
            keycapCircle(ctx, cx, cy, h / 2 - 0.9);
            const R = h * 0.235, oy = cy + h * 0.03;  // optical centre sits a touch low
            ctx.beginPath();
            ctx.moveTo(cx, oy - R);
            ctx.lineTo(cx + R * 0.866, oy + R * 0.5);
            ctx.lineTo(cx - R * 0.866, oy + R * 0.5);
            ctx.closePath();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = inkColor;
            ctx.stroke();
            break;
        }
        case "l1": case "r1": {
            // Low-profile bumper: a gently arched top edge over a flat, rounded bottom.
            const x0 = 0.75, x1 = w - 0.75, yb = h * 0.86, ys = h * 0.34, yt = h * 0.16, rc = Math.min(3, h * 0.16);
            ctx.beginPath();
            ctx.moveTo(x0 + rc, yb);
            ctx.lineTo(x1 - rc, yb);
            ctx.quadraticCurveTo(x1, yb, x1, yb - rc);
            ctx.lineTo(x1, ys + rc);
            ctx.quadraticCurveTo(x1, ys, x1 - rc, ys - 0.4);
            ctx.quadraticCurveTo(cx, 2 * yt - ys, x0 + rc, ys - 0.4);
            ctx.quadraticCurveTo(x0, ys, x0, ys + rc);
            ctx.lineTo(x0, yb - rc);
            ctx.quadraticCurveTo(x0, yb, x0 + rc, yb);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.lineWidth = 1.25;
            ctx.strokeStyle = rimColor;
            ctx.stroke();
            label(ctx, kind.toUpperCase(), cx, (ys + yb) / 2 + 0.3, h * 0.42);
            break;
        }
        case "l2": case "r2": {
            // Sculpted trigger: a narrow curved top lip flaring out to a wide, concave bottom lip.
            const x0 = 0.75, x1 = w - 0.75, yt = h * 0.12, yb = h * 0.9;
            const xl = w * 0.2, xr = w * 0.8, rc = Math.min(2.5, h * 0.13);
            ctx.beginPath();
            ctx.moveTo(xl, yt + 1.2);
            ctx.quadraticCurveTo(cx, yt - h * 0.1, xr, yt + 1.2);
            ctx.bezierCurveTo(xr + w * 0.09, yt + h * 0.16, x1, yb - h * 0.36, x1, yb - rc);
            ctx.quadraticCurveTo(x1, yb, x1 - rc, yb);
            ctx.quadraticCurveTo(cx, yb - h * 0.2, x0 + rc, yb);
            ctx.quadraticCurveTo(x0, yb, x0, yb - rc);
            ctx.bezierCurveTo(x0, yb - h * 0.36, xl - w * 0.09, yt + h * 0.16, xl, yt + 1.2);
            ctx.closePath();
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.lineWidth = 1.25;
            ctx.strokeStyle = rimColor;
            ctx.stroke();
            // The lip's highlight.
            ctx.beginPath();
            ctx.moveTo(xl + 1.5, yt + 2.6);
            ctx.quadraticCurveTo(cx, yt - h * 0.06, xr - 1.5, yt + 2.6);
            ctx.lineWidth = 1;
            ctx.strokeStyle = tint(Theme.ink, active ? 0.2 : 0.12);
            ctx.stroke();
            label(ctx, kind.toUpperCase(), cx, h * 0.53, h * 0.42);
            break;
        }
        case "l3": case "r3": case "lstick": case "rstick": {
            // Analog stick from above: base ring, concave cap, its letter; L3 / R3 add a press chevron.
            const r = h / 2 - 0.9, rc = h * 0.31;
            keycapCircle(ctx, cx, cy, r);
            ctx.beginPath();
            ctx.arc(cx, cy, rc, 0, 2 * Math.PI);
            ctx.fillStyle = active ? Theme.addBg : Theme.card;
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = rimColor;
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(cx, cy, rc - 1.6, Math.PI * 0.15, Math.PI * 0.85);  // the dish's lower lip
            ctx.strokeStyle = tint(Theme.ink, 0.12);
            ctx.stroke();
            const press = kind === "l3" || kind === "r3";
            if (press) {
                const yc = cy - (r + rc) / 2, d = h * 0.075;
                ctx.beginPath();
                ctx.moveTo(cx - d, yc - d * 0.55);
                ctx.lineTo(cx, yc + d * 0.55);
                ctx.lineTo(cx + d, yc - d * 0.55);
                ctx.lineWidth = 1.1;
                ctx.strokeStyle = inkColor;
                ctx.stroke();
            }
            label(ctx, kind.charAt(0).toUpperCase(), cx, cy + 0.4, h * 0.34);
            break;
        }
        case "options": {
            // DualSense Options: a vertical stadium with three bars.
            const pw = h * 0.56, ph = h - 1.8;
            ctx.beginPath();
            ctx.roundedRect(cx - pw / 2, cy - ph / 2, pw, ph, pw / 2, pw / 2);
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.lineWidth = 1.25;
            ctx.strokeStyle = rimColor;
            ctx.stroke();
            const bw = pw * 0.5;
            ctx.beginPath();
            for (let i = -1; i <= 1; ++i) {
                ctx.moveTo(cx - bw / 2, cy + i * h * 0.15);
                ctx.lineTo(cx + bw / 2, cy + i * h * 0.15);
            }
            ctx.lineWidth = 1.2;
            ctx.strokeStyle = inkColor;
            ctx.stroke();
            break;
        }
        case "touchpad": {
            // The touchpad plate: wider at the top, rounded, with a row of micro-dots.
            const yt = h * 0.14, yb = h * 0.86, x0 = 0.75, x1 = w - 0.75, inset = w * 0.07;
            roundedPoly(ctx, [[x0, yt], [x1, yt], [x1 - inset, yb], [x0 + inset, yb]], Math.min(3, h * 0.16));
            ctx.fillStyle = fillColor;
            ctx.fill();
            ctx.lineWidth = 1.25;
            ctx.strokeStyle = rimColor;
            ctx.stroke();
            ctx.fillStyle = tint(inkColor, 0.9);
            const n = 5, span = w * 0.42;
            for (let i = 0; i < n; ++i) {
                ctx.beginPath();
                ctx.arc(cx - span / 2 + span * i / (n - 1), h * 0.34, Math.max(0.7, h * 0.045), 0, 2 * Math.PI);
                ctx.fill();
            }
            // The click seam across the plate.
            ctx.beginPath();
            ctx.moveTo(cx - w * 0.18, h * 0.6);
            ctx.lineTo(cx + w * 0.18, h * 0.6);
            ctx.lineWidth = 1.2;
            ctx.strokeStyle = tint(inkColor, 0.75);
            ctx.stroke();
            break;
        }
        default: {
            // D-pad: a rounded 4-way cross; the arms this icon names are filled.
            const a = h * 0.36, e0 = 0.8, e1 = h - 0.8, lw = 1.25;
            const arms = kind === "dpad_ud" ? ["up", "down"] : kind === "dpad_lr" ? ["left", "right"]
                       : kind === "dpad" ? [] : [kind.substring(5)];
            const cross = (inset, color) => {
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.roundedRect(cx - a / 2 + inset, e0 + inset, a - 2 * inset, e1 - e0 - 2 * inset, 2, 2);
                ctx.fill();
                ctx.beginPath();
                ctx.roundedRect(e0 + inset, cy - a / 2 + inset, e1 - e0 - 2 * inset, a - 2 * inset, 2, 2);
                ctx.fill();
            };
            cross(0, rimColor);
            cross(lw, fillColor);
            // Each arm: its tip area (filled when named) and a small direction wedge.
            const reach = (h - a) / 2 - 1.6;
            const armRect = {
                up: [cx - a / 2 + lw, e0 + lw, a - 2 * lw, reach],
                down: [cx - a / 2 + lw, e1 - lw - reach, a - 2 * lw, reach],
                left: [e0 + lw, cy - a / 2 + lw, reach, a - 2 * lw],
                right: [e1 - lw - reach, cy - a / 2 + lw, reach, a - 2 * lw]
            };
            const dirs = ["up", "down", "left", "right"];
            for (let i = 0; i < 4; ++i) {
                const d = dirs[i], r = armRect[d], on = arms.indexOf(d) >= 0;
                if (on) {
                    ctx.fillStyle = tint(inkColor, 0.92);
                    ctx.beginPath();
                    ctx.roundedRect(r[0], r[1], r[2], r[3], 1.2, 1.2);
                    ctx.fill();
                }
                const mx = r[0] + r[2] / 2, my = r[1] + r[3] / 2, t = h * 0.075;
                ctx.beginPath();
                if (d === "up") { ctx.moveTo(mx, my - t); ctx.lineTo(mx + t, my + t * 0.6); ctx.lineTo(mx - t, my + t * 0.6); }
                else if (d === "down") { ctx.moveTo(mx, my + t); ctx.lineTo(mx + t, my - t * 0.6); ctx.lineTo(mx - t, my - t * 0.6); }
                else if (d === "left") { ctx.moveTo(mx - t, my); ctx.lineTo(mx + t * 0.6, my - t); ctx.lineTo(mx + t * 0.6, my + t); }
                else { ctx.moveTo(mx + t, my); ctx.lineTo(mx - t * 0.6, my - t); ctx.lineTo(mx - t * 0.6, my + t); }
                ctx.closePath();
                ctx.fillStyle = on ? fillColor : tint(inkColor, 0.8);
                ctx.fill();
            }
            break;
        }
        }
    }
}
