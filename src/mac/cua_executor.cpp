#include "mac/cua_executor.hpp"
#include "mac/cua_reticle.hpp"

#include <QBuffer>
#include <QCoreApplication>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QThread>
#include <curl/curl.h>
#include <algorithm>
#include <cmath>
#include <mutex>

namespace studio::cua {

namespace {

bool finitePoint(QPointF p) { return std::isfinite(p.x()) && std::isfinite(p.y()); }

// [x, y] or {"x": .., "y": ..}; false when neither shape holds two finite numbers.
bool jsonPoint(const QJsonValue &v, QPointF *out)
{
    double x = NAN, y = NAN;
    if (v.isArray()) {
        const QJsonArray a = v.toArray();
        if (a.size() < 2 || !a.at(0).isDouble() || !a.at(1).isDouble())
            return false;
        x = a.at(0).toDouble();
        y = a.at(1).toDouble();
    } else if (v.isObject()) {
        const QJsonObject o = v.toObject();
        if (!o.value(QStringLiteral("x")).isDouble() || !o.value(QStringLiteral("y")).isDouble())
            return false;
        x = o.value(QStringLiteral("x")).toDouble();
        y = o.value(QStringLiteral("y")).toDouble();
    } else {
        return false;
    }
    if (!std::isfinite(x) || !std::isfinite(y))
        return false;
    *out = {x, y};
    return true;
}

bool jsonSize(const QJsonValue &v, QSizeF *out)
{
    QPointF p;
    if (v.isObject()) {
        const QJsonObject o = v.toObject();
        if (!o.value(QStringLiteral("width")).isDouble() || !o.value(QStringLiteral("height")).isDouble())
            return false;
        p = {o.value(QStringLiteral("width")).toDouble(), o.value(QStringLiteral("height")).toDouble()};
    } else if (!jsonPoint(v, &p)) {
        return false;
    }
    if (!(p.x() > 0) || !(p.y() > 0) || !std::isfinite(p.x()) || !std::isfinite(p.y()))
        return false;
    *out = {p.x(), p.y()};
    return true;
}

size_t appendBody(char *data, size_t size, size_t n, void *user)
{
    auto *buf = static_cast<QByteArray *>(user);
    if (buf->size() > 1 << 20)
        return 0;  // a grounder reply is a few hundred bytes; refuse anything absurd
    buf->append(data, qsizetype(size * n));
    return size * n;
}

bool clickableRole(const QString &role)
{
    static const QStringList roles{QStringLiteral("AXButton"),   QStringLiteral("AXLink"),
                                   QStringLiteral("AXMenuItem"), QStringLiteral("AXCheckBox"),
                                   QStringLiteral("AXTextField"), QStringLiteral("AXTab"),
                                   QStringLiteral("AXRadioButton"), QStringLiteral("AXPopUpButton"),
                                   QStringLiteral("AXMenuBarItem"), QStringLiteral("AXTextArea"),
                                   QStringLiteral("AXSearchField"), QStringLiteral("AXComboBox")};
    return roles.contains(role);
}

QSizeF jpegSize(const QByteArray &jpeg)
{
    QByteArray copy = jpeg;
    QBuffer buf(&copy);
    if (!buf.open(QIODevice::ReadOnly))
        return {};
    QImageReader reader(&buf);
    const QSize s = reader.size();
    return s.isValid() && !s.isEmpty() ? QSizeF(s) : QSizeF();
}

// Process-lifetime overlay, created and used on the main thread only. Deliberately never
// destroyed: AppKit teardown order at exit is not ours to fight.
CuaReticle *sharedReticle()
{
    static CuaReticle *r = new CuaReticle();
    return r;
}

constexpr int kGlideSettleMs = 170;  // the reticle glide (140 ms) plus a beat, so the click lands visibly
constexpr int kPulseLeadMs = 60;

}  // namespace

GroundResult parseGroundReply(const QByteArray &json)
{
    GroundResult g;
    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) {
        g.detail = QStringLiteral("Grounder reply is not JSON");
        return g;
    }
    const QJsonObject o = doc.object();
    if (o.value(QStringLiteral("ok")).isBool() && !o.value(QStringLiteral("ok")).toBool()) {
        const QString e = o.value(QStringLiteral("error")).toString();
        g.detail = e.isEmpty() ? QStringLiteral("Grounder found no target") : e;
        return g;
    }
    g.hasPoint1000 = jsonPoint(o.value(QStringLiteral("point_1000")), &g.point1000);
    g.hasPointPx = jsonPoint(o.value(QStringLiteral("point_px")), &g.pointPx);
    if (!jsonSize(o.value(QStringLiteral("image_size")), &g.imageSize))
        g.imageSize = {};
    if (g.hasPoint1000
        && (g.point1000.x() < 0 || g.point1000.x() > 1000 || g.point1000.y() < 0 || g.point1000.y() > 1000))
        g.hasPoint1000 = false;
    g.ok = g.hasPoint1000 || g.hasPointPx;
    if (!g.ok) {
        const QString e = o.value(QStringLiteral("error")).toString();
        g.detail = e.isEmpty() ? QStringLiteral("Grounder reply has no point") : e;
    }
    return g;
}

QByteArray buildGroundRequest(const QByteArray &jpeg, const QString &instruction, bool refineCrop)
{
    QJsonObject o;
    o.insert(QStringLiteral("image_b64"), QString::fromLatin1(jpeg.toBase64()));
    o.insert(QStringLiteral("instruction"), instruction);
    o.insert(QStringLiteral("refine_crop"), refineCrop);
    return QJsonDocument(o).toJson(QJsonDocument::Compact);
}

QString grounderUrl()
{
    const QString env = qEnvironmentVariable("LAPIS_CUA_GROUNDER_URL").trimmed();
    return env.isEmpty() ? QStringLiteral("http://127.0.0.1:18120/ground") : env;
}

GroundResult httpGroundPixel(const QByteArray &jpeg, const QString &instruction, bool refineCrop)
{
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    GroundResult fail;
    CURL *h = curl_easy_init();
    if (!h) {
        fail.detail = QStringLiteral("libcurl init failed");
        return fail;
    }
    const QByteArray body = buildGroundRequest(jpeg, instruction, refineCrop);
    const QByteArray url = grounderUrl().toUtf8();
    QByteArray reply;
    curl_slist *headers = curl_slist_append(nullptr, "Content-Type: application/json");
    curl_easy_setopt(h, CURLOPT_URL, url.constData());
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS, body.constData());
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE_LARGE, curl_off_t(body.size()));
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, 4000L);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, 4000L);
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &appendBody);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &reply);
    const CURLcode rc = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(h);
    if (rc != CURLE_OK) {
        fail.detail = QStringLiteral("Grounder offline (%1)").arg(QString::fromUtf8(curl_easy_strerror(rc)));
        return fail;
    }
    if (status != 200) {
        fail.detail = QStringLiteral("Grounder HTTP %1").arg(status);
        return fail;
    }
    return parseGroundReply(reply);
}

QSizeF expectedCaptureSize(const DisplayInfo &d, int maxWidth)
{
    if (!d.globalPoints.isValid() || !(d.scale > 0) || maxWidth <= 0)
        return {};
    const double w = std::max(1.0, std::min(double(maxWidth), d.globalPoints.width() * d.scale));
    const double h = std::max(1.0, std::round(w * d.globalPoints.height() / std::max(1.0, d.globalPoints.width())));
    return {w, h};
}

void runOnMainThread(const std::function<void()> &fn)
{
    QCoreApplication *app = QCoreApplication::instance();
    if (!app || !fn)
        return;
    if (QThread::currentThread() == app->thread())
        fn();
    else
        QMetaObject::invokeMethod(app, [&fn] { fn(); }, Qt::BlockingQueuedConnection);
}

CuaHooks defaultCuaHooks()
{
    CuaHooks h;
    h.listDisplays = [] { return studio::cua::listDisplays(); };
    h.inspectDisplay = [](uint32_t id) { return studio::cua::inspectDisplay(id, 400, 0.4); };
    h.captureDisplayJpeg = [](uint32_t id, int w, uint32_t exclude, QString *err) {
        return studio::cua::captureDisplayJpeg(id, w, exclude, err);
    };
    h.groundPixel = &httpGroundPixel;
    h.clickAt = [](QPointF p, bool dbl) { return studio::cua::clickAt(p, dbl); };
    h.scrollAt = [](QPointF p, int dy) { return studio::cua::scrollAt(p, dy); };
    h.typeText = [](const QString &t) { return studio::cua::typeText(t); };
    h.axPress = [](int pid, const QString &k, const QRectF &r) { return studio::cua::axPress(pid, k, r); };
    h.reticleMove = [](QPointF p, const QString &label) { runOnMainThread([&] { sharedReticle()->moveTo(p, label); }); };
    h.reticleHighlight = [](QRectF r, const QString &label) {
        runOnMainThread([&] { sharedReticle()->highlightRect(r, label); });
    };
    h.reticlePulse = [](QPointF p, const QString &label) {
        runOnMainThread([&] { sharedReticle()->pulseClick(p, label); });
    };
    h.reticleHide = [] { runOnMainThread([] { sharedReticle()->hide(); }); };
    h.reticleWindowId = [] {
        uint32_t id = 0;
        runOnMainThread([&] { id = sharedReticle()->windowNumber(); });
        return id;
    };
    h.pause = [](int ms) {
        if (ms > 0)
            QThread::msleep(static_cast<unsigned long>(ms));
    };
    return h;
}

int axMatchScore(const AxNode &node, const QString &targetDesc)
{
    const QString q = targetDesc.simplified().toLower();
    if (q.isEmpty() || !node.globalRect.isValid() || node.globalRect.isEmpty()
        || !std::isfinite(node.globalRect.x()) || !std::isfinite(node.globalRect.y())
        || !std::isfinite(node.globalRect.width()) || !std::isfinite(node.globalRect.height()))
        return 0;
    const QString title = node.title.simplified().toLower();
    const QString value = node.value.simplified().toLower();
    int score = 0;
    if (!title.isEmpty() && title == q)
        score = 100;
    else if (!value.isEmpty() && value == q)
        score = 90;
    else if (!title.isEmpty() && title.contains(q))
        score = 60;
    else if (!value.isEmpty() && value.contains(q))
        score = 50;
    if (score && clickableRole(node.role))
        score += 20;
    return score;
}

int bestAxMatch(const QVector<AxNode> &nodes, const QString &targetDesc)
{
    int best = -1, bestScore = 0;
    bool tied = false;
    for (int i = 0; i < nodes.size(); ++i) {
        const int s = axMatchScore(nodes[i], targetDesc);
        if (s > bestScore) {
            best = i;
            bestScore = s;
            tied = false;
        } else if (s && s == bestScore && nodes[i].globalRect != nodes[best].globalRect) {
            tied = true;  // a parent and child sharing one rect is still one target
        }
    }
    return tied ? -1 : best;
}

CuaExecutor::CuaExecutor(CuaHooks hooks) : m_hooks(std::move(hooks)) {}

bool CuaExecutor::primaryDisplay(DisplayInfo *out, QString *err) const
{
    const QVector<DisplayInfo> displays = m_hooks.listDisplays ? m_hooks.listDisplays() : QVector<DisplayInfo>();
    for (const DisplayInfo &d : displays)
        if (d.primary) {
            *out = d;
            return true;
        }
    if (!displays.isEmpty()) {
        *out = displays.first();
        return true;
    }
    *err = QStringLiteral("No display");
    return false;
}

CuaExecutor::Located CuaExecutor::locate(const QString &targetDesc, bool refineCrop, const QString &label)
{
    Located loc;
    DisplayInfo display;
    if (!primaryDisplay(&display, &loc.detail))
        return loc;

    // L1: Accessibility tree of the primary display.
    if (m_hooks.inspectDisplay) {
        const QVector<AxNode> nodes = m_hooks.inspectDisplay(display.displayId);
        const int i = bestAxMatch(nodes, targetDesc);
        if (i >= 0) {
            loc.ok = true;
            loc.tier = QStringLiteral("L1_AX");
            loc.node = nodes[i];
            loc.rect = nodes[i].globalRect;
            loc.pt = loc.rect.center();
            loc.detail = QStringLiteral("%1 \"%2\" (%3)").arg(loc.node.role, loc.node.title.isEmpty() ? loc.node.value : loc.node.title,
                                                               loc.node.appName);
            if (m_hooks.reticleHighlight)
                m_hooks.reticleHighlight(loc.rect, label);
            return loc;
        }
    }

    // L3: local VLM pixel grounding on a capture that leaves our own overlay out.
    if (!m_hooks.captureDisplayJpeg || !m_hooks.groundPixel) {
        loc.detail = QStringLiteral("No AX match and no grounder");
        return loc;
    }
    constexpr int kMaxWidth = 1280;
    const uint32_t exclude = m_hooks.reticleWindowId ? m_hooks.reticleWindowId() : 0;
    QString err;
    const QByteArray jpeg = m_hooks.captureDisplayJpeg(display.displayId, kMaxWidth, exclude, &err);
    if (jpeg.isEmpty()) {
        loc.detail = QStringLiteral("Capture failed: %1").arg(err.isEmpty() ? QStringLiteral("empty image") : err);
        return loc;
    }
    const GroundResult g = m_hooks.groundPixel(jpeg, targetDesc, refineCrop);
    if (!g.ok) {
        loc.detail = g.detail.isEmpty() ? QStringLiteral("Grounder failed") : g.detail;
        return loc;
    }
    QSizeF size = g.imageSize;
    if (!size.isValid() || size.isEmpty())
        size = jpegSize(jpeg);
    if (!size.isValid() || size.isEmpty())
        size = expectedCaptureSize(display, kMaxWidth);
    const QPointF px = g.hasPoint1000 ? QPointF(g.point1000.x() / 1000.0 * size.width(), g.point1000.y() / 1000.0 * size.height())
                                      : g.pointPx;
    const QPointF pt = capturePixelToGlobalPoint(px, display, size);
    const QRectF bounds = display.globalPoints.adjusted(0, 0, 0.001, 0.001);  // the far edge is on screen
    if (!finitePoint(pt) || !bounds.contains(pt)) {
        loc.detail = QStringLiteral("Grounder point is off the display");
        return loc;
    }
    loc.ok = true;
    loc.tier = QStringLiteral("L3_VLM");
    loc.pt = pt;
    loc.detail = QStringLiteral("VLM point (%1, %2)").arg(pt.x(), 0, 'f', 1).arg(pt.y(), 0, 'f', 1);
    if (m_hooks.reticleMove)
        m_hooks.reticleMove(pt, label);
    return loc;
}

CuaExecResult CuaExecutor::clickTarget(const QString &targetDesc, bool doubleClick, bool refineCrop)
{
    const QString target = targetDesc.simplified();
    if (target.isEmpty())
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Empty target")};
    const QString label = QStringLiteral("Clicking %1").arg(target);
    const Located loc = locate(target, refineCrop, label);
    if (!loc.ok) {
        if (m_hooks.reticleHide)
            m_hooks.reticleHide();
        return {false, QStringLiteral("ERR"), {}, loc.detail};
    }
    if (m_hooks.pause)
        m_hooks.pause(kGlideSettleMs);
    if (m_hooks.reticlePulse)
        m_hooks.reticlePulse(loc.pt, label);
    if (m_hooks.pause)
        m_hooks.pause(kPulseLeadMs);
    ActionResult act{false, QStringLiteral("No actuator")};
    if (loc.tier == QLatin1String("L1_AX") && m_preferAxPress && m_hooks.axPress && !doubleClick) {
        const QString key = loc.node.title.isEmpty() ? loc.node.role : loc.node.title;
        act = m_hooks.axPress(loc.node.pid, key, loc.rect);
    } else if (m_hooks.clickAt) {
        act = m_hooks.clickAt(loc.pt, doubleClick);
    }
    if (!act.ok)
        return {false, QStringLiteral("ERR"), loc.pt, QStringLiteral("%1: %2").arg(loc.tier, act.detail)};
    return {true, loc.tier, loc.pt, loc.detail};
}

CuaExecResult CuaExecutor::typeIntoTarget(const QString &targetDesc, const QString &text)
{
    if (text.isEmpty())
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Nothing to type")};
    CuaExecResult r = clickTarget(targetDesc);
    if (!r.ok)
        return r;
    if (m_hooks.pause)
        m_hooks.pause(kPulseLeadMs);  // let the field take focus
    const ActionResult t = m_hooks.typeText ? m_hooks.typeText(text) : ActionResult{false, QStringLiteral("No typist")};
    if (!t.ok)
        return {false, QStringLiteral("ERR"), r.globalPt, QStringLiteral("Typing failed: %1").arg(t.detail)};
    r.detail += QStringLiteral("; typed %1 chars").arg(text.size());
    return r;
}

CuaExecResult CuaExecutor::scrollTarget(const QString &targetDesc, int deltaY)
{
    if (deltaY == 0)
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Zero scroll")};
    const QString target = targetDesc.simplified();
    const QString label = target.isEmpty() ? QStringLiteral("Scrolling") : QStringLiteral("Scrolling %1").arg(target);
    Located loc;
    if (!target.isEmpty())
        loc = locate(target, false, label);
    if (!loc.ok) {
        // No target, or it was not found: scroll the middle of the primary display.
        DisplayInfo display;
        QString err;
        if (!primaryDisplay(&display, &err))
            return {false, QStringLiteral("ERR"), {}, err};
        const QString why = loc.detail;
        loc = Located{};
        loc.ok = true;
        loc.tier = QStringLiteral("CENTER");
        loc.pt = display.globalPoints.center();
        loc.detail = target.isEmpty() ? QStringLiteral("display centre") : QStringLiteral("display centre (%1)").arg(why);
        if (m_hooks.reticleMove)
            m_hooks.reticleMove(loc.pt, label);
    } else if (loc.tier == QLatin1String("L1_AX") && m_hooks.reticleMove) {
        m_hooks.reticleMove(loc.pt, label);  // the ring is up; bring the pointer onto it
    }
    if (m_hooks.pause)
        m_hooks.pause(kGlideSettleMs);
    const ActionResult s = m_hooks.scrollAt ? m_hooks.scrollAt(loc.pt, deltaY) : ActionResult{false, QStringLiteral("No scroller")};
    if (!s.ok)
        return {false, QStringLiteral("ERR"), loc.pt, QStringLiteral("Scroll failed: %1").arg(s.detail)};
    return {true, loc.tier, loc.pt, loc.detail};
}

}  // namespace studio::cua
