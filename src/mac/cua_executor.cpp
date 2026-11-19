#include "mac/cua_executor.hpp"
#include "mac/cua_reticle.hpp"

#include <QBuffer>
#include <QCoreApplication>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QRegularExpression>
#include <QThread>
#include <curl/curl.h>
#include <unistd.h>
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
constexpr int kActivateSettleMs = 220;  // the activated app comes forward on its own monitor

// Apps a target may name, with the short hint each maps to. Bare generic words ("code",
// "cursor", "edge", "notes") are left out on purpose: they name UI more often than apps.
struct AppPattern { QRegularExpression re; QString hint; };
const QVector<AppPattern> &appPatterns()
{
    static const QVector<AppPattern> patterns = [] {
        const QList<std::pair<QString, QString>> raw{
            {"google chrome", "Chrome"}, {"chrome", "Chrome"}, {"safari", "Safari"}, {"firefox", "Firefox"},
            {"arc browser", "Arc"}, {"brave browser", "Brave"}, {"microsoft edge", "Microsoft Edge"},
            {"discord", "Discord"}, {"slack", "Slack"}, {"finder", "Finder"}, {"terminal", "Terminal"},
            {"iterm2?", "iTerm"}, {"ghostty", "Ghostty"}, {"spotify", "Spotify"},
            {"visual studio code", "VS Code"}, {"vs ?code", "VS Code"}, {"xcode", "Xcode"},
            {"zoom", "Zoom"}, {"notion", "Notion"}, {"obsidian", "Obsidian"}, {"telegram", "Telegram"},
            {"whatsapp", "WhatsApp"}, {"figma", "Figma"}, {"system settings", "System Settings"}};
        QVector<AppPattern> out;
        for (const auto &[re, hint] : raw)
            out.push_back({QRegularExpression(QStringLiteral("\\b%1\\b").arg(re), QRegularExpression::CaseInsensitiveOption), hint});
        return out;
    }();
    return patterns;
}

// The leftmost (then longest) app mention in desc; false when none.
bool findApp(const QString &desc, QString *hint, qsizetype *start, qsizetype *len)
{
    bool found = false;
    for (const AppPattern &p : appPatterns()) {
        const QRegularExpressionMatch m = p.re.match(desc);
        if (!m.hasMatch())
            continue;
        if (!found || m.capturedStart() < *start || (m.capturedStart() == *start && m.capturedLength() > *len)) {
            found = true;
            *hint = p.hint;
            *start = m.capturedStart();
            *len = m.capturedLength();
        }
    }
    return found;
}

// "send button" -> "send": the role noun a model adds that AX titles never carry.
QString withoutRoleNoun(const QString &q)
{
    static const QRegularExpression tail(QStringLiteral("\\s+(button|field|tab|link|icon|menu item|checkbox|text box|box)$"));
    QString out = q;
    out.remove(tail);
    return out == q ? QString() : out;
}

bool editableRole(const QString &role)
{
    return role == QLatin1String("AXTextField") || role == QLatin1String("AXComboBox") || role == QLatin1String("AXSearchField");
}

// Index of the single best positive score, -1 when none or tied across different rects.
int bestIndex(const QVector<AxNode> &nodes, const std::function<int(const AxNode &)> &score, int *scoreOut = nullptr)
{
    int best = -1, bestScore = 0;
    bool tied = false;
    for (int i = 0; i < nodes.size(); ++i) {
        const int s = score(nodes[i]);
        if (s > bestScore) {
            best = i;
            bestScore = s;
            tied = false;
        } else if (s && s == bestScore && nodes[i].globalRect != nodes[best].globalRect) {
            tied = true;  // a parent and child sharing one rect is still one target
        }
    }
    if (scoreOut)
        *scoreOut = tied ? 0 : bestScore;
    return tied ? -1 : best;
}

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
    h.listAppWindows = [] { return studio::cua::listAppWindows(); };
    h.activateApp = [](const QString &hint, uint32_t *outDisplayId) {
        bool ok = false;
        runOnMainThread([&] { ok = activateAppByName(hint, outDisplayId); });
        return ok;
    };
    h.selfPid = [] { return int(getpid()); };
    // lapisDisplayId stays unset: the executor derives it from the listAppWindows it already has.
    return h;
}

QString extractAppHint(const QString &targetDesc)
{
    QString hint;
    qsizetype start = 0, len = 0;
    return findApp(targetDesc, &hint, &start, &len) ? hint : QString();
}

QString stripAppHint(const QString &targetDesc, const QString &appHint)
{
    QString hint;
    qsizetype start = 0, len = 0;
    if (appHint.isEmpty() || !findApp(targetDesc, &hint, &start, &len) || hint != appHint)
        return targetDesc.simplified();
    QString before = targetDesc.left(start), after = targetDesc.mid(start + len);
    static const QRegularExpression connector(QStringLiteral("\\s*\\b(inside of|inside|within|in|on|of|from|at)(\\s+the)?\\s*$"),
                                              QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression article(QStringLiteral("\\s*\\bthe\\s*$"), QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression suffix(QStringLiteral("^('s|\\s+(app|window))\\b"), QRegularExpression::CaseInsensitiveOption);
    before.remove(connector);
    before.remove(article);
    after.remove(suffix);
    const QString out = (before + QLatin1Char(' ') + after).simplified();
    return out.isEmpty() ? targetDesc.simplified() : out;
}

QString normalizeTypedText(const QString &text)
{
    QString out = text;
    out.replace(QLatin1String("\\r\\n"), QLatin1String("\n"));
    out.replace(QLatin1String("\\n"), QLatin1String("\n"));
    return out;
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
    if (!score) {
        // "address bar" / "url bar": browsers name the field "Address and search bar" or similar.
        static const QStringList addressBar{QStringLiteral("address bar"), QStringLiteral("url bar"),
                                            QStringLiteral("location bar"), QStringLiteral("omnibox"),
                                            QStringLiteral("address field"), QStringLiteral("url field")};
        if (addressBar.contains(q) && editableRole(node.role)
            && (title.contains(QLatin1String("address")) || title.contains(QLatin1String("url"))))
            score = 80;
    }
    if (!score) {
        // "Send button" -> a node titled "Send"; a little below a direct hit.
        const QString bare = withoutRoleNoun(q);
        if (!bare.isEmpty()) {
            const int s = axMatchScore(node, bare);
            return s > 5 ? s - 5 : 0;
        }
    }
    if (score && clickableRole(node.role))
        score += 20;
    return score;
}

int bestAxMatch(const QVector<AxNode> &nodes, const QString &targetDesc)
{
    return bestIndex(nodes, [&](const AxNode &n) { return axMatchScore(n, targetDesc); });
}

CuaExecutor::CuaExecutor(CuaHooks hooks) : m_hooks(std::move(hooks)) {}

bool CuaExecutor::route(const QString &targetDesc, bool activate, Route *r, QString *err)
{
    *r = Route{};
    const QVector<DisplayInfo> displays = m_hooks.listDisplays ? m_hooks.listDisplays() : QVector<DisplayInfo>();
    if (displays.isEmpty()) {
        *err = QStringLiteral("No display");
        return false;
    }
    r->appHint = extractAppHint(targetDesc);
    r->allowSelf = targetDesc.contains(QLatin1String("lapis"), Qt::CaseInsensitive);
    r->selfPid = m_hooks.selfPid ? m_hooks.selfPid() : 0;
    QVector<AppWindowInfo> windows = m_hooks.listAppWindows ? m_hooks.listAppWindows() : QVector<AppWindowInfo>();
    // The display holding the largest window matching pred; 0 when none.
    const auto largestOn = [&](const std::function<bool(const AppWindowInfo &)> &pred) {
        uint32_t id = 0;
        double area = 0;
        for (const AppWindowInfo &w : windows)
            if (w.displayId && pred(w) && w.globalRect.width() * w.globalRect.height() > area) {
                area = w.globalRect.width() * w.globalRect.height();
                id = w.displayId;
            }
        return id;
    };
    if (m_hooks.lapisDisplayId)
        r->lapisDisplayId = m_hooks.lapisDisplayId();
    else if (r->selfPid > 0)
        r->lapisDisplayId = largestOn([&](const AppWindowInfo &w) { return w.pid == r->selfPid; });

    if (!r->appHint.isEmpty()) {
        uint32_t id = 0;
        if (activate && m_hooks.activateApp && m_hooks.activateApp(r->appHint, &id)) {
            if (m_hooks.pause)
                m_hooks.pause(kActivateSettleMs);
            if (!id && m_hooks.listAppWindows)
                windows = m_hooks.listAppWindows();  // a window may only just have come on screen
        }
        if (!id)
            id = largestOn([&](const AppWindowInfo &w) { return w.pid != r->selfPid && appNameMatches(w.appName, r->appHint); });
        r->targetDisplayId = id;
    }

    // Search order: hinted app's display, (Lapis's own when the target names Lapis), displays
    // without Lapis with the primary first, then Lapis's display last.
    const auto add = [&](uint32_t id) {
        if (!id)
            return;
        for (const DisplayInfo &d : r->displays)
            if (d.displayId == id)
                return;
        for (const DisplayInfo &d : displays)
            if (d.displayId == id)
                r->displays.push_back(d);
    };
    add(r->targetDisplayId);
    if (r->allowSelf)
        add(r->lapisDisplayId);
    for (const DisplayInfo &d : displays)
        if (d.primary && d.displayId != r->lapisDisplayId)
            add(d.displayId);
    for (const DisplayInfo &d : displays)
        if (d.displayId != r->lapisDisplayId)
            add(d.displayId);
    for (const DisplayInfo &d : displays)
        add(d.displayId);
    return true;
}

CuaExecutor::Located CuaExecutor::locate(const QString &targetDesc, bool refineCrop, const QString &label, Route *routeOut)
{
    Located loc;
    Route rt;
    if (!route(targetDesc, true, &rt, &loc.detail))
        return loc;
    if (routeOut)
        *routeOut = rt;
    const DisplayInfo display = rt.displays.first();

    // L1: Accessibility trees of every display in route order, never Lapis's own nodes unless
    // the target names Lapis. With an app hint, a hit inside that app beats any other hit.
    if (m_hooks.inspectDisplay) {
        const QString query = rt.appHint.isEmpty() ? targetDesc : stripAppHint(targetDesc, rt.appHint);
        AxNode fallback;
        bool haveFallback = false;
        bool found = false;
        for (const DisplayInfo &d : rt.displays) {
            QVector<AxNode> nodes = m_hooks.inspectDisplay(d.displayId);
            if (rt.selfPid > 0 && !rt.allowSelf)
                nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](const AxNode &n) { return n.pid == rt.selfPid; }),
                            nodes.end());
            const int i = bestIndex(nodes, [&](const AxNode &n) {
                const int s = axMatchScore(n, query);
                return s && !rt.appHint.isEmpty() && appNameMatches(n.appName, rt.appHint) ? s + 30 : s;
            });
            if (i < 0)
                continue;
            if (rt.appHint.isEmpty() || appNameMatches(nodes[i].appName, rt.appHint)) {
                loc.node = nodes[i];
                found = true;
                break;
            }
            if (!haveFallback) {
                fallback = nodes[i];
                haveFallback = true;
            }
        }
        if (!found && haveFallback) {
            loc.node = fallback;
            found = true;
        }
        if (found) {
            loc.ok = true;
            loc.tier = QStringLiteral("L1_AX");
            loc.rect = loc.node.globalRect;
            loc.pt = loc.rect.center();
            loc.detail = QStringLiteral("%1 \"%2\" (%3)").arg(loc.node.role, loc.node.title.isEmpty() ? loc.node.value : loc.node.title,
                                                               loc.node.appName);
            if (m_hooks.reticleHighlight)
                m_hooks.reticleHighlight(loc.rect, label);
            return loc;
        }
    }

    // L3: local VLM pixel grounding on the first routed display (the hinted app's monitor, else
    // one Lapis is not on), on a capture that leaves our own overlay out.
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
    const QString typed = normalizeTypedText(text);
    if (typed.isEmpty())
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Nothing to type")};
    CuaExecResult r = clickTarget(targetDesc);
    if (!r.ok)
        return r;
    if (m_hooks.pause)
        m_hooks.pause(kPulseLeadMs);  // let the field take focus
    const ActionResult t = m_hooks.typeText ? m_hooks.typeText(typed) : ActionResult{false, QStringLiteral("No typist")};
    if (!t.ok)
        return {false, QStringLiteral("ERR"), r.globalPt, QStringLiteral("Typing failed: %1").arg(t.detail)};
    r.detail += QStringLiteral("; typed %1 chars").arg(typed.size());
    if (typed.contains(QLatin1Char('\n')))
        r.detail += QStringLiteral(" + Return");
    return r;
}

CuaExecResult CuaExecutor::scrollTarget(const QString &targetDesc, int deltaY)
{
    if (deltaY == 0)
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Zero scroll")};
    const QString target = targetDesc.simplified();
    const QString label = target.isEmpty() ? QStringLiteral("Scrolling") : QStringLiteral("Scrolling %1").arg(target);
    Located loc;
    Route rt;
    if (!target.isEmpty())
        loc = locate(target, false, label, &rt);
    if (!loc.ok) {
        // No target, or it was not found: scroll the middle of the routed display (never
        // fullscreen Lapis when another monitor has the user's apps).
        QString err;
        if (rt.displays.isEmpty() && !route(target, !target.isEmpty(), &rt, &err))
            return {false, QStringLiteral("ERR"), {}, err};
        const DisplayInfo display = rt.displays.first();
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
