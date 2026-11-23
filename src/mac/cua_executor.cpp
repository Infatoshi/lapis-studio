#include "mac/cua_executor.hpp"
#include "mac/cua_reticle.hpp"
#include "mac/cua_spotlight.hpp"

#include <QBuffer>
#include <QCoreApplication>
#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QRegularExpression>
#include <QSaveFile>
#include <QSet>
#include <QThread>
#include <QTimeZone>
#include <curl/curl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
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
std::atomic<bool> g_overlayLapisBlue{false};      // default: Gold, like the live app
std::atomic<CuaReticle *> g_sharedReticle{nullptr};  // set once, on the main thread

CuaReticle *sharedReticle()
{
    static CuaReticle *r = [] {
        auto *made = new CuaReticle();
        made->setGoldTheme(!g_overlayLapisBlue.load());
        g_sharedReticle.store(made);
        return made;
    }();
    return r;
}

// The longest reticle glide plus a beat, so the click lands visibly after the cartoon hand settles.
constexpr int kGlideSettleMs = int(kGlideMaxSec * 1000) + 30;
constexpr int kPulseLeadMs = 60;

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

GroundResult parseInklingGroundResponse(const QString &content)
{
    GroundResult g;
    const QString text = content.trimmed();
    static const QRegularExpression kNotFound(QStringLiteral(R"("?not_found"?\s*:\s*true)"),
                                              QRegularExpression::CaseInsensitiveOption);
    static const QRegularExpression kPair(QStringLiteral(R"(\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\])"));
    double x = NAN, y = NAN;
    bool found = false;
    // The first {...} span, so a ```json fence or a stray word around the object still parses.
    const qsizetype open = text.indexOf(QLatin1Char('{')), close = text.lastIndexOf(QLatin1Char('}'));
    if (open >= 0 && close > open) {
        const QJsonDocument doc = QJsonDocument::fromJson(text.mid(open, close - open + 1).toUtf8());
        if (doc.isObject()) {
            const QJsonObject o = doc.object();
            if (o.value(QStringLiteral("not_found")).toBool(false)) {
                g.notFound = true;
                g.detail = QStringLiteral("Inkling: target not visible");
                return g;
            }
            QPointF p;
            for (const char *key : {"point", "point_1000"})
                if (!found && jsonPoint(o.value(QLatin1String(key)), &p)) {
                    x = p.x();
                    y = p.y();
                    found = true;
                }
        }
    }
    if (!found && kNotFound.match(text).hasMatch()) {
        g.notFound = true;  // a truncated {"not_found": true
        g.detail = QStringLiteral("Inkling: target not visible");
        return g;
    }
    if (!found) {
        const QRegularExpressionMatch m = kPair.match(text);
        if (m.hasMatch()) {
            x = m.captured(1).toDouble();
            y = m.captured(2).toDouble();
            found = true;
        }
    }
    if (!found || !std::isfinite(x) || !std::isfinite(y)) {
        g.detail = QStringLiteral("Inkling reply has no point");
        return g;
    }
    // 0..1 fractions ([0.283, 0.733]): both within [0, 1] and at least one not a whole number.
    const bool fraction = x >= 0 && x <= 1.0 && y >= 0 && y <= 1.0 && (x != std::floor(x) || y != std::floor(y));
    if (fraction) {
        x *= 1000.0;
        y *= 1000.0;
    }
    if (x < 0 || x > 1000 || y < 0 || y > 1000) {
        g.detail = QStringLiteral("Inkling point (%1, %2) is outside 0-1000").arg(x).arg(y);
        return g;
    }
    g.ok = true;
    g.hasPoint1000 = true;
    g.point1000 = {std::round(x), std::round(y)};
    return g;
}

QByteArray buildInklingGroundRequest(const QByteArray &jpeg, const QString &target, const QString &model)
{
    const QString prompt = QStringLiteral("Locate the exact center of UI element: \"%1\". Reply ONLY with JSON "
                                          "{\"point\":[x,y]} using integer coordinates 0-1000 (x=0 left, y=0 top), or "
                                          "{\"not_found\":true} if it is not clearly visible on screen.")
                               .arg(target.simplified());
    const QString url = QStringLiteral("data:image/jpeg;base64,") + QString::fromLatin1(jpeg.toBase64());
    const QJsonArray content{
        QJsonObject{{QStringLiteral("type"), QStringLiteral("image_url")},
                    {QStringLiteral("image_url"), QJsonObject{{QStringLiteral("url"), url}}}},
        QJsonObject{{QStringLiteral("type"), QStringLiteral("text")}, {QStringLiteral("text"), prompt}}};
    const QJsonArray msgs{QJsonObject{{QStringLiteral("role"), QStringLiteral("user")}, {QStringLiteral("content"), content}},
                          QJsonObject{{QStringLiteral("role"), QStringLiteral("assistant")}, {QStringLiteral("content"), QString()}}};
    QJsonObject o;
    o.insert(QStringLiteral("model"), model.trimmed().isEmpty() ? QString::fromLatin1(kInklingGroundDefaultModel) : model.trimmed());
    o.insert(QStringLiteral("stream"), false);
    o.insert(QStringLiteral("temperature"), 0.0);
    o.insert(QStringLiteral("repeat_penalty"), 1.0);
    o.insert(QStringLiteral("max_tokens"), 48);
    o.insert(QStringLiteral("chat_template_kwargs"), QJsonObject{{QStringLiteral("reasoning_effort"), 0.0}});
    o.insert(QStringLiteral("continue_final_message"), true);
    o.insert(QStringLiteral("add_generation_prompt"), false);
    o.insert(QStringLiteral("messages"), msgs);
    return QJsonDocument(o).toJson(QJsonDocument::Compact);
}

QString inklingChatUrl(const QString &endpoint)
{
    QString base = endpoint.trimmed();
    while (base.endsWith(QLatin1Char('/')))
        base.chop(1);
    if (base.isEmpty() || base.endsWith(QLatin1String("/chat/completions")))
        return base;
    return base + QStringLiteral("/chat/completions");
}

GroundResult httpGroundWithInkling(const QString &endpoint, const QString &model, const QByteArray &jpeg,
                                   const QString &target)
{
    static std::once_flag once;
    std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
    GroundResult fail;
    fail.unreachable = true;
    const QByteArray url = inklingChatUrl(endpoint).toUtf8();
    if (url.isEmpty()) {
        fail.detail = QStringLiteral("No Inkling endpoint");
        return fail;
    }
    CURL *h = curl_easy_init();
    if (!h) {
        fail.detail = QStringLiteral("libcurl init failed");
        return fail;
    }
    const QByteArray body = buildInklingGroundRequest(jpeg, target, model);
    QByteArray reply;
    curl_slist *headers = curl_slist_append(nullptr, "Content-Type: application/json");
    curl_easy_setopt(h, CURLOPT_URL, url.constData());
    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_POST, 1L);
    curl_easy_setopt(h, CURLOPT_POSTFIELDS, body.constData());
    curl_easy_setopt(h, CURLOPT_POSTFIELDSIZE_LARGE, curl_off_t(body.size()));
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_TIMEOUT_MS, long(kInklingGroundTimeoutMs));
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT_MS, long(kInklingGroundConnectMs));
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, &appendBody);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &reply);
    const CURLcode rc = curl_easy_perform(h);
    long status = 0;
    curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
    curl_slist_free_all(headers);
    curl_easy_cleanup(h);
    if (rc != CURLE_OK) {
        fail.detail = QStringLiteral("Inkling grounder offline (%1)").arg(QString::fromUtf8(curl_easy_strerror(rc)));
        return fail;
    }
    if (status != 200) {
        fail.detail = QStringLiteral("Inkling grounder HTTP %1").arg(status);
        return fail;
    }
    const QJsonObject o = QJsonDocument::fromJson(reply).object();
    const QJsonObject msg = o.value(QStringLiteral("choices")).toArray().at(0).toObject().value(QStringLiteral("message")).toObject();
    if (!msg.value(QStringLiteral("content")).isString()) {
        fail.detail = QStringLiteral("Inkling grounder reply has no message content");
        return fail;  // a 200 that is not a chat completion: treat as unreachable, let LFM3 try
    }
    GroundResult g = parseInklingGroundResponse(msg.value(QStringLiteral("content")).toString());
    if (!g.ok && !g.notFound)
        g.detail = QStringLiteral("%1: %2").arg(g.detail, msg.value(QStringLiteral("content")).toString().simplified().left(60));
    return g;
}

QSizeF expectedCaptureSize(const DisplayInfo &d, int maxWidth)
{
    if (!d.globalPoints.isValid() || !(d.scale > 0) || maxWidth <= 0)
        return {};
    const double w = std::max(1.0, std::min(double(maxWidth), d.globalPoints.width() * d.scale));
    const double h = std::max(1.0, std::round(w * d.globalPoints.height() / std::max(1.0, d.globalPoints.width())));
    return {w, h};
}

namespace {

// "Chrome · " when the route names an app, else nothing: the HUD status prefix.
QString hudScope(const QString &appHint)
{
    return appHint.isEmpty() ? QString() : appHint + QStringLiteral(" \u00B7 ");
}

// What a typed string looks like on the HUD: one line, short, a return arrow for Return.
QString typedPreview(const QString &typed)
{
    const bool submits = typed.contains(QLatin1Char('\n'));
    QString t = typed;
    t.replace(QLatin1Char('\n'), QLatin1Char(' '));
    t = t.simplified();
    if (t.size() > 40)
        t = t.left(39) + QChar(0x2026);
    return submits ? t + QStringLiteral(" \u21A9") : t;
}

}  // namespace

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

void showCuaHud(const QString &status, HudTone tone)
{
    runOnMainThread([&] { sharedReticle()->showHud(status, tone); });
}

void hideCuaHud(double afterSeconds)
{
    runOnMainThread([&] { sharedReticle()->hideHud(afterSeconds); });
}

void hideCuaReticle(double afterSeconds)
{
    runOnMainThread([afterSeconds] {
        if (CuaReticle *r = g_sharedReticle.load())
            r->hideAfter(afterSeconds);
    });
}

void setOverlayTheme(bool lapisBlue)
{
    g_overlayLapisBlue.store(lapisBlue);
    setSpotlightLapisBlue(lapisBlue);  // the flag only; the live bar re-renders below
    // Never constructs the shared overlay: one made later picks the flag up in sharedReticle().
    runOnMainThread([lapisBlue] {
        if (CuaReticle *r = g_sharedReticle.load())
            r->setGoldTheme(!lapisBlue);
        refreshSpotlightTheme();
    });
}

bool overlayLapisBlue() { return g_overlayLapisBlue.load(); }

QVector<uint32_t> overlayWindowIds()
{
    QVector<uint32_t> ids;
    runOnMainThread([&] {
        for (const uint32_t id : {sharedReticle()->windowNumber(), sharedReticle()->hudWindowNumber(), spotlightWindowNumber()})
            if (id)
                ids.push_back(id);
    });
    return ids;
}

CuaHooks defaultCuaHooks()
{
    CuaHooks h;
    h.listDisplays = [] { return studio::cua::listDisplays(); };
    // Web pages nest posts 18-30 levels deep: a wider node cap and a longer budget than native UI needs.
    h.inspectDisplay = [](uint32_t id) { return studio::cua::inspectDisplay(id, 500, 0.9); };
    h.captureDisplayJpeg = [](uint32_t id, int w, const QVector<uint32_t> &exclude, QString *err) {
        return studio::cua::captureDisplayJpeg(id, w, exclude, err);
    };
    h.groundPixel = &httpGroundPixel;
    h.inklingGround = &httpGroundWithInkling;
    h.captureObservationJpeg = [](uint32_t id, const QVector<uint32_t> &exclude) {
        QString err;
        return studio::cua::captureDisplayJpeg(id, kObservationJpegWidth, exclude, &err, kObservationJpegQuality);
    };
    h.clickAt = [](QPointF p, bool dbl) { return studio::cua::clickAt(p, dbl); };
    h.scrollAt = [](QPointF p, int dy) { return studio::cua::scrollAt(p, dy); };
    h.typeText = [](const QString &t) { return studio::cua::typeText(t); };
    h.selectAllAndDelete = [] { return studio::cua::selectAllAndDelete(); };
    h.axPress = [](int pid, const QString &k, const QRectF &r) { return studio::cua::axPress(pid, k, r); };
    h.pressHotkey = [](const QString &combo) { return studio::cua::pressHotkey(combo); };
    h.dragBetween = [](QPointF from, QPointF to) { return studio::cua::dragBetween(from, to); };
    h.reticleMove = [](QPointF p, const QString &label) { runOnMainThread([&] { sharedReticle()->moveTo(p, label); }); };
    h.reticleHighlight = [](QRectF r, const QString &label) {
        runOnMainThread([&] { sharedReticle()->highlightRect(r, label); });
    };
    h.reticlePulse = [](QPointF p, const QString &label) {
        runOnMainThread([&] { sharedReticle()->pulseClick(p, label); });
    };
    h.reticleHide = [] { runOnMainThread([] { sharedReticle()->hide(); }); };
    h.reticleHideAfter = [](double seconds) { hideCuaReticle(seconds); };
    h.reticleWindowId = [] {
        uint32_t id = 0;
        runOnMainThread([&] { id = sharedReticle()->windowNumber(); });
        return id;
    };
    h.hudWindowId = [] {
        uint32_t id = 0;
        runOnMainThread([&] { id = sharedReticle()->hudWindowNumber(); });
        return id;
    };
    h.reticleHud = [](const QString &status, HudTone tone) { showCuaHud(status, tone); };
    h.pause = [](int ms) {
        if (ms > 0)
            QThread::msleep(static_cast<unsigned long>(ms));
    };
    h.listAppWindows = [] { return studio::cua::listAppWindows(); };
    h.activateApp = [](const QString &hint) {
        AppActivation a;
        runOnMainThread([&] { a.ok = activateAppByName(hint, &a.displayId, &a.pid, &a.launched); });
        return a;
    };
    h.openUrl = [](const QString &hint, const QString &url) {
        AppActivation a;
        runOnMainThread([&] { a.ok = openUrlInApp(hint, url, &a.displayId, &a.pid, &a.launched, &a.appName); });
        return a;
    };
    h.selfPid = [] { return int(getpid()); };
    // Read on the main thread, whose run loop is free between polls, so NSWorkspace stays current.
    h.frontmostPid = [] {
        int pid = 0;
        runOnMainThread([&] { pid = frontmostAppPid(); });
        return pid;
    };
    h.hideSelf = [] {
        bool ok = false;
        runOnMainThread([&] { ok = hideSelfApp(); });
        return ok;
    };
    // lapisDisplayId stays unset: the executor derives it from the listAppWindows it already has.
    h.traceDir = defaultTraceDir();
    return h;
}

QString defaultTraceDir()
{
    const QString env = qEnvironmentVariable("LAPIS_CUA_TRACE").trimmed().toLower();
    if (env == QLatin1String("0") || env == QLatin1String("off") || env == QLatin1String("false") || env == QLatin1String("no"))
        return {};
    return QDir::homePath() + QStringLiteral("/.lapis-next/cua_traces");
}

bool traceBlocked(bool redacted, const QStringList &fields)
{
    if (redacted)
        return true;
    return std::any_of(fields.cbegin(), fields.cend(),
                       [](const QString &f) { return f.contains(QLatin1String("infinity"), Qt::CaseInsensitive); });
}

QPointF toDisplay1000(QPointF globalPt, const DisplayInfo &display)
{
    const QRectF g = display.globalPoints;
    if (g.width() <= 0 || g.height() <= 0)
        return {};
    return {std::clamp((globalPt.x() - g.x()) / g.width() * 1000.0, 0.0, 1000.0),
            std::clamp((globalPt.y() - g.y()) / g.height() * 1000.0, 0.0, 1000.0)};
}

QRectF toDisplay1000(const QRectF &globalRect, const DisplayInfo &display)
{
    const QPointF a = toDisplay1000(globalRect.topLeft(), display), b = toDisplay1000(globalRect.bottomRight(), display);
    return QRectF(a, b).normalized();
}

namespace {

constexpr int kTraceFrameWidth = 1536;  // the L3 capture width, so a reused L3 frame matches
constexpr const char *kTraceLog = "traces.jsonl";

QString frameName(const QJsonValue &v)
{
    const QString s = v.toString();
    return s.isEmpty() ? QString() : s.section(QLatin1Char('/'), -1);
}

// Keeps the newest maxSteps lines (with 10% slack, so it rewrites rarely) and deletes every
// frame no kept line names, orphans from an interrupted write included.
void pruneTraces(const QString &dir, int maxSteps)
{
    if (maxSteps <= 0)
        return;
    QFile f(dir + QLatin1Char('/') + QLatin1String(kTraceLog));
    if (!f.open(QIODevice::ReadOnly))
        return;
    QList<QByteArray> lines = f.readAll().split('\n');
    f.close();
    lines.removeIf([](const QByteArray &l) { return l.trimmed().isEmpty(); });
    if (lines.size() <= maxSteps + maxSteps / 10)
        return;
    lines = lines.mid(lines.size() - maxSteps);
    QSaveFile out(f.fileName());
    if (!out.open(QIODevice::WriteOnly))
        return;
    QSet<QString> keep;
    for (const QByteArray &l : lines) {
        out.write(l);
        out.write("\n");
        const QJsonObject o = QJsonDocument::fromJson(l).object();
        keep << frameName(o.value(QStringLiteral("before_image"))) << frameName(o.value(QStringLiteral("after_image")));
    }
    if (!out.commit())
        return;
    QDir frames(dir + QStringLiteral("/frames"));
    for (const QString &name : frames.entryList(QDir::Files))
        if (!keep.contains(name))
            frames.remove(name);
}

}  // namespace

QByteArray CuaExecutor::captureFrame(uint32_t displayId) const
{
    if (!m_hooks.captureDisplayJpeg || !displayId)
        return {};
    QVector<uint32_t> exclude;  // the reticle and HUD bar never appear in a training frame
    for (const uint32_t id : {m_hooks.reticleWindowId ? m_hooks.reticleWindowId() : 0u,
                              m_hooks.hudWindowId ? m_hooks.hudWindowId() : 0u})
        if (id)
            exclude.push_back(id);
    QString err;
    return m_hooks.captureDisplayJpeg(displayId, kTraceFrameWidth, exclude, &err);
}

DisplayInfo CuaExecutor::traceDisplay(const Route &rt, QPointF pt, bool hasPt) const
{
    if (hasPt)
        for (const DisplayInfo &d : rt.displays)
            if (d.globalPoints.adjusted(0, 0, 0.001, 0.001).contains(pt))
                return d;
    return rt.displays.isEmpty() ? DisplayInfo{} : rt.displays.first();
}

void CuaExecutor::traceBefore(const Route &rt, QPointF pt, bool hasPt)
{
    if (!tracing() || !m_step.before.isEmpty() || rt.displays.isEmpty())
        return;
    // Privacy first: a redacted step never even captures (the target is known by now).
    if (traceBlocked(m_trace.redacted, {m_trace.goal, m_trace.projectKey, m_step.target, m_step.text, m_step.app, rt.appHint}))
        return;
    const DisplayInfo d = traceDisplay(rt, pt, hasPt);
    m_step.before = captureFrame(d.displayId);
    m_step.beforeDisplayId = d.displayId;
}

void CuaExecutor::traceStep(const QString &action, const QString &target, const QString &text, const Route &rt,
                            const CuaExecResult &r, int deltaY)
{
    if (!tracing() || rt.displays.isEmpty())
        return;
    const QString app = m_step.app.isEmpty() ? rt.appHint : m_step.app;
    if (traceBlocked(m_trace.redacted, {m_trace.goal, m_trace.projectKey, target, text, app, rt.appHint, r.detail,
                                        r.observation}))
        return;
    const bool observe = action == QLatin1String("observe");
    const bool pointless = observe || action == QLatin1String("hotkey") || action == QLatin1String("open");  // no screen point
    const bool hasPt = !pointless && (r.ok || !r.globalPt.isNull());
    const DisplayInfo d = traceDisplay(rt, r.globalPt, hasPt);
    if (m_step.before.isEmpty() || m_step.beforeDisplayId != d.displayId) {
        // No action ran (a miss, or observe): the screen now is the screen before.
        m_step.before = captureFrame(d.displayId);
        m_step.beforeDisplayId = d.displayId;
    }
    if (m_step.before.isEmpty())
        return;  // no frame (capture denied): nothing worth training on
    // The action path already settled kObserveSettleMs inside observeAfterAction.
    const QByteArray after = observe ? QByteArray() : captureFrame(d.displayId);

    const QString dir = m_hooks.traceDir;
    if (!QDir().mkpath(dir + QStringLiteral("/frames")))
        return;
    qint64 ts = QDateTime::currentMSecsSinceEpoch();
    while (QFile::exists(QStringLiteral("%1/frames/%2_before.jpg").arg(dir).arg(ts)))
        ++ts;  // two steps in one millisecond keep separate frames
    auto writeFrame = [&](const QByteArray &jpeg, const char *suffix) -> QString {
        if (jpeg.isEmpty())
            return {};
        const QString rel = QStringLiteral("frames/%1_%2.jpg").arg(ts).arg(QLatin1String(suffix));
        QFile f(dir + QLatin1Char('/') + rel);
        if (!f.open(QIODevice::WriteOnly) || f.write(jpeg) != jpeg.size())
            return {};
        return rel;
    };
    const QString beforeRel = writeFrame(m_step.before, "before");
    if (beforeRel.isEmpty())
        return;
    const QString afterRel = writeFrame(after, "after");

    QJsonObject o;
    o[QStringLiteral("ts")] = QDateTime::fromMSecsSinceEpoch(ts, QTimeZone::UTC).toString(Qt::ISODateWithMs);
    o[QStringLiteral("goal")] = m_trace.goal;
    o[QStringLiteral("action")] = action;
    o[QStringLiteral("target")] = target;
    o[QStringLiteral("text")] = text;
    if (deltaY)
        o[QStringLiteral("delta_y")] = deltaY;
    o[QStringLiteral("tier")] = r.ok ? r.tier : QStringLiteral("ERR");
    o[QStringLiteral("ok")] = r.ok;
    o[QStringLiteral("app")] = app;
    o[QStringLiteral("display_w")] = d.globalPoints.width();
    o[QStringLiteral("display_h")] = d.globalPoints.height();
    if (hasPt) {
        const QPointF p = toDisplay1000(r.globalPt, d);
        o[QStringLiteral("point_1000")] = QJsonArray{qRound(p.x()), qRound(p.y())};
    } else {
        o[QStringLiteral("point_1000")] = QJsonValue::Null;
    }
    if (m_step.hasToPt) {
        const QPointF p = toDisplay1000(m_step.toPt, d);
        o[QStringLiteral("to_point_1000")] = QJsonArray{qRound(p.x()), qRound(p.y())};
    }
    if (r.tier == QLatin1String("L1_AX") && m_step.rect.isValid() && !m_step.rect.isEmpty()) {
        const QRectF b = toDisplay1000(m_step.rect, d);
        o[QStringLiteral("bbox_1000")] = QJsonArray{qRound(b.left()), qRound(b.top()), qRound(b.right()), qRound(b.bottom())};
    } else {
        o[QStringLiteral("bbox_1000")] = QJsonValue::Null;
    }
    o[QStringLiteral("before_image")] = beforeRel;
    o[QStringLiteral("after_image")] = afterRel;
    o[QStringLiteral("detail")] = r.detail;
    QFile log(dir + QLatin1Char('/') + QLatin1String(kTraceLog));
    if (!log.open(QIODevice::WriteOnly | QIODevice::Append))
        return;
    log.write(QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n');
    log.close();
    pruneTraces(dir, m_hooks.traceMaxSteps);
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

bool wholeViewScrollTarget(const QString &targetDesc, const QString &appHint)
{
    QString t = targetDesc.simplified();
    if (!appHint.isEmpty() && t.compare(appHint, Qt::CaseInsensitive) != 0)
        t = stripAppHint(t, appHint);
    else if (!appHint.isEmpty())
        t.clear();  // just the app
    // "Chrome|x.com page" leaves "|x.com page"; a leftover '|' is a separator, not a target.
    while (!t.isEmpty() && (t.front() == QLatin1Char('|') || t.front().isSpace()))
        t.remove(0, 1);
    while (!t.isEmpty() && (t.back() == QLatin1Char('|') || t.back().isSpace()))
        t.chop(1);
    t = t.toLower();
    static const QRegularExpression article(QStringLiteral("^(the|this|that)\\s+"));
    t.remove(article);
    if (t.isEmpty())
        return true;
    static const QStringList nouns{QStringLiteral("page"),     QStringLiteral("window"),  QStringLiteral("screen"),
                                   QStringLiteral("timeline"), QStringLiteral("feed"),    QStringLiteral("content"),
                                   QStringLiteral("body"),     QStringLiteral("main"),    QStringLiteral("web page"),
                                   QStringLiteral("webpage")};
    if (nouns.contains(t))
        return true;
    for (const char *suffix : {" page", " window", " timeline", " feed"})
        if (t.endsWith(QLatin1String(suffix)))
            return true;
    return false;
}

bool navigationHotkey(const QString &combo)
{
    const ParsedHotkey hk = parseHotkeyCombo(combo);
    if (!hk.valid)
        return false;
    return hk.display == QLatin1String("Cmd+R") || hk.display == QLatin1String("Cmd+Shift+R")
           || hk.display == QLatin1String("Cmd+[") || hk.display == QLatin1String("Cmd+]");
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

bool CuaExecutor::route(const QString &targetDesc, bool activate, Route *r, QString *err, const QString &hintOverride)
{
    *r = Route{};
    const QVector<DisplayInfo> displays = m_hooks.listDisplays ? m_hooks.listDisplays() : QVector<DisplayInfo>();
    if (displays.isEmpty()) {
        *err = QStringLiteral("No display");
        return false;
    }
    r->appHint = hintOverride.isEmpty() ? extractAppHint(targetDesc) : hintOverride;
    r->allowSelf = targetDesc.contains(QLatin1String("lapis"), Qt::CaseInsensitive);
    // No app named: the default (the app behind the Spotlight bar) is where "click Sign In" or
    // "press cmd+t" means to act, so it is activated like a named one.
    if (r->appHint.isEmpty() && !r->allowSelf && !m_defaultAppHint.isEmpty()) {
        r->appHint = m_defaultAppHint;
        activate = true;
    }
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
        if (activate && m_hooks.activateApp) {
            hud(QStringLiteral("Switching to %1\u2026").arg(r->appHint), HudTone::Working);
            const AppActivation a = m_hooks.activateApp(r->appHint);
            if (a.ok) {
                id = a.displayId;
                r->activated = true;
                r->appPid = a.pid;
                // On one screen this is the fullscreen Space slide: wait for the app to be in
                // front and the slide to finish before anything is inspected or captured.
                waitFrontmost(a.pid, a.launched, r->selfPid);
                if (m_hooks.listAppWindows)
                    windows = m_hooks.listAppWindows();  // its windows may only just have come on screen
                if (a.pid > 0)
                    if (const uint32_t fresh = largestOn([&](const AppWindowInfo &w) { return w.pid == a.pid; }))
                        id = fresh;
            }
        }
        if (!id)
            id = largestOn([&](const AppWindowInfo &w) { return w.pid != r->selfPid && appNameMatches(w.appName, r->appHint); });
        r->targetDisplayId = id;
    }

    // One screen and Lapis still in front (no hint, or the app could not be brought forward):
    // hide Lapis so the AX scan and the capture see the desktop, never fullscreen Lapis itself.
    if (displays.size() == 1 && !r->allowSelf && r->selfPid > 0 && m_hooks.frontmostPid && m_hooks.hideSelf
        && m_hooks.frontmostPid() == r->selfPid) {
        hud(QStringLiteral("Switching to the desktop\u2026"), HudTone::Working);
        if (m_hooks.hideSelf()) {
            waitFrontmost(0, false, r->selfPid);
            if (m_hooks.listAppWindows)
                windows = m_hooks.listAppWindows();
            r->lapisDisplayId = 0;
        }
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

bool CuaExecutor::waitFrontmost(int pid, bool launched, int selfPid)
{
    const auto pause = [&](int ms) {
        if (m_hooks.pause)
            m_hooks.pause(ms);
    };
    if (!m_hooks.frontmostPid) {
        pause(kActivateSettleMs);  // no front-app signal: a fixed beat
        return true;
    }
    const int timeout = launched ? kLaunchWaitMs : kFrontWaitMs;
    for (int waited = 0;; waited += kFrontPollMs) {
        const int front = m_hooks.frontmostPid();
        if (pid > 0 ? front == pid : (front > 0 && front != selfPid)) {
            pause(kSpaceSlideMs);
            return true;
        }
        if (waited + kFrontPollMs > timeout)
            return false;
        pause(kFrontPollMs);
    }
}

void CuaExecutor::endStep() const
{
    if (m_hooks.reticleHideAfter)
        m_hooks.reticleHideAfter(kStepHideSec);
}

void CuaExecutor::hud(const QString &status, HudTone tone) const
{
    if (m_hooks.reticleHud)
        m_hooks.reticleHud(status, tone);
}

void CuaExecutor::hudFail(const QString &targetDesc, const QString &why) const
{
    hud(why.isEmpty() ? QStringLiteral("Couldn't find %1").arg(targetDesc) : why, HudTone::Error);
}

CuaExecutor::Located CuaExecutor::locate(const QString &targetDesc, bool refineCrop, const QString &label,
                                         const QString &verb, Route *routeOut)
{
    Located loc;
    Route rt;
    if (!route(targetDesc, true, &rt, &loc.detail))
        return loc;
    if (routeOut)
        *routeOut = rt;
    const DisplayInfo display = rt.displays.first();
    hud(hudScope(rt.appHint) + verb + QLatin1Char(' ')
            + (rt.appHint.isEmpty() ? targetDesc : stripAppHint(targetDesc, rt.appHint)),
        HudTone::Working);

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
            m_step.rect = loc.rect;
            m_step.app = loc.node.appName;
            if (m_hooks.reticleHighlight)
                m_hooks.reticleHighlight(loc.rect, label);
            return loc;
        }
    }

    // L3: local VLM pixel grounding on the first routed display (the hinted app's monitor, else
    // one Lapis is not on), on a capture that leaves our own overlay out.
    const bool inkling = !m_vlmEndpoint.isEmpty() && m_hooks.inklingGround;
    if (!m_hooks.captureDisplayJpeg || (!m_hooks.groundPixel && !inkling)) {
        loc.detail = QStringLiteral("No AX match and no grounder");
        return loc;
    }
    constexpr int kMaxWidth = 1536;
    QVector<uint32_t> exclude;  // our own overlays: the cursor and the HUD bar
    for (const uint32_t id : {m_hooks.reticleWindowId ? m_hooks.reticleWindowId() : 0u,
                              m_hooks.hudWindowId ? m_hooks.hudWindowId() : 0u})
        if (id)
            exclude.push_back(id);
    QString err;
    const QByteArray jpeg = m_hooks.captureDisplayJpeg(display.displayId, kMaxWidth, exclude, &err);
    if (jpeg.isEmpty()) {
        loc.detail = QStringLiteral("Capture failed: %1").arg(err.isEmpty() ? QStringLiteral("empty image") : err);
        return loc;
    }
    if (tracing()) {
        m_step.before = jpeg;  // the trace reuses this frame instead of capturing again
        m_step.beforeDisplayId = display.displayId;
    }
    // Inkling on the multimodal endpoint first; LFM3 only when that is unset or unreachable. An
    // explicit not_found stands: LFM3 would only guess a stray click.
    GroundResult g;
    bool grounded = false;
    if (inkling) {
        g = m_hooks.inklingGround(m_vlmEndpoint, m_vlmModel, jpeg, targetDesc);
        grounded = !g.unreachable;
    }
    if (!grounded && m_hooks.groundPixel)
        g = m_hooks.groundPixel(jpeg, targetDesc, refineCrop);
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
    const double localY = pt.y() - display.globalPoints.top();
    const bool wantsMenu = targetDesc.contains(QLatin1String("menu"), Qt::CaseInsensitive)
                           || targetDesc.contains(QLatin1String("apple"), Qt::CaseInsensitive);
    if (!wantsMenu && localY >= 0.0 && localY < 24.0) {
        loc.detail = QStringLiteral("Grounder hit macOS menu bar (%1, %2) for non-menu target")
                         .arg(pt.x(), 0, 'f', 1).arg(pt.y(), 0, 'f', 1);
        return loc;
    }
    loc.ok = true;
    loc.tier = QStringLiteral("L3_VLM");
    loc.pt = pt;
    loc.detail = QStringLiteral("VLM point (%1, %2)").arg(pt.x(), 0, 'f', 1).arg(pt.y(), 0, 'f', 1)
                 + (grounded ? QStringLiteral(" via Inkling") : QString());
    if (m_hooks.reticleMove)
        m_hooks.reticleMove(pt, label);
    return loc;
}

CuaExecResult CuaExecutor::clickTarget(const QString &targetDesc, bool doubleClick, bool refineCrop)
{
    m_step = {};
    m_step.target = targetDesc;
    Route rt;
    const CuaExecResult r = click(targetDesc, doubleClick, refineCrop, true, &rt);
    endStep();
    traceStep(QStringLiteral("click"), targetDesc.simplified(), {}, rt, r);
    return withShot(r);
}

CuaExecResult CuaExecutor::click(const QString &targetDesc, bool doubleClick, bool refineCrop, bool finish, Route *routeOut)
{
    const QString target = targetDesc.simplified();
    if (target.isEmpty())
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Empty target")};
    const QString label = QStringLiteral("Clicking %1").arg(target);
    Route rt;
    const Located loc = locate(target, refineCrop, label, QStringLiteral("Clicking"), &rt);
    if (routeOut)
        *routeOut = rt;
    if (!loc.ok) {
        if (m_hooks.reticleHide)
            m_hooks.reticleHide();
        hudFail(target, {});
        // A miss shows the model what is there instead, so its next try names a real control.
        return {false, QStringLiteral("ERR"), {}, loc.detail, observeAfterAction(rt)};
    }
    if (m_hooks.pause)
        m_hooks.pause(kGlideSettleMs);
    if (m_hooks.reticlePulse)
        m_hooks.reticlePulse(loc.pt, label);
    if (m_hooks.pause)
        m_hooks.pause(kPulseLeadMs);
    traceBefore(rt, loc.pt, true);  // the last frame before the input lands
    ActionResult act{false, QStringLiteral("No actuator")};
    if (loc.tier == QLatin1String("L1_AX") && m_preferAxPress && m_hooks.axPress && !doubleClick) {
        const QString key = loc.node.title.isEmpty() ? loc.node.role : loc.node.title;
        act = m_hooks.axPress(loc.node.pid, key, loc.rect);
    } else if (m_hooks.clickAt) {
        act = m_hooks.clickAt(loc.pt, doubleClick);
    }
    if (!act.ok) {
        hudFail(target, QStringLiteral("Click failed · %1").arg(target));
        return {false, QStringLiteral("ERR"), loc.pt, QStringLiteral("%1: %2").arg(loc.tier, act.detail)};
    }
    if (!finish)
        return {true, loc.tier, loc.pt, loc.detail};
    hud(QStringLiteral("Done · %1").arg(target), HudTone::Done);
    return {true, loc.tier, loc.pt, loc.detail, observeAfterAction(rt)};
}

QVector<AxNode> CuaExecutor::visibleNodes(const Route &rt, uint32_t displayId) const
{
    if (!m_hooks.inspectDisplay || !displayId)
        return {};
    QVector<AxNode> nodes = m_hooks.inspectDisplay(displayId);
    if (rt.selfPid > 0 && !rt.allowSelf)
        nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](const AxNode &n) { return n.pid == rt.selfPid; }),
                    nodes.end());
    if (!rt.appHint.isEmpty())
        std::stable_partition(nodes.begin(), nodes.end(),
                              [&](const AxNode &n) { return appNameMatches(n.appName, rt.appHint); });
    return nodes;
}

void CuaExecutor::preferDisplay(Route *rt, uint32_t displayId)
{
    if (!displayId)
        return;
    for (qsizetype i = 0; i < rt->displays.size(); ++i)
        if (rt->displays[i].displayId == displayId) {
            rt->displays.move(i, 0);
            rt->targetDisplayId = displayId;
            return;
        }
}

bool CuaExecutor::appWindow(const Route &rt, AppWindowInfo *out) const
{
    if (!m_hooks.listAppWindows || (rt.appPid <= 0 && rt.appHint.isEmpty()))
        return false;
    // Front to back: the first match is the app's frontmost window (a cmd+n window included).
    for (const AppWindowInfo &w : m_hooks.listAppWindows()) {
        if (!w.displayId || w.globalRect.isEmpty() || (rt.selfPid > 0 && w.pid == rt.selfPid))
            continue;
        if (rt.appPid > 0 ? w.pid == rt.appPid : appNameMatches(w.appName, rt.appHint)) {
            *out = w;
            return true;
        }
    }
    return false;
}

QString CuaExecutor::observeAfterAction(const Route &routed, bool waitPageLoad, bool followWindow)
{
    if (!m_hooks.inspectDisplay || routed.displays.isEmpty())
        return {};
    if (m_hooks.pause)
        m_hooks.pause(waitPageLoad ? kNavSettleMs : kObserveSettleMs);  // let the UI repaint before reading
    Route rt = routed;
    if (followWindow) {
        AppWindowInfo w;
        if (appWindow(rt, &w))
            preferDisplay(&rt, w.displayId);
    }
    const uint32_t displayId = rt.displays.first().displayId;
    QVector<AxNode> nodes = visibleNodes(rt, displayId);
    if (waitPageLoad) {
        // A page that just started loading shows a spinner and an empty web area: re-read until
        // the app has real content (enough nodes, and some text or a link) or the polls run out.
        const auto loaded = [&](const QVector<AxNode> &ns) {
            int inApp = 0;
            bool content = false;
            for (const AxNode &n : ns) {
                if (!rt.appHint.isEmpty() && !appNameMatches(n.appName, rt.appHint) && (rt.appPid <= 0 || n.pid != rt.appPid))
                    continue;
                ++inApp;
                content = content || n.role == QLatin1String("AXStaticText") || n.role == QLatin1String("AXLink")
                          || n.role == QLatin1String("AXHeading");
            }
            return inApp >= kNavMinNodes && content;
        };
        for (int i = 0; i < kNavPolls && !loaded(nodes); ++i) {
            if (m_hooks.pause)
                m_hooks.pause(kNavPollMs);
            nodes = visibleNodes(rt, displayId);
        }
    }
    captureObservationShot(displayId);  // the settled screen, same moment as the AX read
    return formatNodesForModel(nodes, kPostActionChars);
}

void CuaExecutor::captureObservationShot(uint32_t displayId)
{
    if (!m_attachScreenshots || !m_hooks.captureObservationJpeg || !displayId)
        return;
    QVector<uint32_t> exclude;  // the cursor and HUD bar never appear in what the model sees
    for (const uint32_t id : {m_hooks.reticleWindowId ? m_hooks.reticleWindowId() : 0u,
                              m_hooks.hudWindowId ? m_hooks.hudWindowId() : 0u})
        if (id)
            exclude.push_back(id);
    m_step.obsJpeg = m_hooks.captureObservationJpeg(displayId, exclude);
}

CuaExecResult CuaExecutor::withShot(CuaExecResult r)
{
    r.screenshotJpeg = std::exchange(m_step.obsJpeg, {});
    return r;
}

CuaExecResult CuaExecutor::observeApp(const QString &appHint)
{
    m_step = {};
    m_step.target = appHint;
    Route rt;
    const CuaExecResult r = observeImpl(appHint, &rt);
    traceStep(QStringLiteral("observe"), appHint.simplified(), {}, rt, r);
    return withShot(r);
}

CuaExecResult CuaExecutor::typeIntoTarget(const QString &targetDesc, const QString &text, bool clearFirst)
{
    m_step = {};
    m_step.target = targetDesc;
    m_step.text = text;
    Route rt;
    const CuaExecResult r = typeImpl(targetDesc, text, clearFirst, &rt);
    endStep();
    traceStep(QStringLiteral("type"), targetDesc.simplified(), normalizeTypedText(text), rt, r);
    return withShot(r);
}

CuaExecResult CuaExecutor::scrollTarget(const QString &targetDesc, int deltaY)
{
    m_step = {};
    m_step.target = targetDesc;
    Route rt;
    const CuaExecResult r = scrollImpl(targetDesc, deltaY, &rt);
    endStep();
    traceStep(QStringLiteral("scroll"), targetDesc.simplified(), {}, rt, r, deltaY);
    return withShot(r);
}

CuaExecResult CuaExecutor::observeImpl(const QString &appHint, Route *rtOut)
{
    const QString raw = appHint.simplified();
    QString hint = extractAppHint(raw);
    if (hint.isEmpty())
        hint = raw;  // an app the hint table does not list: activate it by name anyway
    Route &rt = *rtOut;
    QString err;
    if (!route(raw, !hint.isEmpty(), &rt, &err, hint)) {
        hudFail(raw, QStringLiteral("Observe failed · %1").arg(err));
        return {false, QStringLiteral("ERR"), {}, err};
    }
    hud(hudScope(rt.appHint) + QStringLiteral("Reading the screen"), HudTone::Working);
    const DisplayInfo display = rt.displays.first();
    const QVector<AxNode> nodes = visibleNodes(rt, display.displayId);
    CuaExecResult r{true, QStringLiteral("L1_OBS"), display.globalPoints.center(), {}};
    r.observation = formatNodesForModel(nodes, kObserveChars);
    captureObservationShot(display.displayId);
    qsizetype inApp = 0;
    if (!rt.appHint.isEmpty())
        inApp = std::count_if(nodes.begin(), nodes.end(), [&](const AxNode &n) { return appNameMatches(n.appName, rt.appHint); });
    r.detail = rt.appHint.isEmpty() ? QStringLiteral("%1 AX nodes on display %2").arg(nodes.size()).arg(display.displayId)
                                    : QStringLiteral("%1 AX nodes on display %2 (%3 in %4)")
                                          .arg(nodes.size())
                                          .arg(display.displayId)
                                          .arg(inApp)
                                          .arg(rt.appHint);
    hud(QStringLiteral("Done · Read %1").arg(rt.appHint.isEmpty() ? QStringLiteral("the screen") : rt.appHint), HudTone::Done);
    return r;
}

CuaExecResult CuaExecutor::typeImpl(const QString &targetDesc, const QString &text, bool clearFirst, Route *rtOut)
{
    const QString typed = normalizeTypedText(text);
    if (typed.isEmpty())
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Nothing to type")};
    Route &rt = *rtOut;
    CuaExecResult r = click(targetDesc, false, false, false, &rt);
    if (!r.ok)
        return r;
    hud(hudScope(rt.appHint) + QStringLiteral("Typing %1").arg(typedPreview(typed)), HudTone::Working);
    if (m_hooks.pause)
        m_hooks.pause(kPulseLeadMs);  // let the field take focus
    if (clearFirst && m_hooks.selectAllAndDelete) {
        // A retry ("John" after "John Gout") replaces the field instead of typing into it.
        const ActionResult c = m_hooks.selectAllAndDelete();
        if (!c.ok) {
            hudFail(targetDesc, QStringLiteral("Clearing failed · %1").arg(targetDesc.simplified()));
            return {false, QStringLiteral("ERR"), r.globalPt, QStringLiteral("Clearing failed: %1").arg(c.detail)};
        }
    }
    const ActionResult t = m_hooks.typeText ? m_hooks.typeText(typed) : ActionResult{false, QStringLiteral("No typist")};
    if (!t.ok) {
        hudFail(targetDesc, QStringLiteral("Typing failed · %1").arg(targetDesc.simplified()));
        return {false, QStringLiteral("ERR"), r.globalPt, QStringLiteral("Typing failed: %1").arg(t.detail)};
    }
    r.detail += QStringLiteral("; typed %1 chars").arg(typed.size());
    if (typed.contains(QLatin1Char('\n')))
        r.detail += QStringLiteral(" + Return");
    hud(QStringLiteral("Done · Typed %1").arg(typedPreview(typed)), HudTone::Done);
    // A Return usually submits (an address bar, a search): wait for what it loads.
    r.observation = observeAfterAction(rt, typed.contains(QLatin1Char('\n')));
    return r;
}

CuaExecResult CuaExecutor::scrollImpl(const QString &targetDesc, int deltaY, Route *rtOut)
{
    if (deltaY == 0)
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Zero scroll")};
    const QString target = targetDesc.simplified();
    const QString label = target.isEmpty() ? QStringLiteral("Scrolling") : QStringLiteral("Scrolling %1").arg(target);
    Located loc;
    Route &rt = *rtOut;
    const QString namedApp = extractAppHint(target);
    const QString viewApp = namedApp.isEmpty() ? m_defaultAppHint : namedApp;
    const bool wholeView = wholeViewScrollTarget(target, viewApp);
    if (wholeView) {
        // "Chrome", "Chrome|x.com page", "timeline": the page itself, never a control. Asking AX
        // or the VLM for "page" lands on the toolbar (the address bar at y=93), so scroll the
        // middle of the app's frontmost window, a little below centre to clear its toolbar.
        QString err;
        if (!route(target, !namedApp.isEmpty(), &rt, &err)) {
            hudFail(target, QStringLiteral("Scroll failed · %1").arg(err));
            return {false, QStringLiteral("ERR"), {}, err};
        }
        hud(hudScope(rt.appHint) + QStringLiteral("Scrolling"), HudTone::Working);
        AppWindowInfo w;
        if (appWindow(rt, &w)) {
            preferDisplay(&rt, w.displayId);
            const QRectF area = w.globalRect.intersected(rt.displays.first().globalPoints);
            const QRectF box = area.isEmpty() ? w.globalRect : area;
            loc.ok = true;
            loc.tier = QStringLiteral("WINDOW");
            loc.pt = QPointF(box.center().x(), box.top() + box.height() * 0.55);
            loc.detail = QStringLiteral("%1 window centre").arg(w.appName.isEmpty() ? rt.appHint : w.appName);
            m_step.app = w.appName;
            if (m_hooks.reticleMove)
                m_hooks.reticleMove(loc.pt, label);
        }
    } else if (!target.isEmpty()) {
        loc = locate(target, false, label, QStringLiteral("Scrolling"), &rt);
    }
    if (!loc.ok) {
        // No target, or it was not found: scroll the middle of the routed display (never
        // fullscreen Lapis when another monitor has the user's apps).
        QString err;
        if (rt.displays.isEmpty() && !route(target, !target.isEmpty(), &rt, &err)) {
            hudFail(target, QStringLiteral("Scroll failed · %1").arg(err));
            return {false, QStringLiteral("ERR"), {}, err};
        }
        if (target.isEmpty() && !wholeView)
            hud(hudScope(rt.appHint) + QStringLiteral("Scrolling"), HudTone::Working);
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
    traceBefore(rt, loc.pt, true);
    const ActionResult s = m_hooks.scrollAt ? m_hooks.scrollAt(loc.pt, deltaY) : ActionResult{false, QStringLiteral("No scroller")};
    if (!s.ok) {
        hudFail(target, QStringLiteral("Scroll failed · %1").arg(s.detail));
        return {false, QStringLiteral("ERR"), loc.pt, QStringLiteral("Scroll failed: %1").arg(s.detail)};
    }
    hud(target.isEmpty() ? QStringLiteral("Done · Scrolled") : QStringLiteral("Done · Scrolled %1").arg(target), HudTone::Done);
    return {true, loc.tier, loc.pt, loc.detail, observeAfterAction(rt)};
}

CuaExecResult CuaExecutor::hotkeyAction(const QString &appHint, const QString &combo)
{
    m_step = {};
    m_step.target = appHint;
    m_step.text = combo;
    Route rt;
    const CuaExecResult r = hotkeyImpl(appHint, combo, &rt);
    endStep();
    traceStep(QStringLiteral("hotkey"), appHint.simplified(), combo.simplified(), rt, r);
    return withShot(r);
}

CuaExecResult CuaExecutor::openAction(const QString &appHint, const QString &url)
{
    m_step = {};
    m_step.target = appHint;
    m_step.text = url;
    Route rt;
    const CuaExecResult r = openImpl(appHint, url, &rt);
    endStep();
    traceStep(QStringLiteral("open"), appHint.simplified(), url.trimmed(), rt, r);
    return withShot(r);
}

CuaExecResult CuaExecutor::openImpl(const QString &appHint, const QString &url, Route *rtOut)
{
    const QString raw = appHint.simplified();
    QString hint = extractAppHint(raw);
    if (hint.isEmpty())
        hint = raw;  // an app the hint table does not list: open it by name anyway
    Route &rt = *rtOut;
    QString err;
    if (url.trimmed().isEmpty()) {
        if (hint.isEmpty()) {
            hudFail(raw, QStringLiteral("Open failed · name an app or a URL"));
            return {false, QStringLiteral("ERR"), {}, QStringLiteral("cua_open needs an app or a URL")};
        }
        // route() activates, and activateApp reopens a window when the app is running without one.
        if (!route(raw, true, &rt, &err, hint)) {
            hudFail(raw, QStringLiteral("Open failed · %1").arg(err));
            return {false, QStringLiteral("ERR"), {}, err};
        }
        if (m_hooks.activateApp && !rt.activated) {
            hudFail(raw, QStringLiteral("Couldn't open %1").arg(hint));
            return {false, QStringLiteral("ERR"), {}, QStringLiteral("Could not open %1 (not running and not installed)").arg(hint),
                    observeAfterAction(rt)};
        }
        hud(QStringLiteral("Done · Opened %1").arg(hint), HudTone::Done);
        return {true, QStringLiteral("OPEN"), {}, QStringLiteral("Opened %1").arg(hint), observeAfterAction(rt, false, true)};
    }

    const QString full = normalizeOpenUrl(url);
    if (full.isEmpty()) {
        hudFail(url, QStringLiteral("Open failed · not a URL"));
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Not a URL: %1").arg(url.trimmed())};
    }
    if (!m_hooks.openUrl) {
        hudFail(url, QStringLiteral("Open failed · no URL opener"));
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("No URL opener")};
    }
    QString shown = full;
    shown.remove(QRegularExpression(QStringLiteral("^https?://")));
    hud(hudScope(hint) + QStringLiteral("Opening %1").arg(shown), HudTone::Working);
    const AppActivation a = m_hooks.openUrl(hint, full);
    if (!a.ok) {
        hudFail(url, QStringLiteral("Couldn't open %1").arg(shown));
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Could not open %1%2").arg(full,
                    hint.isEmpty() ? QString() : QStringLiteral(" in %1").arg(hint))};
    }
    const int selfPid = m_hooks.selfPid ? m_hooks.selfPid() : 0;
    waitFrontmost(a.pid, a.launched, selfPid);
    // The browser that took the URL is already in front: route to it without activating again.
    const QString app = hint.isEmpty() ? a.appName : hint;
    if (!route(app, false, &rt, &err, app)) {
        hudFail(url, QStringLiteral("Open failed · %1").arg(err));
        return {false, QStringLiteral("ERR"), {}, err};
    }
    rt.appPid = a.pid;
    rt.activated = true;
    AppWindowInfo w;
    preferDisplay(&rt, appWindow(rt, &w) ? w.displayId : a.displayId);
    m_step.app = app;
    hud(QStringLiteral("Done · Opened %1").arg(shown), HudTone::Done);
    const QString detail = QStringLiteral("Opened %1 in a new tab%2").arg(full, app.isEmpty() ? QString() : QStringLiteral(" (%1)").arg(app));
    return {true, QStringLiteral("OPEN"), {}, detail, observeAfterAction(rt, true)};
}

CuaExecResult CuaExecutor::dragTarget(const QString &fromDesc, const QString &toDesc)
{
    m_step = {};
    m_step.target = fromDesc;
    m_step.text = toDesc;
    Route rt;
    const CuaExecResult r = dragImpl(fromDesc, toDesc, &rt);
    endStep();
    traceStep(QStringLiteral("drag"), fromDesc.simplified(), toDesc.simplified(), rt, r);
    return withShot(r);
}

CuaExecResult CuaExecutor::hotkeyImpl(const QString &appHint, const QString &combo, Route *rtOut)
{
    const ParsedHotkey hk = parseHotkeyCombo(combo);
    if (!hk.valid) {
        hudFail(combo, QStringLiteral("Unknown shortcut · %1").arg(combo.simplified()));
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Invalid hotkey: %1").arg(combo.simplified())};
    }
    const QString raw = appHint.simplified();
    QString hint = extractAppHint(raw);
    if (hint.isEmpty())
        hint = raw;  // an app the hint table does not list: activate it by name anyway
    Route &rt = *rtOut;
    QString err;
    // No app: nothing to activate, but route() still hides a single-screen fullscreen Lapis so
    // the shortcut reaches the desktop instead of Lapis.
    if (!route(raw, !hint.isEmpty(), &rt, &err, hint)) {
        hudFail(hk.display, QStringLiteral("Shortcut failed · %1").arg(err));
        return {false, QStringLiteral("ERR"), {}, err};
    }
    hud(hudScope(rt.appHint) + QStringLiteral("Pressing %1").arg(hk.display), HudTone::Working);
    traceBefore(rt, {}, false);
    const ActionResult a = m_hooks.pressHotkey ? m_hooks.pressHotkey(combo) : ActionResult{false, QStringLiteral("No keyboard")};
    if (!a.ok) {
        hudFail(hk.display, QStringLiteral("Shortcut failed · %1").arg(hk.display));
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("Hotkey failed: %1").arg(a.detail)};
    }
    hud(QStringLiteral("Done · Pressed %1").arg(hk.display), HudTone::Done);
    const QString detail = rt.appHint.isEmpty() ? QStringLiteral("Pressed %1").arg(hk.display)
                                                : QStringLiteral("Pressed %1 in %2").arg(hk.display, rt.appHint);
    // Follow the app's frontmost window: cmd+n may have opened it on another monitor.
    return {true, QStringLiteral("KEY"), {}, detail, observeAfterAction(rt, navigationHotkey(combo), true)};
}

CuaExecResult CuaExecutor::dragImpl(const QString &fromDesc, const QString &toDesc, Route *rtOut)
{
    const QString from = fromDesc.simplified(), to = toDesc.simplified();
    if (from.isEmpty() || to.isEmpty())
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("A drag needs both a from and a to target")};
    const QString label = QStringLiteral("Dragging %1 to %2").arg(from, to);
    Route &rt = *rtOut;
    const Located a = locate(from, false, label, QStringLiteral("Dragging from"), &rt);
    const auto miss = [&](const QString &which, const QString &desc, const QString &why) -> CuaExecResult {
        if (m_hooks.reticleHide)
            m_hooks.reticleHide();
        hudFail(desc, {});
        return {false, QStringLiteral("ERR"), {}, QStringLiteral("%1 target \"%2\": %3").arg(which, desc, why),
                observeAfterAction(rt)};
    };
    if (!a.ok)
        return miss(QStringLiteral("From"), from, a.detail);
    const QRectF fromRect = a.rect;
    const QString fromApp = m_step.app;
    // "volume max" after "Chrome slider handle" means the one in Chrome too.
    const QString toQuery = extractAppHint(to).isEmpty() && !rt.appHint.isEmpty()
                                ? QStringLiteral("%1 in %2").arg(to, rt.appHint)
                                : to;
    Route toRoute;
    const Located b = locate(toQuery, false, label, QStringLiteral("Dragging to"), &toRoute);
    if (!b.ok)
        return miss(QStringLiteral("To"), to, b.detail);
    // The trace's bbox and app describe where the drag starts.
    m_step.rect = a.tier == QLatin1String("L1_AX") ? fromRect : QRectF();
    m_step.app = fromApp;
    hud(hudScope(rt.appHint) + label, HudTone::Working);
    if (m_hooks.reticleMove)
        m_hooks.reticleMove(a.pt, label);
    if (m_hooks.pause)
        m_hooks.pause(kGlideSettleMs);
    if (m_hooks.reticlePulse)
        m_hooks.reticlePulse(a.pt, label);
    if (m_hooks.pause)
        m_hooks.pause(kPulseLeadMs);
    traceBefore(rt, a.pt, true);
    if (m_hooks.reticleMove)
        m_hooks.reticleMove(b.pt, label);  // glides alongside the drag
    const ActionResult d = m_hooks.dragBetween ? m_hooks.dragBetween(a.pt, b.pt) : ActionResult{false, QStringLiteral("No pointer")};
    const QString tier = a.tier == QLatin1String("L1_AX") && b.tier == QLatin1String("L1_AX") ? QStringLiteral("L1_AX")
                                                                                               : QStringLiteral("L3_VLM");
    if (!d.ok) {
        hudFail(from, QStringLiteral("Drag failed · %1").arg(from));
        return {false, QStringLiteral("ERR"), a.pt, QStringLiteral("%1: %2").arg(tier, d.detail)};
    }
    m_step.hasToPt = true;
    m_step.toPt = b.pt;
    hud(QStringLiteral("Done · Dragged %1 to %2").arg(from, to), HudTone::Done);
    return {true, tier, a.pt, QStringLiteral("from %1; to %2").arg(a.detail, b.detail), observeAfterAction(rt)};
}

}  // namespace studio::cua
