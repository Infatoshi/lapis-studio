#include "ingest/spotify.hpp"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPointer>
#include <QProcess>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QTimeZone>
#include <QUrl>

#include <algorithm>

namespace studio {

namespace {

constexpr int kTimeoutMs = 10000;
constexpr int kPollMs = 4000;
constexpr int kExpirySlackSecs = 30;
constexpr int kDryRun = -2;  // Done exit code when a process was only logged, never spawned

QString stripTrackPrefix(QString id)
{
    id = id.trimmed();
    static const QString prefix = QStringLiteral("spotify:track:");
    if (id.startsWith(prefix))
        id = id.mid(prefix.size());
    return id;
}

// First line of stderr, tidied: "Bad request: get current playback: Token is not valid".
QString firstLine(const QByteArray &err)
{
    const QString s = QString::fromUtf8(err).trimmed();
    return s.section(QLatin1Char('\n'), 0, 0).trimmed().left(160);
}

QDateTime parseExpiry(const QJsonValue &v)
{
    if (v.isDouble()) {
        const double n = v.toDouble();
        const qint64 ms = n > 1e11 ? qint64(n) : qint64(n * 1000.0);  // epoch ms or epoch seconds
        return QDateTime::fromMSecsSinceEpoch(ms, QTimeZone::UTC);
    }
    const QString s = v.toString().trimmed();
    if (s.isEmpty())
        return {};
    QDateTime dt = QDateTime::fromString(s, Qt::ISODateWithMs);
    if (!dt.isValid()) {
        // chrono writes microseconds ("…30.199972Z"); keep the whole seconds.
        QString t = s;
        t.remove(QRegularExpression(QStringLiteral("\\.\\d+")));
        dt = QDateTime::fromString(t, Qt::ISODate);
    }
    return dt;
}

// Status for the playback card, without ever launching Spotify.app.
const QString kStatusScript = QStringLiteral(
    "if application \"Spotify\" is running then\n"
    "tell application \"Spotify\"\n"
    "set s to player state as string\n"
    "set t to \"\"\n"
    "set a to \"\"\n"
    "try\n"
    "set t to name of current track\n"
    "set a to artist of current track\n"
    "end try\n"
    "return s & linefeed & t & linefeed & a & linefeed & (sound volume as string)\n"
    "end tell\n"
    "else\n"
    "return \"not running\"\n"
    "end if");

// A control that must not launch Spotify.app just to act on nothing (pause, skip, volume).
QString ifRunning(const QString &command)
{
    return QStringLiteral("if application \"Spotify\" is running then tell application \"Spotify\" to %1").arg(command);
}

QString tellSpotify(const QString &command)
{
    return QStringLiteral("tell application \"Spotify\" to %1").arg(command);
}

}  // namespace

SpotifyToken parseSpotifyToken(const QByteArray &json, const QDateTime &now)
{
    SpotifyToken out;
    const QJsonDocument doc = QJsonDocument::fromJson(json.trimmed());
    if (!doc.isObject())
        return out;
    const QJsonObject o = doc.object();
    out.accessToken = o.value(QStringLiteral("access_token")).toString().trimmed();
    out.ok = !out.accessToken.isEmpty();
    out.expiresAt = parseExpiry(o.value(QStringLiteral("expires_at")));
    out.expired = out.expiresAt.isValid() && now.secsTo(out.expiresAt) <= kExpirySlackSecs;
    return out;
}

SpotifyPlayback parseSpotifyPlayback(const QByteArray &json)
{
    SpotifyPlayback out;
    const QJsonDocument doc = QJsonDocument::fromJson(json.trimmed());
    if (!doc.isObject())
        return out;
    const QJsonObject o = doc.object();
    if (!o.contains(QStringLiteral("is_playing")) && !o.contains(QStringLiteral("item")))
        return out;
    out.ok = true;
    out.playing = o.value(QStringLiteral("is_playing")).toBool();
    const QJsonObject item = o.value(QStringLiteral("item")).toObject();
    out.track = item.value(QStringLiteral("name")).toString();
    const QJsonArray artists = item.value(QStringLiteral("artists")).toArray();
    if (!artists.isEmpty())
        out.artist = artists.first().toObject().value(QStringLiteral("name")).toString();
    if (out.artist.isEmpty())  // podcasts: the show stands in for the artist
        out.artist = item.value(QStringLiteral("show")).toObject().value(QStringLiteral("name")).toString();
    const QJsonValue vol = o.value(QStringLiteral("device")).toObject().value(QStringLiteral("volume_percent"));
    if (vol.isDouble())
        out.volume = std::clamp(vol.toInt(), 0, 100);
    return out;
}

QString parseSpotifyFirstTrackId(const QByteArray &json)
{
    const QJsonObject o = QJsonDocument::fromJson(json.trimmed()).object();
    QJsonValue tracks = o.value(QStringLiteral("tracks"));
    if (tracks.isObject())  // Web API shape: {"tracks": {"items": [...]}}
        tracks = tracks.toObject().value(QStringLiteral("items"));
    const QJsonArray arr = tracks.toArray();
    if (arr.isEmpty())
        return {};
    const QJsonObject t = arr.first().toObject();
    const QJsonValue id = t.value(QStringLiteral("id"));
    QString s = id.isObject() ? id.toObject().value(QStringLiteral("id")).toString() : id.toString();
    if (s.isEmpty())
        s = t.value(QStringLiteral("uri")).toString();
    return stripTrackPrefix(s);
}

QString parseSpotifyFirstTrackUri(const QByteArray &json)
{
    // Base62 only: the URI goes into an AppleScript string literal and a JSON body.
    static const QRegularExpression base62(QStringLiteral("^[A-Za-z0-9]{1,64}$"));
    const QString id = parseSpotifyFirstTrackId(json);
    if (!base62.match(id).hasMatch())
        return {};
    return QStringLiteral("spotify:track:") + id;
}

SpotifyPlayback parseSpotifyAppleScriptStatus(const QByteArray &out)
{
    SpotifyPlayback pb;
    QString text = QString::fromUtf8(out);
    text.remove(QLatin1Char('\r'));
    const QStringList lines = text.split(QLatin1Char('\n'));
    const QString state = lines.value(0).trimmed().toLower();
    if (state != QLatin1String("playing") && state != QLatin1String("paused") && state != QLatin1String("stopped"))
        return pb;
    pb.ok = true;
    pb.playing = state == QLatin1String("playing");
    pb.track = lines.value(1).trimmed();
    pb.artist = lines.value(2).trimmed();
    bool okVol = false;
    const double vol = lines.value(3).trimmed().toDouble(&okVol);
    if (okVol)
        pb.volume = std::clamp(int(vol + 0.5), 0, 100);
    return pb;
}

SpotifyController::SpotifyController(QObject *parent)
    : QObject(parent)
    , m_bin(findBinary())
    , m_tokenPath(defaultTokenPath())
    , m_apiBase(defaultApiBase())
    , m_appPath(QStringLiteral("/Applications/Spotify.app"))
    , m_net(new QNetworkAccessManager(this))
{
    m_poll.setInterval(kPollMs);
    connect(&m_poll, &QTimer::timeout, this, &SpotifyController::refresh);
}

SpotifyController::~SpotifyController() = default;

QString SpotifyController::findBinary()
{
    const QString env = qEnvironmentVariable("LAPIS_SPOTIFY_BIN").trimmed();
    if (!env.isEmpty())
        return env;
    const QString brew = QStringLiteral("/opt/homebrew/bin/spotify_player");
    if (QFileInfo(brew).isExecutable())
        return brew;
    return QStandardPaths::findExecutable(QStringLiteral("spotify_player"));
}

QString SpotifyController::defaultTokenPath()
{
    return QDir::homePath() + QStringLiteral("/.cache/spotify-player/user_client_token.json");
}

QString SpotifyController::defaultApiBase()
{
    return QStringLiteral("https://api.spotify.com");
}

void SpotifyController::setSpawnAllowed(bool on)
{
    if (m_spawnAllowed == on)
        return;
    m_spawnAllowed = on;
    if (!on)
        m_poll.stop();
    emit availableChanged();
    if (!on)
        return;
    refresh();  // one look at startup, so slowmix knows whether Spotify needs pausing
    if (m_polling)
        setPolling(true);
}

void SpotifyController::setBinary(const QString &path)
{
    m_bin = path;
    emit availableChanged();
}

void SpotifyController::setTokenPath(const QString &path)
{
    m_tokenPath = path;
    emit availableChanged();
}

void SpotifyController::setApiBase(const QString &base)
{
    m_apiBase = base;
    while (m_apiBase.endsWith(QLatin1Char('/')))
        m_apiBase.chop(1);
}

void SpotifyController::setAppPath(const QString &path)
{
    m_appPath = path;
    emit availableChanged();
}

SpotifyToken SpotifyController::token() const
{
    QFile f(m_tokenPath);
    if (m_tokenPath.isEmpty() || !f.open(QIODevice::ReadOnly))
        return {};
    return parseSpotifyToken(f.read(64 * 1024), QDateTime::currentDateTimeUtc());
}

bool SpotifyController::appInstalled() const
{
    return !m_appPath.isEmpty() && QFileInfo(m_appPath).isDir();
}

bool SpotifyController::available() const
{
    // Dry runs always "work"; live needs at least one route.
    return !m_spawnAllowed || token().usable() || appInstalled()
           || (!m_bin.isEmpty() && QFileInfo(m_bin).isExecutable());
}

bool SpotifyController::networkAllowed() const
{
    return m_spawnAllowed || m_apiBase != defaultApiBase();
}

void SpotifyController::logCommand(const QString &line)
{
    m_log << line;
    if (m_log.size() > 64)
        m_log.removeFirst();
}

bool SpotifyController::api(const QByteArray &verb, const QString &pathAndQuery, const QByteArray &body, ApiDone done)
{
    logCommand(QStringLiteral("api %1 %2").arg(QString::fromLatin1(verb), pathAndQuery));
    if (!networkAllowed())
        return false;
    const SpotifyToken t = token();
    if (!t.usable()) {
        if (t.ok)
            m_error = QStringLiteral("Spotify token expired (start spotify_player to refresh it)");
        return false;
    }
    QNetworkRequest req(QUrl::fromEncoded((m_apiBase + pathAndQuery).toUtf8()));
    req.setRawHeader("Authorization", "Bearer " + t.accessToken.toUtf8());
    req.setTransferTimeout(kTimeoutMs);
    if (verb != "GET") {
        // Spotify answers 411 to a body-less PUT/POST without an explicit length.
        req.setHeader(QNetworkRequest::ContentLengthHeader, body.size());
        if (!body.isEmpty())
            req.setHeader(QNetworkRequest::ContentTypeHeader, QStringLiteral("application/json"));
    }
    QNetworkReply *reply = m_net->sendCustomRequest(req, verb, body);
    ++m_inFlight;
    ++m_requestCount;
    connect(reply, &QNetworkReply::finished, this, [this, reply, done] {
        --m_inFlight;
        const int status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        const QByteArray data = reply->readAll();
        reply->deleteLater();
        if (status == 401)
            m_error = QStringLiteral("Spotify rejected the token (start spotify_player to refresh it)");
        if (done)
            done(status, data);
        emit stateChanged();
    });
    emit stateChanged();
    return true;
}

void SpotifyController::runProcess(const QString &program, const QStringList &args, const QString &logLine, Done done)
{
    logCommand(logLine);
    if (!m_spawnAllowed) {
        // Dry run: the caller already updated the state optimistically.
        if (done)
            done(kDryRun, {}, {});
        return;
    }
    auto *p = new QProcess(this);
    p->setProgram(program);
    p->setArguments(args);
    p->setStandardInputFile(QProcess::nullDevice());
    ++m_inFlight;
    ++m_spawnCount;
    QPointer<QProcess> guard(p);
    QTimer::singleShot(kTimeoutMs, p, [guard] {
        if (guard && guard->state() != QProcess::NotRunning)
            guard->kill();
    });
    const bool isCli = program == m_bin;
    auto finish = [this, p, done, isCli](int code, bool crashed) {
        if (p->property("lapisDone").toBool())
            return;
        p->setProperty("lapisDone", true);
        --m_inFlight;
        const QByteArray out = p->readAllStandardOutput();
        const QByteArray err = p->readAllStandardError();
        const int exitCode = crashed ? -1 : code;
        if (exitCode != 0) {
            const QString line = firstLine(err);
            const QString name = QFileInfo(p->program()).fileName();
            m_error = line.isEmpty() ? QStringLiteral("%1 exited %2").arg(name).arg(exitCode) : line;
            if (isCli && m_error.contains(QLatin1String("Token"), Qt::CaseInsensitive))
                m_error += QStringLiteral(" (run: spotify_player authenticate)");
        } else {
            m_error.clear();
        }
        if (done)
            done(exitCode, out, err);
        emit stateChanged();
        p->deleteLater();
    };
    connect(p, &QProcess::finished, this,
            [finish](int code, QProcess::ExitStatus st) { finish(code, st == QProcess::CrashExit); });
    connect(p, &QProcess::errorOccurred, this, [finish](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart)
            finish(-1, true);
    });
    p->start();
    emit stateChanged();
}

void SpotifyController::run(const QStringList &args, Done done)
{
    if (m_spawnAllowed && (m_bin.isEmpty() || !QFileInfo(m_bin).isExecutable())) {
        logCommand(QStringLiteral("cli ") + args.join(QLatin1Char(' ')));
        m_error = QStringLiteral("spotify_player not found");
        emit stateChanged();
        if (done)
            done(-1, {}, {});
        return;
    }
    runProcess(m_bin, args, QStringLiteral("cli ") + args.join(QLatin1Char(' ')), std::move(done));
}

void SpotifyController::osascript(const QString &script, const QString &label, Done done)
{
    QStringList args;
    for (const QString &line : script.split(QLatin1Char('\n')))
        args << QStringLiteral("-e") << line;
    runProcess(QStringLiteral("/usr/bin/osascript"), args, QStringLiteral("osascript ") + label, std::move(done));
}

void SpotifyController::dispatch(const Route &r, std::function<void(bool ok)> after)
{
    if (!networkAllowed()) {
        // Dry run: record the first route and stop.
        logCommand(r.verb.isEmpty() ? QStringLiteral("osascript ") + r.label
                                    : QStringLiteral("api %1 %2").arg(QString::fromLatin1(r.verb), r.path));
        if (after)
            after(true);
        return;
    }
    if (!r.verb.isEmpty()) {
        const bool sent = api(r.verb, r.path, r.body, [this, r, after](int status, const QByteArray &) {
            if (status >= 200 && status < 300) {
                m_error.clear();
                if (after)
                    after(true);
                return;
            }
            // 404 NO_ACTIVE_DEVICE, 401, 403 (premium), network error: try the local app.
            fallback(r, after);
        });
        if (sent)
            return;
    }
    fallback(r, after);
}

void SpotifyController::fallback(const Route &r, std::function<void(bool ok)> after)
{
    auto finish = [after](int code, const QByteArray &, const QByteArray &) {
        if (after)
            after(code == 0 || code == kDryRun);
    };
    if (appInstalled() && !r.script.isEmpty()) {
        osascript(r.script, r.label, finish);
    } else if (!r.cli.isEmpty()) {
        run(r.cli, finish);
    } else {
        if (m_error.isEmpty())
            m_error = QStringLiteral("No Spotify route for %1").arg(r.label);
        emit stateChanged();
        if (after)
            after(false);
    }
}

void SpotifyController::refreshSoon()
{
    if (m_spawnAllowed)
        QTimer::singleShot(700, this, &SpotifyController::refresh);
}

void SpotifyController::markPlaying(bool on)
{
    const bool started = on && !m_playing;
    m_playing = on;
    emit stateChanged();
    if (started)
        emit playbackStarted();
}

void SpotifyController::applyPlayback(const SpotifyPlayback &pb)
{
    m_playing = pb.playing;
    m_track = pb.track;
    m_artist = pb.artist;
    if (pb.volume >= 0)
        m_volume = pb.volume;
}

void SpotifyController::refresh()
{
    if (!networkAllowed() || m_refreshing)
        return;
    m_refreshing = true;
    const bool sent = api("GET", QStringLiteral("/v1/me/player"), {}, [this](int status, const QByteArray &body) {
        if (status == 200) {
            const SpotifyPlayback pb = parseSpotifyPlayback(body);
            if (pb.ok) {
                applyPlayback(pb);
                m_error.clear();
                m_refreshing = false;
                return;
            }
        }
        // 204: no Web API device is active, but Spotify.app may still be playing locally.
        refreshLocal();
    });
    if (!sent)
        refreshLocal();
}

void SpotifyController::refreshLocal()
{
    if (appInstalled()) {
        osascript(kStatusScript, QStringLiteral("status"), [this](int code, const QByteArray &out, const QByteArray &) {
            m_refreshing = false;
            if (code != 0)
                return;
            const SpotifyPlayback pb = parseSpotifyAppleScriptStatus(out);
            if (pb.ok)
                applyPlayback(pb);
            else
                m_playing = false;  // "not running"
        });
        return;
    }
    run({QStringLiteral("get"), QStringLiteral("key"), QStringLiteral("playback")},
        [this](int code, const QByteArray &out, const QByteArray &) {
            m_refreshing = false;
            if (code != 0)
                return;
            const SpotifyPlayback pb = parseSpotifyPlayback(out);
            if (pb.ok)
                applyPlayback(pb);
            else
                m_playing = false;  // "null": nothing is playing on any device
        });
}

void SpotifyController::togglePlay()
{
    const bool toPlay = !m_playing;
    dispatch({"PUT", toPlay ? QStringLiteral("/v1/me/player/play") : QStringLiteral("/v1/me/player/pause"), {},
              tellSpotify(QStringLiteral("playpause")), QStringLiteral("playpause"),
              {QStringLiteral("playback"), QStringLiteral("play-pause")}},
             [this](bool) { refreshSoon(); });
    markPlaying(toPlay);
}

void SpotifyController::play()
{
    dispatch({"PUT", QStringLiteral("/v1/me/player/play"), {}, tellSpotify(QStringLiteral("play")), QStringLiteral("play"),
              {QStringLiteral("playback"), QStringLiteral("play")}},
             [this](bool) { refreshSoon(); });
    markPlaying(true);
}

void SpotifyController::pause()
{
    dispatch({"PUT", QStringLiteral("/v1/me/player/pause"), {}, ifRunning(QStringLiteral("pause")),
              QStringLiteral("pause"), {QStringLiteral("playback"), QStringLiteral("pause")}},
             [this](bool) { refreshSoon(); });
    markPlaying(false);
}

void SpotifyController::nextTrack()
{
    dispatch({"POST", QStringLiteral("/v1/me/player/next"), {}, ifRunning(QStringLiteral("next track")),
              QStringLiteral("next track"), {QStringLiteral("playback"), QStringLiteral("next")}},
             [this](bool) { refreshSoon(); });
}

void SpotifyController::prevTrack()
{
    dispatch({"POST", QStringLiteral("/v1/me/player/previous"), {}, ifRunning(QStringLiteral("previous track")),
              QStringLiteral("previous track"), {QStringLiteral("playback"), QStringLiteral("previous")}},
             [this](bool) { refreshSoon(); });
}

void SpotifyController::setVolume(int pct)
{
    pct = std::clamp(pct, 0, 100);
    const QString n = QString::number(pct);
    dispatch({"PUT", QStringLiteral("/v1/me/player/volume?volume_percent=") + n, {},
              ifRunning(QStringLiteral("set sound volume to ") + n), QStringLiteral("set sound volume to ") + n,
              {QStringLiteral("playback"), QStringLiteral("volume"), n}});
    m_volume = pct;
    emit stateChanged();
}

void SpotifyController::searchAndPlay(const QString &query)
{
    const QString q = query.simplified();
    if (q.isEmpty())
        return;
    m_track = q;  // shown until the refresh brings the real title
    m_artist.clear();
    const QString searchPath = QStringLiteral("/v1/search?type=track&limit=1&q=")
                               + QString::fromLatin1(QUrl::toPercentEncoding(q));
    if (!networkAllowed()) {
        logCommand(QStringLiteral("api GET ") + searchPath);
        logCommand(QStringLiteral("api PUT /v1/me/player/play"));
        markPlaying(true);
        return;
    }
    // Without a usable token there is no search; spotify_player resolves the name itself.
    auto viaCli = [this, q] {
        run({QStringLiteral("playback"), QStringLiteral("start"), QStringLiteral("track"), QStringLiteral("--name"), q},
            [this](int c, const QByteArray &, const QByteArray &) {
                if (c == 0)
                    markPlaying(true);
                refreshSoon();
            });
    };
    const bool sent = api("GET", searchPath, {}, [this, viaCli](int status, const QByteArray &body) {
        const QString uri = status == 200 ? parseSpotifyFirstTrackUri(body) : QString();
        if (uri.isEmpty()) {
            if (status == 200) {
                m_error = QStringLiteral("No Spotify track matches \"%1\"").arg(m_track);
                emit stateChanged();
            } else {
                viaCli();
            }
            return;
        }
        const QByteArray playBody =
            QJsonDocument(QJsonObject{{QStringLiteral("uris"), QJsonArray{uri}}}).toJson(QJsonDocument::Compact);
        dispatch({"PUT", QStringLiteral("/v1/me/player/play"), playBody,
                  tellSpotify(QStringLiteral("play track \"%1\"").arg(uri)), QStringLiteral("play track ") + uri,
                  {QStringLiteral("playback"), QStringLiteral("start"), QStringLiteral("track"), QStringLiteral("--id"),
                   uri.mid(QStringLiteral("spotify:track:").size())}},
                 [this](bool ok) {
                     if (ok)
                         markPlaying(true);
                     refreshSoon();
                 });
    });
    if (!sent)
        viaCli();
    emit playbackStarted();  // hand the speakers over now, not after the search round trip
}

void SpotifyController::setPolling(bool on)
{
    if (on != m_polling) {
        m_polling = on;
        emit pollingChanged();
    }
    if (!on || !m_spawnAllowed) {
        m_poll.stop();
        return;
    }
    if (!m_poll.isActive()) {
        refresh();
        m_poll.start();
    }
}

}  // namespace studio
