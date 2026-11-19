#include "ingest/spotify.hpp"

#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPointer>
#include <QProcess>
#include <QStandardPaths>

#include <algorithm>

namespace studio {

namespace {

constexpr int kTimeoutMs = 10000;
constexpr int kPollMs = 4000;

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

}  // namespace

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

SpotifyController::SpotifyController(QObject *parent)
    : QObject(parent)
    , m_bin(findBinary())
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

bool SpotifyController::available() const
{
    // Dry runs always "work"; live needs the binary.
    return !m_spawnAllowed || (!m_bin.isEmpty() && QFileInfo(m_bin).isExecutable());
}

void SpotifyController::run(const QStringList &args, Done done)
{
    m_log << args.join(QLatin1Char(' '));
    if (m_log.size() > 64)
        m_log.removeFirst();
    if (!m_spawnAllowed)
        return;  // dry run: the caller already updated the state optimistically
    if (!available()) {
        m_error = QStringLiteral("spotify_player not found");
        emit stateChanged();
        return;
    }
    auto *p = new QProcess(this);
    p->setProgram(m_bin);
    p->setArguments(args);
    p->setStandardInputFile(QProcess::nullDevice());
    ++m_inFlight;
    ++m_spawnCount;
    QPointer<QProcess> guard(p);
    QTimer::singleShot(kTimeoutMs, p, [guard] {
        if (guard && guard->state() != QProcess::NotRunning)
            guard->kill();
    });
    auto finish = [this, p, done](int code, bool crashed) {
        if (p->property("lapisDone").toBool())
            return;
        p->setProperty("lapisDone", true);
        --m_inFlight;
        const QByteArray out = p->readAllStandardOutput();
        const QByteArray err = p->readAllStandardError();
        const int exitCode = crashed ? -1 : code;
        if (exitCode != 0) {
            const QString line = firstLine(err);
            m_error = line.isEmpty() ? QStringLiteral("spotify_player exited %1").arg(exitCode) : line;
            if (m_error.contains(QLatin1String("Token"), Qt::CaseInsensitive))
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

void SpotifyController::refresh()
{
    if (!m_spawnAllowed || m_refreshing)
        return;
    m_refreshing = true;
    run({QStringLiteral("get"), QStringLiteral("key"), QStringLiteral("playback")},
        [this](int code, const QByteArray &out, const QByteArray &) {
            m_refreshing = false;
            if (code != 0)
                return;
            const SpotifyPlayback pb = parseSpotifyPlayback(out);
            if (!pb.ok) {
                // "null": nothing is playing on any device.
                m_playing = false;
                return;
            }
            m_playing = pb.playing;
            m_track = pb.track;
            m_artist = pb.artist;
            if (pb.volume >= 0)
                m_volume = pb.volume;
        });
}

void SpotifyController::togglePlay()
{
    const bool toPlay = !m_playing;
    run({QStringLiteral("playback"), QStringLiteral("play-pause")});
    markPlaying(toPlay);
    refreshSoon();
}

void SpotifyController::play()
{
    run({QStringLiteral("playback"), QStringLiteral("play")});
    markPlaying(true);
    refreshSoon();
}

void SpotifyController::pause()
{
    run({QStringLiteral("playback"), QStringLiteral("pause")});
    markPlaying(false);
    refreshSoon();
}

void SpotifyController::nextTrack()
{
    run({QStringLiteral("playback"), QStringLiteral("next")});
    refreshSoon();
}

void SpotifyController::prevTrack()
{
    run({QStringLiteral("playback"), QStringLiteral("previous")});
    refreshSoon();
}

void SpotifyController::setVolume(int pct)
{
    pct = std::clamp(pct, 0, 100);
    run({QStringLiteral("playback"), QStringLiteral("volume"), QString::number(pct)});
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
    if (!m_spawnAllowed) {
        run({QStringLiteral("search"), q});
        run({QStringLiteral("playback"), QStringLiteral("start"), QStringLiteral("track"), QStringLiteral("--id"),
             QStringLiteral("<first>")});
        markPlaying(true);
        return;
    }
    run({QStringLiteral("search"), q}, [this, q](int code, const QByteArray &out, const QByteArray &) {
        const QString id = code == 0 ? parseSpotifyFirstTrackId(out) : QString();
        // No parsable id: let spotify_player resolve the name itself.
        const QStringList args = id.isEmpty()
                                     ? QStringList{QStringLiteral("playback"), QStringLiteral("start"),
                                                   QStringLiteral("track"), QStringLiteral("--name"), q}
                                     : QStringList{QStringLiteral("playback"), QStringLiteral("start"),
                                                   QStringLiteral("track"), QStringLiteral("--id"), id};
        run(args, [this](int c, const QByteArray &, const QByteArray &) {
            if (c == 0)
                markPlaying(true);
            refreshSoon();
        });
    });
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
