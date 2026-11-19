#include "ingest/slowmix.hpp"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QElapsedTimer>
#include <QJsonObject>
#include <QLocalSocket>
#include <QProcess>
#include <QRegularExpression>
#include <QSet>
#include <QThread>

#include <algorithm>

#include <cerrno>
#include <signal.h>

namespace studio {

namespace {

const QSet<QString> &audioExts()
{
    // Same list slowmix's own picker accepts.
    static const QSet<QString> exts{"mp3", "wav", "flac", "ogg", "oga", "opus", "m4a",
                                    "aac", "aiff", "aif", "wma", "caf"};
    return exts;
}

QString cleanName(const QString &fileName)
{
    QString stem = QFileInfo(fileName).completeBaseName();
    stem.replace(QLatin1Char('_'), QLatin1Char(' '));
    stem.replace(QLatin1Char('-'), QLatin1Char(' '));
    return stem.simplified();
}

// Key with the most seconds; ties go to the lexically smaller key so results are stable.
template <typename K>
K argmax(const QHash<K, double> &h, const K &fallback)
{
    K best = fallback;
    double bestSecs = -1;
    for (auto it = h.cbegin(); it != h.cend(); ++it) {
        if (it.value() > bestSecs || (it.value() == bestSecs && it.key() < best)) {
            best = it.key();
            bestSecs = it.value();
        }
    }
    return best;
}

QVariantMap toVariant(const SlowmixTrack &t)
{
    return QVariantMap{{"name", t.name},
                       {"fileName", t.fileName},
                       {"path", t.path},
                       {"relPath", t.relPath},
                       {"recentMinutes", t.recentMinutes},
                       {"totalMinutes", t.totalMinutes},
                       {"lastPlayed", t.lastPlayed},
                       {"favoriteSpeed", t.favoriteSpeed},
                       {"favoriteReverb", t.favoriteReverb}};
}

}  // namespace

double slowmixSpeedFromTag(const QString &tag)
{
    const QString t = tag.trimmed().toLower();
    if (t == QLatin1String("slowed"))
        return 0.85;
    if (t == QLatin1String("normal"))
        return 1.0;
    if (t == QLatin1String("nightcore"))
        return 1.25;
    if (t == QLatin1String("fast"))
        return 1.6;
    if (t.endsWith(QLatin1Char('x'))) {
        bool ok = false;
        const double v = t.chopped(1).toDouble(&ok);
        if (ok && v > 0)
            return v;
    }
    return 0;
}

QHash<QString, SlowmixPlayStats> parseSlowmixPlays(const QString &playsPath, qint64 nowEpochSec, int maxStaleDays)
{
    struct Acc {
        SlowmixPlayStats s;
        QHash<double, double> speedAll, speedRecent;
        QHash<QString, double> reverbAll, reverbRecent;
    };
    QHash<QString, Acc> acc;
    QFile f(playsPath);
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text))
        return {};
    const qint64 cutoff = nowEpochSec - qint64(maxStaleDays) * 86400;
    while (!f.atEnd()) {
        const QString line = QString::fromUtf8(f.readLine()).trimmed();
        if (line.isEmpty())
            continue;
        const QStringList cols = line.split(QLatin1Char('\t'));
        if (cols.size() < 5)
            continue;
        bool okEnd = false, okSecs = false;
        const qint64 end = cols[0].toLongLong(&okEnd);
        const double secs = cols[1].toDouble(&okSecs);
        if (!okEnd || !okSecs || secs <= 0)
            continue;  // header row or junk
        const QString track = cols.mid(4).join(QLatin1Char('\t'));
        if (track.isEmpty())
            continue;
        Acc &a = acc[track];
        const bool recent = end >= cutoff && end <= nowEpochSec + 86400;
        a.s.totalSeconds += secs;
        if (recent)
            a.s.recentSeconds += secs;
        a.s.lastPlayed = std::max(a.s.lastPlayed, end);
        const double speed = slowmixSpeedFromTag(cols[2]);
        const QString reverb = cols[3].trimmed().toLower();
        if (speed > 0) {
            a.speedAll[speed] += secs;
            if (recent)
                a.speedRecent[speed] += secs;
        }
        if (reverb == QLatin1String("off") || reverb == QLatin1String("hall") || reverb == QLatin1String("heavy")) {
            a.reverbAll[reverb] += secs;
            if (recent)
                a.reverbRecent[reverb] += secs;
        }
    }
    QHash<QString, SlowmixPlayStats> out;
    for (auto it = acc.begin(); it != acc.end(); ++it) {
        Acc &a = it.value();
        const bool useRecent = a.s.recentSeconds > 0;
        const auto &sp = useRecent && !a.speedRecent.isEmpty() ? a.speedRecent : a.speedAll;
        const auto &rv = useRecent && !a.reverbRecent.isEmpty() ? a.reverbRecent : a.reverbAll;
        a.s.favoriteSpeed = argmax(sp, 1.0);
        a.s.favoriteReverb = argmax(rv, QStringLiteral("hall"));
        out.insert(it.key(), a.s);
    }
    return out;
}

bool isSlowmixJunk(const QString &relPath)
{
    static const QSet<QString> junkDirs{"gen", "takes", "experiments", "decompiled", "graft", "ddsp"};
    static const QStringList junkWords{"runnerup", "_raw_", "sfx", "_take"};
    static const QRegularExpression baked(
        QStringLiteral("_(slowed|normal|nightcore|fast|\\d+\\.\\d+x)_(hall|heavy|off)\\.|_nightcore\\."),
        QRegularExpression::CaseInsensitiveOption);
    const QStringList parts = QDir::fromNativeSeparators(relPath).split(QLatin1Char('/'), Qt::SkipEmptyParts);
    if (parts.isEmpty())
        return true;
    for (const QString &p : parts)
        if (p.startsWith(QLatin1Char('.')))
            return true;  // dot-dirs and dotfiles, like slowmix's own picker
    for (int i = 0; i + 1 < parts.size(); ++i)
        if (junkDirs.contains(parts[i].toLower()))
            return true;
    const QString file = parts.last().toLower();
    for (const QString &w : junkWords)
        if (file.contains(w))
            return true;
    return baked.match(file).hasMatch();
}

QList<SlowmixTrack> curateSlowmixTracks(const QString &clipsDir, const QString &playsPath, bool includeArchive,
                                        qint64 nowEpochSec)
{
    QList<SlowmixTrack> out;
    if (clipsDir.isEmpty() || !QFileInfo(clipsDir).isDir())
        return out;
    if (nowEpochSec <= 0)
        nowEpochSec = QDateTime::currentSecsSinceEpoch();
    const QHash<QString, SlowmixPlayStats> stats = parseSlowmixPlays(playsPath, nowEpochSec, 30);
    const QDir root(clipsDir);
    // Dot-dirs and dotfiles (.plays.tsv, .git) are dropped by isSlowmixJunk.
    QDirIterator it(clipsDir, QDir::Files | QDir::Dirs | QDir::NoDotAndDotDot, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QFileInfo fi(it.next());
        if (!fi.isFile() || !audioExts().contains(fi.suffix().toLower()))
            continue;
        const QString rel = root.relativeFilePath(fi.absoluteFilePath());
        if (isSlowmixJunk(rel))
            continue;
        SlowmixTrack t;
        t.fileName = fi.fileName();
        t.name = cleanName(t.fileName);
        t.path = fi.absoluteFilePath();
        t.relPath = rel;
        auto s = stats.constFind(rel);
        if (s == stats.cend())
            s = stats.constFind(t.path);
        if (s != stats.cend()) {
            t.recentMinutes = s->recentSeconds / 60.0;
            t.totalMinutes = s->totalSeconds / 60.0;
            t.lastPlayed = s->lastPlayed;
            t.favoriteSpeed = s->favoriteSpeed;
            t.favoriteReverb = s->favoriteReverb;
        }
        if (!includeArchive && t.recentMinutes <= 0)
            continue;  // never played, or nothing in the last 30 days
        out.append(t);
    }
    std::sort(out.begin(), out.end(), [](const SlowmixTrack &a, const SlowmixTrack &b) {
        if (a.recentMinutes != b.recentMinutes)
            return a.recentMinutes > b.recentMinutes;
        if (a.totalMinutes != b.totalMinutes)
            return a.totalMinutes > b.totalMinutes;
        return a.relPath < b.relPath;
    });
    return out;
}

// --- SlowmixController ------------------------------------------------------------------------

SlowmixController::SlowmixController(QObject *parent)
    : QObject(parent)
    , m_bin(defaultBinary())
    , m_sockPath(defaultSocketPath())
{
    m_poll.setInterval(1000);
    connect(&m_poll, &QTimer::timeout, this, [this] {
        if (running()) {
            ++m_statusInFlight;
            m_sock->write("status\n");
        }
    });
}

SlowmixController::~SlowmixController()
{
    // Never `quit` the daemon: the music outlives this process.
    closeSocket();
}

QString SlowmixController::defaultSocketPath()
{
    const QString env = qEnvironmentVariable("LAPIS_SLOWMIX_SOCK");
    if (!env.isEmpty())
        return env;
    return QDir::homePath() + QStringLiteral("/.lapis-next/slowmix.sock");
}

void SlowmixController::setSocketPath(const QString &path)
{
    if (path == m_sockPath)
        return;
    closeSocket();
    m_sockPath = path;
    emit availableChanged();
}

QString SlowmixController::defaultBinary()
{
    const QString env = qEnvironmentVariable("LAPIS_SLOWMIX_BIN");
    if (!env.isEmpty())
        return env;
    return QDir::homePath() + QStringLiteral("/slowmix/target/release/slowmix");
}

QString SlowmixController::defaultClipsDir()
{
    return QDir::homePath() + QStringLiteral("/clips");
}

void SlowmixController::setSpawnAllowed(bool on)
{
    if (m_spawnAllowed == on)
        return;
    m_spawnAllowed = on;
    if (!on) {
        closeSocket();
    } else if (connectSocket(250)) {
        // A daemon from an earlier session is still playing: show what it plays.
        m_syncPending = true;
        ++m_statusInFlight;
        m_sock->write("status\n");
        m_poll.start();
        emit stateChanged();
    }
    emit availableChanged();
}

void SlowmixController::setBinary(const QString &path)
{
    m_bin = path;
    emit availableChanged();
}

void SlowmixController::setLibrary(const QString &clipsDir, const QString &playsPath)
{
    m_clipsDir = clipsDir;
    m_playsPath = playsPath;
    if (!m_tracks.isEmpty())
        refresh();
}

bool SlowmixController::available() const
{
    // A dry run (tests, offscreen snaps) always "plays"; live needs a daemon or the built binary.
    return !m_spawnAllowed || running() || QFileInfo(m_bin).isExecutable();
}

bool SlowmixController::running() const
{
    return m_sock && m_sock->state() == QLocalSocket::ConnectedState;
}

bool SlowmixController::connectSocket(int timeoutMs)
{
    if (running())
        return true;
    closeSocket();
    if (m_sockPath.isEmpty())
        return false;
    auto *s = new QLocalSocket(this);
    s->connectToServer(m_sockPath);
    if (!s->waitForConnected(timeoutMs)) {
        delete s;
        return false;
    }
    m_sock = s;
    m_buf.clear();
    connect(s, &QLocalSocket::readyRead, this, &SlowmixController::readStatus);
    connect(s, &QLocalSocket::disconnected, this, [this, s] {
        if (s != m_sock)
            return;
        closeSocket();
        // The connection dropped. If the daemon still answers, reattach and resync; if it is
        // gone, so is the music.
        QTimer::singleShot(0, this, [this] {
            if (m_spawnAllowed && connectSocket(100)) {
                m_syncPending = true;
                ++m_statusInFlight;
                m_sock->write("status\n");
            } else if (m_playing) {
                m_playing = false;
            }
            emit stateChanged();
            emit availableChanged();
        });
    });
    m_poll.start();
    return true;
}

bool SlowmixController::ensureConnected(bool *spawned)
{
    if (spawned)
        *spawned = false;
    if (running())
        return true;
    if (!m_spawnAllowed)
        return false;
    if (connectSocket(250))
        return true;  // a daemon is already up (started by an earlier session)
    if (!QFileInfo(m_bin).isExecutable())
        return false;
    QDir().mkpath(QFileInfo(m_sockPath).absolutePath());
    QProcess p;
    p.setProgram(m_bin);
    p.setArguments({QStringLiteral("--socket"), m_sockPath});
    p.setWorkingDirectory(QDir::homePath());
    p.setStandardInputFile(QProcess::nullDevice());
    p.setStandardOutputFile(QProcess::nullDevice());
    p.setStandardErrorFile(QProcess::nullDevice());
    // Own session: Ctrl+C on the app's terminal or its process group never reaches the music.
    p.setUnixProcessParameters(QProcess::UnixProcessFlag::CreateNewSession
                               | QProcess::UnixProcessFlag::ResetSignalHandlers);
    qint64 pid = 0;
    if (!p.startDetached(&pid))
        return false;
    ++m_spawnCount;
    QElapsedTimer t;
    t.start();
    while (t.elapsed() < 3000) {
        if (connectSocket(100)) {
            if (spawned)
                *spawned = true;
            // A fresh daemon starts at slowmix defaults; push what the UI shows.
            send(QByteArray("speed ") + QByteArray::number(m_speed, 'f', 2));
            send(QByteArray("reverb ") + m_reverb.toUtf8());
            send(QByteArray("bass ") + QByteArray::number(m_bass, 'f', 1));
            send(QByteArray("loop ") + (m_looping ? "1" : "0"));
            emit availableChanged();
            return true;
        }
        if (pid > 0 && ::kill(pid_t(pid), 0) != 0 && errno == ESRCH)
            break;  // the daemon exited before listening (bad binary, unusable socket path)
        QThread::msleep(20);
    }
    return false;
}

void SlowmixController::send(const QByteArray &line)
{
    if (!running())
        return;
    // A status reply already in flight predates this command; don't let it undo the UI.
    m_skipReplies = m_statusInFlight;
    m_sock->write(line + '\n');
}

void SlowmixController::applyStatus(const QJsonObject &o)
{
    m_playing = o.value("playing").toBool();
    m_looping = o.value("loop").toBool(m_looping);
    const double speed = o.value("speed").toDouble(0);
    if (speed > 0)
        m_speed = speed;
    const QString reverb = o.value("reverb").toString().toLower();
    if (reverb == QLatin1String("off") || reverb == QLatin1String("hall") || reverb == QLatin1String("heavy"))
        m_reverb = reverb;
    m_bass = o.value("bass").toDouble(m_bass);
    QString path = o.value("path").toString();
    const QString track = o.value("track").toString();
    if (path.isEmpty() && !track.isEmpty()) {
        // Older daemons report only the track, relative to ~/clips like the log.
        path = QDir::isAbsolutePath(track)
                   ? track
                   : QDir(m_clipsDir.isEmpty() ? defaultClipsDir() : m_clipsDir).filePath(track);
    }
    if (path.isEmpty())
        return;  // nothing loaded yet
    m_currentPath = path;
    m_currentTrack.clear();
    for (const QVariant &v : std::as_const(m_tracks))
        if (v.toMap().value("path").toString() == path) {
            m_currentTrack = v.toMap().value("name").toString();
            break;
        }
    if (m_currentTrack.isEmpty())
        m_currentTrack = cleanName(QFileInfo(path).fileName());
}

void SlowmixController::readStatus()
{
    if (!m_sock)
        return;
    m_buf += m_sock->readAll();
    int nl;
    while ((nl = m_buf.indexOf('\n')) >= 0) {
        const QByteArray line = m_buf.left(nl).trimmed();
        m_buf.remove(0, nl + 1);
        const QJsonObject o = QJsonDocument::fromJson(line).object();
        if (o.isEmpty())
            continue;
        if (m_statusInFlight > 0)
            --m_statusInFlight;
        const double pos = o.value("pos").toDouble(), dur = o.value("dur").toDouble();
        if (pos != m_pos || dur != m_dur) {
            m_pos = pos;
            m_dur = dur;
            emit positionChanged();
        }
        if (m_skipReplies > 0) {
            --m_skipReplies;
            continue;  // a pending sync waits for the first reply that postdates every command
        }
        if (m_syncPending) {
            m_syncPending = false;
            applyStatus(o);
            emit stateChanged();
            continue;
        }
        const bool playing = o.value("playing").toBool();
        const bool looping = o.value("loop").toBool(m_looping);
        if (playing != m_playing || looping != m_looping) {
            m_playing = playing;
            m_looping = looping;
            emit stateChanged();
        }
    }
}

void SlowmixController::closeSocket()
{
    m_poll.stop();
    m_statusInFlight = 0;
    m_skipReplies = 0;
    m_syncPending = false;
    if (!m_sock)
        return;
    QLocalSocket *s = m_sock;
    m_sock.clear();
    s->disconnect(this);
    // Hand over queued commands, then hang up. Never `quit`: the daemon keeps playing.
    if (s->state() == QLocalSocket::ConnectedState && s->bytesToWrite() > 0)
        s->waitForBytesWritten(200);
    s->abort();
    s->deleteLater();  // may be inside this socket's own disconnected() signal
}

void SlowmixController::selectTrack(const QString &path)
{
    if (path.isEmpty())
        return;
    QVariantMap track;
    for (const QVariant &v : std::as_const(m_tracks))
        if (v.toMap().value("path").toString() == path) {
            track = v.toMap();
            break;
        }
    m_currentPath = path;
    m_currentTrack = track.isEmpty() ? cleanName(QFileInfo(path).fileName()) : track.value("name").toString();
    if (!track.isEmpty()) {
        m_speed = track.value("favoriteSpeed").toDouble();
        m_reverb = track.value("favoriteReverb").toString();
    }
    m_playing = true;
    m_pos = 0;
    m_dur = 0;
    bool spawned = false;
    if (ensureConnected(&spawned) && !spawned) {
        // ensureConnected already pushed speed / reverb / bass / loop to a new daemon.
        send(QByteArray("speed ") + QByteArray::number(m_speed, 'f', 2));
        send(QByteArray("reverb ") + m_reverb.toUtf8());
    }
    send(QByteArray("load ") + path.toUtf8());
    send(QByteArray("loop ") + (m_looping ? "1" : "0"));  // load resets the loop region
    emit stateChanged();
    emit positionChanged();
}

void SlowmixController::togglePlay()
{
    if (m_currentPath.isEmpty()) {
        if (m_tracks.isEmpty())
            refresh();
        if (!m_tracks.isEmpty())
            selectTrack(m_tracks.first().toMap().value("path").toString());
        return;
    }
    m_playing = !m_playing;
    bool spawned = false;
    if (ensureConnected(&spawned)) {
        if (spawned)
            send(QByteArray("load ") + m_currentPath.toUtf8() + (m_playing ? "" : " paused"));
        else
            send(m_playing ? "play" : "pause");
    }
    emit stateChanged();
}

void SlowmixController::setSpeed(double speed)
{
    speed = std::clamp(speed, 0.5, 2.0);
    if (qFuzzyCompare(speed, m_speed))
        return;
    m_speed = speed;
    send(QByteArray("speed ") + QByteArray::number(m_speed, 'f', 2));
    emit stateChanged();
}

void SlowmixController::setReverb(const QString &reverb)
{
    const QString r = reverb.trimmed().toLower();
    if (r != QLatin1String("off") && r != QLatin1String("hall") && r != QLatin1String("heavy"))
        return;
    if (r == m_reverb)
        return;
    m_reverb = r;
    send(QByteArray("reverb ") + m_reverb.toUtf8());
    emit stateChanged();
}

void SlowmixController::setBass(double db)
{
    db = std::clamp(db, -6.0, 15.0);
    if (db == m_bass)
        return;
    m_bass = db;
    send(QByteArray("bass ") + QByteArray::number(m_bass, 'f', 1));
    emit stateChanged();
}

void SlowmixController::setLooping(bool on)
{
    if (on == m_looping)
        return;
    m_looping = on;
    send(QByteArray("loop ") + (on ? "1" : "0"));
    emit stateChanged();
}

void SlowmixController::setShowArchive(bool on)
{
    if (on == m_showArchive)
        return;
    m_showArchive = on;
    refresh();
}

void SlowmixController::refresh()
{
    QVariantList list;
    for (const SlowmixTrack &t : curateSlowmixTracks(m_clipsDir, m_playsPath, m_showArchive))
        list.append(toVariant(t));
    m_tracks = list;
    emit tracksChanged();
}

}  // namespace studio
