#include "ingest/slowmix.hpp"

#include <QDateTime>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QRegularExpression>
#include <QSet>

#include <algorithm>

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
{
    m_poll.setInterval(1000);
    connect(&m_poll, &QTimer::timeout, this, [this] {
        if (running()) {
            ++m_statusInFlight;
            m_proc->write("status\n");
        }
    });
}

SlowmixController::~SlowmixController()
{
    stopProcess();
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
    if (!on)
        stopProcess();
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
    // A dry run (tests, offscreen snaps) always "plays"; live needs the built binary.
    return !m_spawnAllowed || QFileInfo(m_bin).isExecutable();
}

bool SlowmixController::running() const
{
    return m_proc && m_proc->state() == QProcess::Running;
}

bool SlowmixController::ensureProcess(bool *spawned)
{
    if (spawned)
        *spawned = false;
    if (running())
        return true;
    if (!m_spawnAllowed || !QFileInfo(m_bin).isExecutable())
        return false;
    if (m_proc)
        m_proc->deleteLater();
    auto *p = new QProcess(this);
    p->setProgram(m_bin);
    p->setArguments({QStringLiteral("--stdio")});
    p->setWorkingDirectory(QDir::homePath());
    p->setStandardErrorFile(QProcess::nullDevice());
    connect(p, &QProcess::readyReadStandardOutput, this, &SlowmixController::readStatus);
    connect(p, &QProcess::finished, this, [this, p] {
        if (p != m_proc)
            return;
        m_poll.stop();
        m_statusInFlight = 0;
        m_skipReplies = 0;
        if (m_playing) {
            m_playing = false;
            emit stateChanged();
        }
    });
    p->start();
    if (!p->waitForStarted(2000)) {
        p->deleteLater();
        return false;
    }
    m_proc = p;
    m_buf.clear();
    ++m_spawnCount;
    if (spawned)
        *spawned = true;
    // A fresh child starts at slowmix defaults; push what the UI shows.
    send(QByteArray("speed ") + QByteArray::number(m_speed, 'f', 2));
    send(QByteArray("reverb ") + m_reverb.toUtf8());
    send(QByteArray("bass ") + QByteArray::number(m_bass, 'f', 1));
    send(QByteArray("loop ") + (m_looping ? "1" : "0"));
    m_poll.start();
    emit stateChanged();
    return true;
}

void SlowmixController::send(const QByteArray &line)
{
    if (!running())
        return;
    // A status reply already in flight predates this command; don't let it undo the UI.
    m_skipReplies = m_statusInFlight;
    m_proc->write(line + '\n');
}

void SlowmixController::readStatus()
{
    if (!m_proc)
        return;
    m_buf += m_proc->readAllStandardOutput();
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

void SlowmixController::stopProcess()
{
    m_poll.stop();
    if (!m_proc)
        return;
    QProcess *p = m_proc;
    m_proc.clear();
    p->disconnect(this);
    if (p->state() != QProcess::NotRunning) {
        // quit flushes listening time to ~/clips/.plays.tsv; EOF does too if quit is missed.
        p->write("quit\n");
        p->closeWriteChannel();
        if (!p->waitForFinished(1500)) {
            p->terminate();
            if (!p->waitForFinished(500)) {
                p->kill();
                p->waitForFinished(500);
            }
        }
    }
    delete p;
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
    if (ensureProcess(&spawned) && !spawned) {
        // ensureProcess already pushed speed / reverb / bass / loop to a new child.
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
    if (ensureProcess(&spawned)) {
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
