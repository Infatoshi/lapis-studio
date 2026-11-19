// slowmix music: the listening log, the curated rotation, and a headless player controller.
//
// slowmix appends one row per listening chunk to ~/clips/.plays.tsv:
//   end_unix \t seconds \t speed \t reverb \t track
// where speed is slowed | normal | nightcore | fast | 1.10x, reverb is off | hall | heavy, and
// track is the file relative to ~/clips (absolute when outside it).
//
// SlowmixController talks to a persistent `slowmix --socket <path>` daemon over a local socket
// (load / play / pause / speed / reverb / bass / loop / status, one command per line). The daemon
// outlives the app: the controller attaches to a running one and syncs the UI from its status,
// spawns one detached on first play when none is running, and on destruction only closes its
// connection, so music keeps playing across app restarts. It never connects or spawns unless
// setSpawnAllowed(true), which only main()'s live window calls; every offscreen and test mode
// runs it as a dry run that updates the same properties without touching audio.
#pragma once

#include <QHash>
#include <QJsonObject>
#include <QObject>
#include <QPointer>
#include <QString>
#include <QTimer>
#include <QVariantList>

class QLocalSocket;

namespace studio {

struct SlowmixPlayStats {
    double totalSeconds = 0;
    double recentSeconds = 0;  // ended within maxStaleDays of now
    qint64 lastPlayed = 0;     // newest end_unix
    double favoriteSpeed = 1.0;
    QString favoriteReverb = QStringLiteral("hall");
};

// "slowed" 0.85, "normal" 1.0, "nightcore" 1.25, "fast" 1.6, "1.10x" 1.10; 0 when unknown.
double slowmixSpeedFromTag(const QString &tag);

// Per-track stats keyed by the log's track column. Favorite speed / reverb are the settings with
// the most listening seconds (last-window seconds when the track has any, else all-time).
QHash<QString, SlowmixPlayStats> parseSlowmixPlays(const QString &playsPath, qint64 nowEpochSec,
                                                   int maxStaleDays = 30);

struct SlowmixTrack {
    QString name;       // human stem: "night_drive_v2.wav" -> "night drive v2"
    QString fileName;
    QString path;       // absolute
    QString relPath;    // relative to the clips dir, as the log writes it
    double recentMinutes = 0;
    double totalMinutes = 0;
    qint64 lastPlayed = 0;
    double favoriteSpeed = 1.0;
    QString favoriteReverb = QStringLiteral("hall");
};

// True for generated takes, stems, experiments, sfx and pre-baked slowmix exports.
bool isSlowmixJunk(const QString &relPath);

// Audio files under clipsDir minus junk. includeArchive == false keeps only tracks played within
// the last 30 days, sorted by those minutes (descending); true keeps every song, still sorted by
// recent minutes, then all-time minutes, then name. nowEpochSec <= 0 means the current time.
QList<SlowmixTrack> curateSlowmixTracks(const QString &clipsDir, const QString &playsPath,
                                        bool includeArchive = false, qint64 nowEpochSec = 0);

class SlowmixController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool available READ available NOTIFY availableChanged)
    Q_PROPERTY(bool running READ running NOTIFY stateChanged)
    Q_PROPERTY(bool playing READ playing NOTIFY stateChanged)
    Q_PROPERTY(QString currentTrack READ currentTrack NOTIFY stateChanged)
    Q_PROPERTY(QString currentPath READ currentPath NOTIFY stateChanged)
    Q_PROPERTY(double speed READ speed NOTIFY stateChanged)
    Q_PROPERTY(QString reverb READ reverb NOTIFY stateChanged)
    Q_PROPERTY(double bass READ bass NOTIFY stateChanged)
    Q_PROPERTY(bool looping READ looping NOTIFY stateChanged)
    Q_PROPERTY(double position READ position NOTIFY positionChanged)
    Q_PROPERTY(double duration READ duration NOTIFY positionChanged)
    Q_PROPERTY(bool showArchive READ showArchive WRITE setShowArchive NOTIFY tracksChanged)
    Q_PROPERTY(QVariantList tracks READ tracks NOTIFY tracksChanged)

public:
    explicit SlowmixController(QObject *parent = nullptr);
    ~SlowmixController() override;

    // $LAPIS_SLOWMIX_BIN, else ~/slowmix/target/release/slowmix.
    static QString defaultBinary();
    static QString defaultClipsDir();  // ~/clips
    // $LAPIS_SLOWMIX_SOCK, else ~/.lapis-next/slowmix.sock.
    static QString defaultSocketPath();

    // true probes the socket: a running daemon is attached and the UI syncs to its status.
    // false closes the connection (the daemon keeps playing).
    void setSpawnAllowed(bool on);
    bool spawnAllowed() const { return m_spawnAllowed; }
    void setBinary(const QString &path);
    void setSocketPath(const QString &path);  // closes any current connection
    QString socketPath() const { return m_sockPath; }
    void setLibrary(const QString &clipsDir, const QString &playsPath);
    QString clipsDir() const { return m_clipsDir; }

    bool available() const;
    bool running() const;  // connected to the daemon
    bool playing() const { return m_playing; }
    QString currentTrack() const { return m_currentTrack; }
    QString currentPath() const { return m_currentPath; }
    double speed() const { return m_speed; }
    QString reverb() const { return m_reverb; }
    double bass() const { return m_bass; }
    bool looping() const { return m_looping; }
    double position() const { return m_pos; }
    double duration() const { return m_dur; }
    bool showArchive() const { return m_showArchive; }
    QVariantList tracks() const { return m_tracks; }
    int spawnCount() const { return m_spawnCount; }  // daemons started; stays 0 in dry runs

    Q_INVOKABLE void selectTrack(const QString &path);  // applies its favorite preset, then plays
    Q_INVOKABLE void togglePlay();
    Q_INVOKABLE void setSpeed(double speed);
    Q_INVOKABLE void setReverb(const QString &reverb);
    Q_INVOKABLE void setBass(double db);
    Q_INVOKABLE void setLooping(bool on);
    Q_INVOKABLE void setShowArchive(bool on);
    Q_INVOKABLE void refresh();  // rescan the clips dir and the log

signals:
    void availableChanged();
    void stateChanged();
    void positionChanged();
    void tracksChanged();

private:
    // Connects to the daemon, spawning it detached when none answers. False in dry runs or when
    // nothing could be reached; *spawned is true when a fresh daemon just started.
    bool ensureConnected(bool *spawned = nullptr);
    bool connectSocket(int timeoutMs);
    void send(const QByteArray &line);
    void readStatus();
    void applyStatus(const QJsonObject &o);  // full sync from a daemon we just attached to
    void closeSocket();

    bool m_spawnAllowed = false;
    QString m_bin;
    QString m_clipsDir;
    QString m_playsPath;
    QString m_sockPath;
    QPointer<QLocalSocket> m_sock;
    bool m_syncPending = false;  // next unskipped status reply carries the daemon's whole state
    QTimer m_poll;
    QByteArray m_buf;
    int m_spawnCount = 0;
    int m_statusInFlight = 0;  // `status` lines sent, replies not yet read
    int m_skipReplies = 0;     // replies that predate the last command: position only

    bool m_playing = false;
    QString m_currentTrack;
    QString m_currentPath;
    double m_speed = 1.0;
    QString m_reverb = QStringLiteral("hall");
    double m_bass = 0.0;
    bool m_looping = true;
    double m_pos = 0;
    double m_dur = 0;
    bool m_showArchive = false;
    QVariantList m_tracks;
};

}  // namespace studio
