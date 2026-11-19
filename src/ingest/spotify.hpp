// Spotify through the `spotify_player` CLI (https://github.com/aome510/spotify-player).
//
// SpotifyController drives one-shot, non-blocking `spotify_player` invocations (QProcess, 10 s
// cap each): `get key playback` for state, `playback play-pause|play|pause|next|previous|volume`,
// and `search <q>` followed by `playback start track --id <id>`. The binary is
// $LAPIS_SPOTIFY_BIN, else /opt/homebrew/bin/spotify_player, else the first on PATH.
//
// Like SlowmixController it never spawns anything unless setSpawnAllowed(true), which only
// main()'s live window calls: Mode::Mock, --test and every offscreen mode are dry runs that
// update the same properties optimistically and record each command line in commandLog().
#pragma once

#include <QByteArray>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>

#include <functional>

namespace studio {

struct SpotifyPlayback {
    bool ok = false;       // the reply parsed as a playback object
    bool playing = false;  // is_playing
    QString track;         // item.name
    QString artist;        // item.artists[0].name
    int volume = -1;       // device.volume_percent, -1 when absent
};

// `spotify_player get key playback` output. "null" / empty / garbage -> ok == false.
SpotifyPlayback parseSpotifyPlayback(const QByteArray &json);
// `spotify_player search <q>` output: the first track's bare id ("spotify:track:" stripped),
// from tracks[0].id (string or {id}), else tracks[0].uri. Empty when there is none.
QString parseSpotifyFirstTrackId(const QByteArray &json);

class SpotifyController : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool available READ available NOTIFY availableChanged)
    Q_PROPERTY(bool playing READ playing NOTIFY stateChanged)
    Q_PROPERTY(QString track READ track NOTIFY stateChanged)
    Q_PROPERTY(QString artist READ artist NOTIFY stateChanged)
    Q_PROPERTY(int volume READ volume NOTIFY stateChanged)
    Q_PROPERTY(bool busy READ busy NOTIFY stateChanged)
    Q_PROPERTY(QString error READ error NOTIFY stateChanged)
    // The Spotify tab binds this while it is on screen: refresh now, then every 4 s.
    Q_PROPERTY(bool polling READ polling WRITE setPolling NOTIFY pollingChanged)

public:
    explicit SpotifyController(QObject *parent = nullptr);
    ~SpotifyController() override;

    // $LAPIS_SPOTIFY_BIN, else /opt/homebrew/bin/spotify_player, else PATH; empty when none.
    static QString findBinary();

    void setSpawnAllowed(bool on);
    bool spawnAllowed() const { return m_spawnAllowed; }
    void setBinary(const QString &path);
    QString binary() const { return m_bin; }

    bool available() const;
    bool playing() const { return m_playing; }
    QString track() const { return m_track; }
    QString artist() const { return m_artist; }
    int volume() const { return m_volume; }
    bool busy() const { return m_inFlight > 0; }
    QString error() const { return m_error; }
    // Every command issued, as "playback next" / "search night drive" (dry runs included).
    QStringList commandLog() const { return m_log; }
    int spawnCount() const { return m_spawnCount; }

    Q_INVOKABLE void refresh();
    Q_INVOKABLE void togglePlay();
    Q_INVOKABLE void play();
    Q_INVOKABLE void pause();
    Q_INVOKABLE void nextTrack();
    Q_INVOKABLE void prevTrack();
    Q_INVOKABLE void setVolume(int pct);
    Q_INVOKABLE void searchAndPlay(const QString &query);
    bool polling() const { return m_polling; }
    // Poll `get key playback` every 4 s while on (live only; a dry run just records the flag).
    Q_INVOKABLE void setPolling(bool on);

signals:
    void pollingChanged();
    void availableChanged();
    void stateChanged();
    // We just asked Spotify to start sound (play, toggle to play, search-and-play). Backend pauses
    // slowmix on it so the two never play over each other.
    void playbackStarted();

private:
    using Done = std::function<void(int exitCode, const QByteArray &out, const QByteArray &err)>;
    void run(const QStringList &args, Done done = {});
    void refreshSoon();
    void markPlaying(bool on);

    bool m_spawnAllowed = false;
    QString m_bin;
    QTimer m_poll;
    QStringList m_log;
    int m_spawnCount = 0;
    int m_inFlight = 0;
    bool m_refreshing = false;
    bool m_polling = false;

    bool m_playing = false;
    QString m_track;
    QString m_artist;
    int m_volume = -1;
    QString m_error;
};

}  // namespace studio
