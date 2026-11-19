// Spotify control, three routes in order of preference:
//
//  1. The Spotify Web API (QNetworkAccessManager), authorised with the access token
//     `spotify_player` keeps fresh in ~/.cache/spotify-player/user_client_token.json. The file is
//     re-read on every request, so a token the daemon rotates is picked up without a restart.
//  2. The local Spotify.app over AppleScript (`osascript`), when the Web API has no active device
//     (204 / 404), rejects the token (401), is unreachable, or the token is missing or expired.
//  3. The `spotify_player` CLI ($LAPIS_SPOTIFY_BIN, else /opt/homebrew/bin/spotify_player, else
//     PATH), only when Spotify.app is not installed.
//
// Like SlowmixController it never spawns a process or touches the network unless
// setSpawnAllowed(true), which only main()'s live window calls: Mode::Mock, --test and every
// offscreen mode are dry runs that update the same properties optimistically and record each
// command in commandLog(). Tests may opt the network back in by pointing setApiBase() at a local
// mock server; osascript and the CLI still never run without setSpawnAllowed(true).
#pragma once

#include <QByteArray>
#include <QDateTime>
#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>

#include <functional>

class QNetworkAccessManager;

namespace studio {

struct SpotifyPlayback {
    bool ok = false;       // the reply parsed as a playback object
    bool playing = false;  // is_playing
    QString track;         // item.name
    QString artist;        // item.artists[0].name
    int volume = -1;       // device.volume_percent, -1 when absent
};

struct SpotifyToken {
    bool ok = false;       // the file held a non-empty access_token
    bool expired = false;  // expires_at is at most 30 s away (or past)
    QString accessToken;
    QDateTime expiresAt;   // invalid when the file had no expires_at
    bool usable() const { return ok && !expired; }
};

// `user_client_token.json`: {"access_token": "...", "expires_at": "2026-11-19T07:42:30.19Z"}.
// expires_at may also be epoch seconds. A missing expires_at counts as not expired.
SpotifyToken parseSpotifyToken(const QByteArray &json, const QDateTime &now);
// `GET /v1/me/player` (or `spotify_player get key playback`). "null" / empty / garbage -> ok == false.
SpotifyPlayback parseSpotifyPlayback(const QByteArray &json);
// `GET /v1/search?type=track` (or `spotify_player search`): the first track's bare id
// ("spotify:track:" stripped), from tracks[0].id (string or {id}), else tracks[0].uri.
QString parseSpotifyFirstTrackId(const QByteArray &json);
// Same, as a full "spotify:track:<id>" URI; empty unless the id is base62.
QString parseSpotifyFirstTrackUri(const QByteArray &json);
// The status AppleScript's reply: "<player state>\n<track>\n<artist>\n<sound volume>", or
// "not running". ok == true only for playing / paused / stopped.
SpotifyPlayback parseSpotifyAppleScriptStatus(const QByteArray &out);

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
    static QString defaultTokenPath();  // ~/.cache/spotify-player/user_client_token.json
    static QString defaultApiBase();    // https://api.spotify.com

    void setSpawnAllowed(bool on);
    bool spawnAllowed() const { return m_spawnAllowed; }
    void setBinary(const QString &path);
    QString binary() const { return m_bin; }
    void setTokenPath(const QString &path);
    QString tokenPath() const { return m_tokenPath; }
    // A non-default base (a test's local mock) enables the network even in a dry run.
    void setApiBase(const QString &base);
    QString apiBase() const { return m_apiBase; }
    void setAppPath(const QString &path);  // /Applications/Spotify.app
    QString appPath() const { return m_appPath; }

    SpotifyToken token() const;  // the token file as it is on disk right now
    bool appInstalled() const;

    bool available() const;
    bool playing() const { return m_playing; }
    QString track() const { return m_track; }
    QString artist() const { return m_artist; }
    int volume() const { return m_volume; }
    bool busy() const { return m_inFlight > 0; }
    QString error() const { return m_error; }
    // Every command issued (dry runs included): "api PUT /v1/me/player/pause",
    // "osascript next track", "cli playback next".
    QStringList commandLog() const { return m_log; }
    int spawnCount() const { return m_spawnCount; }
    int requestCount() const { return m_requestCount; }

    Q_INVOKABLE void refresh();
    Q_INVOKABLE void togglePlay();
    Q_INVOKABLE void play();
    Q_INVOKABLE void pause();
    Q_INVOKABLE void nextTrack();
    Q_INVOKABLE void prevTrack();
    Q_INVOKABLE void setVolume(int pct);
    Q_INVOKABLE void searchAndPlay(const QString &query);
    bool polling() const { return m_polling; }
    // Poll the playback state every 4 s while on (live only; a dry run just records the flag).
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
    // HTTP status (0 on a network error or timeout) and body.
    using ApiDone = std::function<void(int status, const QByteArray &body)>;
    // One control action through the route chain.
    struct Route {
        QByteArray verb;     // "PUT" / "POST"; empty: skip the Web API
        QString path;        // "/v1/me/player/next"
        QByteArray body;     // JSON, may be empty
        QString script;      // AppleScript for Spotify.app
        QString label;       // log label for the script: "next track"
        QStringList cli;     // spotify_player arguments
    };

    bool networkAllowed() const;
    void logCommand(const QString &line);
    // Sends the request when the network is allowed and the token usable; false otherwise
    // (the request is still logged).
    bool api(const QByteArray &verb, const QString &pathAndQuery, const QByteArray &body, ApiDone done);
    void runProcess(const QString &program, const QStringList &args, const QString &logLine, Done done);
    void run(const QStringList &args, Done done = {});
    void osascript(const QString &script, const QString &label, Done done = {});
    void dispatch(const Route &r, std::function<void(bool ok)> after = {});
    void fallback(const Route &r, std::function<void(bool ok)> after);
    void refreshLocal();
    void applyPlayback(const SpotifyPlayback &pb);
    void refreshSoon();
    void markPlaying(bool on);

    bool m_spawnAllowed = false;
    QString m_bin;
    QString m_tokenPath;
    QString m_apiBase;
    QString m_appPath;
    QNetworkAccessManager *m_net = nullptr;
    QTimer m_poll;
    QStringList m_log;
    int m_spawnCount = 0;
    int m_requestCount = 0;
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
