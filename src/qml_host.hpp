// One long-lived QQuickWindow (and the C++ Backend behind it) hosting a StudioApp root item that
// can be rebuilt in place. Each reload builds a *fresh* QQmlEngine, instantiates StudioApp from
// the qml/ directory on disk, and only on success swaps the new root in and drops the old root
// and engine; a broken edit logs its errors and leaves the running UI untouched. The window,
// its fullscreen state, the Backend, Slowmix audio and LLM streams never notice.
//
// Without a QML directory the host loads the compiled-in module (qrc) once, as release builds do.
#pragma once

#include <QObject>
#include <QPointer>
#include <QQuickItem>
#include <QQuickWindow>
#include <QStringList>
#include <QTimer>
#include <QVariantMap>

class QFileSystemWatcher;
class QQmlEngine;
class QSocketNotifier;

namespace studio {

class Backend;

class QmlHost : public QObject {
    Q_OBJECT
public:
    QmlHost(QQuickWindow *window, Backend *backend, QObject *parent = nullptr);
    ~QmlHost() override;

    // The qml/ directory to load from (hot-reload source). Empty: the compiled-in module.
    void setQmlDir(const QString &dir);
    QString qmlDir() const { return m_qmlDir; }
    // Watch qml/*.qml (150 ms debounce), Cmd+Shift+L on the window, and SIGUSR1.
    void setHotReload(bool on);
    bool hotReload() const { return m_hotReload; }

    // First load, and every reload after it. True when a new root replaced the old one.
    bool load();
    Q_INVOKABLE bool reload();

    QQuickWindow *window() const { return m_window; }
    QQuickItem *root() const { return m_root; }
    QQmlEngine *engine() const { return m_engine; }
    int generation() const { return m_generation; }  // successful loads so far
    int failures() const { return m_failures; }
    QStringList lastErrors() const { return m_lastErrors; }

    // UI state carried across a reload (lens, project, slide, Needs Me filter).
    QVariantMap saveUiState() const;
    void restoreUiState(const QVariantMap &state);

    // LAPIS_QML_DIR, else <repo>/qml when StudioApp.qml exists there (repo = the executable's
    // parent directory, i.e. build/..). Empty when neither exists.
    static QString findQmlDir();
    // Write qml/qmldir (Theme singleton) when the directory lacks one. False if it cannot.
    static bool ensureQmldir(const QString &dir);

signals:
    void reloaded(bool ok, int generation);

protected:
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    void scheduleReload();
    void rewatch();
    void fitRoot();

    QPointer<QQuickWindow> m_window;
    Backend *m_backend = nullptr;
    QString m_qmlDir;
    bool m_hotReload = false;
    QQmlEngine *m_engine = nullptr;
    QPointer<QQuickItem> m_root;
    int m_generation = 0;
    int m_failures = 0;
    QStringList m_lastErrors;
    QFileSystemWatcher *m_fsWatcher = nullptr;
    QSocketNotifier *m_sigNotifier = nullptr;
    QTimer m_debounce;
};

}  // namespace studio
