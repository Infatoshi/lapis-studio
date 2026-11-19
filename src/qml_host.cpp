#include "qml_host.hpp"

#include "backend.hpp"
#include "providers.hpp"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QFileSystemWatcher>
#include <QKeyEvent>
#include <QQmlComponent>
#include <QQmlContext>
#include <QQmlEngine>
#include <QQuickItem>
#include <QQuickWindow>
#include <QSocketNotifier>
#include <QUrl>

#include <csignal>
#include <cstdio>
#include <unistd.h>

namespace studio {

namespace {

// SIGUSR1 -> self-pipe -> QSocketNotifier on the GUI thread (the handler only writes a byte).
int g_sigPipe[2] = {-1, -1};
struct sigaction g_oldUsr1;
bool g_usr1Installed = false;

void onSigUsr1(int)
{
    const char b = 1;
    [[maybe_unused]] const ssize_t n = ::write(g_sigPipe[1], &b, 1);
}

// The properties a reload carries over, set back in this order (project before its slide and lens).
const char *const kUiProps[] = {"islandFilter", "selectedIsland", "projectKey", "artifactIndex", "lens"};

}  // namespace

QmlHost::QmlHost(QQuickWindow *window, Backend *backend, QObject *parent)
    : QObject(parent)
    , m_window(window)
    , m_backend(backend)
{
    m_debounce.setSingleShot(true);
    m_debounce.setInterval(150);
    connect(&m_debounce, &QTimer::timeout, this, [this] { reload(); });
}

QmlHost::~QmlHost()
{
    setHotReload(false);
    delete m_root.data();  // the root before the engine that created it
    delete m_engine;
}

void QmlHost::setQmlDir(const QString &dir)
{
    m_qmlDir = dir.isEmpty() ? QString() : QDir(dir).absolutePath();
    if (m_hotReload)
        rewatch();
}

void QmlHost::setHotReload(bool on)
{
    if (on == m_hotReload)
        return;
    m_hotReload = on;
    if (on) {
        m_fsWatcher = new QFileSystemWatcher(this);
        auto changed = [this](const QString &) { scheduleReload(); };
        connect(m_fsWatcher, &QFileSystemWatcher::fileChanged, this, changed);
        connect(m_fsWatcher, &QFileSystemWatcher::directoryChanged, this, changed);
        rewatch();
        if (m_window)
            m_window->installEventFilter(this);
        if (!g_usr1Installed && ::pipe(g_sigPipe) == 0) {
            struct sigaction sa {};
            sa.sa_handler = onSigUsr1;
            sigemptyset(&sa.sa_mask);
            sa.sa_flags = SA_RESTART;
            if (::sigaction(SIGUSR1, &sa, &g_oldUsr1) == 0) {
                g_usr1Installed = true;
                m_sigNotifier = new QSocketNotifier(g_sigPipe[0], QSocketNotifier::Read, this);
                connect(m_sigNotifier, &QSocketNotifier::activated, this, [this] {
                    char buf[16];
                    [[maybe_unused]] const ssize_t n = ::read(g_sigPipe[0], buf, sizeof buf);
                    std::fprintf(stderr, "lapis_studio: SIGUSR1, reloading QML\n");
                    QMetaObject::invokeMethod(this, &QmlHost::reload, Qt::QueuedConnection);
                });
            } else {
                ::close(g_sigPipe[0]);
                ::close(g_sigPipe[1]);
                g_sigPipe[0] = g_sigPipe[1] = -1;
            }
        }
        return;
    }
    m_debounce.stop();
    delete m_fsWatcher;
    m_fsWatcher = nullptr;
    if (m_window)
        m_window->removeEventFilter(this);
    if (m_sigNotifier) {
        delete m_sigNotifier;
        m_sigNotifier = nullptr;
        ::sigaction(SIGUSR1, &g_oldUsr1, nullptr);
        g_usr1Installed = false;
        ::close(g_sigPipe[0]);
        ::close(g_sigPipe[1]);
        g_sigPipe[0] = g_sigPipe[1] = -1;
    }
}

void QmlHost::rewatch()
{
    if (!m_fsWatcher || m_qmlDir.isEmpty())
        return;
    // Editors save by atomic rename, which drops a file watch; re-add everything each time.
    const QStringList old = m_fsWatcher->files() + m_fsWatcher->directories();
    if (!old.isEmpty())
        m_fsWatcher->removePaths(old);
    QStringList paths{m_qmlDir};
    for (const QString &f : QDir(m_qmlDir).entryList({QStringLiteral("*.qml")}, QDir::Files))
        paths << m_qmlDir + QLatin1Char('/') + f;
    m_fsWatcher->addPaths(paths);
}

void QmlHost::scheduleReload()
{
    rewatch();
    m_debounce.start();
}

bool QmlHost::eventFilter(QObject *watched, QEvent *event)
{
    if (watched == m_window && event->type() == QEvent::KeyPress) {
        auto *k = static_cast<QKeyEvent *>(event);
        // Qt maps Cmd to ControlModifier on macOS.
        const auto mods = k->modifiers() & (Qt::ControlModifier | Qt::ShiftModifier | Qt::AltModifier | Qt::MetaModifier);
        if (k->key() == Qt::Key_L && mods == (Qt::ControlModifier | Qt::ShiftModifier)) {
            QMetaObject::invokeMethod(this, &QmlHost::reload, Qt::QueuedConnection);
            return true;
        }
    }
    return QObject::eventFilter(watched, event);
}

QVariantMap QmlHost::saveUiState() const
{
    QVariantMap s;
    if (m_root) {
        for (const char *p : kUiProps) {
            const QVariant v = m_root->property(p);
            if (v.isValid())
                s.insert(QString::fromLatin1(p), v);
        }
    }
    if (m_backend)
        s.insert(QStringLiteral("focusedProject"), m_backend->focusedProject());
    return s;
}

void QmlHost::restoreUiState(const QVariantMap &state)
{
    if (m_backend && state.contains(QStringLiteral("focusedProject"))
        && m_backend->focusedProject() != state.value(QStringLiteral("focusedProject")).toString())
        m_backend->setFocusedProject(state.value(QStringLiteral("focusedProject")).toString());
    if (!m_root)
        return;
    for (const char *p : kUiProps) {
        const auto it = state.constFind(QString::fromLatin1(p));
        if (it != state.cend() && m_root->property(p).isValid())
            m_root->setProperty(p, *it);
    }
}

void QmlHost::fitRoot()
{
    if (!m_root || !m_window)
        return;
    m_root->setSize(m_window->size());
}

bool QmlHost::load()
{
    return reload();
}

bool QmlHost::reload()
{
    if (!m_window)
        return false;
    const bool first = m_root.isNull();
    const QVariantMap ui = saveUiState();

    auto *engine = new QQmlEngine(this);
    engine->addImageProvider(QStringLiteral("pdf"), new PdfImageProvider);
    QStringList errors;
    QQuickItem *item = nullptr;
    {
        QQmlComponent component(engine);
        if (m_qmlDir.isEmpty())
            component.loadFromModule("Studio", "StudioApp", QQmlComponent::PreferSynchronous);
        else
            component.loadUrl(QUrl::fromLocalFile(m_qmlDir + QStringLiteral("/StudioApp.qml")),
                              QQmlComponent::PreferSynchronous);
        if (component.isReady()) {
            // The C++-owned Backend goes in as StudioApp's own `backend` property: no singleton
            // registration, so every fresh engine binds to the same live object.
            QObject *obj = component.createWithInitialProperties(
                {{QStringLiteral("backend"), QVariant::fromValue(static_cast<QObject *>(m_backend))},
                 {QStringLiteral("demoMode"), false}},
                engine->rootContext());
            item = qobject_cast<QQuickItem *>(obj);
            if (!item) {
                delete obj;
                for (const QQmlError &e : component.errors())
                    errors << e.toString();
                if (errors.isEmpty())
                    errors << QStringLiteral("StudioApp did not create a QQuickItem");
            }
        } else {
            for (const QQmlError &e : component.errors())
                errors << e.toString();
            if (errors.isEmpty())
                errors << QStringLiteral("StudioApp component not ready");
        }
    }
    if (!item) {
        // Keep whatever is running: a typo in a .qml file never blanks the window.
        delete engine;
        ++m_failures;
        m_lastErrors = errors;
        for (const QString &e : errors)
            std::fprintf(stderr, "lapis_studio: QML reload failed: %s\n", qPrintable(e));
        emit reloaded(false, m_generation);
        return false;
    }

    QQmlEngine::setObjectOwnership(item, QQmlEngine::CppOwnership);
    QPointer<QQuickItem> oldRoot = m_root;
    QQmlEngine *oldEngine = m_engine;
    if (oldRoot) {
        oldRoot->setVisible(false);
        oldRoot->setEnabled(false);
    }
    item->setParentItem(m_window->contentItem());
    m_root = item;
    m_engine = engine;
    fitRoot();
    connect(m_window, &QWindow::widthChanged, item, [this] { fitRoot(); });
    connect(m_window, &QWindow::heightChanged, item, [this] { fitRoot(); });
    if (!first)
        restoreUiState(ui);
    item->forceActiveFocus();
    if (oldRoot) {
        oldRoot->setParentItem(nullptr);
        oldRoot->deleteLater();  // posted before the engine's, so it is destroyed first
    }
    if (oldEngine)
        oldEngine->deleteLater();
    ++m_generation;
    m_lastErrors.clear();
    if (!first)
        std::fprintf(stderr, "lapis_studio: QML reloaded (generation %d)\n", m_generation);
    emit reloaded(true, m_generation);
    return true;
}

QString QmlHost::findQmlDir()
{
    const QString env = qEnvironmentVariable("LAPIS_QML_DIR").trimmed();
    if (!env.isEmpty() && QFileInfo::exists(env + QStringLiteral("/StudioApp.qml")))
        return QDir(env).absolutePath();
    const QString repo = QDir(QCoreApplication::applicationDirPath() + QStringLiteral("/../qml")).absolutePath();
    if (QFileInfo::exists(repo + QStringLiteral("/StudioApp.qml")))
        return repo;
    return {};
}

bool QmlHost::ensureQmldir(const QString &dir)
{
    const QString path = dir + QStringLiteral("/qmldir");
    if (QFileInfo::exists(path))
        return true;
    QFile f(path);
    if (!f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        return false;
    return f.write("singleton Theme 1.0 Theme.qml\n") > 0;
}

}  // namespace studio
