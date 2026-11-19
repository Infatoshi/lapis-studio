#include "ingest/phonon.hpp"

#include <QDateTime>
#include <QDir>
#include <QFile>
#include <QStringList>

#include <algorithm>

#include <sys/stat.h>

namespace studio {

namespace {

constexpr qint64 kHeadBytes = 4096;   // the header always fits here
constexpr qint64 kTailBytes = 16384;  // enough for far more than maxLines turns

qint64 fdMtimeMs(int fd)
{
    struct stat st {};
    if (fd < 0 || ::fstat(fd, &st) != 0)
        return 0;
#if defined(__APPLE__)
    return qint64(st.st_mtimespec.tv_sec) * 1000 + st.st_mtimespec.tv_nsec / 1000000;
#else
    return qint64(st.st_mtim.tv_sec) * 1000 + st.st_mtim.tv_nsec / 1000000;
#endif
}

QString headerStatus(const QString &header)
{
    for (const QString &raw : header.split(QLatin1Char('\n'))) {
        const QString s = raw.trimmed();
        if (s.startsWith(QLatin1String("Status:"), Qt::CaseInsensitive))
            return s.mid(7).trimmed().toLower();
    }
    return {};
}

}  // namespace

QString phononLatestPath()
{
    return QDir::homePath() + QStringLiteral("/Library/Application Support/Phonon/Meetings/latest.txt");
}

PhononSnapshot readPhononSnapshot(const QString &path, int maxLines, qint64 maxAgeMs)
{
    PhononSnapshot snap;
    QFile f(path);  // follows a symlink; the open fd pins one version of the file
    if (!f.open(QIODevice::ReadOnly))
        return snap;
    snap.mtimeMs = fdMtimeMs(f.handle());
    const qint64 size = f.size();

    // Header: everything up to and including the first "\n---\n" near the top.
    const QByteArray head = f.read(std::min(size, kHeadBytes));
    qint64 bodyStart = 0;
    const qsizetype sep = head.indexOf("\n---\n");
    if (sep >= 0) {
        snap.status = headerStatus(QString::fromUtf8(head.left(sep)));
        bodyStart = sep + 5;
    }

    const qint64 ageMs = QDateTime::currentMSecsSinceEpoch() - snap.mtimeMs;
    if (maxAgeMs > 0 && ageMs > maxAgeMs)
        return snap;  // stale: status and mtime only

    const qint64 from = std::max(bodyStart, size - kTailBytes);
    if (!f.seek(from))
        return snap;
    QString text = QString::fromUtf8(f.readAll());
    if (from > bodyStart) {
        // Started mid-file: drop the partial first line.
        const qsizetype nl = text.indexOf(QLatin1Char('\n'));
        text = nl >= 0 ? text.mid(nl + 1) : QString();
    }
    QStringList lines;
    for (const QString &raw : text.split(QLatin1Char('\n'))) {
        const QString s = raw.trimmed();
        if (!s.isEmpty())
            lines << s;
    }
    if (maxLines >= 0 && lines.size() > maxLines)
        lines = lines.mid(lines.size() - maxLines);
    snap.tail = lines.join(QLatin1Char('\n'));
    snap.live = !snap.tail.isEmpty() && snap.status != QLatin1String("complete");
    return snap;
}

QString readPhononTail(const QString &path, int maxLines, qint64 maxAgeMs)
{
    return readPhononSnapshot(path, maxLines, maxAgeMs).tail;
}

}  // namespace studio
