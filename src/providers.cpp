#include "providers.hpp"

#include "pdfrender.hpp"

#include <QUrl>

namespace studio {

PdfImageProvider::PdfImageProvider()
    : QQuickImageProvider(QQuickImageProvider::Image)
{
}

QImage PdfImageProvider::requestImage(const QString &id, QSize *size, const QSize &requestedSize)
{
    const qsizetype slash = id.indexOf(QLatin1Char('/'));
    const int page = slash > 0 ? id.left(slash).toInt() : 0;
    const QString path = QUrl::fromPercentEncoding(id.mid(slash + 1).toUtf8());
    const int width = requestedSize.width() > 0 ? requestedSize.width() : 1200;
    QImage img = renderPdfPage(path, page, width, true);
    if (size)
        *size = img.size();
    return img;
}

}  // namespace studio
