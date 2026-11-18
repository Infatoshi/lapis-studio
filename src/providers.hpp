#pragma once

#include <QQuickImageProvider>

namespace studio {

// image://pdf/<page>/<percent-encoded absolute path>  -> PDFKit-rendered page.
class PdfImageProvider : public QQuickImageProvider {
public:
    PdfImageProvider();
    QImage requestImage(const QString &id, QSize *size, const QSize &requestedSize) override;
};

}  // namespace studio
