// Native PDFKit page rendering (macOS). Implemented in pdfrender.mm.
#pragma once

#include <QImage>
#include <QString>

namespace studio {

int pdfPageCount(const QString &path);

// Renders one page at `targetWidth` pixels wide. With `ink` the page is mapped to
// warm parchment ink on a near-black card so it sits on the OLED stage instead of
// glaring white; saturated pixels (figures) keep their colour.
QImage renderPdfPage(const QString &path, int page, int targetWidth, bool ink);

}  // namespace studio
