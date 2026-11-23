#pragma once

#include <QString>

namespace studio::cua {

// lapis_studio --bench-cua "Chrome|Me" [RUNS [VLM_ENDPOINT]]: times the real click path (AX scans, captures,
// JPEG encodes, trace writes, fixed pauses) per hook with no input, overlay or activation.
int benchCua(const QString &target, int runs, const QString &vlmEndpoint = {});

}  // namespace studio::cua
