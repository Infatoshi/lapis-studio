// Phonon live meeting / audio transcript ingest. Phonon writes
//   <header lines, including "Status: recording|transcribing|complete">
//   ---
//   0:12 You: ...
//   0:15 Call: ...
// to Meetings/latest.txt and replaces it by atomic rename (or repoints a symlink).
#pragma once

#include <QString>

namespace studio {

// ~/Library/Application Support/Phonon/Meetings/latest.txt
QString phononLatestPath();

struct PhononSnapshot {
    QString tail;        // last N non-empty turn lines after the header; empty when missing or stale
    QString status;      // "recording" | "transcribing" | "complete" | "" (no header)
    qint64 mtimeMs = 0;  // 0 when the file is missing
    bool live = false;   // fresh, has turns, and not complete
};

// One consistent read: the file is opened once, so an atomic replace mid-read yields either the
// old or the new transcript, never a mix. Fresh means modified within maxAgeMs (<= 0: any age).
PhononSnapshot readPhononSnapshot(const QString &path, int maxLines = 15, qint64 maxAgeMs = 10 * 60 * 1000);

// Read the last `maxLines` non-empty turn lines from Phonon's live transcript if modified within
// `maxAgeMs` (returns empty if missing or stale). The header up to "\n---\n" is never included.
QString readPhononTail(const QString &path, int maxLines = 15, qint64 maxAgeMs = 10 * 60 * 1000);

}  // namespace studio
