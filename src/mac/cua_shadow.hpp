#pragma once

// Passive human-click shadow recorder: every real left / right mouse-down the user makes becomes
// one labeled grounding sample (display JPEG + AX element under the pointer) in
// <traceDir>/human_traces.jsonl + human_frames/. The event tap is listen-only (it never blocks or
// delays input); AX lookup, capture and disk I/O run on one background worker thread.

#include "mac/cua.hpp"
#include <QPointF>
#include <QRectF>
#include <QString>
#include <QVector>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>

namespace studio::cua {

constexpr int kShadowMinGapMs = 350;       // at most one capture per 350 ms
constexpr int kShadowFrameWidth = 1536;    // same width as the CUA L3 / trace frames
constexpr int kShadowDefaultMaxSteps = 3000;
constexpr const char *kShadowLog = "human_traces.jsonl";
constexpr const char *kShadowFrames = "human_frames";

// One captured mouse-down, as the tap saw it.
struct ShadowClick {
    QPointF globalPt;   // Quartz global points
    bool right = false;
    qint64 tsMs = 0;    // ms since epoch (UTC)
};

// The AX element under the click point.
struct ShadowAxHit {
    bool ok = false;
    QString role, subrole, title, description, appName;
    int pid = 0;
    QRectF globalRect;  // Quartz global points; empty when unknown
};

// Dependency-injected side effects, so tests never touch a real tap, AX or ScreenCaptureKit.
struct ShadowHooks {
    std::function<QVector<DisplayInfo>()> listDisplays;
    std::function<ShadowAxHit(QPointF)> elementAt;
    std::function<QByteArray(uint32_t displayId, int maxWidth)> captureJpeg;
    int selfPid = 0;  // clicks on this process (Lapis) are skipped; 0 = no self filter
};
// The real hooks: listDisplays, AXUIElementCopyElementAtPosition, captureDisplayJpeg excluding
// excludeIds() (called per capture; may be empty).
ShadowHooks defaultShadowHooks(std::function<QVector<uint32_t>()> excludeIds = {});

enum class ShadowOutcome { Recorded, SkippedSensitive, SkippedNoDisplay, SkippedNoFrame, WriteFailed };
struct ShadowResult {
    ShadowOutcome outcome = ShadowOutcome::WriteFailed;
    QString image;  // "human_frames/<ts_ms>.jpg" when recorded
    QByteArray line;  // the JSONL line written (no trailing newline)
};

// True for a password field (role or subrole AXSecureTextField: macOS reports password inputs as
// role AXTextField + subrole AXSecureTextField) or when the app / title / description mentions
// infinity, 1password, keychain or bitwarden (case-insensitive).
bool shadowSensitive(const ShadowAxHit &hit);
// True when a click at nowMs may be captured, given the last captured one (0 = none yet).
bool shadowRateAllows(qint64 lastMs, qint64 nowMs, int minGapMs = kShadowMinGapMs);
// Pure-ish core: privacy check -> display lookup -> capture -> frame + one JSONL line -> cap.
// Never captures a frame for a sensitive target.
ShadowResult recordClickSample(const QString &traceDir, const ShadowClick &click, const ShadowHooks &hooks,
                               int maxSteps = kShadowDefaultMaxSteps);
// Keeps the newest maxSteps lines once the log exceeds maxSteps * 1.1, and deletes every frame
// in human_frames/ no kept line names.
void pruneHumanTraces(const QString &traceDir, int maxSteps);

class CuaShadowRecorder {
public:
    CuaShadowRecorder();
    ~CuaShadowRecorder();  // stops
    CuaShadowRecorder(const CuaShadowRecorder &) = delete;
    CuaShadowRecorder &operator=(const CuaShadowRecorder &) = delete;

    // Installs the listen-only tap on its own run-loop thread and starts the worker. False when
    // already running, traceDir is empty, or the tap cannot be created (no Input Monitoring /
    // Accessibility permission). Never prompts.
    bool start(const QString &traceDir, int maxSteps = kShadowDefaultMaxSteps);
    // Removes the tap, drops queued clicks, waits for the in-flight sample to finish.
    void stop();
    bool isRunning() const;
    int recordedCount() const;
    // The overlay windows (reticle, HUD bar) every frame excludes. Thread-safe.
    void setExcludeWindowIds(const QVector<uint32_t> &ids);
    // Tests: replace the real hooks (call before start).
    void setHooks(ShadowHooks hooks);

    struct Impl;

private:
    std::unique_ptr<Impl> d;
};

}  // namespace studio::cua
