# Lapis Studio

Standalone Qt 6 / QML + Metal 240 Hz workstation UI for supervising concurrent agentic coding sessions, inspecting artifacts (charts, diffs, tables, PDFs, media, write-ups), and asking rapid-feedback questions across multiple model providers.

## Features

- **240 Hz Metal Constellation & Zoom Morph**: Overview constellation of active workspaces with smooth spring-damped focus navigation (`vim`, `wasd`, or `mouse` modes) and DualShock 4 / extended gamepad support via macOS `GameController.framework`.
- **Artifact Stage**: Inspect charts, diffs, CSV tables, PDFs (`PDFKit`), images, video (`QtMultimedia`), and agent write-ups with keyboard, mouse, or controller scrolling.
- **Multi-Provider Fast Q&A**:
  - `claude` (Anthropic)
  - `codex app-server` (OpenAI)
  - `agy` (Google)
  - OpenAI-compatible HTTP/SSE endpoints (e.g. vLLM / `qwen3.8-flash-next`) with prefix-cached prompt layout and `reasoning_content` / `reasoning` delta streaming.

## Build (macOS)

Requires CMake 3.24+, Ninja, Xcode Command Line Tools (`PDFKit`, `AppKit`, `CoreGraphics`, `Metal`, `GameController`), and Qt 6.11+.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<path-to-qt6>
cmake --build build -j
```

## Verification

```bash
QT_QPA_PLATFORM=offscreen ./build/lapis_studio --test
QT_QPA_PLATFORM=offscreen ./build/lapis_studio --bench-fps --metal
```

## OpenAI-Compatible Endpoint Configuration

Configure a custom OpenAI-compatible server in `~/.lapis-next/studio_prefs.json` or via environment variables:

```json
{
  "openaiCompatUrl": "http://localhost:8000/v1",
  "openaiCompatModel": "qwen3.8-flash-next"
}
```

Or:

```bash
export LAPIS_OPENAI_COMPAT_URL="http://localhost:8000/v1"
export LAPIS_OPENAI_COMPAT_MODEL="qwen3.8-flash-next"
```
