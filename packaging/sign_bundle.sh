#!/bin/bash
# Run from any directory. Set LAPIS_CODESIGN_IDENTITY to a signing identity.
set -euo pipefail
: "${LAPIS_CODESIGN_IDENTITY:?Set LAPIS_CODESIGN_IDENTITY}"
root="$(cd "$(dirname "$0")/.." && pwd)"
build="$root/build"
bundle="$build/LapisStudio.app"
test -x "$build/lapis_studio" || { echo 'Build lapis_studio first.' >&2; exit 1; }
# macdeployqt must come from the same Qt installation used to build the executable.
deploy="$(command -v macdeployqt || true)"
test -n "$deploy" || { echo 'Put the build Qt installation bin directory on PATH (macdeployqt required).' >&2; exit 1; }
test ! -e "$bundle" || { echo 'LapisStudio.app already exists; move it aside before packaging.' >&2; exit 1; }
stage="$(mktemp -d "$build/.lapis-bundle.XXXXXX")"
trap 'rm -rf "$stage"' EXIT
app="$stage/LapisStudio.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$build/lapis_studio" "$app/Contents/MacOS/lapis_studio"
cp "$root/packaging/Info.plist.in" "$app/Contents/Info.plist"
plutil -lint "$app/Contents/Info.plist" "$root/packaging/entitlements.plist"
"$deploy" "$app" "-qmldir=$root/qml" -always-overwrite
codesign --sign "${LAPIS_CODESIGN_IDENTITY:?Set LAPIS_CODESIGN_IDENTITY}" \
    --options runtime --entitlements "$root/packaging/entitlements.plist" --deep --force "$app"
codesign --verify --deep --strict "$app"
mv "$app" "$bundle"
echo 'Created build/LapisStudio.app; signature verified.'
