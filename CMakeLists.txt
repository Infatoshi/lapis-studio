cmake_minimum_required(VERSION 3.24)
project(lapis_studio LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_AUTOMOC ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Qt6 6.11 REQUIRED COMPONENTS Core Gui Qml Quick QuickControls2 QuickShapes Multimedia Svg Network Concurrent)
find_package(CURL REQUIRED)
qt_standard_project_setup(REQUIRES 6.11)

function(lapis_warnings target)
  target_compile_options(${target} PRIVATE -Wall -Wextra)
endfunction()

# --- Layered static libraries (one-way: core <- llm <- ingest) ----------

# Value types and the lock-free SPSC ring. QtCore only.
add_library(lapis_core STATIC
  src/core/types.hpp src/core/types.cpp
  src/core/window_geometry.hpp src/core/window_geometry.cpp
  src/core/ring_buffer.hpp
)
target_include_directories(lapis_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lapis_core PUBLIC Qt6::Core)
lapis_warnings(lapis_core)

# Transcript parsing and Phonon live-transcript ingest.
add_library(lapis_ingest STATIC
  src/ingest/phonon.hpp src/ingest/phonon.cpp
  src/ingest/slowmix.hpp src/ingest/slowmix.cpp
  src/ingest/memory.hpp src/ingest/memory.cpp
  src/ingest/ambient.hpp src/ingest/ambient.cpp
  src/transcript.hpp src/transcript.cpp
)
# ambient.cpp reuses lapis_llm's Tier 0/1 builders so its prefix stays byte-identical (llm never links ingest).
target_link_libraries(lapis_ingest PUBLIC lapis_core lapis_llm Qt6::Core Qt6::Concurrent Qt6::Network)
lapis_warnings(lapis_ingest)

# APC-aligned prompt builder, SSE parser and the libcurl-multi LlmRouter.
add_library(lapis_llm STATIC
  src/llm/prefix.hpp src/llm/prefix.cpp
  src/llm/sse.hpp src/llm/sse.cpp
  src/llm/router.hpp src/llm/router.cpp
)
target_link_libraries(lapis_llm PUBLIC lapis_core Qt6::Core PRIVATE CURL::libcurl)
lapis_warnings(lapis_llm)

# Hover / pin semantic targets and their Tier 3 slice. QtCore + QtGui only.
add_library(lapis_targets STATIC
  src/targets/target_registry.hpp src/targets/target_registry.cpp
)
target_link_libraries(lapis_targets PUBLIC lapis_core Qt6::Core Qt6::Gui)
lapis_warnings(lapis_targets)

# macOS-only rendering and input: PDFKit pages, GameController pad, offscreen Metal target.
add_library(lapis_mac STATIC
  src/pdfrender.hpp src/pdfrender.mm
  src/gamepad.hpp src/gamepad.cpp src/gamepad_mac.mm
  src/metalbench.hpp src/metalbench.mm
  src/mac/window_state.hpp src/mac/window_state.mm
)
target_include_directories(lapis_mac PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(lapis_mac PUBLIC lapis_core Qt6::Core Qt6::Gui Qt6::Quick
  "-framework ApplicationServices" "-framework PDFKit" "-framework AppKit" "-framework CoreGraphics" "-framework Metal" "-framework GameController")
lapis_warnings(lapis_mac)

# --- App -----------------------------------------------------------------------------------

qt_add_executable(lapis_studio
  src/main.cpp
  src/backend.hpp src/backend.cpp
  src/providers.hpp src/providers.cpp
  src/fixtures.hpp src/fixtures.cpp
  src/demo.hpp src/demo.cpp
  src/tests.cpp
  src/bench.cpp
  src/qml_host.hpp src/qml_host.cpp
)
# A plain Unix executable at build/lapis_studio, as the brief asks (no .app bundle).
set_target_properties(lapis_studio PROPERTIES MACOSX_BUNDLE OFF)

set_source_files_properties(qml/Theme.qml PROPERTIES QT_QML_SINGLETON_TYPE TRUE)
qt_add_qml_module(lapis_studio
  URI Studio
  VERSION 1.0
  QML_FILES
    qml/Theme.qml
    qml/Main.qml
    qml/StudioApp.qml
    qml/Constellation.qml
    qml/Island.qml
    qml/Ladder.qml
    qml/Stage.qml
    qml/Composer.qml
    qml/NewSessionDialog.qml
    qml/Caret.qml
    qml/ChartView.qml
    qml/DiffView.qml
    qml/CsvTable.qml
    qml/PdfView.qml
    qml/ImageView.qml
    qml/MediaView.qml
    qml/ProsePanel.qml
    qml/PredictorRing.qml
    qml/FocusCursor.qml
    qml/TermZoomBar.qml
    qml/SettingsOverlay.qml
    qml/ControllerMapOverlay.qml
    qml/OnboardingModal.qml
    qml/MorningBriefingOverlay.qml
    qml/SessionReviewOverlay.qml
    qml/SlowmixOverlay.qml
    qml/FloatingOmnibar.qml
)

lapis_warnings(lapis_studio)
target_link_libraries(lapis_studio PRIVATE
  lapis_core lapis_ingest lapis_llm lapis_targets lapis_mac
  Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::QuickShapes
  Qt6::Multimedia Qt6::Svg Qt6::Network Qt6::Concurrent)

# --- Fast per-module tests (no QML, no GUI) ------------------------------------------------

qt_add_executable(test_core_ingest tests/test_core_ingest.cpp)
set_target_properties(test_core_ingest PROPERTIES MACOSX_BUNDLE OFF)
target_link_libraries(test_core_ingest PRIVATE lapis_core lapis_ingest Qt6::Core)
lapis_warnings(test_core_ingest)

qt_add_executable(test_llm tests/test_llm.cpp)
set_target_properties(test_llm PROPERTIES MACOSX_BUNDLE OFF)
target_link_libraries(test_llm PRIVATE lapis_llm lapis_core Qt6::Core)
lapis_warnings(test_llm)

qt_add_executable(test_targets tests/test_targets.cpp)
set_target_properties(test_targets PROPERTIES MACOSX_BUNDLE OFF)
target_link_libraries(test_targets PRIVATE lapis_targets lapis_core Qt6::Core Qt6::Gui)
lapis_warnings(test_targets)
