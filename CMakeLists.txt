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
qt_standard_project_setup(REQUIRES 6.11)

qt_add_executable(lapis_studio
  src/main.cpp
  src/backend.hpp src/backend.cpp
  src/transcript.hpp src/transcript.cpp
  src/pdfrender.hpp src/pdfrender.mm
  src/providers.hpp src/providers.cpp
  src/fixtures.hpp src/fixtures.cpp
  src/demo.hpp src/demo.cpp
  src/tests.cpp
  src/bench.cpp
  src/metalbench.hpp src/metalbench.mm
  src/gamepad.hpp src/gamepad.cpp src/gamepad_mac.mm
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
)

target_compile_options(lapis_studio PRIVATE -Wall -Wextra)
target_link_libraries(lapis_studio PRIVATE
  Qt6::Core Qt6::Gui Qt6::Qml Qt6::Quick Qt6::QuickControls2 Qt6::QuickShapes
  Qt6::Multimedia Qt6::Svg Qt6::Network Qt6::Concurrent
  "-framework PDFKit" "-framework AppKit" "-framework CoreGraphics" "-framework Metal" "-framework GameController")
