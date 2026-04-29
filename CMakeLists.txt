cmake_minimum_required(VERSION 3.18)

# Detect CUDA availability before project() so we can declare languages.
set(_DASGRAIN_CUDA_AVAILABLE OFF)
if(NOT APPLE)
    include(CheckLanguage)
    # We declare CXX first; CUDA is enabled below on platforms that support it.
endif()

project(DasGrainOFX
    VERSION 1.0.0
    DESCRIPTION "DasGrain regrain plug-in for DaVinci Resolve (Fusion)"
    LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE)
    if(NOT CMAKE_OSX_ARCHITECTURES)
        set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "OSX archs" FORCE)
    endif()
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Min macOS" FORCE)
    endif()
endif()

# ---------------------------------------------------------------------------
# Backend toggles
# ---------------------------------------------------------------------------
option(DASGRAIN_ENABLE_METAL  "Build the Metal kernel backend (Apple only)" ON)
option(DASGRAIN_ENABLE_CUDA   "Build the CUDA kernel backend"               ON)
option(DASGRAIN_ENABLE_OPENCL "Build the OpenCL kernel backend"             ON)
option(DASGRAIN_ENABLE_OPENMP "Use OpenMP for the CPU backend"              ON)

if(APPLE)
    set(DASGRAIN_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
endif()

# Try to enable CUDA as a project language if requested.
if(DASGRAIN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "all-major")
        endif()
        set(_DASGRAIN_CUDA_AVAILABLE ON)
    else()
        message(STATUS "DasGrain: CUDA compiler not found; disabling CUDA backend.")
        set(DASGRAIN_ENABLE_CUDA OFF)
    endif()
endif()

# Locate OpenCL.
set(_DASGRAIN_OPENCL_AVAILABLE OFF)
if(DASGRAIN_ENABLE_OPENCL)
    if(APPLE)
        find_library(OPENCL_FRAMEWORK OpenCL)
        if(OPENCL_FRAMEWORK)
            set(_DASGRAIN_OPENCL_AVAILABLE ON)
        else()
            message(STATUS "DasGrain: OpenCL framework not found on macOS; disabling OpenCL backend.")
            set(DASGRAIN_ENABLE_OPENCL OFF)
        endif()
    else()
        find_package(OpenCL QUIET)
        if(OpenCL_FOUND)
            set(_DASGRAIN_OPENCL_AVAILABLE ON)
        else()
            message(STATUS "DasGrain: OpenCL not found; disabling OpenCL backend.")
            set(DASGRAIN_ENABLE_OPENCL OFF)
        endif()
    endif()
endif()

set(_DASGRAIN_METAL_AVAILABLE OFF)
if(APPLE AND DASGRAIN_ENABLE_METAL)
    set(_DASGRAIN_METAL_AVAILABLE ON)
endif()

# ---------------------------------------------------------------------------
# OpenFX SDK (vendored as a submodule under third_party/openfx)
# ---------------------------------------------------------------------------
set(OFX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openfx")
set(OFX_HEADER_DIR "${OFX_ROOT}/include")
set(OFX_SUPPORT_HEADER_DIR "${OFX_ROOT}/Support/include")
set(OFX_SUPPORT_LIBRARY_DIR "${OFX_ROOT}/Support/Library")

if(NOT EXISTS "${OFX_HEADER_DIR}/ofxImageEffect.h")
    message(FATAL_ERROR
        "OpenFX SDK headers not found at ${OFX_HEADER_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

file(GLOB OFX_SUPPORT_LIBRARY_FILES CONFIGURE_DEPENDS
    "${OFX_SUPPORT_LIBRARY_DIR}/*.cpp")

add_library(OfxSupport STATIC ${OFX_SUPPORT_LIBRARY_FILES})
target_include_directories(OfxSupport PUBLIC
    "${OFX_HEADER_DIR}"
    "${OFX_SUPPORT_HEADER_DIR}")
target_compile_features(OfxSupport PUBLIC cxx_std_17)
set_target_properties(OfxSupport PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_definitions(OfxSupport PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(OfxSupport PRIVATE -Wno-deprecated -Wno-deprecated-declarations)
endif()

# ---------------------------------------------------------------------------
# Plugin sources
# ---------------------------------------------------------------------------
set(DASGRAIN_CORE_SOURCES
    src/core/ColorSpace.cpp
    src/core/FrameRange.cpp
    src/core/ResponseCurve.cpp
    src/core/Analyser.cpp
    src/core/Voronoi.cpp)

set(DASGRAIN_PLUGIN_SOURCES
    src/plugin/DasGrainFactory.cpp
    src/plugin/DasGrainPlugin.cpp
    src/plugin/DasGrainInteract.cpp
    src/plugin/DegrainHelperPlugin.cpp
    src/plugin/Params.cpp
    src/plugin/PluginRegistration.cpp)

set(DASGRAIN_KERNEL_SOURCES
    src/kernels/cpu/GrainApplyCPU.cpp
    src/kernels/cpu/VoronoiCPU.cpp
    src/kernels/cpu/AnalyserCPU.cpp
    src/kernels/KernelDispatch.cpp)

set(DASGRAIN_ALL_SOURCES
    ${DASGRAIN_CORE_SOURCES}
    ${DASGRAIN_PLUGIN_SOURCES}
    ${DASGRAIN_KERNEL_SOURCES})

if(_DASGRAIN_METAL_AVAILABLE)
    list(APPEND DASGRAIN_ALL_SOURCES
        src/kernels/metal/GrainApplyMetal.mm
        src/kernels/metal/VoronoiMetal.mm
        src/kernels/metal/AnalyserMetal.mm)
endif()
if(_DASGRAIN_CUDA_AVAILABLE)
    list(APPEND DASGRAIN_ALL_SOURCES
        src/kernels/cuda/GrainApplyCuda.cu
        src/kernels/cuda/VoronoiCuda.cu
        src/kernels/cuda/AnalyserCuda.cu)
endif()
if(_DASGRAIN_OPENCL_AVAILABLE)
    list(APPEND DASGRAIN_ALL_SOURCES
        src/kernels/opencl/GrainApplyOpenCL.cpp
        src/kernels/opencl/VoronoiOpenCL.cpp
        src/kernels/opencl/AnalyserOpenCL.cpp)
endif()

# Plugin bundle layout
if(APPLE)
    set(DASGRAIN_ARCHDIR "MacOS")
elseif(WIN32)
    set(DASGRAIN_ARCHDIR "Win64")
elseif(UNIX)
    set(DASGRAIN_ARCHDIR "Linux-x86-64")
else()
    set(DASGRAIN_ARCHDIR "unknown-arch")
endif()

if(NOT PLUGIN_INSTALLDIR)
    if(APPLE)
        set(PLUGIN_INSTALLDIR "/Library/OFX/Plugins")
    elseif(WIN32)
        set(PLUGIN_INSTALLDIR "C:/Program Files/Common Files/OFX/Plugins")
    elseif(UNIX)
        set(PLUGIN_INSTALLDIR "/usr/OFX/Plugins")
    endif()
endif()

set(DASGRAIN_TARGET DasGrain)
set(DASGRAIN_BUNDLE_NAME "DasGrain.ofx.bundle")
set(DASGRAIN_BUNDLE_DIR  "${CMAKE_BINARY_DIR}/${DASGRAIN_BUNDLE_NAME}")

if(APPLE)
    add_library(${DASGRAIN_TARGET} MODULE ${DASGRAIN_ALL_SOURCES})
else()
    add_library(${DASGRAIN_TARGET} SHARED ${DASGRAIN_ALL_SOURCES})
endif()

set_target_properties(${DASGRAIN_TARGET} PROPERTIES
    SUFFIX ".ofx"
    PREFIX ""
    OUTPUT_NAME "DasGrain"
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden)

target_link_libraries(${DASGRAIN_TARGET} PRIVATE OfxSupport)

target_include_directories(${DASGRAIN_TARGET} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${OFX_HEADER_DIR}"
    "${OFX_SUPPORT_HEADER_DIR}")

target_compile_definitions(${DASGRAIN_TARGET} PRIVATE
    DASGRAIN_VERSION_STR="${PROJECT_VERSION}")

if(_DASGRAIN_METAL_AVAILABLE)
    target_compile_definitions(${DASGRAIN_TARGET} PRIVATE DASGRAIN_HAS_METAL=1)
    target_link_libraries(${DASGRAIN_TARGET} PRIVATE
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore")
endif()

# OpenGL: needed by the overlay interact (DasGrainInteract draws the sample
# box). The host provides the live GL context; we only need link-time
# satisfaction.
if(APPLE)
    target_link_libraries(${DASGRAIN_TARGET} PRIVATE "-framework OpenGL")
elseif(WIN32)
    target_link_libraries(${DASGRAIN_TARGET} PRIVATE opengl32)
else()
    find_package(OpenGL QUIET)
    if(OpenGL_FOUND)
        target_link_libraries(${DASGRAIN_TARGET} PRIVATE OpenGL::GL)
    endif()
endif()

if(_DASGRAIN_CUDA_AVAILABLE)
    target_compile_definitions(${DASGRAIN_TARGET} PRIVATE DASGRAIN_HAS_CUDA=1)
endif()

if(_DASGRAIN_OPENCL_AVAILABLE)
    target_compile_definitions(${DASGRAIN_TARGET} PRIVATE DASGRAIN_HAS_OPENCL=1)
    if(APPLE)
        target_link_libraries(${DASGRAIN_TARGET} PRIVATE "-framework OpenCL")
    else()
        target_link_libraries(${DASGRAIN_TARGET} PRIVATE OpenCL::OpenCL)
    endif()
endif()

if(DASGRAIN_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${DASGRAIN_TARGET} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_definitions(${DASGRAIN_TARGET} PRIVATE DASGRAIN_HAS_OPENMP=1)
    else()
        message(STATUS "DasGrain: OpenMP not found; CPU backend will be single-threaded inside kernels (the host still calls multiThread).")
    endif()
endif()

if(MSVC)
    target_compile_definitions(${DASGRAIN_TARGET} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

# ---------------------------------------------------------------------------
# Bundle layout (Info.plist + binary)
# ---------------------------------------------------------------------------
set(PLUGIN_NAME       "DasGrain")
set(PLUGIN_EXE        "DasGrain.ofx")
set(BUNDLE_IDENTIFIER "com.dasgrain.ofx.DasGrain")
set(BUNDLE_SIGNATURE  "????")
set(PLUGIN_VERSION    "${PROJECT_VERSION}")
set(DASGRAIN_RESOURCE_DIR "${DASGRAIN_BUNDLE_DIR}/Contents/Resources")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/packaging/mac/Info.plist.in"
    "${DASGRAIN_BUNDLE_DIR}/Contents/Info.plist"
    @ONLY)

# Place the built binary inside the bundle's arch dir during build, so we
# can also test-load directly from the build tree.
set_target_properties(${DASGRAIN_TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${DASGRAIN_BUNDLE_DIR}/Contents/${DASGRAIN_ARCHDIR}"
    RUNTIME_OUTPUT_DIRECTORY "${DASGRAIN_BUNDLE_DIR}/Contents/${DASGRAIN_ARCHDIR}")

add_custom_command(TARGET ${DASGRAIN_TARGET} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${DASGRAIN_RESOURCE_DIR}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE"
            "${DASGRAIN_RESOURCE_DIR}/LICENSE.txt"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/NOTICE"
            "${DASGRAIN_RESOURCE_DIR}/NOTICE.txt"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openfx/LICENSE.md"
            "${DASGRAIN_RESOURCE_DIR}/OPENFX-LICENSE.txt"
    COMMENT "Copying license and notice files into DasGrain.ofx bundle")

# macOS XProtect / Gatekeeper will kill unsigned universal binaries on first
# launch. Ad-hoc codesign every freshly built binary so Resolve can load it.
if(APPLE)
    add_custom_command(TARGET ${DASGRAIN_TARGET} POST_BUILD
        COMMAND /usr/bin/codesign --force --sign - "$<TARGET_FILE:${DASGRAIN_TARGET}>"
        COMMAND /usr/bin/codesign --force --sign - "${DASGRAIN_BUNDLE_DIR}"
        COMMENT "Ad-hoc codesigning DasGrain.ofx bundle")
endif()

# Install the bundle to the system OFX location.
install(DIRECTORY "${DASGRAIN_BUNDLE_DIR}/"
        DESTINATION "${PLUGIN_INSTALLDIR}/${DASGRAIN_BUNDLE_NAME}"
        USE_SOURCE_PERMISSIONS)

message(STATUS "DasGrain backends: "
    "Metal=${_DASGRAIN_METAL_AVAILABLE} "
    "CUDA=${_DASGRAIN_CUDA_AVAILABLE} "
    "OpenCL=${_DASGRAIN_OPENCL_AVAILABLE} "
    "OpenMP=${OpenMP_CXX_FOUND}")
message(STATUS "DasGrain bundle dir: ${DASGRAIN_BUNDLE_DIR}")
message(STATUS "DasGrain install dir: ${PLUGIN_INSTALLDIR}/${DASGRAIN_BUNDLE_NAME}")
