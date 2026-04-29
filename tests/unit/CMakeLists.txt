# Tiny, dependency-free unit-test harness. Keeps Phase-1 startup time low.

add_executable(dasgrain_unit_tests
    test_response_curve.cpp
    test_color_space.cpp
    test_frame_range.cpp
    test_voronoi.cpp
    test_grain_apply.cpp
    test_analyser.cpp
    test_scatter.cpp
    test_golden.cpp
    main.cpp)

target_include_directories(dasgrain_unit_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/src")

target_sources(dasgrain_unit_tests PRIVATE
    "${CMAKE_SOURCE_DIR}/src/core/ResponseCurve.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/ColorSpace.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/FrameRange.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/Voronoi.cpp"
    "${CMAKE_SOURCE_DIR}/src/core/Analyser.cpp"
    "${CMAKE_SOURCE_DIR}/src/kernels/cpu/GrainApplyCPU.cpp"
    "${CMAKE_SOURCE_DIR}/src/kernels/cpu/VoronoiCPU.cpp")

target_compile_features(dasgrain_unit_tests PRIVATE cxx_std_17)

if(APPLE)
    # Same XProtect workaround as the plugin bundle.
    add_custom_command(TARGET dasgrain_unit_tests POST_BUILD
        COMMAND /usr/bin/codesign --force --sign - "$<TARGET_FILE:dasgrain_unit_tests>"
        COMMENT "Ad-hoc codesigning dasgrain_unit_tests")
endif()

add_test(NAME dasgrain_unit_tests COMMAND dasgrain_unit_tests)
