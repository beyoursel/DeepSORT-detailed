cmake_minimum_required(VERSION 3.5)

project(DeepSORT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(yaml-cpp REQUIRED)


# ONNXRuntime backend selection:
# Use GPU version by default. GPU library also contains CPU EP, so it works for both.
option(USE_ONNXRUNTIME_GPU "Use ONNXRuntime GPU library" ON)

if(USE_ONNXRUNTIME_GPU)
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-x64-gpu-1.16.3")
    message(STATUS "Using ONNXRuntime GPU: ${ONNXRUNTIME_DIR}")
else()
    set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/lib/onnxruntime-linux-x64-1.12.1")
    message(STATUS "Using ONNXRuntime CPU: ${ONNXRUNTIME_DIR}")
endif()

include_directories("${ONNXRUNTIME_DIR}/include")
link_directories("${ONNXRUNTIME_DIR}/lib")
find_package(OpenCV 4 REQUIRED)

get_filename_component(ONNXRUNTIME_DIR_NAME ${ONNXRUNTIME_DIR} NAME)
set(ONNXRUNTIME_RPATH "$ORIGIN/../lib/${ONNXRUNTIME_DIR_NAME}/lib")


include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/backend/include
    ${CMAKE_SOURCE_DIR}/tracker/deepsort/include
    ${CMAKE_SOURCE_DIR}/tracker/bytetrack/include
    ${CMAKE_SOURCE_DIR}/detector/include
    ${CMAKE_SOURCE_DIR}/detector/yolo/include
    ${CMAKE_SOURCE_DIR}/config

    )


set(BACKEND_SOURCES
    backend/src/BackendFactory.cpp
    backend/src/ONNXRuntimeBackend.cpp
)

add_executable(DeepSORT
    config/AppConfig.cpp

    ${BACKEND_SOURCES}

    detector/src/detector_utils.cpp
    detector/src/DetectorFactory.cpp
    detector/yolo/src/YOLOv5Detector.cpp
    detector/yolo/src/YOLOv8Detector.cpp
    detector/yolo/src/YOLOv26Detector.cpp

    tracker/deepsort/src/FeatureTensor.cpp
    tracker/deepsort/src/model.cpp
    tracker/deepsort/src/kalmanfilter.cpp
    tracker/deepsort/src/linear_assignment.cpp
    tracker/deepsort/src/nn_matching.cpp
    tracker/deepsort/src/track.cpp
    tracker/deepsort/src/tracker.cpp
    tracker/deepsort/src/munkres.cpp
    tracker/deepsort/src/hungarianoper.cpp

    tracker/bytetrack/src/BytekalmanFilter.cpp
    tracker/bytetrack/src/BYTETracker.cpp
    tracker/bytetrack/src/lapjv.cpp
    tracker/bytetrack/src/STrack.cpp
    tracker/bytetrack/src/utils.cpp

    main.cpp)

target_link_libraries(DeepSORT PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp)

# Set RPATH so the executable finds libonnxruntime.so and its provider libraries at runtime.
# Use $ORIGIN relative paths so the project is relocatable.
set_target_properties(DeepSORT PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_RPATH}"
    INSTALL_RPATH "${ONNXRUNTIME_RPATH}"
)


add_executable(test_detector
    config/AppConfig.cpp

    ${BACKEND_SOURCES}

    detector/src/detector_utils.cpp
    detector/src/DetectorFactory.cpp
    detector/yolo/src/YOLOv5Detector.cpp
    detector/yolo/src/YOLOv8Detector.cpp
    detector/yolo/src/YOLOv26Detector.cpp

    test_detector.cpp)

target_link_libraries(test_detector PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp)

set_target_properties(test_detector PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_RPATH}"
    INSTALL_RPATH "${ONNXRUNTIME_RPATH}"
)
