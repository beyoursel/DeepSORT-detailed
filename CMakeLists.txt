cmake_minimum_required(VERSION 3.5)

project(yolo_tracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release (-O3) when no build type is given; an empty build type
# means no optimization, which severely slows down Eigen and the trackers.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")


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
    ${CMAKE_SOURCE_DIR}/tracker/include
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

# TensorRT backend (optional): -DUSE_TENSORRT=ON
option(USE_TENSORRT "Enable TensorRT backend" OFF)
set(TENSORRT_DIR "/media/taole/mydisk/TensorRT-8.5.1.7" CACHE PATH "TensorRT install dir")
set(TENSORRT_LIBS "")
if(USE_TENSORRT)
    message(STATUS "Using TensorRT: ${TENSORRT_DIR}")
    # CMake 3.16 has no FindCUDAToolkit (needs 3.17+); locate cudart manually.
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8" CACHE PATH "CUDA toolkit dir")
    find_library(CUDART_LIBRARY cudart
        PATHS "${CUDA_TOOLKIT_ROOT_DIR}/lib64" "${CUDA_TOOLKIT_ROOT_DIR}/lib"
        REQUIRED)
    include_directories("${CUDA_TOOLKIT_ROOT_DIR}/include")
    # SYSTEM: TRT 8.5 headers use deprecated APIs internally; treat them as
    # system headers so their warnings don't drown out ours.
    include_directories(SYSTEM "${TENSORRT_DIR}/include")
    link_directories("${TENSORRT_DIR}/lib")
    list(APPEND BACKEND_SOURCES backend/src/TensorRTBackend.cpp)
    # TRT 8.5 deprecates the binding-based API (enqueueV3 tensor API is the
    # replacement); silence it only for this file since we pin to 8.5.
    set_source_files_properties(backend/src/TensorRTBackend.cpp
        PROPERTIES COMPILE_OPTIONS "-Wno-deprecated-declarations")
    add_compile_definitions(USE_TENSORRT)
    set(TENSORRT_LIBS nvinfer nvonnxparser ${CUDART_LIBRARY})
endif()

add_executable(YoloTracker
    config/AppConfig.cpp

    ${BACKEND_SOURCES}

    detector/src/detector_utils.cpp
    detector/src/DetectorFactory.cpp
    detector/yolo/src/YOLODetectorBase.cpp
    detector/yolo/src/YOLOv5Detector.cpp
    detector/yolo/src/YOLOv8Detector.cpp
    detector/yolo/src/YOLOv26Detector.cpp

    tracker/src/TrackerFactory.cpp

    tracker/deepsort/src/DeepSORTAdapter.cpp
    tracker/deepsort/src/FeatureTensor.cpp
    tracker/deepsort/src/model.cpp
    tracker/deepsort/src/kalmanfilter.cpp
    tracker/deepsort/src/linear_assignment.cpp
    tracker/deepsort/src/nn_matching.cpp
    tracker/deepsort/src/track.cpp
    tracker/deepsort/src/tracker.cpp
    tracker/deepsort/src/munkres.cpp
    tracker/deepsort/src/hungarianoper.cpp

    tracker/bytetrack/src/ByteTrackAdapter.cpp
    tracker/bytetrack/src/BytekalmanFilter.cpp
    tracker/bytetrack/src/BYTETracker.cpp
    tracker/bytetrack/src/lapjv.cpp
    tracker/bytetrack/src/STrack.cpp
    tracker/bytetrack/src/utils.cpp

    main.cpp)

target_link_libraries(YoloTracker PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp
    ${TENSORRT_LIBS})

# Set RPATH so the executable finds libonnxruntime.so and its provider libraries at runtime.
# Use $ORIGIN relative paths so the project is relocatable.
if(USE_TENSORRT)
    set(EXTRA_RPATH ";${TENSORRT_DIR}/lib")
else()
    set(EXTRA_RPATH "")
endif()
set_target_properties(YoloTracker PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_RPATH}${EXTRA_RPATH}"
    INSTALL_RPATH "${ONNXRUNTIME_RPATH}${EXTRA_RPATH}"
)


add_executable(test_detector
    config/AppConfig.cpp

    ${BACKEND_SOURCES}

    detector/src/detector_utils.cpp
    detector/src/DetectorFactory.cpp
    detector/yolo/src/YOLODetectorBase.cpp
    detector/yolo/src/YOLOv5Detector.cpp
    detector/yolo/src/YOLOv8Detector.cpp
    detector/yolo/src/YOLOv26Detector.cpp

    test_detector.cpp)

target_link_libraries(test_detector PRIVATE
    onnxruntime
    ${OpenCV_LIBS}
    Eigen3::Eigen
    yaml-cpp
    ${TENSORRT_LIBS})

set_target_properties(test_detector PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_RPATH}${EXTRA_RPATH}"
    INSTALL_RPATH "${ONNXRUNTIME_RPATH}${EXTRA_RPATH}"
)
