#include "DetectorFactory.h"
#include "YOLOv26Detector.h"
#include "YOLOv5Detector.h"
#include "YOLOv8Detector.h"
#include <stdexcept>

namespace detector {

std::shared_ptr<IDetector> DetectorFactory::Create(const std::string& type) {
  if (type == "yolov5") {
    return std::make_shared<YOLOv5Detector>();
  }
  if (type == "yolov8" || type == "yolov9" || type == "yolov10" ||
      type == "yolov11") {
    return std::make_shared<YOLOv8Detector>();
  }
  if (type == "yolov26") {
    return std::make_shared<YOLOv26Detector>();
  }
  throw std::runtime_error("Unsupported detector type: " + type);
}

}  // namespace detector
