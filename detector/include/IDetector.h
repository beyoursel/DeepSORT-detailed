#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

// Keep detect_result in global namespace for compatibility with existing
// tracker code.
struct detect_result {
  int classId;
  float confidence;
  cv::Rect_<float> box;
};

namespace detector {

class IDetector {
 public:
  virtual ~IDetector() = default;

  // Throws std::runtime_error on initialization failure (fail-fast).
  virtual void Init() = 0;
  virtual void Detect(cv::Mat& frame, std::vector<detect_result>& results) = 0;
  virtual void DrawFrame(cv::Mat& frame,
                         std::vector<detect_result>& results) = 0;
  virtual size_t NumClasses() const = 0;
};

}  // namespace detector
