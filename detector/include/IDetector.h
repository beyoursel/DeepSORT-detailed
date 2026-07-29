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

  virtual bool init() = 0;
  virtual void detect(cv::Mat& frame, std::vector<detect_result>& results) = 0;
  virtual void draw_frame(cv::Mat& frame,
                          std::vector<detect_result>& results) = 0;
  virtual size_t num_classes() const = 0;
};

}  // namespace detector
