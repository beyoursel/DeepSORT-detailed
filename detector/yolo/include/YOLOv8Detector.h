#pragma once

#include "IBackend.h"
#include "IDetector.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace detector {

class YOLOv8Detector : public IDetector {
 public:
  YOLOv8Detector() = default;
  ~YOLOv8Detector() override = default;

  bool init() override;
  void detect(cv::Mat& frame, std::vector<detect_result>& results) override;
  void draw_frame(cv::Mat& frame, std::vector<detect_result>& results) override;
  size_t num_classes() const override { return classes_.size(); }

 private:
  void load_classes();
  std::vector<float> preprocess(const cv::Mat& letterboxed);

  std::shared_ptr<backend::IBackend> backend_;
  std::vector<std::string> classes_;

  float confidence_threshold_ = 0.25f;
  float nms_threshold_ = 0.4f;
  int model_input_width_ = 640;
  int model_input_height_ = 640;
  std::string classes_path_;
};

}  // namespace detector
