#pragma once

#include "IBackend.h"
#include "IDetector.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace detector {

class YOLOv26Detector : public IDetector {
 public:
  YOLOv26Detector() = default;
  ~YOLOv26Detector() override = default;

  void Init() override;
  void Detect(cv::Mat& frame, std::vector<detect_result>& results) override;
  void DrawFrame(cv::Mat& frame, std::vector<detect_result>& results) override;
  size_t NumClasses() const override { return classes_.size(); }

 private:
  void LoadClasses();
  std::vector<float> Preprocess(const cv::Mat& letterboxed);

  std::shared_ptr<backend::IBackend> backend_;
  std::vector<std::string> classes_;

  float confidence_threshold_ = 0.25f;
  float nms_threshold_ = 0.4f;
  int model_input_width_ = 640;
  int model_input_height_ = 640;
  std::string classes_path_;
};

}  // namespace detector
