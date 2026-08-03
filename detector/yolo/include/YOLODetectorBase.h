#pragma once

#include "IBackend.h"
#include "IDetector.h"
#include "detector_utils.h"
#include <memory>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace detector {

// Base class for YOLO-family detectors sharing the
// letterbox -> preprocess -> inference -> post-process pipeline.
// Subclasses only implement ParseOutput() for their specific ONNX output
// layout.
class YOLODetectorBase : public IDetector {
 public:
  void Init() override;
  void Detect(cv::Mat& frame, std::vector<detect_result>& results) override;
  void DrawFrame(cv::Mat& frame, std::vector<detect_result>& results) override;
  size_t NumClasses() const override { return classes_.size(); }

 protected:
  // Parse raw model output into candidate boxes (original-frame coordinates),
  // confidences and class ids. Called inside the "det_post" timing scope.
  virtual void ParseOutput(const std::vector<float>& output_data,
                           const std::vector<int64_t>& output_shape,
                           const ScaleInfo& scale_info,
                           std::vector<cv::Rect>& boxes,
                           std::vector<float>& confidences,
                           std::vector<int>& class_ids) = 0;

  // True when the model already applies NMS internally (e.g. YOLOv26 end2end
  // export); the base class then skips the extra NmsFilter pass.
  virtual bool HasBuiltinNms() const { return false; }

  // Pick which model output to parse when the backend reports several
  // (e.g. YOLOv5's ONNX also exposes the three raw detection heads, and
  // TensorRT binding order does not follow the ONNX output order).
  virtual std::string SelectOutputName(
      const std::vector<std::string>& names) const {
    return names.front();
  }

  std::shared_ptr<backend::IBackend> backend_;
  std::vector<std::string> classes_;

  float confidence_threshold_ = 0.25f;
  float nms_threshold_ = 0.4f;
  int model_input_width_ = 640;
  int model_input_height_ = 640;

 private:
  void LoadClasses();
  std::vector<float> Preprocess(const cv::Mat& letterboxed);

  std::string classes_path_;
  std::string input_name_;   // queried from the model at Init
  std::string output_name_;  // queried from the model at Init
};

}  // namespace detector
