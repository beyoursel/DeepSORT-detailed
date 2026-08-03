#pragma once

#include "YOLODetectorBase.h"

namespace detector {

class YOLOv5Detector : public YOLODetectorBase {
 protected:
  void ParseOutput(const std::vector<float>& output_data,
                   const std::vector<int64_t>& output_shape,
                   const ScaleInfo& scale_info,
                   std::vector<cv::Rect>& boxes,
                   std::vector<float>& confidences,
                   std::vector<int>& class_ids) override;

  // YOLOv5's ONNX exposes the decoded (1, 25200, 85) tensor as "output"
  // alongside the three raw heads; make sure we parse that one regardless
  // of the backend's binding order.
  std::string SelectOutputName(
      const std::vector<std::string>& names) const override {
    for (const std::string& name : names) {
      if (name == "output") return name;
    }
    return names.front();
  }
};

}  // namespace detector
