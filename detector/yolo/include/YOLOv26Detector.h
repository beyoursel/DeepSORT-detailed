#pragma once

#include "YOLODetectorBase.h"

namespace detector {

class YOLOv26Detector : public YOLODetectorBase {
 protected:
  void ParseOutput(const std::vector<float>& output_data,
                   const std::vector<int64_t>& output_shape,
                   const ScaleInfo& scale_info,
                   std::vector<cv::Rect>& boxes,
                   std::vector<float>& confidences,
                   std::vector<int>& class_ids) override;
  bool HasBuiltinNms() const override { return true; }
};

}  // namespace detector
