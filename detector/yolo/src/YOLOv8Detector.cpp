/*!
    @Description : YOLOv8 detector via IBackend (ONNXRuntime CPU/GPU)
*/
#include "YOLOv8Detector.h"
#include <stdexcept>

namespace detector {

void YOLOv8Detector::ParseOutput(const std::vector<float>& output_data,
                                 const std::vector<int64_t>& output_shape,
                                 const ScaleInfo& scale_info,
                                 std::vector<cv::Rect>& boxes,
                                 std::vector<float>& confidences,
                                 std::vector<int>& class_ids) {
  // YOLOv8 output shape: (1, 4 + num_classes, num_anchors)
  if (output_shape.size() != 3 || output_shape[1] <= 4) {
    throw std::runtime_error("Unexpected YOLOv8 output shape");
  }

  int64_t num_anchors = output_shape[2];
  int num_classes = static_cast<int>(output_shape[1]) - 4;

  if (num_classes != static_cast<int>(classes_.size())) {
    throw std::runtime_error(
        "YOLOv8 model output classes (" + std::to_string(num_classes) +
        ") does not match label file classes (" +
        std::to_string(classes_.size()) +
        "); check detector.type/model_path/coco_labels in the config");
  }

  for (int64_t i = 0; i < num_anchors; ++i) {
    float cx = output_data[i];
    float cy = output_data[num_anchors + i];
    float ow = output_data[2 * num_anchors + i];
    float oh = output_data[3 * num_anchors + i];

    float best_score = 0.0f;
    int best_class = 0;
    for (int c = 0; c < num_classes; ++c) {
      float score = output_data[(4 + c) * num_anchors + i];
      if (score > best_score) {
        best_score = score;
        best_class = c;
      }
    }

    if (best_score > confidence_threshold_) {
      float x = (cx - scale_info.pad_x - 0.5f * ow) * scale_info.x_factor;
      float y = (cy - scale_info.pad_y - 0.5f * oh) * scale_info.y_factor;
      float width = ow * scale_info.x_factor;
      float height = oh * scale_info.y_factor;

      boxes.emplace_back(static_cast<int>(x), static_cast<int>(y),
                         static_cast<int>(width), static_cast<int>(height));
      class_ids.push_back(best_class);
      confidences.push_back(best_score);
    }
  }
}

}  // namespace detector
