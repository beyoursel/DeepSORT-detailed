/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-23 02:52:41
*/
#include "YOLOv5Detector.h"
#include <stdexcept>

namespace detector {

void YOLOv5Detector::ParseOutput(const std::vector<float>& output_data,
                                 const std::vector<int64_t>& output_shape,
                                 const ScaleInfo& scale_info,
                                 std::vector<cv::Rect>& boxes,
                                 std::vector<float>& confidences,
                                 std::vector<int>& class_ids) {
  // YOLOv5 ONNX output shape: (1, N, 5 + num_classes)
  if (output_shape.size() != 3 || output_shape[2] <= 5) {
    throw std::runtime_error("Unexpected YOLOv5 output shape");
  }

  int rows = static_cast<int>(output_shape[1]);
  int cols = static_cast<int>(output_shape[2]);
  int num_classes = cols - 5;

  if (num_classes != static_cast<int>(classes_.size())) {
    throw std::runtime_error(
        "YOLOv5 model output classes (" + std::to_string(num_classes) +
        ") does not match label file classes (" +
        std::to_string(classes_.size()) +
        "); check detector.type/model_path/coco_labels in the config");
  }

  for (int i = 0; i < rows; ++i) {
    const float* row = output_data.data() + i * cols;
    float box_conf = row[4];
    if (box_conf < confidence_threshold_) {
      continue;
    }

    int best_class = 0;
    float best_score = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
      if (row[5 + c] > best_score) {
        best_score = row[5 + c];
        best_class = c;
      }
    }

    if (best_score > confidence_threshold_) {
      float cx = row[0];
      float cy = row[1];
      float ow = row[2];
      float oh = row[3];

      float x = (cx - scale_info.pad_x - 0.5f * ow) * scale_info.x_factor;
      float y = (cy - scale_info.pad_y - 0.5f * oh) * scale_info.y_factor;
      float width = ow * scale_info.x_factor;
      float height = oh * scale_info.y_factor;

      boxes.emplace_back(static_cast<int>(x), static_cast<int>(y),
                         static_cast<int>(width), static_cast<int>(height));
      class_ids.push_back(best_class);
      confidences.push_back(best_score * box_conf);
    }
  }
}

}  // namespace detector
