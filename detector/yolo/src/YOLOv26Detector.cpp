/*!
    @Description : YOLOv26 detector via IBackend (ONNXRuntime CPU/GPU)
*/
#include "YOLOv26Detector.h"
#include <stdexcept>

namespace detector {

void YOLOv26Detector::ParseOutput(const std::vector<float>& output_data,
                                  const std::vector<int64_t>& output_shape,
                                  const ScaleInfo& scale_info,
                                  std::vector<cv::Rect>& boxes,
                                  std::vector<float>& confidences,
                                  std::vector<int>& class_ids) {
  // YOLOv26 default (end2end) ONNX output shape: (1, 300, 6)
  // Each row: [x1, y1, x2, y2, confidence, class_id]
  // NMS is already applied inside the model (see HasBuiltinNms()).
  if (output_shape.size() != 3 || output_shape[2] != 6) {
    throw std::runtime_error(
        "Unexpected YOLOv26 output shape, expected (1, 300, 6). "
        "If you exported with end2end=False, use YOLOv8-style "
        "post-processing.");
  }

  int64_t num_dets = output_shape[1];  // typically 300

  for (int64_t i = 0; i < num_dets; ++i) {
    float x1 = output_data[i * 6 + 0];
    float y1 = output_data[i * 6 + 1];
    float x2 = output_data[i * 6 + 2];
    float y2 = output_data[i * 6 + 3];
    float conf = output_data[i * 6 + 4];
    int class_id = static_cast<int>(output_data[i * 6 + 5]);

    if (conf < confidence_threshold_) {
      continue;
    }

    // Map coordinates from letterboxed image back to original frame.
    float orig_x1 = (x1 - scale_info.pad_x) * scale_info.x_factor;
    float orig_y1 = (y1 - scale_info.pad_y) * scale_info.y_factor;
    float orig_x2 = (x2 - scale_info.pad_x) * scale_info.x_factor;
    float orig_y2 = (y2 - scale_info.pad_y) * scale_info.y_factor;

    int x = static_cast<int>(orig_x1);
    int y = static_cast<int>(orig_y1);
    int w = static_cast<int>(orig_x2 - orig_x1);
    int h = static_cast<int>(orig_y2 - orig_y1);

    if (w <= 0 || h <= 0) {
      continue;
    }

    boxes.emplace_back(x, y, w, h);
    class_ids.push_back(class_id);
    confidences.push_back(conf);
  }
}

}  // namespace detector
