/*!
    @Description : YOLOv26 detector via IBackend (ONNXRuntime CPU/GPU)
*/
#include "YOLOv26Detector.h"
#include "AppConfig.h"
#include "BackendFactory.h"
#include "Timer.h"
#include "detector_utils.h"
#include <fstream>
#include <iostream>

namespace detector {

void YOLOv26Detector::Init() {
  const DetectorConfig& cfg = AppConfig::GetInstance()->detector;

  confidence_threshold_ = cfg.confidence_threshold;
  nms_threshold_ = cfg.nms_threshold;
  model_input_width_ = cfg.input_width;
  model_input_height_ = cfg.input_height;
  classes_path_ = AppConfig::GetInstance()->dataset.coco_labels;

  LoadClasses();

  backend::BackendConfig backend_cfg;
  backend_cfg.type = cfg.backend;
  backend_cfg.model_path = cfg.model_path;
  backend_cfg.device_id = AppConfig::GetInstance()->backend.device_id;

  backend_ = backend::BackendFactory::Create(backend_cfg);
  if (!backend_->LoadModel(backend_cfg.model_path)) {
    throw std::runtime_error("YOLOv26Detector init failed: cannot load model " +
                             backend_cfg.model_path);
  }
}

void YOLOv26Detector::LoadClasses() {
  classes_.clear();
  std::ifstream ifs(classes_path_);
  if (!ifs.is_open()) {
    throw std::runtime_error("File " + classes_path_ + " not found");
  }
  std::string line;
  while (std::getline(ifs, line)) {
    classes_.push_back(line);
  }
}

std::vector<float> YOLOv26Detector::Preprocess(const cv::Mat& letterboxed) {
  // YOLOv26 expects RGB, normalized to [0, 1], NCHW format.
  cv::Mat blob;
  cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                         cv::Size(model_input_width_, model_input_height_),
                         cv::Scalar(0, 0, 0), true, false);

  return std::vector<float>(blob.begin<float>(), blob.end<float>());
}

void YOLOv26Detector::Detect(cv::Mat& frame,
                             std::vector<detect_result>& results) {
  results.clear();

  ScaleInfo scale_info;
  cv::Mat letterboxed;
  std::vector<float> input_tensor_values;
  {
    ScopedTimer timer("det_pre");
    letterboxed =
        Letterbox(frame, model_input_width_, model_input_height_, scale_info);
    input_tensor_values = Preprocess(letterboxed);
  }

  std::vector<int64_t> input_shape = {1, 3, model_input_height_,
                                      model_input_width_};
  std::vector<float> output_data;
  std::vector<int64_t> output_shape;
  {
    ScopedTimer timer("det_infer");
    if (!backend_->Run("images", input_tensor_values, input_shape, "output0",
                       output_data, output_shape)) {
      throw std::runtime_error("YOLOv26 inference failed");
    }
  }

  // YOLOv26 default (end2end) ONNX output shape: (1, 300, 6)
  // Each row: [x1, y1, x2, y2, confidence, class_id]
  // NMS is already applied inside the model.
  if (output_shape.size() != 3 || output_shape[2] != 6) {
    throw std::runtime_error(
        "Unexpected YOLOv26 output shape, expected (1, 300, 6). "
        "If you exported with end2end=False, use YOLOv8-style "
        "post-processing.");
  }

  {
    ScopedTimer timer("det_post");
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

      detect_result dr;
      dr.classId = class_id;
      dr.confidence = conf;
      dr.box = cv::Rect_<float>(x, y, w, h);
      results.push_back(dr);
    }
  }
}

void YOLOv26Detector::DrawFrame(cv::Mat& frame,
                                std::vector<detect_result>& results) {
  DrawResults(frame, results, classes_);
}

}  // namespace detector
