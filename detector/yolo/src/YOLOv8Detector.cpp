/*!
    @Description : YOLOv8 detector via IBackend (ONNXRuntime CPU/GPU)
*/
#include "YOLOv8Detector.h"
#include "AppConfig.h"
#include "BackendFactory.h"
#include "Timer.h"
#include "detector_utils.h"
#include <fstream>
#include <iostream>

namespace detector {

void YOLOv8Detector::Init() {
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
    throw std::runtime_error("YOLOv8Detector init failed: cannot load model " +
                             backend_cfg.model_path);
  }

  if (backend_->GetInputNames().empty() || backend_->GetOutputNames().empty()) {
    throw std::runtime_error(
        "YOLOv8Detector init failed: model has no input/output nodes");
  }
  input_name_ = backend_->GetInputNames().front();
  output_name_ = backend_->GetOutputNames().front();
}

void YOLOv8Detector::LoadClasses() {
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

std::vector<float> YOLOv8Detector::Preprocess(const cv::Mat& letterboxed) {
  // YOLOv8 expects RGB, normalized to [0, 1], NCHW format.
  cv::Mat blob;
  cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                         cv::Size(model_input_width_, model_input_height_),
                         cv::Scalar(0, 0, 0), true, false);

  return std::vector<float>(blob.begin<float>(), blob.end<float>());
}

void YOLOv8Detector::Detect(cv::Mat& frame,
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
    if (!backend_->Run(input_name_, input_tensor_values, input_shape,
                       output_name_, output_data, output_shape)) {
      throw std::runtime_error("YOLOv8 inference failed");
    }
  }

  // YOLOv8 output shape: (1, 4 + num_classes, num_anchors)
  if (output_shape.size() != 3 || output_shape[1] <= 4) {
    throw std::runtime_error("Unexpected YOLOv8 output shape");
  }

  int64_t num_anchors = output_shape[2];
  int num_classes = static_cast<int>(output_shape[1]) - 4;

  if (num_classes != static_cast<int>(classes_.size())) {
    std::cerr << "Warning: model output classes (" << num_classes
              << ") does not match label file classes (" << classes_.size()
              << ")" << std::endl;
  }

  std::vector<cv::Rect> boxes;
  std::vector<int> classIds;
  std::vector<float> confidences;

  {
    ScopedTimer timer("det_post");
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
        classIds.push_back(best_class);
        confidences.push_back(best_score);
      }
    }

    results = NmsFilter(boxes, confidences, classIds, confidence_threshold_,
                        nms_threshold_);
  }
}

void YOLOv8Detector::DrawFrame(cv::Mat& frame,
                               std::vector<detect_result>& results) {
  DrawResults(frame, results, classes_);
}

}  // namespace detector
