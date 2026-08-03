#include "YOLODetectorBase.h"
#include "AppConfig.h"
#include "BackendFactory.h"
#include "Timer.h"
#include <fstream>
#include <stdexcept>

namespace detector {

void YOLODetectorBase::Init() {
  const DetectorConfig& cfg = AppConfig::GetInstance().detector;

  confidence_threshold_ = cfg.confidence_threshold;
  nms_threshold_ = cfg.nms_threshold;
  model_input_width_ = cfg.input_width;
  model_input_height_ = cfg.input_height;
  classes_path_ = AppConfig::GetInstance().dataset.coco_labels;

  LoadClasses();

  backend::BackendConfig backend_cfg;
  backend_cfg.type = cfg.backend;
  backend_cfg.model_path = cfg.model_path;
  backend_cfg.device_id = AppConfig::GetInstance().backend.device_id;
  backend_cfg.fp16 = AppConfig::GetInstance().backend.fp16;

  backend_ = backend::BackendFactory::Create(backend_cfg);
  if (!backend_->LoadModel(backend_cfg.model_path)) {
    throw std::runtime_error("YOLO detector init failed: cannot load model " +
                             backend_cfg.model_path);
  }

  if (backend_->GetInputNames().empty() || backend_->GetOutputNames().empty()) {
    throw std::runtime_error(
        "YOLO detector init failed: model has no input/output nodes");
  }
  input_name_ = backend_->GetInputNames().front();
  output_name_ = backend_->GetOutputNames().front();
}

void YOLODetectorBase::LoadClasses() {
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

std::vector<float> YOLODetectorBase::Preprocess(const cv::Mat& letterboxed) {
  // YOLO expects RGB, normalized to [0, 1], NCHW format.
  cv::Mat blob;
  cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                         cv::Size(model_input_width_, model_input_height_),
                         cv::Scalar(0, 0, 0), true, false); // true means swapRB to get RGB
  return std::vector<float>(blob.begin<float>(), blob.end<float>());
}

void YOLODetectorBase::Detect(cv::Mat& frame,
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
      throw std::runtime_error("YOLO inference failed");
    }
  }

  {
    ScopedTimer timer("det_post");
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    ParseOutput(output_data, output_shape, scale_info, boxes, confidences,
                class_ids);

    if (HasBuiltinNms()) {
      results.reserve(boxes.size());
      for (size_t i = 0; i < boxes.size(); ++i) {
        detect_result dr;
        dr.classId = class_ids[i];
        dr.confidence = confidences[i];
        dr.box = cv::Rect_<float>(boxes[i]);
        results.push_back(dr);
      }
    } else {
      results = NmsFilter(boxes, confidences, class_ids, confidence_threshold_,
                          nms_threshold_);
    }
  }
}

void YOLODetectorBase::DrawFrame(cv::Mat& frame,
                                 std::vector<detect_result>& results) {
  DrawResults(frame, results, classes_);
}

}  // namespace detector
