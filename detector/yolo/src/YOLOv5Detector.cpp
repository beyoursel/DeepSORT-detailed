/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-23 02:52:41
*/
#include "YOLOv5Detector.h"
#include "AppConfig.h"
#include "BackendFactory.h"
#include "Timer.h"
#include "detector_utils.h"
#include <iostream>

namespace detector {

void YOLOv5Detector::Init() {
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
    throw std::runtime_error("YOLOv5Detector init failed: cannot load model " +
                             backend_cfg.model_path);
  }

  if (backend_->GetInputNames().empty() || backend_->GetOutputNames().empty()) {
    throw std::runtime_error(
        "YOLOv5Detector init failed: model has no input/output nodes");
  }
  input_name_ = backend_->GetInputNames().front();
  output_name_ = backend_->GetOutputNames().front();
}

void YOLOv5Detector::LoadClasses() {
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

std::vector<float> YOLOv5Detector::Preprocess(const cv::Mat& letterboxed) {
  cv::Mat blob;
  cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                         cv::Size(model_input_width_, model_input_height_),
                         cv::Scalar(0, 0, 0), true, false);
  return std::vector<float>(blob.begin<float>(), blob.end<float>());
}

void YOLOv5Detector::Detect(cv::Mat& frame,
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
      throw std::runtime_error("YOLOv5 inference failed");
    }
  }

  // YOLOv5 ONNX output shape: (1, N, 5 + num_classes)
  if (output_shape.size() != 3 || output_shape[2] <= 5) {
    throw std::runtime_error("Unexpected YOLOv5 output shape");
  }

  int rows = static_cast<int>(output_shape[1]);
  int cols = static_cast<int>(output_shape[2]);
  int num_classes = cols - 5;

  if (num_classes != static_cast<int>(classes_.size())) {
    std::cerr << "Warning: model output classes (" << num_classes
              << ") does not match label file classes (" << classes_.size()
              << ")" << std::endl;
  }

  cv::Mat det_output(rows, cols, CV_32F, output_data.data());

  std::vector<cv::Rect> boxes;
  std::vector<int> classIds;
  std::vector<float> confidences;

  {
    ScopedTimer timer("det_post");
    for (int i = 0; i < det_output.rows; i++) {
      float box_conf = det_output.at<float>(i, 4);
      if (box_conf < confidence_threshold_) {
        continue;
      }

      cv::Mat classes_confidences =
          det_output.row(i).colRange(5, det_output.cols);
      cv::Point classIdPoint;
      double cls_conf;
      cv::minMaxLoc(classes_confidences, 0, &cls_conf, 0, &classIdPoint);

      if (cls_conf > confidence_threshold_) {
        float cx = det_output.at<float>(i, 0);
        float cy = det_output.at<float>(i, 1);
        float ow = det_output.at<float>(i, 2);
        float oh = det_output.at<float>(i, 3);

        float x = (cx - scale_info.pad_x - 0.5f * ow) * scale_info.x_factor;
        float y = (cy - scale_info.pad_y - 0.5f * oh) * scale_info.y_factor;
        float width = ow * scale_info.x_factor;
        float height = oh * scale_info.y_factor;

        boxes.emplace_back(static_cast<int>(x), static_cast<int>(y),
                           static_cast<int>(width), static_cast<int>(height));
        classIds.push_back(classIdPoint.x);
        confidences.push_back(static_cast<float>(cls_conf * box_conf));
      }
    }

    results = NmsFilter(boxes, confidences, classIds, confidence_threshold_,
                        nms_threshold_);
  }
}

void YOLOv5Detector::DrawFrame(cv::Mat& frame,
                               std::vector<detect_result>& results) {
  DrawResults(frame, results, classes_);
}

}  // namespace detector
