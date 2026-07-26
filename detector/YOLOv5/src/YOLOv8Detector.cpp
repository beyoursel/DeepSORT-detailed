/*!
    @Description : YOLOv8 detector via ONNXRuntime
*/
#include <YOLOv8Detector.h>
#include <detector_utils.h>
#include <AppConfig.h>
#include <fstream>
#include <iostream>

namespace detector {

bool YOLOv8Detector::init()
{
    const DetectorConfig& cfg = AppConfig::getInstance()->detector;

    confidence_threshold_ = cfg.confidence_threshold;
    nms_threshold_ = cfg.nms_threshold;
    model_input_width_ = cfg.input_width;
    model_input_height_ = cfg.input_height;
    model_path_ = cfg.model_path;
    classes_path_ = AppConfig::getInstance()->dataset.coco_labels;

    load_classes();

    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8Detector");
    session_options_ = Ort::SessionOptions();
    session_options_.SetIntraOpNumThreads(1);

    session_ = Ort::Session(env_, model_path_.c_str(), session_options_);

    return true;
}

void YOLOv8Detector::load_classes()
{
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

std::vector<float> YOLOv8Detector::preprocess(const cv::Mat& letterboxed)
{
    // YOLOv8 expects RGB, normalized to [0, 1], NCHW format.
    cv::Mat blob;
    cv::dnn::blobFromImage(letterboxed, blob, 1.0 / 255.0,
                           cv::Size(model_input_width_, model_input_height_),
                           cv::Scalar(0, 0, 0), true, false);

    return std::vector<float>(blob.begin<float>(), blob.end<float>());
}

void YOLOv8Detector::detect(cv::Mat& frame, std::vector<detect_result>& results)
{
    results.clear();

    ScaleInfo scale_info;
    cv::Mat letterboxed = letterbox(frame, model_input_width_, model_input_height_, scale_info);

    std::vector<float> input_tensor_values = preprocess(letterboxed);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::vector<int64_t> input_shape = {1, 3, model_input_height_, model_input_width_};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    const char* input_names[] = {"images"};
    const char* output_names[] = {"output0"};

    Ort::RunOptions run_options;
    std::vector<Ort::Value> output_tensors = session_.Run(
        run_options, input_names, &input_tensor, 1, output_names, 1);

    Ort::Value& output_tensor = output_tensors.front();
    Ort::TensorTypeAndShapeInfo shape_info = output_tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = shape_info.GetShape();

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

    const float* raw_output = output_tensor.GetTensorData<float>();

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;

    for (int64_t i = 0; i < num_anchors; ++i) {
        float cx = raw_output[i];
        float cy = raw_output[num_anchors + i];
        float ow = raw_output[2 * num_anchors + i];
        float oh = raw_output[3 * num_anchors + i];

        float best_score = 0.0f;
        int best_class = 0;
        for (int c = 0; c < num_classes; ++c) {
            float score = raw_output[(4 + c) * num_anchors + i];
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

    results = nms_filter(boxes, confidences, classIds,
                         confidence_threshold_, nms_threshold_);
}

void YOLOv8Detector::draw_frame(cv::Mat& frame, std::vector<detect_result>& results)
{
    draw_results(frame, results, classes_);
}

} // namespace detector
