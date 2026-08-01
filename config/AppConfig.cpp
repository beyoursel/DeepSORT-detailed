#include "AppConfig.h"
#include <iostream>
#include <stdexcept>

AppConfig& AppConfig::GetInstance() {
  static AppConfig instance;  // thread-safe since C++11
  return instance;
}

static std::string SafeGetString(const YAML::Node& node,
                                 const std::string& key) {
  if (!node[key]) {
    throw std::runtime_error("Missing required config key: " + key);
  }
  return node[key].as<std::string>();
}

template <typename T>
static T SafeGet(const YAML::Node& node, const std::string& key) {
  if (!node[key]) {
    throw std::runtime_error("Missing required config key: " + key);
  }
  return node[key].as<T>();
}

template <typename T>
static T GetWithDefault(const YAML::Node& node, const std::string& key,
                        const T& default_val) {
  if (!node[key]) {
    return default_val;
  }
  return node[key].as<T>();
}

bool AppConfig::Load(const std::string& yaml_path) {
  try {
    YAML::Node config = YAML::LoadFile(yaml_path);

    // dataset
    if (config["dataset"]) {
      dataset.coco_labels = SafeGetString(config["dataset"], "coco_labels");
    }

    // input
    if (config["input"]) {
      input.source = SafeGetString(config["input"], "source");
      input.type =
          GetWithDefault(config["input"], "type", std::string("video"));
    }

    // output
    if (config["output"]) {
      output.video = GetWithDefault(config["output"], "video", std::string(""));
      output.image = SafeGetString(config["output"], "image");
      output.fourcc =
          GetWithDefault(config["output"], "fourcc", std::string("MJPG"));
      output.fps = GetWithDefault(config["output"], "fps", 10);
      output.width = GetWithDefault(config["output"], "width", 1920);
      output.height = GetWithDefault(config["output"], "height", 1080);
      output.show = GetWithDefault(config["output"], "show", true);
      output.result =
          GetWithDefault(config["output"], "result", std::string(""));
    }

    // global backend
    if (config["backend"]) {
      backend.type = GetWithDefault(config["backend"], "type",
                                    std::string("onnxruntime_cpu"));
      backend.device_id = GetWithDefault(config["backend"], "device_id", 0);
    }

    // detector
    if (config["detector"]) {
      detector.type =
          GetWithDefault(config["detector"], "type", std::string("yolov5"));
      detector.backend =
          GetWithDefault(config["detector"], "backend", backend.type);
      detector.model_path = SafeGetString(config["detector"], "model_path");
      detector.input_width =
          GetWithDefault(config["detector"], "input_width", 640);
      detector.input_height =
          GetWithDefault(config["detector"], "input_height", 640);
      detector.confidence_threshold =
          GetWithDefault(config["detector"], "confidence_threshold", 0.25f);
      detector.nms_threshold =
          GetWithDefault(config["detector"], "nms_threshold", 0.4f);
    }

    // tracker
    if (config["tracker"]) {
      tracker.type =
          GetWithDefault(config["tracker"], "type", std::string("bytetrack"));
    }

    // deepsort
    if (config["deepsort"]) {
      deepsort.feature_model_path =
          SafeGetString(config["deepsort"], "feature_model_path");
      deepsort.backend =
          GetWithDefault(config["deepsort"], "backend", backend.type);
      deepsort.feature_dim =
          GetWithDefault(config["deepsort"], "feature_dim", 512);
      deepsort.max_cosine_distance =
          GetWithDefault(config["deepsort"], "max_cosine_distance", 0.2f);
      deepsort.nn_budget = GetWithDefault(config["deepsort"], "nn_budget", 100);
      deepsort.max_iou_distance =
          GetWithDefault(config["deepsort"], "max_iou_distance", 0.7f);
      deepsort.max_age = GetWithDefault(config["deepsort"], "max_age", 30);
      deepsort.n_init = GetWithDefault(config["deepsort"], "n_init", 3);
    }

    // bytetrack
    if (config["bytetrack"]) {
      bytetrack.track_buffer =
          GetWithDefault(config["bytetrack"], "track_buffer", 30);
      bytetrack.track_thresh =
          GetWithDefault(config["bytetrack"], "track_thresh", 0.5f);
      bytetrack.high_thresh =
          GetWithDefault(config["bytetrack"], "high_thresh", 0.6f);
      bytetrack.match_thresh =
          GetWithDefault(config["bytetrack"], "match_thresh", 0.8f);
    }

    std::cout << "Config loaded from: " << yaml_path << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load config: " << e.what() << std::endl;
    return false;
  }
}
