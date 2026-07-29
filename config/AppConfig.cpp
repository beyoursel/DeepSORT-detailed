#include "AppConfig.h"
#include <iostream>
#include <stdexcept>

AppConfig* AppConfig::instance = nullptr;

AppConfig* AppConfig::getInstance() {
  if (instance == nullptr) {
    instance = new AppConfig();
  }
  return instance;
}

static std::string safeGetString(const YAML::Node& node,
                                 const std::string& key) {
  if (!node[key]) {
    throw std::runtime_error("Missing required config key: " + key);
  }
  return node[key].as<std::string>();
}

template <typename T>
static T safeGet(const YAML::Node& node, const std::string& key) {
  if (!node[key]) {
    throw std::runtime_error("Missing required config key: " + key);
  }
  return node[key].as<T>();
}

template <typename T>
static T getWithDefault(const YAML::Node& node, const std::string& key,
                        const T& default_val) {
  if (!node[key]) {
    return default_val;
  }
  return node[key].as<T>();
}

bool AppConfig::load(const std::string& yaml_path) {
  try {
    YAML::Node config = YAML::LoadFile(yaml_path);

    // dataset
    if (config["dataset"]) {
      dataset.coco_labels = safeGetString(config["dataset"], "coco_labels");
    }

    // input
    if (config["input"]) {
      input.source = safeGetString(config["input"], "source");
      input.type =
          getWithDefault(config["input"], "type", std::string("video"));
    }

    // output
    if (config["output"]) {
      output.video = getWithDefault(config["output"], "video", std::string(""));
      output.image = safeGetString(config["output"], "image");
      output.fourcc =
          getWithDefault(config["output"], "fourcc", std::string("MJPG"));
      output.fps = getWithDefault(config["output"], "fps", 10);
      output.width = getWithDefault(config["output"], "width", 1920);
      output.height = getWithDefault(config["output"], "height", 1080);
      output.show = getWithDefault(config["output"], "show", true);
      output.result =
          getWithDefault(config["output"], "result", std::string(""));
    }

    // global backend
    if (config["backend"]) {
      backend.type = getWithDefault(config["backend"], "type",
                                    std::string("onnxruntime_cpu"));
      backend.device_id = getWithDefault(config["backend"], "device_id", 0);
    }

    // detector
    if (config["detector"]) {
      detector.type =
          getWithDefault(config["detector"], "type", std::string("yolov5"));
      detector.backend =
          getWithDefault(config["detector"], "backend", backend.type);
      detector.model_path = safeGetString(config["detector"], "model_path");
      detector.input_width =
          getWithDefault(config["detector"], "input_width", 640);
      detector.input_height =
          getWithDefault(config["detector"], "input_height", 640);
      detector.confidence_threshold =
          getWithDefault(config["detector"], "confidence_threshold", 0.25f);
      detector.nms_threshold =
          getWithDefault(config["detector"], "nms_threshold", 0.4f);
    }

    // tracker
    if (config["tracker"]) {
      tracker.type =
          getWithDefault(config["tracker"], "type", std::string("bytetrack"));
    }

    // deepsort
    if (config["deepsort"]) {
      deepsort.feature_model_path =
          safeGetString(config["deepsort"], "feature_model_path");
      deepsort.backend =
          getWithDefault(config["deepsort"], "backend", backend.type);
      deepsort.feature_dim =
          getWithDefault(config["deepsort"], "feature_dim", 512);
      deepsort.max_cosine_distance =
          getWithDefault(config["deepsort"], "max_cosine_distance", 0.2f);
      deepsort.nn_budget = getWithDefault(config["deepsort"], "nn_budget", 100);
      deepsort.max_iou_distance =
          getWithDefault(config["deepsort"], "max_iou_distance", 0.7f);
      deepsort.max_age = getWithDefault(config["deepsort"], "max_age", 30);
      deepsort.n_init = getWithDefault(config["deepsort"], "n_init", 3);
    }

    // bytetrack
    if (config["bytetrack"]) {
      bytetrack.fps = getWithDefault(config["bytetrack"], "fps", 20);
      bytetrack.track_buffer =
          getWithDefault(config["bytetrack"], "track_buffer", 30);
      bytetrack.track_thresh =
          getWithDefault(config["bytetrack"], "track_thresh", 0.5f);
      bytetrack.high_thresh =
          getWithDefault(config["bytetrack"], "high_thresh", 0.6f);
      bytetrack.match_thresh =
          getWithDefault(config["bytetrack"], "match_thresh", 0.8f);
    }

    std::cout << "Config loaded from: " << yaml_path << std::endl;
    return true;
  } catch (const std::exception& e) {
    std::cerr << "Failed to load config: " << e.what() << std::endl;
    return false;
  }
}
