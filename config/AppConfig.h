#pragma once

#include "BackendConfig.h"
#include <string>
#include <yaml-cpp/yaml.h>

struct DetectorConfig {
  std::string type;
  std::string backend;  // optional override, fallback to global backend.type
  std::string model_path;
  int input_width;
  int input_height;
  float confidence_threshold;
  float nms_threshold;
};

struct DeepSORTConfig {
  std::string feature_model_path;
  std::string backend;  // optional override, fallback to global backend.type
  int feature_dim;
  float max_cosine_distance;
  int nn_budget;
  float max_iou_distance;
  int max_age;
  int n_init;
};

struct ByteTrackConfig {
  int fps;
  int track_buffer;
  float track_thresh;
  float high_thresh;
  float match_thresh;
};

struct InputConfig {
  std::string source;
  std::string type;  // "video" or "image"
};

struct OutputConfig {
  std::string video;
  std::string image;
  std::string fourcc;
  int fps;
  int width;
  int height;
  bool show = true;  // real-time preview window; set false on headless servers
  std::string result;  // MOT-format track output file; empty = disabled
};

struct DatasetConfig {
  std::string coco_labels;
};

struct TrackerConfig {
  std::string type;  // "deepsort" or "bytetrack"
};

class AppConfig {
 public:
  static AppConfig* getInstance();

  bool load(const std::string& yaml_path);

  backend::BackendConfig backend;
  DetectorConfig detector;
  DeepSORTConfig deepsort;
  ByteTrackConfig bytetrack;
  InputConfig input;
  OutputConfig output;
  DatasetConfig dataset;
  TrackerConfig tracker;

 private:
  AppConfig() = default;
  AppConfig(const AppConfig&) = delete;
  AppConfig& operator=(const AppConfig&) = delete;

  static AppConfig* instance;
};
