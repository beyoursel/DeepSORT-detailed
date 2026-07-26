#pragma once

#include <string>
#include <yaml-cpp/yaml.h>

struct DetectorConfig {
    std::string type;
    std::string model_path;
    int input_width;
    int input_height;
    float confidence_threshold;
    float nms_threshold;
};

struct DeepSORTConfig {
    std::string feature_model_path;
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
    std::string type; // "video" or "image"
};

struct OutputConfig {
    std::string video;
    std::string image;
    std::string fourcc;
    int fps;
    int width;
    int height;
};

struct DatasetConfig {
    std::string coco_labels;
};

struct TrackerConfig {
    std::string type; // "deepsort" or "bytetrack"
};

class AppConfig {
public:
    static AppConfig* getInstance();

    bool load(const std::string& yaml_path);

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
