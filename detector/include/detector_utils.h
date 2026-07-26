#pragma once

#include "IDetector.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace detector {

struct ScaleInfo {
    float x_factor;
    float y_factor;
    float pad_x;    // left padding in network input coordinates
    float pad_y;    // top padding in network input coordinates
};

cv::Mat letterbox(const cv::Mat& src,
                  int target_width,
                  int target_height,
                  ScaleInfo& scale_info);

std::vector<detect_result> nms_filter(std::vector<cv::Rect>& boxes,
                                       std::vector<float>& confidences,
                                       std::vector<int>& classIds,
                                       float confidence_threshold,
                                       float nms_threshold);

void draw_results(cv::Mat& frame,
                  std::vector<detect_result>& results,
                  const std::vector<std::string>& classes);

} // namespace detector
