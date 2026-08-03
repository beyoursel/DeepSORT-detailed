#include "detector_utils.h"
#include <opencv2/imgproc.hpp>

namespace detector {

cv::Mat Letterbox(const cv::Mat& src, int target_width, int target_height,
                  ScaleInfo& scale_info) {
  int src_w = src.cols;
  int src_h = src.rows;

  float scale = std::min(static_cast<float>(target_width) / src_w,
                         static_cast<float>(target_height) / src_h);

  int new_w = static_cast<int>(src_w * scale);
  int new_h = static_cast<int>(src_h * scale);

  cv::Mat resized;
  cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR); // INTER_LINEAR Resize

  cv::Mat dst =
      cv::Mat::zeros(cv::Size(target_width, target_height), src.type());
  int pad_x = (target_width - new_w) / 2;
  int pad_y = (target_height - new_h) / 2;
  cv::Rect roi(pad_x, pad_y, new_w, new_h);
  resized.copyTo(dst(roi)); // copy the resized img to the roi in dst

  scale_info.x_factor = static_cast<float>(src_w) / new_w;
  scale_info.y_factor = static_cast<float>(src_h) / new_h;
  scale_info.pad_x = static_cast<float>(pad_x);
  scale_info.pad_y = static_cast<float>(pad_y);

  return dst;
}

std::vector<detect_result> NmsFilter(std::vector<cv::Rect>& boxes,
                                     std::vector<float>& confidences,
                                     std::vector<int>& classIds,
                                     float confidence_threshold,
                                     float nms_threshold) {
  std::vector<detect_result> results;
  std::vector<int> indexes;
  cv::dnn::NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold,
                    indexes);

  results.reserve(indexes.size());
  for (size_t i = 0; i < indexes.size(); i++) {
    int index = indexes[i];
    detect_result dr;
    dr.classId = classIds[index];
    dr.confidence = confidences[index];
    dr.box = boxes[index];
    results.push_back(dr);
  }
  return results;
}

void DrawResults(cv::Mat& frame, std::vector<detect_result>& results,
                 const std::vector<std::string>& classes) {
  for (const auto& dr : results) {
    cv::rectangle(frame, dr.box, cv::Scalar(0, 0, 255), 2, 8);
    cv::rectangle(frame, cv::Point(dr.box.tl().x, dr.box.tl().y - 20),
                  cv::Point(dr.box.br().x, dr.box.tl().y),
                  cv::Scalar(255, 0, 0), -1);

    std::string label = cv::format("%.2f", dr.confidence);
    if (dr.classId >= 0 && dr.classId < static_cast<int>(classes.size())) {
      label = classes[dr.classId] + ":" + label;
    }

    cv::putText(frame, label, cv::Point(dr.box.x, dr.box.y + 6), 1, 2,
                cv::Scalar(0, 255, 0), 2);
  }
}

}  // namespace detector
