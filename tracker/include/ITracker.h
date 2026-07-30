#pragma once

#include "IDetector.h"  // detect_result
#include <opencv2/core.hpp>
#include <vector>

namespace tracking {

// Unified tracking output: one confirmed track with its current bounding box.
struct TrackResult {
  int track_id;
  cv::Rect_<float> box;  // top-left x/y + width/height
};

class ITracker {
 public:
  virtual ~ITracker() = default;

  // Update the tracker with detections of the current frame and return
  // the currently confirmed tracks.
  // `frame` is only used by appearance-based trackers (e.g. DeepSORT ReID);
  // motion-only trackers (e.g. ByteTrack) ignore it.
  virtual std::vector<TrackResult> Update(
      const cv::Mat& frame, const std::vector<detect_result>& detections) = 0;
};

}  // namespace tracking
