#pragma once

#include "AppConfig.h"
#include "ITracker.h"
#include <memory>

class ByteTracker;  // legacy ByteTrack implementation

namespace tracking {

// ITracker adapter around the legacy ByteTrack implementation.
class ByteTrackAdapter : public ITracker {
 public:
  explicit ByteTrackAdapter(const ByteTrackConfig& cfg);
  ~ByteTrackAdapter() override;

  std::vector<TrackResult> Update(
      const cv::Mat& frame,
      const std::vector<detect_result>& detections) override;

 private:
  std::unique_ptr<ByteTracker> impl_;
};

}  // namespace tracking
