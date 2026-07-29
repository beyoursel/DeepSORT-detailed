#include "ByteTrackAdapter.h"
#include "BYTETracker.h"
#include "Timer.h"

ByteTrackAdapter::ByteTrackAdapter(const ByteTrackConfig& cfg)
    : impl_(new BYTETracker(cfg)) {}

ByteTrackAdapter::~ByteTrackAdapter() = default;

std::vector<TrackResult> ByteTrackAdapter::update(
    const cv::Mat& /*frame*/, const std::vector<detect_result>& detections) {
  ScopedTimer timer("track");

  std::vector<TrackResult> out;
  std::vector<STrack> stracks = impl_->update(detections);
  for (const STrack& st : stracks) {
    const std::vector<float>& tlwh = st.tlwh;
    bool vertical = tlwh[2] / tlwh[3] > 1.6;
    if (tlwh[2] * tlwh[3] > 20 && !vertical) {
      out.push_back(
          {st.track_id, cv::Rect_<float>(tlwh[0], tlwh[1], tlwh[2], tlwh[3])});
    }
  }
  return out;
}
