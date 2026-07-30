#pragma once

#include "AppConfig.h"
#include "IDetector.h"  // detect_result
#include "STrack.h"
#include <climits>  // LONG_MAX

class ByteTracker {
 public:
  explicit ByteTracker(const ByteTrackConfig& cfg);
  ~ByteTracker();

  std::vector<STrack> Update(const std::vector<detect_result>& objects);
  cv::Scalar GetColor(int idx);

 private:
  std::vector<STrack*> JointStracks(std::vector<STrack*>& tlista,
                                    std::vector<STrack>& tlistb);
  std::vector<STrack> JointStracks(std::vector<STrack>& tlista,
                                   std::vector<STrack>& tlistb);

  std::vector<STrack> SubStracks(std::vector<STrack>& tlista,
                                 std::vector<STrack>& tlistb);
  void RemoveDuplicateStracks(std::vector<STrack>& resa,
                              std::vector<STrack>& resb,
                              std::vector<STrack>& stracksa,
                              std::vector<STrack>& stracksb);

  void LinearAssignment(std::vector<std::vector<float>>& cost_matrix,
                        int cost_matrix_size, int cost_matrix_size_size,
                        float thresh, std::vector<std::vector<int>>& matches,
                        std::vector<int>& unmatched_a,
                        std::vector<int>& unmatched_b);
  std::vector<std::vector<float>> IouDistance(std::vector<STrack*>& atracks,
                                              std::vector<STrack>& btracks,
                                              int& dist_size,
                                              int& dist_size_size);
  std::vector<std::vector<float>> IouDistance(std::vector<STrack>& atracks,
                                              std::vector<STrack>& btracks);
  std::vector<std::vector<float>> Ious(std::vector<std::vector<float>>& atlbrs,
                                       std::vector<std::vector<float>>& btlbrs);

  double Lapjv(const std::vector<std::vector<float>>& cost,
               std::vector<int>& rowsol, std::vector<int>& colsol,
               bool extend_cost = false, float cost_limit = LONG_MAX,
               bool return_cost = true);

 private:
  float track_thresh_;
  float high_thresh_;
  float match_thresh_;
  int frame_id_;
  int max_time_lost_;

  std::vector<STrack> tracked_stracks_;
  std::vector<STrack> lost_stracks_;
  std::vector<STrack> removed_stracks_;
  byte_kalman::ByteKalmanFilter kalman_filter_;
};
