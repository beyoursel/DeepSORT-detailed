#pragma once

#include "dataType.h"

#include <map>

// A tool to calculate distance;
class NearNeighborDisMetric {
 public:
  enum METRIC_TYPE { euclidean = 1, cosine };
  NearNeighborDisMetric(METRIC_TYPE metric, float matching_threshold,
                        int budget);
  DYNAMICM Distance(const FEATURESS& features, const std::vector<int>& targets);
  //    void partial_fit(FEATURESS& features, std::vector<int> targets,
  //    std::vector<int> active_targets);
  void PartialFit(std::vector<TRACKER_DATA>& tid_feats,
                   std::vector<int>& active_targets);
  float mating_threshold;

 private:
  typedef Eigen::VectorXf (NearNeighborDisMetric::*PTRFUN)(const FEATURESS&,
                                                           const FEATURESS&);
  Eigen::VectorXf NnCosineDistance(const FEATURESS& x, const FEATURESS& y);
  Eigen::VectorXf NnEuclideanDistance(const FEATURESS& x, const FEATURESS& y);

  Eigen::MatrixXf Pdist(const FEATURESS& x, const FEATURESS& y);
  Eigen::MatrixXf CosineDistance(const FEATURESS& a, const FEATURESS& b,
                                   bool data_is_normalized = false);

 private:
  PTRFUN metric_;
  int budget_;
  std::map<int, FEATURESS> samples_;
};
