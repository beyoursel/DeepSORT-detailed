#include "nn_matching.h"

using namespace Eigen;

NearNeighborDisMetric::NearNeighborDisMetric(
    NearNeighborDisMetric::METRIC_TYPE metric, float matching_threshold,
    int budget) {
  if (metric == euclidean) {
    metric_ = &NearNeighborDisMetric::NnEuclideanDistance;
  } else if (metric == cosine) {
    metric_ = &NearNeighborDisMetric::NnCosineDistance;
  }

  this->mating_threshold = matching_threshold;
  this->budget_ = budget;
  this->samples_.clear();
}

DYNAMICM
NearNeighborDisMetric::Distance(const FEATURESS& features,
                                const std::vector<int>& targets) {
  DYNAMICM cost_matrix = Eigen::MatrixXf::Zero(targets.size(), features.rows());
  int idx = 0;
  for (int target : targets) {
    cost_matrix.row(idx) = (this->*metric_)(this->samples_[target], features);
    idx++;
  }
  return cost_matrix;
}

void NearNeighborDisMetric::PartialFit(std::vector<TRACKER_DATA>& tid_feats,
                                        std::vector<int>& active_targets) {
  /*python code:
   * let feature(target_id) append to samples;
   * && delete not comfirmed target_id from samples.
   * update samples;
   */

  for (TRACKER_DATA& data : tid_feats) {
    int track_id = data.first;
    FEATURESS newFeatOne = data.second;

    if (samples_.find(track_id) != samples_.end()) {  // append
      int oldSize = samples_[track_id].rows();
      int addSize = newFeatOne.rows();
      int newSize = oldSize + addSize;

      int feature_dim = samples_[track_id].cols();
      if (newSize <= this->budget_) {
        FEATURESS newSampleFeatures(newSize, feature_dim);
        newSampleFeatures.block(0, 0, oldSize, feature_dim) = samples_[track_id];
        newSampleFeatures.block(oldSize, 0, addSize, feature_dim) = newFeatOne;
        samples_[track_id] = newSampleFeatures;
      } else {
        if (oldSize < this->budget_) {  // original space is not enough;
          FEATURESS newSampleFeatures(this->budget_, feature_dim);
          if (addSize >= this->budget_) {
            newSampleFeatures =
                newFeatOne.block(0, 0, this->budget_, feature_dim);
          } else {
            newSampleFeatures.block(0, 0, this->budget_ - addSize, feature_dim) =
                samples_[track_id]
                    .block(addSize, 0, this->budget_ - addSize, feature_dim)
                    .eval();
            newSampleFeatures.block(this->budget_ - addSize, 0, addSize,
                                    feature_dim) = newFeatOne;
          }
          samples_[track_id] = newSampleFeatures;
        } else {  // original space is ok;
          if (addSize >= this->budget_) {
            samples_[track_id] =
                newFeatOne.block(0, 0, this->budget_, feature_dim);
          } else {
            samples_[track_id].block(0, 0, this->budget_ - addSize, feature_dim) =
                samples_[track_id]
                    .block(addSize, 0, this->budget_ - addSize, feature_dim)
                    .eval();
            samples_[track_id].block(this->budget_ - addSize, 0, addSize,
                                    feature_dim) = newFeatOne;
          }
        }
      }
    } else {  // not exit, create new one;
      samples_[track_id] = newFeatOne;
    }
  }  // add features;

  // erase the samples which not in active_targets;
  for (std::map<int, FEATURESS>::iterator i = samples_.begin();
       i != samples_.end();) {
    bool flag = false;
    for (int j : active_targets)
      if (j == i->first) {
        flag = true;
        break;
      }
    if (flag == false)
      samples_.erase(i++);
    else
      i++;
  }
}

Eigen::VectorXf NearNeighborDisMetric::NnCosineDistance(const FEATURESS& x,
                                                          const FEATURESS& y) {
  MatrixXf distances = CosineDistance(x, y);
  VectorXf res = distances.colwise().minCoeff().transpose();
  return res;
}

Eigen::VectorXf NearNeighborDisMetric::NnEuclideanDistance(
    const FEATURESS& x, const FEATURESS& y) {
  MatrixXf distances = Pdist(x, y);
  VectorXf res = distances.colwise().minCoeff().transpose();
  res = res.array().max(VectorXf::Zero(res.rows()).array());
  return res;
}

Eigen::MatrixXf NearNeighborDisMetric::Pdist(const FEATURESS& x,
                                              const FEATURESS& y) {
  int len1 = x.rows(), len2 = y.rows();
  if (len1 == 0 || len2 == 0) {
    return Eigen::MatrixXf::Zero(len1, len2);
  }
  MatrixXf res = x * y.transpose() * -2;
  res = res.colwise() + x.rowwise().squaredNorm();
  res = res.rowwise() + y.rowwise().squaredNorm().transpose();
  res = res.array().max(MatrixXf::Zero(res.rows(), res.cols()).array());
  return res;
}

Eigen::MatrixXf NearNeighborDisMetric::CosineDistance(
    const FEATURESS& a, const FEATURESS& b, bool data_is_normalized) {
  if (data_is_normalized == true) {
    return 1. - (a * b.transpose()).array();
  }
  // L2-normalize each row first so the distance is a true cosine distance
  // even when the feature model does not output unit vectors.
  FEATURESS an = a.rowwise().normalized();
  FEATURESS bn = b.rowwise().normalized();
  return 1. - (an * bn.transpose()).array();
}
