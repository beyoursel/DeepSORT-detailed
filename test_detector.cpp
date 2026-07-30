/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-23 02:52:22
*/
#include "AppConfig.h"
#include "DetectorFactory.h"
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>

int main(int argc, char* argv[]) {
  std::string config_path = "./config/config.yaml";
  if (argc > 1) {
    config_path = argv[1];
  }

  AppConfig* cfg = AppConfig::GetInstance();
  if (!cfg->Load(config_path)) {
    std::cerr << "Failed to load config from " << config_path << std::endl;
    return -1;
  }

  std::shared_ptr<detector::IDetector> detector;
  try {
    detector = detector::DetectorFactory::Create(cfg->detector.type);
    detector->Init();
  } catch (const std::exception& e) {
    std::cerr << "Failed to initialize detector: " << e.what() << std::endl;
    return -1;
  }

  std::cout << "Detector type: " << cfg->detector.type << std::endl;

  std::vector<detect_result> results;
  cv::Mat frame = cv::imread(cfg->input.source);
  if (frame.empty()) {
    std::cerr << "Failed to load image: " << cfg->input.source << std::endl;
    return -1;
  }

  auto start = std::chrono::system_clock::now();
  detector->Detect(frame, results);
  auto end = std::chrono::system_clock::now();
  auto detect_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
          .count();

  detector->DrawFrame(frame, results);

  std::string window_title = "Detector: " + cfg->detector.type;
  cv::imshow(window_title, frame);

  cv::imwrite(cfg->output.image, frame);

  results.clear();

  return 0;
}
