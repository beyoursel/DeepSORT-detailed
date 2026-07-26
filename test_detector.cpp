/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-23 02:52:22
*/
#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "YOLOv5Detector.h"
#include "AppConfig.h"


int main(int argc, char *argv[])
{
    std::string config_path = "./config/config.yaml";
    if (argc > 1) {
        config_path = argv[1];
    }

    AppConfig* cfg = AppConfig::getInstance();
    if (!cfg->load(config_path)) {
        std::cerr << "Failed to load config from " << config_path << std::endl;
        return -1;
    }

    std::shared_ptr<YOLOv5Detector> detector(new YOLOv5Detector());
    detector->init();

    std::vector<detect_result> results;
    cv::Mat frame = cv::imread(cfg->input.source);
    if (frame.empty()) {
        std::cerr << "Failed to load image: " << cfg->input.source << std::endl;
        return -1;
    }

    auto start = std::chrono::system_clock::now();
    detector->detect(frame, results);
    auto end = std::chrono::system_clock::now();
    auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    detector->draw_frame(frame, results);

    cv::imshow("YOLOv5-6.x", frame);

    cv::imwrite(cfg->output.image, frame);

    results.clear();

    return 0;
}
