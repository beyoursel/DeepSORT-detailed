#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <sstream>

class detect_result
{
public:
    int classId;
    float confidence;
    cv::Rect_<float> box;
};



class YOLOv5Detector
{
public:
    void init();
    void detect(cv::Mat & frame, std::vector<detect_result> &result);
    std::vector<std::string> classes_;
    void draw_frame(cv::Mat & frame, std::vector<detect_result> &results);
private:

    cv::dnn::Net net;

    float confidence_threshold_ = 0.25f;
    float nms_threshold_ = 0.4f;

    int model_input_width_ = 640;
    int model_input_height_ = 640;

};
