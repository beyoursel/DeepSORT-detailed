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
#include "DetectorFactory.h"
#include "AppConfig.h"

#include "FeatureTensor.h"
#include "BYTETracker.h" //bytetrack
#include "tracker.h"//deepsort
#include "Timer.h"

#include <chrono>

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}


void test_deepsort(cv::Mat& frame, std::vector<detect_result>& results, tracker& mytracker)
{
    ScopedTimer timer("track");
    std::vector<detect_result> objects;
    DETECTIONS detections;

    // 1) 先画检测框（绿色，含类别置信度）
    for (const detect_result& dr : results)
    {
        if (dr.classId == 0) // person
        {
            objects.push_back(dr);
            get_detections(DETECTBOX(dr.box.x, dr.box.y, dr.box.width, dr.box.height),
                           dr.confidence, detections);

            std::string det_label = cv::format("person:%.2f", dr.confidence);
            cv::rectangle(frame, dr.box, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::putText(frame, det_label,
                        cv::Point(dr.box.x, dr.box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }

    if (FeatureTensor::getInstance()->getRectsFeature(frame, detections))
    {
        mytracker.predict();
        mytracker.update(detections);

        // 2) 再画跟踪框（红色，含 ID）
        for (Track& track : mytracker.tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1) continue;
            DETECTBOX tmp = track.to_tlwh();
            cv::Rect rect(tmp(0), tmp(1), tmp(2), tmp(3));
            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            std::string label = cv::format("ID:%d", track.track_id);
            cv::putText(frame, label, cv::Point(rect.x, rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }
}


void test_bytetrack(cv::Mat& frame, std::vector<detect_result>& results, BYTETracker& tracker)
{
    ScopedTimer timer("track");
    std::vector<detect_result> objects;

    // 1) 先画检测框（绿色，含类别置信度）
    for (const detect_result& dr : results)
    {
        if (dr.classId == 0) // person
        {
            objects.push_back(dr);
            std::string det_label = cv::format("person:%.2f", dr.confidence);
            cv::rectangle(frame, dr.box, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
            cv::putText(frame, det_label,
                        cv::Point(dr.box.x, dr.box.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }

    std::vector<STrack> output_stracks = tracker.update(objects);

    // 2) 再画跟踪框（红色，含 ID）
    for (size_t i = 0; i < output_stracks.size(); i++)
    {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        bool vertical = tlwh[2] / tlwh[3] > 1.6;
        if (tlwh[2] * tlwh[3] > 20 && !vertical)
        {
            cv::Rect rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]);
            cv::rectangle(frame, rect, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            std::string label = cv::format("ID:%d", output_stracks[i].track_id);
            cv::putText(frame, label, cv::Point(tlwh[0], tlwh[1] - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
    }
}

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

    // Initialize tracker based on config
    tracker* mytracker = nullptr;
    BYTETracker* bytetracker = nullptr;

    if (cfg->tracker.type == "deepsort") {
        mytracker = new tracker(cfg->deepsort);
    } else if (cfg->tracker.type == "bytetrack") {
        bytetracker = new BYTETracker(cfg->bytetrack);
    } else {
        std::cerr << "Unknown tracker type: " << cfg->tracker.type << std::endl;
        return -1;
    }

    std::shared_ptr<detector::IDetector> detector =
        detector::DetectorFactory::create(cfg->detector.type);
    if (!detector->init()) {
        std::cerr << "Detector initialization failed, exiting." << std::endl;
        delete mytracker;
        delete bytetracker;
        return -1;
    }

    const auto& input_cfg = cfg->input;
    const auto& output_cfg = cfg->output;

    std::cout << "begin read video" << std::endl;
    cv::VideoCapture capture(input_cfg.source);

    if (!capture.isOpened()) {
        printf("could not read this video file...\n");
        return -1;
    }
    std::cout << "end read video" << std::endl;

    std::vector<detect_result> results;
    int num_frames = 0;

    // Use input video's native size and fps for output to avoid resolution mismatch.
    int input_fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
    int input_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int input_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (input_fps <= 0) input_fps = output_cfg.fps;
    if (input_width <= 0) input_width = output_cfg.width;
    if (input_height <= 0) input_height = output_cfg.height;

    int fourcc = cv::VideoWriter::fourcc(
        output_cfg.fourcc[0], output_cfg.fourcc[1],
        output_cfg.fourcc[2], output_cfg.fourcc[3]);
    cv::VideoWriter video(output_cfg.video, fourcc, input_fps,
                          cv::Size(input_width, input_height));
    if (!video.isOpened()) {
        std::cerr << "Failed to open video writer: " << output_cfg.video << std::endl;
        return -1;
    }

    while (true)
    {
        cv::Mat frame;

        if (!capture.read(frame))
        {
            std::cout << "\n Cannot read the video file. please check your video.\n";
            break;
        }

        num_frames++;
        auto det_start = std::chrono::steady_clock::now();
        detector->detect(frame, results);
        auto det_end = std::chrono::steady_clock::now();
        double detect_time = elapsed_ms(det_start, det_end);

        auto track_start = std::chrono::steady_clock::now();
        if (cfg->tracker.type == "deepsort") {
            test_deepsort(frame, results, *mytracker);
        } else {
            test_bytetrack(frame, results, *bytetracker);
        }
        auto track_end = std::chrono::steady_clock::now();
        double track_time = elapsed_ms(track_start, track_end);

        std::cout << "[FRAME] frame=" << num_frames
                  << " dets=" << results.size()
                  << " det_total=" << detect_time << "ms"
                  << " track_total=" << track_time << "ms"
                  << std::endl;

        std::string window_title = "Detector: " + cfg->detector.type;
        cv::imshow(window_title, frame);

        video.write(frame);

        if(cv::waitKey(30) == 27)
        {
            break;
        }

        results.clear();
    }
    capture.release();
    video.release();
    cv::destroyAllWindows();

    delete mytracker;
    delete bytetracker;

    return 0;
}
