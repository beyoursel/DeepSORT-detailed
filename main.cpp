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

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d)
{
    DETECTION_ROW tmpRow;
    tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

    tmpRow.confidence = confidence;
    d.push_back(tmpRow);
}


void test_deepsort(cv::Mat& frame, std::vector<detect_result>& results,tracker& mytracker)
{
    std::vector<detect_result> objects;

    DETECTIONS detections;
    for (detect_result dr : results)
    {
        if(dr.classId == 0) //person
        {
            objects.push_back(dr);
            cv::rectangle(frame, dr.box, cv::Scalar(255, 0, 0), 2);
            get_detections(DETECTBOX(dr.box.x, dr.box.y,dr.box.width,  dr.box.height),dr.confidence,  detections);
        }
    }

    std::cout<<"begin track"<<std::endl;
    if(FeatureTensor::getInstance()->getRectsFeature(frame, detections))
    {
        std::cout << "get feature succeed!"<<std::endl;
        mytracker.predict();
        mytracker.update(detections);
        std::vector<RESULT_DATA> result;
        for(Track& track : mytracker.tracks) {
            if(!track.is_confirmed() || track.time_since_update > 1) continue;
            result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
        }
        for(unsigned int k = 0; k < detections.size(); k++)
        {
            DETECTBOX tmpbox = detections[k].tlwh;
            cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
            cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);

            for(unsigned int k = 0; k < result.size(); k++)
            {
                DETECTBOX tmp = result[k].second;
                cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
                rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);

                std::string label = cv::format("%d", result[k].first);
                cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
            }
        }
    }
    std::cout<<"end track"<<std::endl;
}


void test_bytetrack(cv::Mat& frame, std::vector<detect_result>& results,BYTETracker& tracker)
{
    std::vector<detect_result> objects;


    for (detect_result dr : results)
    {

        if(dr.classId == 0) //person
        {
            objects.push_back(dr);
        }
    }


    std::vector<STrack> output_stracks = tracker.update(objects);

    for (unsigned long i = 0; i < output_stracks.size(); i++)
    {
        std::vector<float> tlwh = output_stracks[i].tlwh;
        bool vertical = tlwh[2] / tlwh[3] > 1.6;
        if (tlwh[2] * tlwh[3] > 20 && !vertical)
        {
            cv::Scalar s = tracker.get_color(output_stracks[i].track_id);
            cv::putText(frame, cv::format("%d", output_stracks[i].track_id), cv::Point(tlwh[0], tlwh[1] - 5),
                    0, 0.6, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
            cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), s, 2);
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
    detector->init();

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
        auto start = std::chrono::system_clock::now();
        detector->detect(frame, results);
        auto end = std::chrono::system_clock::now();
        auto detect_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout << detector->num_classes() << ":" << results.size() << ":" << num_frames << std::endl;

        if (cfg->tracker.type == "deepsort") {
            test_deepsort(frame, results, *mytracker);
        } else {
            test_bytetrack(frame, results, *bytetracker);
        }

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
