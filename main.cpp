/*!
    @Description : https://github.com/shaoshengsong/
    @Author      : shaoshengsong
    @Date        : 2022-09-23 02:52:22
*/
#include "AppConfig.h"
#include "DetectorFactory.h"
#include "ITracker.h"
#include "Timer.h"
#include "TrackerFactory.h"
#include <cctype>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <sstream>

#include <chrono>
#include <memory>

// Write one MOTChallenge-format line:
// <frame>,<id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1
static void WriteMotLine(std::ostream* mot_out, int frame_id, int track_id,
                         const cv::Rect_<float>& box) {
  if (!mot_out) return;
  *mot_out << frame_id << "," << track_id << "," << box.x << "," << box.y << ","
           << box.width << "," << box.height << ",1,-1,-1,-1\n";
}

// Draw a green detection box (class + confidence).
static void DrawDetection(cv::Mat& frame, const detect_result& dr,
                          const char* class_name) {
  std::string det_label = cv::format("%s:%.2f", class_name, dr.confidence);
  cv::rectangle(frame, dr.box, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
  cv::putText(frame, det_label, cv::Point(dr.box.x, dr.box.y - 5),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1,
              cv::LINE_AA);
}

// Draw a red tracking box (track ID).
static void DrawTrack(cv::Mat& frame, const tracking::TrackResult& t) {
  cv::rectangle(frame, t.box, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
  std::string label = cv::format("ID:%d", t.track_id);
  cv::putText(frame, label, cv::Point(t.box.x, t.box.y - 5),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2,
              cv::LINE_AA);
}

// Run one frame through the detect -> track pipeline, draw results and dump
// MOT lines. Returns false on exception (caller should stop the loop).
static bool ProcessFrame(cv::Mat& frame,
                         const std::shared_ptr<detector::IDetector>& detector,
                         const std::unique_ptr<tracking::ITracker>& tracker,
                         std::vector<detect_result>& results, int frame_id,
                         std::ostream* mot_out) {
  try {
    auto det_start = std::chrono::steady_clock::now();
    detector->Detect(frame, results);
    auto det_end = std::chrono::steady_clock::now();
    double detect_time = ElapsedMs(det_start, det_end);

    // Track persons only (historical behavior); draw green detection boxes.
    std::vector<detect_result> objects;
    for (const detect_result& dr : results) {
      if (dr.classId == 0)  // person
      {
        objects.push_back(dr);
        DrawDetection(frame, dr, "person");
      }
    }

    auto track_start = std::chrono::steady_clock::now();
    std::vector<tracking::TrackResult> tracks = tracker->Update(frame, objects);
    auto track_end = std::chrono::steady_clock::now();
    double track_time = ElapsedMs(track_start, track_end);

    // Draw red tracking boxes and dump MOT results.
    for (const tracking::TrackResult& t : tracks) {
      DrawTrack(frame, t);
      WriteMotLine(mot_out, frame_id, t.track_id, t.box);
    }

    std::cout << "[FRAME] frame=" << frame_id << " dets=" << results.size()
              << " det_total=" << detect_time << "ms"
              << " track_total=" << track_time << "ms" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] frame=" << frame_id << ": " << e.what() << std::endl;
    return false;
  }
  return true;
}

// Shared frame loop for video files and camera streams. Returns process exit
// code.
static int RunCaptureLoop(cv::VideoCapture& capture,
                          const std::shared_ptr<detector::IDetector>& detector,
                          const std::unique_ptr<tracking::ITracker>& tracker,
                          const AppConfig* cfg, std::ostream* mot_out) {
  const auto& output_cfg = cfg->output;

  // Use input source's native size and fps for output to avoid resolution
  // mismatch; fall back to configured values when the backend reports none
  // (common for camera devices and network streams).
  int input_fps = static_cast<int>(capture.get(cv::CAP_PROP_FPS));
  int input_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  int input_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  if (input_fps <= 0) input_fps = output_cfg.fps;
  if (input_width <= 0) input_width = output_cfg.width;
  if (input_height <= 0) input_height = output_cfg.height;

  // Video writer is optional: empty output.video disables it.
  cv::VideoWriter video;
  if (!output_cfg.video.empty()) {
    const std::string fourcc_str = (output_cfg.fourcc.size() >= 4)
                                       ? output_cfg.fourcc
                                       : std::string("MJPG");
    int fourcc = cv::VideoWriter::fourcc(fourcc_str[0], fourcc_str[1],
                                         fourcc_str[2], fourcc_str[3]);
    video.open(output_cfg.video, fourcc, input_fps,
               cv::Size(input_width, input_height));
    if (!video.isOpened()) {
      std::cerr << "Failed to open video writer: " << output_cfg.video
                << std::endl;
      return -1;
    }
  }

  std::vector<detect_result> results;
  int num_frames = 0;
  while (true) {
    cv::Mat frame;

    if (!capture.read(frame)) {
      std::cout << "\n Cannot read frame from source, stopping.\n";
      break;
    }

    num_frames++;
    if (!ProcessFrame(frame, detector, tracker, results, num_frames, mot_out)) {
      break;
    }

    if (output_cfg.show) {
      std::string window_title = "Detector: " + cfg->detector.type;
      cv::imshow(window_title, frame);
      if (cv::waitKey(30) == 27) {
        break;
      }
    }

    if (video.isOpened()) {
      video.write(frame);
    }

    results.clear();
  }
  capture.release();
  video.release();
  if (output_cfg.show) {
    cv::destroyAllWindows();
  }

  return 0;
}

// A pure-digit source string (e.g. "0") is a camera device index; anything
// else (file path, rtsp://...) is opened as a string.
static bool ParseDeviceIndex(const std::string& source, int& index) {
  if (source.empty()) return false;
  for (char c : source) {
    if (!std::isdigit(static_cast<unsigned char>(c))) return false;
  }
  index = std::stoi(source);
  return true;
}

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

  std::unique_ptr<tracking::ITracker> tracker;
  try {
    tracker = tracking::TrackerFactory::Create(cfg->tracker.type);
  } catch (const std::exception& e) {
    std::cerr << "Failed to create tracker: " << e.what() << std::endl;
    return -1;
  }

  std::shared_ptr<detector::IDetector> detector;
  try {
    detector = detector::DetectorFactory::Create(cfg->detector.type);
  } catch (const std::exception& e) {
    std::cerr << "Failed to create detector: " << e.what() << std::endl;
    return -1;
  }
  if (!detector->Init()) {
    std::cerr << "Detector initialization failed, exiting." << std::endl;
    return -1;
  }

  const auto& input_cfg = cfg->input;
  const auto& output_cfg = cfg->output;

  // MOT-format track result output (optional).
  std::ofstream mot_file;
  std::ostream* mot_out = nullptr;
  if (!output_cfg.result.empty()) {
    mot_file.open(output_cfg.result);
    if (!mot_file.is_open()) {
      std::cerr << "Failed to open result file: " << output_cfg.result
                << std::endl;
      return -1;
    }
    mot_out = &mot_file;
  }

  if (input_cfg.type == "image") {
    // Single-image mode: detect + track once, save and/or show the result.
    cv::Mat frame = cv::imread(input_cfg.source);
    if (frame.empty()) {
      std::cerr << "could not read this image file: " << input_cfg.source
                << std::endl;
      return -1;
    }

    std::vector<detect_result> results;
    ProcessFrame(frame, detector, tracker, results, 1, mot_out);

    if (!output_cfg.image.empty()) {
      cv::imwrite(output_cfg.image, frame);
    }
    if (output_cfg.show) {
      std::string window_title = "Detector: " + cfg->detector.type;
      cv::imshow(window_title, frame);
      cv::waitKey(0);
      cv::destroyAllWindows();
    }
    return 0;
  }

  cv::VideoCapture capture;
  if (input_cfg.type == "camera") {
    int device_index = 0;
    if (ParseDeviceIndex(input_cfg.source, device_index)) {
      capture.open(device_index);
    } else {
      capture.open(input_cfg.source);  // network stream URL
    }
  } else if (input_cfg.type == "video") {
    capture.open(input_cfg.source);
  } else {
    std::cerr << "Unsupported input.type: \"" << input_cfg.type
              << "\" (expected: video | image | camera)" << std::endl;
    return -1;
  }

  if (!capture.isOpened()) {
    std::cerr << "could not open input source: " << input_cfg.source
              << std::endl;
    return -1;
  }

  return RunCaptureLoop(capture, detector, tracker, cfg, mot_out);
}
