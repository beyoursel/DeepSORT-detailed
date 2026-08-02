#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// Per-frame stage timing recorder. Disabled unless Open() is called with a
// CSV path (output.timing in the config). ScopedTimer records stages into it
// automatically; call EndFrame() once per processed frame to flush one row.
// Columns are the stage names seen in the first frame, in first-seen order;
// a stage that appears only in later frames is ignored. Single-threaded use.
class TimingLogger {
 public:
  static TimingLogger& GetInstance() {
    static TimingLogger instance;  // thread-safe since C++11
    return instance;
  }

  bool Open(const std::string& path) {
    out_.open(path);
    if (!out_.is_open()) {
      std::cerr << "Failed to open timing csv: " << path << std::endl;
      return false;
    }
    return true;
  }

  bool IsOpen() const { return out_.is_open(); }

  void Record(const std::string& stage, double ms) {
    if (!IsOpen()) return;
    if (!header_written_ && row_.find(stage) == row_.end()) {
      columns_.push_back(stage);
    }
    row_[stage] = ms;
  }

  void EndFrame() {
    if (!IsOpen()) return;
    if (!header_written_) {
      out_ << "frame";
      for (const auto& c : columns_) out_ << "," << c;
      out_ << "\n";
      header_written_ = true;
    }
    out_ << ++frame_id_;
    for (const auto& c : columns_) {
      out_ << ",";
      auto it = row_.find(c);
      if (it != row_.end()) out_ << it->second;
    }
    out_ << "\n";
    row_.clear();
  }

 private:
  TimingLogger() = default;
  TimingLogger(const TimingLogger&) = delete;
  TimingLogger& operator=(const TimingLogger&) = delete;

  std::ofstream out_;
  std::vector<std::string> columns_;
  std::map<std::string, double> row_;
  bool header_written_ = false;
  long frame_id_ = 0;
};
