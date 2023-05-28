#ifndef TARS_UTILS_TIMER_H_
#define TARS_UTILS_TIMER_H_

#include <sys/time.h>

#include <cstddef>

#include "tars/core/macro.h"

namespace tars {

inline int64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}

class Timer {
 public:
  DISABLE_COPY_MOVE_ASSIGN(Timer);
  virtual void StartTiming() = 0;
  virtual void StopTiming() = 0;
  virtual void AccumulateTiming() = 0;
  virtual void ClearTiming() = 0;
  virtual double ElapsedMicros() = 0;
  virtual double AccumulatedMicros() = 0;
};

class WallClockTimer : public Timer {
 public:
  WallClockTimer() : accumulated_micros_(0) {}

  void StartTiming() override { start_micros_ = NowMicros(); }

  void StopTiming() override { stop_micros_ = NowMicros(); }

  void AccumulateTiming() override {
    StopTiming();
    accumulated_micros_ += stop_micros_ - start_micros_;
  }

  void ClearTiming() override {
    start_micros_ = 0;
    stop_micros_ = 0;
    accumulated_micros_ = 0;
  }

  double ElapsedMicros() override { return stop_micros_ - start_micros_; }

  double AccumulatedMicros() override { return accumulated_micros_; }

 private:
  double start_micros_;
  double stop_micros_;
  double accumulated_micros_;
};

}  // namespace tars

#endif  // TARS_UTILS_TIMER_H_
