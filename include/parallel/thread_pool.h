#pragma once

#include <atomic>
#include <latch>
#include <barrier>
#include <thread>
#include <vector>
#include <functional>
#include <stop_token>
#include <mutex>
#include <condition_variable>

namespace md {

// ============================================
// 线程池管理（避免重复创建线程的开销）
// ============================================
class thread_pool {
 public:
  static thread_pool& instance() {
    static thread_pool pool(std::thread::hardware_concurrency());
    return pool;
  }

  size_t num_workers() const noexcept { return workers_.size(); }

  template <typename Func>
  void parallel_for(size_t total_size, Func&& func, size_t chunk_size = 0) {
    const size_t num_workers = workers_.size();
    if (chunk_size == 0) {
      chunk_size = (total_size + num_workers - 1) / num_workers;
    }

    std::latch latch(static_cast<ptrdiff_t>(num_workers));
    std::atomic<size_t> next_chunk{0};

    for (size_t w = 0; w < num_workers; ++w) {
      workers_[w]->enqueue([&, w]() {
        while (true) {
          size_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
          size_t start = chunk * chunk_size;
          if (start >= total_size) break;
          size_t end = std::min(start + chunk_size, total_size);
          func(w, start, end);
        }
        latch.count_down();
      });
    }
    latch.wait();
  }

  template <typename Func>
  bool parallel_for_cancellable(size_t total_size, std::stop_token stoken, Func&& func, size_t chunk_size = 0) {
    const size_t num_workers = workers_.size();
    if (chunk_size == 0) {
      chunk_size = (total_size + num_workers - 1) / num_workers;
    }

    std::latch latch(static_cast<ptrdiff_t>(num_workers));
    std::atomic<size_t> next_chunk{0};
    std::atomic<bool> cancelled{false};

    for (size_t w = 0; w < num_workers; ++w) {
      workers_[w]->enqueue([&, w]() {
        while (!stoken.stop_requested() && !cancelled.load(std::memory_order_relaxed)) {
          size_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
          size_t start = chunk * chunk_size;
          if (start >= total_size) break;
          size_t end = std::min(start + chunk_size, total_size);
          func(w, start, end);
        }
        if (stoken.stop_requested()) {
          cancelled.store(true, std::memory_order_relaxed);
        }
        latch.count_down();
      });
    }
    latch.wait();
    return !cancelled.load(std::memory_order_relaxed);
  }

  ~thread_pool() {
    for (auto& w : workers_) {
      w->request_stop();
    }
  }

 private:
  struct worker {
    std::jthread thread;
    std::atomic<bool> has_work{false};
    std::function<void()> task;
    std::mutex mtx;
    std::condition_variable cv;

    worker()
        : thread([this](std::stop_token stoken) {
            while (!stoken.stop_requested()) {
              std::function<void()> local_task;
              {
                std::unique_lock lock(mtx);
                cv.wait(lock, [this, &stoken]() { return has_work.load() || stoken.stop_requested(); });
                if (stoken.stop_requested()) break;
                if (has_work.load()) {
                  local_task = std::move(task);
                  has_work.store(false);
                }
              }
              if (local_task) {
                local_task();
              }
            }
          }) {}

    void enqueue(std::function<void()> f) {
      {
        std::lock_guard lock(mtx);
        task = std::move(f);
        has_work.store(true);
      }
      cv.notify_one();
    }

    void request_stop() {
      thread.request_stop();
      cv.notify_one();
    }
  };

  std::vector<std::unique_ptr<worker>> workers_;

  explicit thread_pool(size_t num_threads) {
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back(std::make_unique<worker>());
    }
  }
};

}  // namespace md
