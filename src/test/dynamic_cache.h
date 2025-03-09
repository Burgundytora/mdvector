#include <immintrin.h>
#include <xmmintrin.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// 默认参数（当无法检测CPU缓存时使用）
constexpr size_t DEFAULT_L1D_CACHE_SIZE = 32768;  // 32KB
constexpr size_t DEFAULT_CACHE_LINE_SIZE = 64;    // 64字节

// 动态获取CPU缓存信息的结构体
struct CPUCacheInfo {
  size_t l1d_cache_size;   // L1数据缓存大小（字节）
  size_t cache_line_size;  // 缓存行大小（字节）
};

// 跨平台获取CPU缓存信息
CPUCacheInfo get_cpu_cache_info() {
  CPUCacheInfo info = {DEFAULT_L1D_CACHE_SIZE, DEFAULT_CACHE_LINE_SIZE};

#ifdef __linux__
  // Linux: 通过/sys文件系统获取
  std::ifstream cache_dir("/sys/devices/system/cpu/cpu0/cache/index0/");
  if (cache_dir.good()) {
    // 获取L1d缓存大小
    std::ifstream size_file("/sys/devices/system/cpu/cpu0/cache/index0/size");
    if (size_file) {
      std::string line;
      if (std::getline(size_file, line)) {
        size_t size_kb = std::stoul(line.substr(0, line.find('K')));
        info.l1d_cache_size = size_kb * 1024;
      }
    }

    // 获取缓存行大小
    std::ifstream line_file("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size");
    if (line_file) {
      std::string line;
      if (std::getline(line_file, line)) {
        info.cache_line_size = std::stoul(line);
      }
    }
  }
#elif defined(_WIN32)
  // Windows: 通过GetLogicalProcessorInformation
  PSYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer = nullptr;
  DWORD length = 0;
  GetLogicalProcessorInformation(nullptr, &length);
  buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)malloc(length);
  if (GetLogicalProcessorInformation(buffer, &length)) {
    DWORD count = length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    for (DWORD i = 0; i < count; ++i) {
      if (buffer[i].Relationship == RelationCache) {
        CACHE_DESCRIPTOR& cache = buffer[i].Cache;
        if (cache.Level == 1 && cache.Type == CacheData) {
          info.l1d_cache_size = cache.Size;
          info.cache_line_size = cache.LineSize;
        }
      }
    }
  }
  free(buffer);
#endif

  return info;
}

// 动态策略参数生成
template <typename T>
struct DynamicConfig {
  size_t small_data_threshold;  // 小数据阈值（元素个数）
  size_t l1_block_elements;     // L1分块元素数
  size_t prefetch_distance;     // 预取距离（缓存行数）

  explicit DynamicConfig(const CPUCacheInfo& info) {
    // 小数据阈值 = (L1缓存大小 / 元素大小) * 0.5
    small_data_threshold = (info.l1d_cache_size / sizeof(T)) / 2;

    // L1分块大小 = (L1缓存大小 * 0.75) / 元素大小
    l1_block_elements = (info.l1d_cache_size * 3 / 4) / sizeof(T);

    // 预取距离 = L2缓存延迟周期数（经验值，可调整）
    // 假设内存延迟约200周期，每周期处理2缓存行
    prefetch_distance = 200 / (info.cache_line_size / sizeof(T)) / 2;
    prefetch_distance = std::max(prefetch_distance, 4UL);  // 最小4行
  }
};

template <typename T>
void optimized_avx2_add(const T* a, const T* b, T* c, size_t n) {
  // 获取CPU缓存信息
  static CPUCacheInfo cache_info = get_cpu_cache_info();
  DynamicConfig<T> config(cache_info);

  constexpr size_t simd_pack = sizeof(__m256) / sizeof(T);
  const size_t cache_line_elements = cache_info.cache_line_size / sizeof(T);

  // 小数据模式
  if (n <= config.small_data_threshold) {
    for (size_t i = 0; i < n; i += simd_pack) {
      if (i + cache_line_elements < n) {
        _mm_prefetch(reinterpret_cast<const char*>(a + i + cache_line_elements), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b + i + cache_line_elements), _MM_HINT_T0);
      }
      // ... AVX2计算代码与之前相同 ...
    }
    return;
  }

  // 大数据模式
  for (size_t base = 0; base < n; base += config.l1_block_elements) {
    const size_t block_end = std::min(base + config.l1_block_elements, n);
    // ... 分块预取和计算代码与之前类似 ...
    // 使用动态计算的prefetch_distance
    const size_t prefetch_pos = i + cache_line_elements * config.prefetch_distance;
  }
  // ... 其余代码 ...
}