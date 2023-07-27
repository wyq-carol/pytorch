#pragma once
// #include <c10/macros/Macros.h>
#include <c10/core/Allocator.h>
#include <c10/core/ATMCommon.h>
// #include <c10/core/TensorImpl.h>
// #include <c10/core/StorageImpl.h>
#include <c10/cuda/CUDAMacros.h>
#include <c10/util/IntrusiveList.h>

#include <mutex>
#include <exception>
#include <map>
#include <vector>
#include <string>
#include <iterator>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <typeinfo>

// 2^15
#define MAX_LOG_PRESERVED 32768
namespace c10 {

struct TensorImpl;
struct StorageImpl;

namespace cuda {
// List of [ Calling Function + Debug Log ] 
typedef std::vector<std::pair<std::string, std::string>> DebugLogList;

class ATMDebugLog;
class ImplProfile;

C10_CUDA_API ATMDebugLog* get_debug_log();
C10_CUDA_API ImplProfile* get_impl_profile();

class ATMConfig {
  public: 
  ATMConfig() = default;
};

enum class ATMLogLevel {
  DEBUG,
  INFO,
  WARNING,
  ERROR
};

class ATMDebugLog {
  public:
  ATMDebugLog() = default;
  void add_debug(const ATMLogLevel level, const std::string &func, const std::string &info) {
    std::unique_lock<std::mutex> lock(mutex_);
    switch (level) {
      case ATMLogLevel::DEBUG : { 
        count_debug_log_++; 
        debug_log_.push_back(std::make_pair(func, info)); 
        if (count_debug_log_ % MAX_LOG_PRESERVED == 0) handle_log_oom("debug", debug_log_, count_debug_log_);
        break; }
      case ATMLogLevel::INFO : { 
        count_info_log_++; 
        info_log_.push_back(std::make_pair(func, info)); 
        if (count_info_log_ % MAX_LOG_PRESERVED == 0) handle_log_oom("info", info_log_, count_info_log_);
        break; }
      case ATMLogLevel::WARNING : { 
        count_warning_log_++; 
        warning_log_.push_back(std::make_pair(func, info)); 
        if (count_warning_log_ % MAX_LOG_PRESERVED == 0) handle_log_oom("warning", warning_log_, count_warning_log_);
        break; }
      case ATMLogLevel::ERROR : { 
        count_error_log_++; 
        error_log_.push_back(std::make_pair(func, info)); 
        if (count_error_log_ % MAX_LOG_PRESERVED == 0) handle_log_oom("error", error_log_, count_error_log_);
        break; }
    }
    
  }
  const DebugLogList& get_debug(ATMLogLevel level) const {
    switch (level) {
      case ATMLogLevel::DEBUG : return debug_log_;
      case ATMLogLevel::INFO  : return info_log_;
      case ATMLogLevel::WARNING : return warning_log_;
      case ATMLogLevel::ERROR : return error_log_;
    }
    return debug_log_;
  }
  void clear_debug(ATMLogLevel level) {
    std::unique_lock<std::mutex> lock(mutex_);
    switch (level) {
      case ATMLogLevel::DEBUG : debug_log_.clear();
                                break;
      case ATMLogLevel::INFO  : info_log_.clear();
                                break;
      case ATMLogLevel::WARNING : warning_log_.clear();
                                break;
      case ATMLogLevel::ERROR : error_log_.clear();
                                break;
    }
  }
  private:
  void handle_log_oom(std::string log_name, DebugLogList& log_list, int log_count) {
    if (log_count % MAX_LOG_PRESERVED) return; 
    log_count -= MAX_LOG_PRESERVED;
    int iter = 0;
    FILE* fd = fopen((log_name + ".atm.log").c_str(), "a+");
    for (auto log_el : log_list) {
      std::string debug_output = "[" + std::to_string((++iter) + log_count) + "]" + log_el.first + "|=>|" + log_el.second + "\n";
      fprintf(fd, "%s", debug_output.c_str());
    }
    log_list.clear();
  }

  std::mutex mutex_;
  // Guarded by mutex_
  DebugLogList debug_log_; 
  DebugLogList info_log_;
  DebugLogList warning_log_;
  DebugLogList error_log_;
  int count_debug_log_;
  int count_info_log_;
  int count_warning_log_;
  int count_error_log_;
};

struct ImplProfileEl {
  uint64_t data_ptr_;
  int64_t life_start_;
  int64_t life_end_;
  uint64_t size_; // in Byte
  std::vector<int64_t> access_seq_;
  uint8_t by_operator;
};
class ImplProfile {
  public:
  ImplProfile() = default;
  void tensorLifeStart(const c10::TensorImpl* tensor_ptr);
  //   const void *data_ptr = tensor_ptr->data();
  //   tensor_profile_.insert(
  //     std::make_pair(reinterpret_cast<uint64_t>(tensor_ptr), 
  //     ImplProfileEl{
  //       reinterpret_cast<uint64_t>(data_ptr),
  //       std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(),
  //       0,
  //       tensor_ptr->storage().nbytes()
  //     })
  //   );
  //   return;
  // }
  void tensorSetStorage(const c10::TensorImpl* tensor_ptr);
  void tensorLifeEnds(const c10::TensorImpl* tensor_ptr);
  //   tensor_profile_[reinterpret_cast<uint64_t>(tensor_ptr)].life_end_ = 
  //     std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  //   return;
  // }
  void storageLifeStart(const c10::StorageImpl* storage_ptr) {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    // const void *data_ptr = tensor_ptr->data();
    storage_profile_.insert(std::make_pair(reinterpret_cast<uint64_t>(storage_ptr), 
      ImplProfileEl{
        0,
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count(),
        0, 0, {}, 0
      })
    );
    return;
  }
  void storageLifeEnds(const c10::StorageImpl* storage_ptr) {
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    if (storage_profile_.find(reinterpret_cast<uint64_t>(storage_ptr)) == storage_profile_.end()) {
      storage_profile_.insert(std::make_pair(reinterpret_cast<uint64_t>(storage_ptr),
        ImplProfileEl{ 0, 0, 0, 0, {}, 0}) );
      #ifdef ATM_DEBUG_4
      get_debug_log()->add_debug(ATMLogLevel::DEBUG, 
                                "ImplProfile::storageLifeEnds", 
                                std::to_string(reinterpret_cast<uint64_t>(storage_ptr)) + "Not Found");
      #endif
      // return;
    }
    storage_profile_[reinterpret_cast<uint64_t>(storage_ptr)].life_end_ = 
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
  }
  void storageSetStorage(const c10::StorageImpl* storage_ptr, void* data_ptr, size_t size) {
    if (storage_profile_.find(reinterpret_cast<uint64_t>(storage_ptr)) == storage_profile_.end()) {
      #ifdef ATM_DEBUG_4
      get_debug_log()->add_debug(ATMLogLevel::DEBUG, 
                                "ImplProfile::storageSetStorage", 
                                std::to_string(reinterpret_cast<uint64_t>(storage_ptr)) + "Not Found");
      #endif
      return;
    }
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    storage_profile_[reinterpret_cast<uint64_t>(storage_ptr)].data_ptr_ = reinterpret_cast<uint64_t>(data_ptr);
    storage_profile_[reinterpret_cast<uint64_t>(storage_ptr)].size_ = size;
  }

  void storageAppendAccess(const c10::StorageImpl* storage_ptr) {
    if (storage_profile_.find(reinterpret_cast<uint64_t>(storage_ptr)) == storage_profile_.end()) {
      storage_profile_.insert(std::make_pair(reinterpret_cast<uint64_t>(storage_ptr),
        ImplProfileEl{ 0, 0, 0, 0, {}, 0}) );
      #ifdef ATM_DEBUG_4
      get_debug_log()->add_debug(ATMLogLevel::DEBUG, 
                                "ImplProfile::storageAppendAccess", 
                                std::to_string(reinterpret_cast<uint64_t>(storage_ptr)) + "Not Found");
      #endif
      // return;
    }
    std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
    storage_profile_[reinterpret_cast<uint64_t>(storage_ptr)].access_seq_.push_back(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
    );
  }
  
  void clear_storage_profile() { storage_profile_.clear(); }
  std::map<uint64_t, ImplProfileEl>& get_storage_profile() { return storage_profile_; }
  private:

  std::mutex mutex_;
  // Guarded by mutex_
  std::map<uint64_t, ImplProfileEl> tensor_profile_;
  std::map<uint64_t, ImplProfileEl> storage_profile_;
  
};



} // cuda
} // c10
