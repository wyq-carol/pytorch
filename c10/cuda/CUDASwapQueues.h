#pragma once

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/UniqueVoidPtr.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/util/llvmMathExtras.h>

#include <c10/core/EntityStorageImpl.h>
#include <c10/core/StorageImpl.h>

#include <mutex>
#include <deque>
#include <thread>

namespace c10 {
namespace cuda {
struct CudaEntityTransferQueue {
 public:
  CudaEntityTransferQueue() : 
          enable_flag_(false),
          active_flag_(false),
          unique_flag_(false) {}
  void               enqueue(EntityStorageImpl* impl);
  int                erase(EntityStorageImpl* impl);
  EntityStorageRef_t dequeue();

  virtual void       start_actions() = 0;
  virtual void       wait_and_stop_actions() = 0;
  virtual void       wait_actions() = 0;

 protected:  
  std::mutex                     action_mutex_;
  // guarded by action_mutex
  std::deque<EntityStorageRef_t> actions_;
  std::atomic_bool               enable_flag_;
  std::atomic_bool               active_flag_;
  std::atomic_bool               unique_flag_;

  std::condition_variable        not_empty_cv_;
  std::condition_variable        empty_cv_;
};

struct CudaEntityEvictQueue final : public CudaEntityTransferQueue {
 public:
  CudaEntityEvictQueue() = default;
  
  static CudaEntityEvictQueue& get_evict_queue();
  
  void               start_actions() override;
  void               wait_and_stop_actions() override;
  void               wait_actions() override;
  
 private:
  static void thread_do_entity_evict(CudaEntityEvictQueue& evict_queue);
  std::thread thread_do_entity_evict_;
};

struct CudaEntityFetchQueue final : public CudaEntityTransferQueue {
 public:
  CudaEntityFetchQueue() = default;

  static CudaEntityFetchQueue& get_fetch_queue();
  
  void               enqueue_front(EntityStorageImpl* impl);

  void               enable_queue();
  void               start_actions() override;
  void               wait_and_stop_actions() override;
  void               wait_actions() override;
 private:  
  static void thread_do_entity_fetch(CudaEntityFetchQueue& fetch_queue);
  std::thread thread_do_entity_fetch_;
};

} // namespace cuda
} // namespace c10