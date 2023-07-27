#pragma once

#include <c10/core/ATMCommon.h>
#include <c10/core/Allocator.h>
#include <c10/cuda/ATMConfig.h>

#include <c10/util/intrusive_ptr.h>
// #include <c10/core/StorageImpl.h>

#include <mutex>
#include <condition_variable>

namespace c10 {

struct StorageImpl;
struct EntityStorageImpl;

struct EntityStorageRef {
  EntityStorageRef(EntityStorageImpl* impl) : 
    impl_(impl) {}
  EntityStorageRef(const EntityStorageRef &impl_ref) : 
    impl_(impl_ref.impl_) {}
  std::shared_ptr<EntityStorageImpl> impl_;
};

typedef EntityStorageRef* EntityStorageRef_t; 

enum class EntityStorageStat : uint8_t {
  kOnline,  // on  device
  kOffline, // off device
  kTrans,   // on  transfer
};

enum class TransStat : uint8_t {
  kNone,    // no mission
  kPgOut,   // on pageout
  kPgIn     // on pagein
};

struct EntityStorageImpl {
  // Abstract class. These methods must be defined for a specific implementation (e.g. CUDA)
  virtual void do_pagein(void* dst, void* src, size_t size, bool sync) = 0;
  virtual void do_pageout(void* dst, void* src, size_t size, bool sync) = 0;
  virtual void do_pagein_cb() {
    std::unique_lock<std::mutex> lock(mutex_);
    trans_stat_ = TransStat::kNone;
    entity_stat_ = EntityStorageStat::kOnline;
  }
  virtual void do_pageout_cb() {
    std::unique_lock<std::mutex> lock(mutex_);
    trans_stat_ = TransStat::kNone;
    entity_stat_ = EntityStorageStat::kOffline;
  }

  EntityStorageImpl(StorageImpl* storage, c10::Allocator* host_allocator) :
    storage_(storage), host_allocator_(host_allocator), dirty_(false),
    trans_stat_(TransStat::kNone), entity_stat_(EntityStorageStat::kOnline) { 
  }

  EntityStorageImpl() = delete;
  virtual ~EntityStorageImpl() {}
  
  void             release_resources();
  virtual void     ensure_data() {
    #ifdef ATM_DEBUG_STORAGE
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                      "EntityStorageImpl::ensure_data", "");
    #endif
    ensure_data_internal(true);
  }

  // StorageImpl accessors defined in StorageImpl.h to avoid circular depencencies
  const Allocator* allocator() const;
  size_t           capacity() const;
  Device           device() const;
  void*            device_ptr() const;
  c10::DataPtr     set_device_ptr(c10::DataPtr&& data_ptr);
  void             mark_dirty();
  /*
  * set synchronize true to use synchronize swapin
  */
  virtual void     ensure_data_internal(bool sync) {
    std::unique_lock<std::mutex> lock(mutex_);
    std::lock_guard<std::mutex> ensure_lock(ensure_mutex_);
  }
  virtual void     prefetch_internal() = 0;
  // Wait transfer done, you should do understand what you're doing !!!
  virtual void     unsafe_wait_transfer() = 0;
  
  virtual void     pageout_internal() { do_pageout_cb(); }
  virtual void     pagein_internal()  { do_pagein_cb(); }
  virtual void     need_prefetch_internal() {}

  virtual void     pageout_internal_sync() { }
  virtual void     pagein_internal_sync()  { }

  uint64_t         id() const { return entity_id_; }

  // Initialized at or soon after construction
  StorageImpl*    const storage_;
  c10::Allocator* const host_allocator_;
  uint64_t              entity_id_;
  
  mutable std::mutex    mutex_;
  mutable std::mutex    ensure_mutex_;

  // Guarded by mutex_
  c10::DataPtr          host_data_ptr_;
  bool                  dirty_;

  TransStat             trans_stat_;
  EntityStorageStat     entity_stat_;
};


namespace cuda {
} // cuda
} // c10