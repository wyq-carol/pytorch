#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>

#include <c10/util/intrusive_ptr.h>

#include <c10/core/ATMCommon.h>
#include <c10/core/EntityStorageImpl.h>

#include <c10/cuda/ATMConfig.h>
namespace c10 {

// A storage represents the underlying backing data buffer for a
// tensor.  This concept was inherited from the original Torch7
// codebase; we'd kind of like to get rid of the concept
// (see https://github.com/pytorch/pytorch/issues/14797) but
// it's hard work and no one has gotten around to doing it.
//
// NB: storage is supposed to uniquely own a data pointer; e.g.,
// two non-null data pointers alias if and only if they are from
// the same storage.  Technically you can violate this invariant
// (e.g., you can create a non-owning StorageImpl with at::from_blob)
// but a lot of things won't work correctly, including:
//
// - An ordinary deleter on such a storage is wrong, because normal deleters
//   assume unique ownership, but if you have two storages at the same data,
//   that implies there is some sort of shared ownership. So your deleter would
//   have to actually be internally doing some sort of refcount thing
// - Deepcopy in Python side relies on storage equality and not data pointer
//   equality; so if there are two separate storages pointing to the same data,
//   the data will actually get duplicated in that case (one data ptr before,
//   two data ptrs after)
// - Version counts won't work correctly, because we do all VC tracking at the
//   level of storages (unless you explicitly disconnect the VC with detach);
//   mutation because data pointers are the same are totally untracked
struct C10_API StorageImpl : public c10::intrusive_ptr_target {
 public:
  struct use_byte_size_t {};

  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable)
      : data_ptr_(std::move(data_ptr)),
        size_bytes_(size_bytes),
        resizable_(resizable),
        received_cuda_(false),
        allocator_(allocator),
        entity_(nullptr) {
        // entity_(allocator ? allocator->as_entity(this) : nullptr) { // wyq
    #ifdef ATM_DEBUG_1
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::constructor", 
                                    "Pre Allocated DataPtr Device: " + device().str());
    #endif
    #ifdef ATM_DEBUG_2
    auto impl_profile_ = c10::cuda::get_impl_profile();
    impl_profile_->storageLifeStart(this);
    impl_profile_->storageSetStorage(this, data_ptr_.get(), nbytes());
    #endif
    if (resizable) {
      TORCH_INTERNAL_ASSERT(
          allocator_, "For resizable storage, allocator must be provided");
    }
  }

  StorageImpl(
      use_byte_size_t /*use_byte_size*/,
      size_t size_bytes,
      at::Allocator* allocator,
      bool resizable)
      : StorageImpl(
            use_byte_size_t(),
            size_bytes,
            allocator->allocate(size_bytes),
            allocator,
            resizable) {
    #ifdef ATM_DEBUG_1
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::constructor", 
                                    "No Pre Allocated DataPtr Device: " + device().str());
    #endif
    #ifdef ATM_DEBUG_2
    c10::cuda::get_impl_profile()->storageLifeStart(this);
    #endif
  }

  StorageImpl& operator=(StorageImpl&& other) = default;
  StorageImpl& operator=(const StorageImpl&) = delete;
  StorageImpl() = delete;
  StorageImpl(StorageImpl&& other) : entity_(other.entity_) {
    #ifdef ATM_DEBUG_1
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::constructor", 
                                    "No Pre Allocated (From Other) DataPtr Device: " + device().str());
    #endif
    // #ifdef ATM_DEBUG_2
    // c10::cuda::get_impl_profile()->storageLifeStart(this);
    // #endif
  }
  StorageImpl(const StorageImpl&) = delete;
  ~StorageImpl() override;

  void reset() {
    data_ptr_.clear();
    size_bytes_ = 0;
  }

  template <typename T>
  inline T* data() const {
    #ifdef ATM_ENSURE_DATA
    if (atm_enabled()) entity_.impl_->ensure_data();
    #endif
    #ifdef ATM_DEBUG_1
    T x_;
    const char* type_name = typeid(x_).name();
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::data"+ std::string(type_name) +"(const)", 
                                    "Accessed Data");
    #endif
    #ifdef ATM_DEBUG_2
    c10::cuda::get_impl_profile()->storageAppendAccess(this);
    #endif
    return unsafe_data<T>();
  }

  template <typename T>
  inline T* unsafe_data() const {
    #ifdef ATM_ENSURE_DATA
    if (atm_enabled()) entity_.impl_->ensure_data();
    #endif
    return static_cast<T*>(this->data_ptr_.get());
  }

  void release_resources() override;

  size_t nbytes() const {
    return size_bytes_;
  }

  // TODO: remove later
  void set_nbytes(size_t size_bytes) {
    size_bytes_ = size_bytes;
  }

  bool resizable() const {
    return resizable_;
  };

  at::DataPtr& data_ptr() {
    #ifdef ATM_ENSURE_DATA
    if (atm_enabled()) entity_.impl_->ensure_data();
    #endif
    return data_ptr_;
  };

  const at::DataPtr& data_ptr() const {
    #ifdef ATM_ENSURE_DATA
    if (atm_enabled()) entity_.impl_->ensure_data();
    #endif
    return data_ptr_;
  };

  // Returns the previous data_ptr
  at::DataPtr set_data_ptr(at::DataPtr&& data_ptr) {
    at::DataPtr old_data_ptr(std::move(data_ptr_));
    data_ptr_ = std::move(data_ptr);
    #ifdef ATM_DEBUG_1
    // printf("Set DataPtr Device: %s\n", device().str().c_str());
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::set_data_ptr", 
                                    "Set DataPtr Device: " + device().str());
    #endif
    #ifdef ATM_DEBUG_2
    c10::cuda::get_impl_profile()->storageSetStorage(this, data_ptr_.get(), nbytes());
    #endif
    return old_data_ptr;
  };

  void set_data_ptr_noswap(at::DataPtr&& data_ptr) {
    data_ptr_ = std::move(data_ptr);
  }

  // TODO: Return const ptr eventually if possible
  void* data() {
    #ifdef ATM_ENSURE_DATA
    if (atm_enabled()) entity_.impl_->ensure_data();
    #endif
    #ifdef ATM_DEBUG_1
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::data",
                                    "Accessed Data");
    #endif
    #ifdef ATM_DEBUG_2
    c10::cuda::get_impl_profile()->storageAppendAccess(this);
    #endif
    return data_ptr_.get();
  }

  void* data() const {
    #ifdef ATM_ENSURE_DATA
    if (atm_enabled()) entity_.impl_->ensure_data();
    #endif
    #ifdef ATM_DEBUG_1
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::data_ptr(const)",
                                    "Accessed Data");
    #endif
    #ifdef ATM_DEBUG_2
    c10::cuda::get_impl_profile()->storageAppendAccess(this);
    #endif
    return data_ptr_.get();
  }

  at::DeviceType device_type() const {
    return data_ptr_.device().type();
  }

  at::Allocator* allocator() {
    #ifdef ATM_DEBUG_1
    // printf("Used Allocator Once\n");
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::allocator", 
                                    "Used Allocator Once");
    #endif
    return allocator_;
  }

  const at::Allocator* allocator() const {
    #ifdef ATM_DEBUG_1
    c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG, 
                                    "StorageImpl::allocator(const)", 
                                    "Used Allocator Once");
    #endif
    return allocator_;
  };

  // You generally shouldn't use this method, but it is occasionally
  // useful if you want to override how a tensor will be reallocated,
  // after it was already allocated (and its initial allocator was
  // set)
  void set_allocator(at::Allocator* allocator) {
    allocator_ = allocator;
  }

  Device device() const {
    return data_ptr_.device();
  }

  void set_resizable(bool resizable) {
    if (resizable) {
      // We need an allocator to be resizable
      AT_ASSERT(allocator_);
    }
    resizable_ = resizable;
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      void* src,
      size_t size_bytes,
      DeleterFnPtr d = nullptr) {
    UniqueStorageShareExternalPointer(
        at::DataPtr(src, src, d, data_ptr_.device()), size_bytes);
  }

  /**
   * Can only be called when use_count is 1
   */
  void UniqueStorageShareExternalPointer(
      at::DataPtr&& data_ptr,
      size_t size_bytes) {
    data_ptr_ = std::move(data_ptr);
    size_bytes_ = size_bytes;
    allocator_ = nullptr;
    resizable_ = false;
  }

  // This method can be used only after storage construction and cannot be used
  // to modify storage status
  void set_received_cuda(bool received_cuda) {
    received_cuda_ = received_cuda;
  }

  bool received_cuda() {
    return received_cuda_;
  }

  // manual method should be deprecated other than debug   
  void pageout_manual();
  void pagein_manual();
  void need_prefetch();
  
  bool atm_enabled() const { return entity_.impl_.get() != nullptr; }
 
  EntityStorageRef& entity() {
    return entity_;
  }

  DataPtr data_ptr_;
 private:
  size_t size_bytes_;
  bool resizable_;
  // Identifies that Storage was received from another process and doesn't have
  // local to process cuda memory allocation
  bool received_cuda_;

  Allocator* allocator_;

  EntityStorageRef entity_;
};

inline const c10::Allocator* EntityStorageImpl::allocator() const {
  return storage_->allocator();
}
inline size_t EntityStorageImpl::capacity() const {
  return storage_->nbytes();
}
inline void* EntityStorageImpl::device_ptr() const {
  return storage_->data_ptr_.get();
}
inline c10::Device EntityStorageImpl::device() const {
  return storage_->device();
}
inline c10::DataPtr EntityStorageImpl::set_device_ptr(c10::DataPtr&& data_ptr) {
  std::swap(storage_->data_ptr_, data_ptr);
  return std::move(data_ptr);
}

} // namespace c10
