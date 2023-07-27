#include <c10/core/StorageImpl.h>
#include <c10/util/Exception.h>
#include <ATen/cuda/CachingHostAllocator.h>

namespace c10 {
    StorageImpl::~StorageImpl() {
        if (atm_enabled())
            entity_.impl_->mark_dirty();
    }
    void StorageImpl::release_resources() {
        #ifdef ATM_DEBUG_2
        c10::cuda::get_impl_profile()->storageLifeEnds(this);
        #endif
        if (atm_enabled())
            entity_.impl_->mark_dirty();
        data_ptr_.clear();
    }
    void StorageImpl::pagein_manual() {
        #ifdef ATM_DEBUG_STORAGE
        c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG,
                                            "StorageImpl::pagein_manual", "");
        #endif
        if (atm_enabled()) {
            entity_.impl_->pagein_internal();
        }
    }
    void StorageImpl::pageout_manual() {
        #ifdef ATM_DEBUG_STORAGE
        c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG,
                                            "StorageImpl::pageout_manual", "");
        #endif
        if (atm_enabled()) {
            entity_.impl_->pageout_internal();
        }
    }
    void StorageImpl::need_prefetch() {
        #ifdef ATM_DEBUG_STORAGE
        c10::cuda::get_debug_log()->add_debug(c10::cuda::ATMLogLevel::DEBUG,
                                            "StorageImpl::need_prefetch", "");
        #endif
        if (atm_enabled()) {
            entity_.impl_->need_prefetch_internal();
        }
    }
    void EntityStorageImpl::release_resources() {}
    void EntityStorageImpl::mark_dirty() {
        std::lock_guard<std::mutex> lock(mutex_);
        dirty_ = true;
    }
}