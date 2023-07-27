#include <c10/cuda/CUDASwapQueues.h>

namespace c10 {
namespace cuda {

void CudaEntityTransferQueue::enqueue(EntityStorageImpl* impl) 
{
  std::unique_lock<std::mutex> lock(action_mutex_);
  if (enable_flag_) { 
    actions_.emplace_back(new EntityStorageRef(impl->storage_->entity()));
    if (active_flag_) {
      lock.unlock(); not_empty_cv_.notify_all();
    }
  }
}

EntityStorageRef_t CudaEntityTransferQueue::dequeue() 
{
  std::lock_guard<std::mutex> lock(action_mutex_);
  if (actions_.empty()) 
    return nullptr; 
  auto impl_ref = actions_.front();
  actions_.pop_front();
  return impl_ref;
}

int CudaEntityTransferQueue::erase(EntityStorageImpl* impl) 
{
  std::lock_guard<std::mutex> lock(action_mutex_);
  for (auto i = actions_.begin(); i != actions_.end(); i++)
    if ((*i)->impl_->entity_id_ == impl->entity_id_) {
      actions_.erase(i); return 0;
    }
  return 1;
}

CudaEntityEvictQueue& CudaEntityEvictQueue::get_evict_queue() 
{
  static CudaEntityEvictQueue evict_queue_;
  return evict_queue_;
}

void CudaEntityEvictQueue::thread_do_entity_evict(CudaEntityEvictQueue& evict_queue) 
{
  std::unique_lock<std::mutex> lock(evict_queue.action_mutex_);
  // unique working thread allowed
  if (evict_queue.unique_flag_) return;
    else evict_queue.unique_flag_ = true;
  while (true) {
    lock.unlock();
    auto impl_ref = evict_queue.dequeue();
    lock.lock();
    while (impl_ref == nullptr && evict_queue.actions_.empty()) {
      evict_queue.empty_cv_.notify_all();
      evict_queue.not_empty_cv_.wait(lock);
      lock.unlock();
      impl_ref = evict_queue.dequeue();
      lock.lock();
      if (!evict_queue.active_flag_) goto post_evict_thread;
    }
    lock.unlock();
    if (impl_ref->impl_.use_count() > 1 && !impl_ref->impl_->dirty_) {
      impl_ref->impl_->pageout_internal_sync();
      impl_ref->impl_->do_pageout_cb();
    }
    delete impl_ref;
    lock.lock();
  }
  post_evict_thread:
  // allow new unique thread to create 
  evict_queue.unique_flag_ = false;
}

void CudaEntityEvictQueue::start_actions() 
{
  std::lock_guard<std::mutex> lock(action_mutex_);
  if (active_flag_ || enable_flag_ || unique_flag_) return; 
  active_flag_ = true;
  enable_flag_ = true;
  thread_do_entity_evict_ = std::thread(thread_do_entity_evict, std::ref(*this));
  thread_do_entity_evict_.detach();
}

void CudaEntityEvictQueue::wait_and_stop_actions() 
{
  std::unique_lock<std::mutex> lock(action_mutex_);
  enable_flag_ = false;
  if (!active_flag_) return;
  // there's running working thread, wait
  if (!actions_.empty()) empty_cv_.wait(lock);
  active_flag_ = false;
  lock.unlock();
  // must be a working thread wait not_empty_cv
  not_empty_cv_.notify_all();
}

void CudaEntityEvictQueue::wait_actions() 
{
  std::unique_lock<std::mutex> lock(action_mutex_);
  if (!active_flag_) return;
  enable_flag_ = false;
  if (!actions_.empty()) empty_cv_.wait(lock);
  enable_flag_ = true;
}


void CudaEntityFetchQueue::enqueue_front(EntityStorageImpl* impl) 
{
  std::unique_lock<std::mutex> lock(action_mutex_);
  if (enable_flag_) { 
    actions_.emplace_front(new EntityStorageRef(impl->storage_->entity()));
    if (active_flag_) {
      lock.unlock(); not_empty_cv_.notify_all();
    }
  }
}

CudaEntityFetchQueue& CudaEntityFetchQueue::get_fetch_queue() 
{
  static CudaEntityFetchQueue fetch_queue_;
  return fetch_queue_;
}

void CudaEntityFetchQueue::thread_do_entity_fetch(CudaEntityFetchQueue& fetch_queue) 
{
  std::unique_lock<std::mutex> lock(fetch_queue.action_mutex_);
  // unique working thread allowed
  if (fetch_queue.unique_flag_) return;
    else fetch_queue.unique_flag_ = true;
  while (true) {
    lock.unlock();
    auto impl_ref = fetch_queue.dequeue();
    lock.lock();
    while (impl_ref == nullptr && fetch_queue.actions_.empty()) {
      fetch_queue.empty_cv_.notify_all();
      fetch_queue.not_empty_cv_.wait(lock);
      lock.unlock();
      impl_ref = fetch_queue.dequeue();
      lock.lock();
      if (!fetch_queue.active_flag_) goto post_fetch_thread;
    }
    lock.unlock();
    if (impl_ref->impl_.use_count() > 1 && !impl_ref->impl_->dirty_) {
      impl_ref->impl_->pagein_internal_sync();
      impl_ref->impl_->do_pagein_cb();
    }
    delete impl_ref;
    lock.lock();
  }
  post_fetch_thread:
  // allow new unique thread to create 
  fetch_queue.unique_flag_ = false;
}

void CudaEntityFetchQueue::enable_queue()
{
  std::lock_guard<std::mutex> lock(action_mutex_);
  enable_flag_ = true;
}

void CudaEntityFetchQueue::start_actions() 
{
  std::lock_guard<std::mutex> lock(action_mutex_);
  if (active_flag_ || unique_flag_) return; 
  active_flag_ = true;
  thread_do_entity_fetch_ = std::thread(thread_do_entity_fetch, std::ref(*this));
  thread_do_entity_fetch_.detach();
}

void CudaEntityFetchQueue::wait_and_stop_actions() 
{
  std::unique_lock<std::mutex> lock(action_mutex_);
  enable_flag_ = false;
  if (!active_flag_) return;
  if (!actions_.empty()) empty_cv_.wait(lock);
  active_flag_ = false;
  lock.unlock();
  not_empty_cv_.notify_all();
}

void CudaEntityFetchQueue::wait_actions() 
{
  std::unique_lock<std::mutex> lock(action_mutex_);
  if (!active_flag_) return;
  enable_flag_ = false;
  if (!actions_.empty()) empty_cv_.wait(lock);
  enable_flag_ = true;
}

} // cuda
} // c10