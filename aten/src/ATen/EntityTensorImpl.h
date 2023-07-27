#pragma once

#include <atomic>
#include <memory>
#include <numeric>
#include <random>

#include <c10/core/Backend.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/core/CopyBytes.h>

#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/Flags.h>
#include <c10/util/Logging.h>
#include <c10/util/python_stub.h>
#include <c10/core/TensorImpl.h>
#include <ATen/Tensor.h>
#include <ATen/ATen.h>

#define likely(x)      __builtin_expect(!!(x), 1)
#define unlikely(x)    __builtin_expect(!!(x), 0)
// #define TORCH_CHECK(a, ...) // profile mode

namespace at {

template<typename T>
struct RefCell final : intrusive_ptr_target {
  mutable T value;
  void release_resources() final { static_release_resources(value); }
  RefCell(const T& t) : value(t) { }
};

template<typename T>
using Ref = intrusive_ptr<RefCell<T>>;

template<typename T>
void static_release_resources(intrusive_ptr<T>& ptr) { ptr.reset(); }

class EntityTensorCell;
using Strong = intrusive_ptr<EntityTensorCell>;
using Strongs = std::vector<Strong>;
using Weak = weak_intrusive_ptr<EntityTensorCell>;
using Weaks = std::vector<Weak>;
using Tensors = std::vector<Tensor>;
using RematFunc_t = std::function<Tensors(const Tensors&)>;
using MutateFunc_t = std::function<void(const Tensors&)>;

using time_t = std::chrono::time_point<std::chrono::system_clock>;
using duration_t = std::chrono::system_clock::duration;

struct Unsafe { };

// The rematerializer could be called to reinvoke an operator.
// Tensor point to remat which point to Tensor.
// To build the cycle remat support a default constructor,
// And allow you to fill in the member later.
struct Rematerializer : intrusive_ptr_target {
  RematFunc_t func;
  Strongs     inputs;
  Weaks       outputs;
  duration_t  compute_cost;
  // when some output in here get evicted, they should belong to this ecn.
  // a rematerializer have to track this,
  // because when multiple output of a rematerializer get evicted,
  // we only want to count the compute cost once.
  Rematerializer(const Unsafe&,
                 const RematFunc_t& func,
                 const Strongs&     inputs,
                 duration_t         compute_cost)  :
          func(func),
          inputs(inputs),
          compute_cost(compute_cost) { }
  void release_resources() final {
    func = RematFunc_t();
    inputs.clear();
    outputs.clear();
  }
  void remat();
};

// Track all Tensor that share the same Storage.
// This is the atomic level of eviction - when evicting, everything here will get evicted.
// When an AliasPool is evicted, the Storage of the underlying tensor must be freed.
// Additionally, the AliasPool contain weak pointer to all children of tensors,
// in order to compute the score of evicting a Storage.
struct AliasPool : intrusive_ptr_target {
  Weaks                         tensors_;
  Weaks                         neighbors_;
  // std::set<ecn_ptr> neighbor_ecn();
  // get() might hold some raw Tensor, rendering them unevictable.
  // it is likely that get() will run out of memory, and when it does so, it will try to evict.
  // so, it is crucial that we dont try to evict those tensors - doing so will not evict anything.
  // lock_count count how many time a tensor is referenced by get.
  size_t                        lock_count_ = 0;
  size_t                        external_count_ = 0;
  size_t lock()   { return ++lock_count_; }
  size_t unlock() { return --lock_count_; }
  
  intrusive_ptr<Rematerializer> head_remat_;
  bool evictable() const { return lock_count_ == 0 && head_remat_; }
  // if it is not evictable it must not be evicted.
  bool                          is_evicted_ = false;
  size_t                        memory_;
  time_t                        last_used_;
  void update_last_used() { last_used_ = std::chrono::system_clock::now();}
  
  // An aliaspool cant register itself to the checkpointpool - you have to do it yourself.
  AliasPool(const Unsafe&, intrusive_ptr<Rematerializer> head_remat, size_t memory) :
          head_remat_(head_remat),
          memory_(memory),
          last_used_(std::chrono::system_clock::now()) { }
  
  // if it is evicted, then hold the evicted tensor group.
  // ecn_ptr ecn;
  // double cost(time_t current_time);
  // void evict();
  void register_external() { ++external_count_; }
  void release_external()  {
    --external_count_;
    if (external_count_ == 0) {
      if (lock_count_ > 0) return;
      TORCH_CHECK(lock_count_ == 0);
      // if (memory_ > 0 && (!ecn) && head_remat) {
      //   evict();
      // }
    }
  }
  // if it was evicted, refresh it. otherwise do nothing.
  // have to check so, because when we rematerialize a single tensor in an aliaspool,
  // we will set it to non-evicted, and when we rematerialize it's tensor they will also reset this.
  void set_not_evicted(const intrusive_ptr<AliasPool>& self);
  void release_resources() final {
    tensors_.clear();
    neighbors_.clear();
    head_remat_.reset();
  }
};

struct EntityTensorCell : intrusive_ptr_target {
  std::unique_ptr<Tensor>       t_;
  bool                          defined_ = false;
  bool                          is_undefined_tensor_;
  DispatchKeySet                key_set_;
  DispatchKeySet                key_set() const {
    TORCH_CHECK(defined_);
    return key_set_;
  }
  caffe2::TypeMeta              dtype_;
  caffe2::TypeMeta              dtype() const {
    TORCH_CHECK(defined_);
    return dtype_;
  }
  c10::optional<Device>         optional_device_;
  c10::optional<Device>         optional_device() const {
    TORCH_CHECK(defined_);
    return optional_device_;
  }
  // A Tensor is evictable iff it's AliasPool is evictable.
  // A evictable tensor must have Rematerializer.
  intrusive_ptr<AliasPool>      pool_;
  intrusive_ptr<Rematerializer> remat_;
  void     evict() {
    TORCH_CHECK(remat_);
    t_.reset();
  }

  void     fill(const Tensor& t);
  explicit EntityTensorCell(const Tensor& t, const intrusive_ptr<AliasPool>& pool) 
          : pool_(pool)                { fill(t); }
  explicit EntityTensorCell(const Tensor& t,
                            const intrusive_ptr<AliasPool>& pool,
                            const intrusive_ptr<Rematerializer>& remat) 
          : pool_(pool), remat_(remat) { fill(t); }
  size_t   memory();
  Tensor   get();
  void     pin();
  void     release_resources() final {
    t_.reset();
    pool_.reset();
    remat_.reset();
  }
};


// An external reference.
// Each strong will have at most one external reference.
// By keeping such an invariant, whenever an external reference die,
// We know that the underlying strong is only used internally.
// Thus, when it die we can apply optimization like banishing/infinite staleness.
// We keep this invariant by only allowing EntityTensorImpl to make new External,
// When new EntityTensorImpl is constructed.
struct External : intrusive_ptr_target {
  External(const Strong& value) : value(value) {
    // value->pool->register_external();
  }
  External(const Tensor& value) :
          External(Strong::make(value, intrusive_ptr<AliasPool>::make(Unsafe(),
                                                                      intrusive_ptr<Rematerializer>(),
                                                                      -1))) { }
  External(const Tensor& value,
           const intrusive_ptr<AliasPool>& pool,
           const intrusive_ptr<Rematerializer>& remat) :
    External(Strong::make(value, pool, remat)) { }
  Strong value;
  void release_resources() override;
};


struct EntityTensorImpl : TensorImpl {
  
  static int counter;
  static int gen_counter()          { return counter++; }
  
  int                           id_ = gen_counter();
  std::string counter_name() const  { return std::string("[ETI") + std::to_string(id_) + "]"; }

  Ref<intrusive_ptr<External>>  ref;
  Tensor ref_tensor() const         { return ref->value->value->get(); }
  void release_resources() final;

  explicit EntityTensorImpl(const Ref<intrusive_ptr<External>>& ref);
  explicit EntityTensorImpl(const intrusive_ptr<External>& e);
  explicit EntityTensorImpl(const Tensor& t);

  static Tensors make(const std::string&  name,
                      const RematFunc_t&  remat,
                      const Tensors&      inputs);
  // mutate_idx indicate which of the inputs will get mutated.
  static void  mutate(const std::string&  name,
                      const MutateFunc_t& mutate,
                      const Tensors&      inputs,
                      const std::vector<size_t>& mutate_idx);

  intrusive_ptr<TensorImpl> shallow_copy_and_detach(const VariableVersion& version_counter,
                                                    bool allow_tensor_metadata_change) const override;
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override;

  int64_t size(int64_t d) const               { return ref_tensor().size(d); }
  int64_t stride(int64_t d) const             { return ref_tensor().stride(d); }

  int64_t dim_custom() const override         { return ref_tensor().dim(); }
  int64_t numel_custom() const override       { return ref_tensor().numel(); }
  IntArrayRef sizes_custom() const override   { return ref_tensor().sizes(); }
  IntArrayRef strides_custom() const override { return ref_tensor().strides(); }
  
  bool has_storage() const override           { return false; }

  const Storage& storage() const override     { return ref_tensor().storage(); }
  
  template <typename T>
  inline T* data_ptr_impl() const             { return ref_tensor().data_ptr(); }
};

inline EntityTensorImpl* get_entity_tensor_impl(const Tensor& t) {
  auto* eti = dynamic_cast<EntityTensorImpl*>(t.unsafeGetTensorImpl());
  TORCH_CHECK(eti != nullptr);
  return eti;
}

inline Ref<intrusive_ptr<External>> get_cell_from_tensor(const Tensor& t) {
  return get_entity_tensor_impl(t)->ref;
}

inline DispatchKeySet convert_key_set(const DispatchKeySet& t) {
  TORCH_CHECK(!t.has(DispatchKey::Checkpoint));
  auto ret = t.add(DispatchKey::Checkpoint);
  return ret;
}

TORCH_API void setDebugCallbackFunction(std::function<int64_t()>);

}