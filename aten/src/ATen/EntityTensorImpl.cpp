#include <ATen/EntityTensorImpl.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <chrono>
#include <string>
#include <random>
#include <cmath>

namespace at {

static std::function<int64_t()> debug_callback = []() -> int64_t { return 0; }; // do nothing

namespace native {
Tensor checkpoint(const Tensor& t) {
  auto eti = intrusive_ptr<EntityTensorImpl>::make(t);
  return Tensor(eti);
}

Tensor uncheckpoint(const Tensor& t) {
  return get_entity_tensor_impl(t)->ref_tensor();
}

Tensor decheckpoint(const Tensor& t) {
  auto* eti = dynamic_cast<EntityTensorImpl*>(t.unsafeGetTensorImpl());
  return eti ? eti->ref->value->value->get() : t;
}

bool is_checkpoint(const Tensor& t) {
  return dynamic_cast<EntityTensorImpl*>(t.unsafeGetTensorImpl()) != nullptr;
}

Tensor try_checkpoint(const Tensor& t) {
  return is_checkpoint(t) ? t : checkpoint(t);
}

bool evict_checkpoint(const Tensor& t) {
  if (!is_checkpoint(t)) return false;
  get_entity_tensor_impl(t)->ref->value->value->evict();
  return true;
}

Tensor remat_checkpoint(const Tensor& t) {
  return is_checkpoint(t) ? get_entity_tensor_impl(t)->ref->value->value->get() : t;
}


bool pageout_manual(const Tensor& t) {
  t.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->pageout_manual();
  return true;
}

bool pagein_manual(const Tensor& t) {
  t.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->pagein_manual();
  return true;
}

bool need_prefech(const Tensor& t) {
  t.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->need_prefetch();
  return true;
}

int64_t get_pointer(const Tensor&t) {
  return t.unsafeGetTensorImpl()->storage().unsafeGetStorageImpl()->entity().impl_->entity_id_;
}
} // namespace native 

[[inline]]
Tensor uncheckpoint(const Strong& input) {
  return input->get();
}

Tensors uncheckpoint(const Strongs& inputs) {
  Tensors ret;
  ret.reserve(inputs.size());
  for (const Strong& input : inputs)
    // inlined manually
    ret.push_back(input->get());
  return ret;
};

Tensors try_checkpoint(const Tensors& inputs) {
  Tensors ret;
  ret.reserve(inputs.size());
  for (const Tensor& input : inputs)
    ret.push_back(at::native::try_checkpoint(input));
  return ret;
}

void Rematerializer::remat() {
  // TODO: refactor using RAII for exception safety.
  for (const Strong& s : inputs)
    s->pool_->lock();
  Tensors ts = uncheckpoint(inputs);
  time_t pre_rematerialize  = std::chrono::system_clock::now();
  auto ret = func(ts);
  time_t post_rematerialize = std::chrono::system_clock::now();
  // pool.auto_evict();
  // remat_compute_time_ += (post_rematerialize - pre_rematerialize).count();
  TORCH_CHECK(ret.size() == outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i)
    if (auto output_cell = outputs[i].lock())
      output_cell->fill(ret[i]);
  // ecn.reset();
  for (const Strong& s : inputs)
    s->pool_->unlock();
}

void AliasPool::set_not_evicted(const intrusive_ptr<AliasPool>& self) {
  if (unlikely(is_evicted_)) {
    is_evicted_ = false;
    // if (ecn) {
    //   TORCH_CHECK(head_remat);
    //   auto cpi = get_t(ecn);
    //   update_t(ecn, CheckpointInfo(cpi.compute_cost - head_remat->compute_cost));
    //   ecn.reset();
    // }
    // pool.add(self);
  }
}

size_t EntityTensorCell::memory() {
  TORCH_CHECK(defined_);
  return pool_->memory_;
}

Tensor EntityTensorCell::get() {
  if (!t_) {
    TORCH_CHECK(remat_);
    remat_->remat();
  }
  TORCH_CHECK(t_);
  // TORCH_CHECK(! t->key_set().has(DispatchKey::CheckpointTensorId));
  pool_->update_last_used(); 
  return *t_;
}

void EntityTensorCell::pin() {
  get();
  pool_->head_remat_.reset();
  remat_.reset();
}

void EntityTensorCell::fill(const Tensor& t) {
  if (t_) return;
  t_ = std::make_unique<Tensor>(t.detach());
  if (!defined_) {
    defined_ = true;
    is_undefined_tensor_ = !t.defined();
    key_set_ = t.key_set();
    if (t.requires_grad()) {
      key_set_ = key_set_.add(DispatchKey::Autograd);
    }
    dtype_ = t.dtype();
    optional_device_ = t.optional_device();
  }
}

void External::release_resources() {
  value->pool_->release_external();
  value.reset();
}

int EntityTensorImpl::counter = 0;

EntityTensorImpl::EntityTensorImpl(const Ref<intrusive_ptr<External>>& ref) 
        : TensorImpl(convert_key_set(ref->value->value->key_set()),
              ref->value->value->dtype(),
              ref->value->value->optional_device()),
  ref(ref) {
  if (key_set().has(DispatchKey::Autograd)) 
    set_requires_grad(true);
  set_sizes_strides_policy(SizesStridesPolicy::CustomSizes);
  enable_remateriazation();
}

EntityTensorImpl::EntityTensorImpl(const intrusive_ptr<External>& e) 
        : EntityTensorImpl(Ref<intrusive_ptr<External>>::make(e)) { 
}

EntityTensorImpl::EntityTensorImpl(const Tensor& t) 
        : EntityTensorImpl(intrusive_ptr<External>::make(t)) {
}

intrusive_ptr<TensorImpl> EntityTensorImpl::shallow_copy_and_detach(const VariableVersion& version_counter,
                                                  bool allow_tensor_metadata_change) const {
  auto ret = intrusive_ptr<EntityTensorImpl>::make(ref);
  return ret;
}

void EntityTensorImpl::shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) {
  auto* eti = dynamic_cast<EntityTensorImpl*>(impl.get());
  ref->value = eti->ref->value;
}

bool is_alias(const Tensor& l, const Tensor& r) {
  return l.defined() && r.defined() && l.is_alias_of(r);
}

// return an index for alias.
// we dont care which one because they all lead to the same alias pool.
// return -1 for no alias.
int get_alias(const Tensors& ts, const Tensor& t) {
  if (!t.defined())
    for (size_t i = 0; i < ts.size(); ++i)
      if (ts[i].defined() && t.is_alias_of(ts[i])) return i;
  return -1;
}

struct MakeRawResult {
  std::vector<intrusive_ptr<External>> outputs;
  std::vector<int> aliases;
  duration_t time;
  intrusive_ptr<Rematerializer> rematerializer;
};

// void add_neighbor(const Strong& l, const Strong& r) {
//   l->pool_->neighbors.push_back(Weak(r));
//   r->pool_->neighbors.push_back(Weak(l));
// }

// remat take a single vector of tensors,
// while there are two vector, one storing nonconstants and one storing constants.
// the constants are small and they will not be considered for eviction.
// however, we have to stitch the two vectors together to pass it in remat.
// the size_t in constants decide the location to stitch them in, while input_values fill in the rest.
MakeRawResult make_raw(const RematFunc_t& remat_f,
                       const Strongs&     inputs) {
  for (const Strong& s : inputs)
    s->pool_->lock();
  Tensors raw_inputs = uncheckpoint(inputs);
  time_t pre_rematerialize  = std::chrono::system_clock::now();
  auto raw_outputs = remat_f(raw_inputs);
  time_t post_rematerialize = std::chrono::system_clock::now();
  // pool.auto_evict();
  // base_compute_time_ += (post - pre).count();
  std::vector<intrusive_ptr<External>>    outputs;
  std::vector<int>                        aliases;
  Weaks                                   weak_outputs;
  auto remat = intrusive_ptr<Rematerializer>::make(Unsafe(), remat_f, inputs, post_rematerialize - pre_rematerialize);

  for (const Tensor& t : raw_outputs) {
    intrusive_ptr<AliasPool> alias_pool;
    int alias = get_alias(raw_inputs, t);
    if (alias == -1) {
      // auto m = memory(t);
      alias_pool = intrusive_ptr<AliasPool>::make(Unsafe(), remat, -1);
      // pool.add(alias_pool)
    }
    else {
      alias_pool = inputs[alias]->pool_;
      if (alias_pool->head_remat_) {
        alias_pool->head_remat_->compute_cost += (post_rematerialize - pre_rematerialize);
      }
    }
    auto e = intrusive_ptr<External>::make(t, alias_pool, remat);
    // pool.exts.push_back(weak_intrusive_ptr<External>(e));
    alias_pool->tensors_.push_back(Weak(e->value));
    outputs.push_back(e);
    aliases.push_back(alias);
    weak_outputs.push_back(Weak(outputs.back()->value));
  }
  remat->outputs = weak_outputs;
  // for (size_t i = 0; i < inputs.size(); ++i) {
    // for (size_t j = 0; j < outputs.size(); ++j) {
      // if (!is_alias(raw_inputs[i], raw_outputs[j])) {
        // add_neighbor(inputs[i], outputs[j]->value);
      // }
    // }
  // }
  for (const Strong& s : inputs)
    s->pool_->unlock();
  return {outputs, aliases, post_rematerialize - pre_rematerialize, remat};
}

Tensors EntityTensorImpl::make(const std::string& name,
                               const RematFunc_t& remat,
                               const Tensors&     inputs) {
  Tensors checkpointed_inputs = try_checkpoint(inputs);
  auto input_size = checkpointed_inputs.size();

  Strongs input_values;
  input_values.reserve(input_size);

  std::vector<std::string> args;
  args.reserve(input_size);

  for (const Tensor& t: checkpointed_inputs)
    input_values.push_back(get_entity_tensor_impl(t)->ref->value->value);

  auto ret = make_raw(remat, input_values);

  Tensors tensors;
  tensors.reserve(ret.outputs.size());

  for (const auto& t: ret.outputs) {
    auto cp = Tensor(intrusive_ptr<EntityTensorImpl>::make(t));
    tensors.push_back(cp);
  }

  return tensors;
}

// TODO: check that mutated value does not have alias.
void EntityTensorImpl::mutate(const std::string& name,
                                  const MutateFunc_t& mutate,
                                  const Tensors& inputs,
                                  const std::vector<size_t>& mutate_idx) {
  auto remat = [=](const Tensors& t) -> Tensors {
                 Tensors new_input_values = t;
                 for (size_t idx: mutate_idx) {
                   new_input_values[idx] = t[idx].clone();
                 }
                 mutate(new_input_values);
                 return new_input_values;
               };
  Tensors checkpointed_inputs = try_checkpoint(inputs);
  Strongs input_values;
  std::vector<std::string> args;
  for (const Tensor& t: checkpointed_inputs)
    input_values.push_back(get_entity_tensor_impl(t)->ref->value->value);

  auto ret = make_raw(remat, input_values);
  const auto& modified = ret.outputs;
  for (size_t idx: mutate_idx) {
    get_cell_from_tensor(inputs[idx])->value = modified[idx];
  }
}

void EntityTensorImpl::release_resources() { 
  ref.reset(); 
}


void setDebugCallbackFunction(std::function<int64_t()> f_) { debug_callback = f_; }

} // namespace at