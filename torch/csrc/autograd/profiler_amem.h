#pragma once
// #include <torch/csrc/python_headers.h>
#include <vector>
#include <utility>
#include <memory>
#include <map>
#include <cstdint>
#include <torch/csrc/autograd/function.h>
// #include <torch/csrc/utils/object_ptr.h>
// #include <torch/csrc/Exceptions.h>

namespace torch{ namespace automem{
    
struct AMemProfiler{
public:
  AMemProfiler() = default;
  void init() { grad_execution_time.clear(); }
  std::map<int32_t,int32_t> grad_execution_time;
private:
  int32_t grad_fn_nums;
};

TORCH_API AMemProfiler* GetAutoMemProfiler();

}}
