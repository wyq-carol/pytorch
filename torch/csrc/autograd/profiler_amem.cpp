
// #include <ATen/ATen.h>
// #include <c10/core/StorageImpl.h>
//#include <ATen/native/Copy.h>
//#include <ATen/Dispatch.h>
//#include <ATen/NativeFunctions.h>
#include <torch/csrc/autograd/profiler_amem.h>

namespace torch{ namespace automem {

static AMemProfiler amem_profiler;
TORCH_API AMemProfiler* GetAutoMemProfiler() { return &amem_profiler; }

}}
