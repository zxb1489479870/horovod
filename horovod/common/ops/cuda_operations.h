//
// Created by Travis Addair on 2018-12-14.
//

#ifndef HOROVOD_CUDA_OPERATIONS_H
#define HOROVOD_CUDA_OPERATIONS_H

#include <queue>
#include <unordered_map>

#include <cuda_runtime.h>

#include "collective_operations.h"

namespace horovod {
namespace common {

#define CUDA_CHECK(entries, timeline, op_name, op)                             \
  {                                                                            \
    auto cuda_result = (op);                                                   \
    if (cuda_result != cudaSuccess) {                                          \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(std::string(op_name) + " failed: " +   \
                                        cudaGetErrorString(cuda_result)));     \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define RECORD_EVENT(entries, timeline, event_queue, name, stream)                    \
  {                                                                                   \
    cudaEvent_t event;                                                                \
    CUDA_CHECK(entries, timeline, "GetCudaEvent", GetCudaEvent(&event))               \
    CUDA_CHECK(entries, timeline, "cudaEventRecord", cudaEventRecord(event, stream))  \
    (event_queue).emplace(name, event);                                               \
  }

#define WAIT_FOR_EVENTS(entries, timeline, event_queue)                               \
  {                                                                                   \
    while (!(event_queue).empty()) {                                                  \
      std::string name;                                                               \
      cudaEvent_t event;                                                              \
      std::tie(name, event) = (event_queue).front();                                  \
      (event_queue).pop();                                                            \
      if (name != "") {                                                               \
        timeline.ActivityStartAll(entries, name);                                     \
      }                                                                               \
      CUDA_CHECK(entries, timeline, "cudaEventSynchronize", cudaEventSynchronize(event)) \
      if (name != "") {                                                               \
        timeline.ActivityEndAll(entries);                                             \
      }                                                                               \
      CUDA_CHECK(entries, timeline, "ReleaseCudaEvent", ReleaseCudaEvent(event))         \
    }                                                                                 \
  }

struct CUDAContext {
  cudaError_t GetCudaEvent(cudaEvent_t* event);

  cudaError_t ReleaseCudaEvent(cudaEvent_t event);

  // The CUDA stream used for data transfers and within-allreduce operations.
  // A naive implementation would use the TensorFlow StreamExecutor CUDA
  // stream. However, the allreduce and allgather require doing memory copies
  // and kernel executions (for accumulation of values on the GPU). However,
  // the subsequent operations must wait for those operations to complete,
  // otherwise MPI (which uses its own stream internally) will begin the data
  // transfers before the CUDA calls are complete. In order to wait for those
  // CUDA operations, if we were using the TensorFlow stream, we would have to
  // synchronize that stream; however, other TensorFlow threads may be
  // submitting more work to that stream, so synchronizing on it can cause the
  // allreduce to be delayed, waiting for compute totally unrelated to it in
  // other parts of the graph. Overlaying memory transfers and compute during
  // backpropagation is crucial for good performance, so we cannot use the
  // TensorFlow stream, and must use our own stream.
  std::unordered_map<int, cudaStream_t> streams;

  // We reuse CUDA events as it appears that their creation carries non-zero cost.
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex cuda_events_mutex;
};

class CUDAAllreduce : public AllreduceOp {
public:
  CUDAAllreduce(CUDAContext* context,
                CommunicationContext* comm_context,
                HorovodGlobalState* global_state);
  void Allreduce(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices) override;

protected:
  void MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                            std::vector<TensorTableEntry>& entries) override;
  void MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                             std::vector<TensorTableEntry>& entries) override;
  void StreamSynchronize(std::vector<TensorTableEntry>& entries) override;

  void InitCUDA(std::vector<TensorTableEntry>& entries);
  bool OnGPU(std::vector<TensorTableEntry>& entries);

  struct CUDAContext* cuda_context_;
};

class CUDACustomAllreduce : public CUDAAllreduce {
public:
  CUDACustomAllreduce(CUDAContext* context,
                      CommunicationContext* comm_context,
                      HorovodGlobalState* global_state);
  void Allreduce(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices) override;

protected:
  virtual void InitComm(std::vector<TensorTableEntry>& entries, std::vector<int32_t>& devices) = 0;
  virtual void CustomAllreduce(std::vector<TensorTableEntry>& entries,
                               cudaStream_t& stream, std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                               const void* fused_input_data, void* buffer_data,
                               int64_t& num_elements, size_t& buffer_len) = 0;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_CUDA_OPERATIONS_H
