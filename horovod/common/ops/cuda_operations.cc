//
// Created by Travis Addair on 2018-12-14.
//

#include "cuda_operations.h"

#include <thread>

namespace horovod {
namespace common {

// This event management code is only used with CUDA
cudaError_t CUDAContext::GetCudaEvent(cudaEvent_t* event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = cuda_events[device];
    if (!queue.empty()) {
      *event = queue.front();
      queue.pop();
      return cudaSuccess;
    }
  }

  return cudaEventCreateWithFlags(event, cudaEventBlockingSync |
                                         cudaEventDisableTiming);
}

cudaError_t CUDAContext::ReleaseCudaEvent(cudaEvent_t event) {
  int device;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }

  auto& mutex = cuda_events_mutex;
  {
    std::lock_guard<std::mutex> guard(mutex);
    auto& queue = cuda_events[device];
    queue.push(event);
  }

  return cudaSuccess;
}

bool CUDAOperation::OnGPU_NoMPI(TensorTableEntry e) const {
  return e.device != CPU_DEVICE_ID;
}

CUDAAllreduce::CUDAAllreduce(CUDAContext* context,
                             CommunicationContext* comm_context,
                             HorovodGlobalState* global_state)
                             : AllreduceOp(comm_context, global_state), cuda_context_(context) {}

void CUDAAllreduce::Allreduce(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices) {
  if (OnGPU(entries)) {
    InitCUDA(entries);
  }
  AllreduceOp::Allreduce(entries, devices);
}

CUDACustomAllreduce::CUDACustomAllreduce(CUDAContext* context,
                                         CommunicationContext* comm_context,
                                         HorovodGlobalState* global_state)
: CUDAAllreduce(context, comm_context, global_state) {}

void CUDACustomAllreduce::Allreduce(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices) {
  if (!OnGPU(entries)) {
    AllreduceOp::Allreduce(entries, devices);
    return;
  }

  InitCUDA(entries);

  auto& first_entry = entries[0];
  auto stream = cuda_context_->streams[first_entry.device];
  auto event_queue = std::queue<std::pair<std::string, cudaEvent_t>>();

  InitComm();

  auto& timeline = global_state_->timeline;
  if (timeline.Initialized()) {
    RECORD_EVENT(entries, timeline, event_queue, QUEUE, stream)
  }

  // If entries.size() > 1, we copy tensors into fusion buffer before
  // allreduce, and distribute results of allreduce back into target
  // tensors after allreduce.

  const void* fused_input_data;
  void* buffer_data;
  int64_t num_elements = 0;
  size_t buffer_len;
  if (entries.size() > 1) {
    // Access the fusion buffer.
    auto& buffer = global_state_->fusion_buffer.GetBuffer(
        first_entry.device, first_entry.context->framework());
    buffer_data =
        const_cast<void*>(buffer->AccessData(first_entry.context));

    // Copy memory into the fusion buffer.
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      CUDA_CHECK(entries, timeline, "cudaMemcpyAsync",
                 cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                                 (size_t)e.tensor->size(),
                                 cudaMemcpyDeviceToDevice, stream))
      offset += e.tensor->size();
    }

    buffer_len = (size_t)offset;

    if (timeline.Initialized() || global_state_->ddl_initialized) {
      RECORD_EVENT(entries, timeline, event_queue, MEMCPY_IN_FUSION_BUFFER, stream)
    }

    // Set the input data to originate from the buffer.
    fused_input_data = buffer_data;

    // Perform the reduction on the fusion buffer.
    for (auto& e : entries) {
      num_elements += e.tensor->shape().num_elements();
    }

  } else {
    fused_input_data = first_entry.tensor->data();
    buffer_data = (void*)first_entry.output->data();
    num_elements = first_entry.tensor->shape().num_elements();
    buffer_len = (size_t)first_entry.output->size();

    if (global_state_->ddl_initialized) {
      // Copy input buffer content to output buffer
      // because DDL only supports in-place allreduce
      CUDA_CHECK(entries, timeline, "cudaMemcpyAsync",
                 cudaMemcpyAsync(buffer_data, fused_input_data, buffer_len,
                                 cudaMemcpyDeviceToDevice, stream))
      RECORD_EVENT(entries, timeline, event_queue, MEMCPY_IN_FUSION_BUFFER, stream)
    }
  }

  void* host_buffer = nullptr;
  CustomAllreduce(entries, num_elements, buffer_len);

  if (entries.size() > 1) {
    // Copy memory out of the fusion buffer.
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      CUDA_CHECK(entries, timeline, "cudaMemcpyAsync",
                 cudaMemcpyAsync((void*)e.output->data(),
                                 buffer_data_at_offset,
                                 (size_t)e.tensor->size(),
                                 cudaMemcpyDeviceToDevice, stream))
      offset += e.tensor->size();
    }
    if (timeline.Initialized()) {
      RECORD_EVENT(entries, timeline, event_queue, MEMCPY_OUT_FUSION_BUFFER, stream)
    }
  }

  // Use completion marker via event because it's faster than
  // blocking cudaStreamSynchronize() in this thread.
  RECORD_EVENT(entries, timeline, event_queue, "", stream)

  // TODO: use thread pool or single thread for callbacks
  std::thread finalizer_thread([entries, first_entry, host_buffer,
                                event_queue, &timeline]() mutable {
    CUDA_CHECK(entries, timeline, "cudaSetDevice", cudaSetDevice(first_entry.device))

    WAIT_FOR_EVENTS(entries, timeline, event_queue)

    if (host_buffer != nullptr) {
      free(host_buffer);
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }
  });

  finalizer_thread.detach();
}

bool CUDAAllreduce::OnGPU(std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  return first_entry.device != CPU_DEVICE_ID;
}

void CUDAAllreduce::InitCUDA(std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  CUDA_CHECK(entries, global_state_->timeline, "cudaSetDevice", cudaSetDevice(first_entry.device))

  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = cuda_context_->streams[first_entry.device];
  if (stream == nullptr) {
    int greatest_priority;
    CUDA_CHECK(entries, global_state_->timeline, "cudaDeviceGetStreamPriorityRange",
               cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority))
    CUDA_CHECK(entries, global_state_->timeline, "cudaStreamCreateWithPriority",
               cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                            greatest_priority))
  }
}

void CUDAAllreduce::MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                         std::vector<TensorTableEntry>& entries) {
  if (OnGPU(entries)) {
    auto& first_entry = entries[0];
    CUDA_CHECK(entries, global_state_->timeline, "cudaMemcpyAsync",
               cudaMemcpyAsync(
                   buffer_data_at_offset, e.tensor->data(),
                   (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                   cuda_context_->streams[first_entry.device]))
  } else {
    AllreduceOp::MemcpyInFusionBuffer(buffer_data_at_offset, e, entries);
  }
}

void CUDAAllreduce::MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                          std::vector<TensorTableEntry>& entries) {
  if (OnGPU(entries)) {
    auto& first_entry = entries[0];
    CUDA_CHECK(entries, global_state_->timeline, "cudaMemcpyAsync",
               cudaMemcpyAsync(
                   (void*)e.output->data(), buffer_data_at_offset,
                   (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                   cuda_context_->streams[first_entry.device]))
  } else {
    AllreduceOp::MemcpyOutFusionBuffer(buffer_data_at_offset, e, entries);
  }
}

void CUDAAllreduce::StreamSynchronize(std::vector<TensorTableEntry>& entries) {
  if (OnGPU(entries)) {
    auto& first_entry = entries[0];
    CUDA_CHECK(
        entries, global_state_->timeline, "cudaStreamSynchronize",
        cudaStreamSynchronize(cuda_context_->streams[first_entry.device]))
  }
}

} // namespace common
} // namespace horovod
