// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

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

void CUDAContext::ErrorCheck(std::string op_name, cudaError_t cuda_result) {
  if (cuda_result != cudaSuccess) {
    throw std::logic_error(std::string(op_name) + " failed: " + cudaGetErrorString(cuda_result));
  }
}

void CUDAContext::RecordEvent(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                              std::string name, cudaStream_t stream) {
  cudaEvent_t event;
  ErrorCheck("GetCudaEvent", GetCudaEvent(&event));
  ErrorCheck("cudaEventRecord", cudaEventRecord(event, stream));
  event_queue.emplace(name, event);
}

void CUDAContext::WaitForEvents(std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                                std::vector<TensorTableEntry>& entries, Timeline& timeline) {
  while (!(event_queue).empty()) {
    std::string name;
    cudaEvent_t event;
    std::tie(name, event) = (event_queue).front();
    (event_queue).pop();
    if (name != "") {
      timeline.ActivityStartAll(entries, name);
    }
    ErrorCheck("cudaEventSynchronize", cudaEventSynchronize(event));
    if (name != "") {
      timeline.ActivityEndAll(entries);
    }
    ErrorCheck("ReleaseCudaEvent", ReleaseCudaEvent(event));
  }
}

CUDAAllreduce::CUDAAllreduce(CUDAContext* context,
                             CommunicationContext* comm_context,
                             HorovodGlobalState* global_state)
                             : AllreduceOp(comm_context, global_state), cuda_context_(context) {}

Status CUDAAllreduce::Execute(std::vector<TensorTableEntry>& entries, const HorovodResponse& response) {
  if (OnGPU(entries)) {
    InitCUDA(entries);
  }
  return AllreduceOp::Execute(entries, response);
}

CUDACustomAllreduce::CUDACustomAllreduce(CUDAContext* context,
                                         CommunicationContext* comm_context,
                                         HorovodGlobalState* global_state)
: CUDAAllreduce(context, comm_context, global_state) {}

Status CUDACustomAllreduce::Execute(std::vector<TensorTableEntry>& entries, const HorovodResponse& response) {
  if (!OnGPU(entries)) {
    return AllreduceOp::Execute(entries, response);
  }

  InitCUDA(entries);

  auto& first_entry = entries[0];
  auto stream = cuda_context_->streams[first_entry.device];
  auto event_queue = std::queue<std::pair<std::string, cudaEvent_t>>();

  InitComm(entries, response.devices());

  auto& timeline = global_state_->timeline;
  if (timeline.Initialized()) {
    cuda_context_->RecordEvent(event_queue, QUEUE, stream);
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
      auto cuda_result = cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                                         (size_t)e.tensor->size(),
                                         cudaMemcpyDeviceToDevice, stream);
      cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
      offset += e.tensor->size();
    }

    buffer_len = (size_t)offset;

    if (timeline.Initialized() || global_state_->ddl_initialized) {
      cuda_context_->RecordEvent(event_queue, MEMCPY_IN_FUSION_BUFFER, stream);
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
      auto cuda_result = cudaMemcpyAsync(buffer_data, fused_input_data, buffer_len,
                                         cudaMemcpyDeviceToDevice, stream);
      cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
      cuda_context_->RecordEvent(event_queue, MEMCPY_IN_FUSION_BUFFER, stream);
    }
  }

  void* host_buffer = nullptr;
  CustomAllreduce(entries, stream, event_queue, fused_input_data, buffer_data, num_elements, buffer_len, host_buffer);

  if (entries.size() > 1) {
    // Copy memory out of the fusion buffer.
    int64_t offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*)buffer_data + offset;
      auto cuda_result = cudaMemcpyAsync((void*)e.output->data(),
                                         buffer_data_at_offset,
                                         (size_t)e.tensor->size(),
                                         cudaMemcpyDeviceToDevice, stream);
      cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
      offset += e.tensor->size();
    }
    if (timeline.Initialized()) {
      cuda_context_->RecordEvent(event_queue, MEMCPY_OUT_FUSION_BUFFER, stream);
    }
  }

  // Use completion marker via event because it's faster than
  // blocking cudaStreamSynchronize() in this thread.
  cuda_context_->RecordEvent(event_queue, "", stream);

  // TODO: use thread pool or single thread for callbacks
  std::thread finalizer_thread([entries, first_entry, host_buffer,
                                event_queue, &timeline, this]() mutable {
    auto cuda_result = cudaSetDevice(first_entry.device);
    cuda_context_->ErrorCheck("cudaSetDevice", cuda_result);

    cuda_context_->WaitForEvents(event_queue, entries, timeline);
    if (host_buffer != nullptr) {
      free(host_buffer);
    }

    for (auto& e : entries) {
      timeline.End(e.tensor_name, e.output);
      e.callback(Status::OK());
    }
  });

  finalizer_thread.detach();

  return Status::Finalizing();
}

bool CUDAAllreduce::OnGPU(std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  return first_entry.device != CPU_DEVICE_ID;
}

void CUDAAllreduce::InitCUDA(std::vector<TensorTableEntry>& entries) {
  auto& first_entry = entries[0];
  cuda_context_->ErrorCheck("cudaSetDevice", cudaSetDevice(first_entry.device));

  // Ensure stream is in the map before executing reduction.
  cudaStream_t& stream = cuda_context_->streams[first_entry.device];
  if (stream == nullptr) {
    int greatest_priority;
    cuda_context_->ErrorCheck("cudaDeviceGetStreamPriorityRange",
                              cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority));
    cuda_context_->ErrorCheck("cudaStreamCreateWithPriority",
                              cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest_priority));
  }
}

void CUDAAllreduce::MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                         std::vector<TensorTableEntry>& entries) {
  if (OnGPU(entries)) {
    auto& first_entry = entries[0];
    auto cuda_result = cudaMemcpyAsync(buffer_data_at_offset, e.tensor->data(),
                                       (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                                       cuda_context_->streams[first_entry.device]);
    cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
  } else {
    AllreduceOp::MemcpyInFusionBuffer(buffer_data_at_offset, e, entries);
  }
}

void CUDAAllreduce::MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                          std::vector<TensorTableEntry>& entries) {
  if (OnGPU(entries)) {
    auto& first_entry = entries[0];
    auto cuda_result = cudaMemcpyAsync((void*)e.output->data(), buffer_data_at_offset,
                                       (size_t)e.tensor->size(), cudaMemcpyDeviceToDevice,
                                       cuda_context_->streams[first_entry.device]);
    cuda_context_->ErrorCheck("cudaMemcpyAsync", cuda_result);
  } else {
    AllreduceOp::MemcpyOutFusionBuffer(buffer_data_at_offset, e, entries);
  }
}

void CUDAAllreduce::StreamSynchronize(std::vector<TensorTableEntry>& entries) {
  if (OnGPU(entries)) {
    auto& first_entry = entries[0];
    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[first_entry.device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);
  }
}

} // namespace common
} // namespace horovod
