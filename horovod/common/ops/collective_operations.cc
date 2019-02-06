//
// Created by Travis Addair on 2018-12-14.
//

#include "collective_operations.h"

namespace horovod {
namespace common {

// Allreduce
AllreduceOp::AllreduceOp(CommunicationContext* comm_context, HorovodGlobalState* global_state)
    : comm_context_(comm_context), global_state_(global_state) {}

void AllreduceOp::Allreduce(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;

  if (entries.size() > 1) {
    // Access the fusion buffer.
    auto &buffer = global_state_->fusion_buffer.GetBuffer(
        first_entry.device, first_entry.context->framework());
    auto buffer_data = buffer->AccessData(first_entry.context);

    // Copy memory into the fusion buffer.
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    int64_t offset = 0;
    for (auto &e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyInFusionBuffer(buffer_data_at_offset, e, entries);
      offset += e.tensor->size();
    }

    StreamSynchronize(entries);
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, comm_context_->AllreduceActivity());
    int64_t num_elements = 0;
    for (auto& e : entries) {
      num_elements += e.tensor->shape().num_elements();
    }
    comm_context_->Allreduce(buffer_data, num_elements, first_entry);
    timeline.ActivityEndAll(entries);

    // Copy memory out of the fusion buffer.
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyOutFusionBuffer(buffer_data_at_offset, e, entries);
      offset += e.tensor->size();
    }

    StreamSynchronize(entries);
    timeline.ActivityEndAll(entries);
  } else {
    auto& e = first_entry;
    timeline.ActivityStartAll(entries, comm_context_->AllreduceActivity());
    const void* sendbuff = e.tensor->data() == e.output->data() ? nullptr : e.tensor->data();
    comm_context_->Allreduce((void*)e.output->data(), e.tensor->shape().num_elements(), first_entry, sendbuff);
    timeline.ActivityEndAll(entries);
  }

  for (auto& e : entries) {
    timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());
  }
}

void AllreduceOp::MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                        std::vector<TensorTableEntry>& entries) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t)e.tensor->size());
}

void AllreduceOp::MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                         std::vector<TensorTableEntry>& entries) {
  std::memcpy((void*)e.output->data(), buffer_data_at_offset,
              (size_t)e.tensor->size());
}

void AllreduceOp::StreamSynchronize(std::vector<TensorTableEntry>& entries) {
}

// Allgather
AllgatherOp::AllgatherOp(CommunicationContext* comm_context, HorovodGlobalState* global_state)
    : comm_context_(comm_context), global_state_(global_state) {}

void AllgatherOp::Allgather(std::vector<TensorTableEntry>& entries, const std::vector<int64_t>& tensor_sizes) {
  assert(entries.size() == 1);
  auto e = entries[0];
  auto& timeline = global_state_->timeline;

  // Copy tensor sizes from the MPI response into a vector of int64_t
  // and compute total size.  This is size of first dimension.
  int64_t total_dimension_size = 0;
  for (auto sz : tensor_sizes) {
    total_dimension_size += sz;
  }

  // Every tensor participating in Allgather operation may have different
  // first dimension size, but the rest of dimensions are same for all
  // tensors.  Here we get shape of tensor sliced by first dimension.
  TensorShape single_slice_shape;
  for (int i = 1; i < e.tensor->shape().dims(); ++i) {
    single_slice_shape.AddDim(e.tensor->shape().dim_size(i));
  }

  // Allgather output will have shape of:
  // (sum of first dimension of every tensor) x (tensor slice shape).
  TensorShape output_shape;
  output_shape.AddDim((int64_t)total_dimension_size);
  output_shape.AppendShape(single_slice_shape);

  timeline.ActivityStartAll(entries, ALLOCATE_OUTPUT);
  Status status = e.context->AllocateOutput(output_shape, &e.output);
  if (!status.ok()) {
    timeline.End(e.tensor_name, nullptr);
    e.callback(status);
    return;
  }
  timeline.ActivityEndAll(entries);

  // Compute all displacements and recvcounts
  auto* recvcounts = new int[tensor_sizes.size()];
  auto* displcmnts = new int[tensor_sizes.size()];
  for (unsigned int i = 0; i < tensor_sizes.size(); i++) {
    recvcounts[i] =
        (int)(single_slice_shape.num_elements() * tensor_sizes[i]);
    if (i == 0) {
      displcmnts[i] = 0;
    } else {
      displcmnts[i] = displcmnts[i - 1] + recvcounts[i - 1];
    }
  }

  int element_size;
  comm_context_->GetTypeSize(e.tensor->dtype(), &element_size);

  int64_t total_size = recvcounts[tensor_sizes.size() - 1] +
                       displcmnts[tensor_sizes.size() - 1];

  bool on_gpu_no_mpi = false;
#if HAVE_CUDA
  on_gpu_no_mpi = e.device != CPU_DEVICE_ID;
#if HOROVOD_GPU_ALLGATHER == 'M'   // 'M' stands for MPI
    on_gpu_no_mpi = false;
#endif
#endif
  // on_gpu_no_mpi == true if the data is in GPU but Horovod was not
  // compiled with CUDA-aware MPI

  if (on_gpu_no_mpi || global_state_->param_manager.HierarchicalAllgather()) {
    // If shared buffer is not initialized or is not large enough, reallocate
    if (global_state_->shared_buffer == nullptr ||
        global_state_->shared_buffer_size < total_size * element_size) {
      if (global_state_->shared_buffer != nullptr) {
        comm_context_->FreeSharedBuffer();
        global_state_->shared_buffer = nullptr;
      }
      int64_t window_size =
          global_state_->local_rank == 0 ? total_size * element_size : 0;

      // Allocate shared memory, give each rank their respective pointer
      timeline.ActivityStartAll(entries, ALLOCATE_SHARED_BUFFER);
      comm_context_->AllocateSharedBuffer(window_size, element_size, &global_state_->shared_buffer,
                                          CommunicationContext::Communicator::LOCAL);
      if (global_state_->local_rank != 0) {
        comm_context_->QuerySharedBuffer(0, &global_state_->shared_buffer);
      }
      global_state_->shared_buffer_size = total_size * element_size;
      timeline.ActivityEndAll(entries);
    }

    // Compute cross-node allgather displacements and recvcounts for
    // homogeneous/parallelized case
    auto* cross_recvcounts =  new int[global_state_->cross_size]();
    auto* cross_displcmnts =  new int[global_state_->cross_size]();

    if (global_state_->is_homogeneous) {
      for (int i = 0; i < global_state_->cross_size; i++) {
        cross_recvcounts[i] = recvcounts[global_state_->local_size * i +
                                         global_state_->local_rank];
        cross_displcmnts[i] = displcmnts[global_state_->local_size * i +
                                         global_state_->local_rank];
      }
    } else if (global_state_->local_rank == 0) {
      // In this case local rank 0 will allgather with all local data
      int offset = 0;
      for (int i = 0; i < global_state_->cross_size; i++) {
        for (int j = offset; j < offset + global_state_->local_sizes[i]; j++) {
          cross_recvcounts[i] += recvcounts[j];
        }
        cross_displcmnts[i] = displcmnts[offset];
        offset += global_state_->local_sizes[i];
      }
    }

    int64_t copy_len = recvcounts[global_state_->rank] * element_size;
    void* shared_buffer_at_offset =
        (uint8_t*)global_state_->shared_buffer +
        displcmnts[global_state_->rank] * element_size;

#if HAVE_CUDA
    // If data is on GPU, create CUDA stream if needed and copy data into
      // shared buffer with appropriate offset
      if (on_gpu_no_mpi) {
        cudaStream_t& stream = horovod_global.streams[e.device];
        CUDA_CHECK(entries, "cudaSetDevice", cudaSetDevice(e.device))

        if (stream == nullptr) {
          int greatest_priority;
          CUDA_CHECK(entries, "cudaDeviceGetStreamPriorityRange",
                     cudaDeviceGetStreamPriorityRange(NULL, &greatest_priority))
          CUDA_CHECK(entries, "cudaStreamCreateWithPriority",
                     cudaStreamCreateWithPriority(
                         &stream, cudaStreamNonBlocking, greatest_priority))
        }
        auto event_queue = std::queue<std::pair<std::string, cudaEvent_t>>();

        // Copy to shared buffer at the cpu
        CUDA_CHECK(entries, "cudaMemcpyAsync",
                   cudaMemcpyAsync(shared_buffer_at_offset, e.tensor->data(),
                                   copy_len, cudaMemcpyDeviceToHost, stream))

        if (timeline.Initialized()) {
          RECORD_EVENT(entries, event_queue, MEMCPY_IN_HOST_BUFFER, stream)
        }
        WAIT_FOR_EVENTS(entries, timeline, event_queue)
      } else {
#endif
    // CPU copy to shared buffer
    timeline.ActivityStartAll(entries, MEMCPY_IN_SHARED_BUFFER);
    memcpy(shared_buffer_at_offset, e.tensor->data(), copy_len);
    comm_context_->Barrier(CommunicationContext::Communicator::GLOBAL);
    timeline.ActivityEndAll(entries);
#if HAVE_CUDA
    }
#endif
    // Perform the cross-node allgather. If the cluster is homogeneous all
    // local ranks participate, otherwise local rank 0 handles all data
    timeline.ActivityStartAll(entries, MPI_CROSS_ALLGATHER);
    if (global_state_->is_homogeneous || global_state_->local_rank == 0) {
      comm_context_->Allgatherv(nullptr, 0, DataType::HOROVOD_NULL, global_state_->shared_buffer,
                                cross_recvcounts, cross_displcmnts, e.tensor->dtype(),
                                CommunicationContext::Communicator::CROSS);
    }
    comm_context_->Barrier(CommunicationContext::Communicator::GLOBAL);
    timeline.ActivityEndAll(entries);

#if HAVE_CUDA
    if (on_gpu_no_mpi) {
        cudaStream_t& stream = horovod_global.streams[e.device];
        auto event_queue = std::queue<std::pair<std::string, cudaEvent_t>>();

        // Copy back to the output buffer at the gpu
        CUDA_CHECK(entries, "cudaMemcpyAsync",
                   cudaMemcpyAsync((void*)e.output->data(),
                                   horovod_global.shared_buffer,
                                   (size_t)(total_size * element_size),
                                   cudaMemcpyHostToDevice, stream))

        if (timeline.Initialized()) {
          RECORD_EVENT(entries, event_queue, MEMCPY_OUT_HOST_BUFFER, stream)
        }
        WAIT_FOR_EVENTS(entries, timeline, event_queue)
      } else {
#endif
    // Copy the result from MPI shared memory to rank-specific output buffer
    timeline.ActivityStartAll(entries, COPY_ALLGATHER_OUTPUT);
    memcpy((void*)e.output->data(), global_state_->shared_buffer,
           total_size * element_size);
    timeline.ActivityEndAll(entries);
#if HAVE_CUDA
    }
#endif
    // Free the buffers
    delete[] cross_displcmnts;
    delete[] cross_recvcounts;
  } else {
    // Data is at the CPU and hierarchical allgather is disabled, or
    // Data is at the GPU and HOROVOD_GPU_ALLGATHER == MPI
    timeline.ActivityStartAll(entries, MPI_ALLGATHER);
    comm_context_->Allgatherv(e.tensor->data(), (int)e.tensor->shape().num_elements(),
                              e.tensor->dtype(), (void*)e.output->data(), recvcounts,
                              displcmnts, e.tensor->dtype(),
                              CommunicationContext::Communicator::GLOBAL);
    timeline.ActivityEndAll(entries);
  }
  delete[] recvcounts;
  delete[] displcmnts;
  timeline.End(e.tensor_name, e.output);
  e.callback(Status::OK());
}

} // namespace common
} // namespace horovod
