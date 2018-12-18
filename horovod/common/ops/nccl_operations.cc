//
// Created by Travis Addair on 2018-12-14.
//

#include "nccl_operations.h"

namespace horovod {
namespace common {

ncclDataType_t GetNCCLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
    case HOROVOD_INT32:
      return ncclInt32;
    case HOROVOD_INT64:
      return ncclInt64;
    case HOROVOD_FLOAT16:
      return ncclFloat16;
    case HOROVOD_FLOAT32:
      return ncclFloat32;
    case HOROVOD_FLOAT64:
      return ncclFloat64;
    default:
      throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                             " is not supported in NCCL mode.");
  }
}

NCCLAllreduce::NCCLAllreduce(horovod::common::NCCLContext* nccl_context,
                             horovod::common::CUDAContext* cuda_context,
                             horovod::common::HorovodGlobalState* global_state)
                             : CUDACustomAllreduce(cuda_context, global_state), nccl_context_(nccl_context) {}

void NCCLAllreduce::InitComm(std::vector<TensorTableEntry>& entries, std::vector<int32_t>& devices) {
  // Determine GPU IDs of the devices participating in this communicator.
  std::vector<int32_t> nccl_device_map = GetDeviceMap(devices);

  // Ensure NCCL communicator is in the map before executing reduction.
  ncclComm_t& nccl_comm = nccl_context_->nccl_comms[nccl_device_map];
  if (nccl_comm == nullptr) {
    auto& timeline = global_state_->timeline;
    timeline.ActivityStartAll(entries, INIT_NCCL);

    int nccl_rank, nccl_size;
    MPI_Comm nccl_id_bcast_comm;
    SetCommStrategy(nccl_rank, nccl_size, nccl_id_bcast_comm);

    ncclUniqueId nccl_id;
    if (nccl_rank == 0) {
      NCCL_CHECK(entries, timeline, "ncclGetUniqueId", ncclGetUniqueId(&nccl_id))
    }

    MPI_CHECK(entries, timeline, "MPI_Bcast",
              MPI_Bcast((void*)&nccl_id, sizeof(nccl_id), MPI_BYTE, 0,
                        nccl_id_bcast_comm));

    ncclComm_t new_nccl_comm;
    NCCL_CHECK(
        entries, timeline, "ncclCommInitRank",
        ncclCommInitRank(&new_nccl_comm, nccl_size, nccl_id, nccl_rank))
    nccl_comm = new_nccl_comm;

    // Barrier helps NCCL to synchronize after initialization and avoid
    // deadlock that we've been seeing without it.
    MPI_CHECK(entries, timeline, "MPI_Barrier", MPI_Barrier(global_state_->mpi_comm));

    timeline.ActivityEndAll(entries);
  }
}

void NCCLAllreduce::CustomAllreduce(std::vector<TensorTableEntry>& entries,
                                    cudaStream_t& stream,
                                    std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                                    const void* fused_input_data, void* buffer_data,
                                    int64_t& num_elements, size_t& buffer_len) {
  auto& first_entry = entries[0];
  NCCL_CHECK(entries, global_state_->timeline, "ncclAllReduce",
             ncclAllReduce(fused_input_data, buffer_data,
                           (size_t)num_elements,
                           GetNCCLDataType(first_entry.tensor), ncclSum,
                           nccl_comm, stream))
  if (global_state_->timeline.Initialized()) {
    RECORD_EVENT(entries, global_state_->timeline, event_queue, NCCL_ALLREDUCE, stream)
  }
}

std::vector<int32_t> NCCLAllreduce::GetDeviceMap(std::vector<int32_t>& devices) {
  return devices;
}

void NCCLAllreduce::SetCommStrategy(int& nccl_rank, int& nccl_size, MPI_Comm& nccl_id_bcast_comm) {
  nccl_rank = global_state_->rank;
  nccl_size = global_state_->size;
  nccl_id_bcast_comm = global_state_->mpi_comm;
}

HierarchicalAllreduce::HierarchicalAllreduce(horovod::common::NCCLContext* nccl_context,
                                             horovod::common::CUDAContext* cuda_context,
                                             horovod::common::HorovodGlobalState* global_state)
                                             : NCCLAllreduce(nccl_context, cuda_context, global_state) {}

void HierarchicalAllreduce::CustomAllreduce(std::vector<TensorTableEntry>& entries,
                                            cudaStream_t& stream,
                                            std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                                            const void* fused_input_data, void* buffer_data,
                                            int64_t& num_elements, size_t& buffer_len) {
  auto& first_entry = entries[0];
  int element_size;
  MPI_Type_size(global_state_->GetMPIDataType(first_entry.tensor), &element_size);

  // If cluster is homogeneous and we are using fusion buffer, include
  // dummy elements from the buffer (if necessary) to make sure the data
  // is divisible by local_size. This is always possible since we
  // set the fusion buffer size divisible by local_size.
  if (global_state_->is_homogeneous && entries.size() > 1) {
    // Making sure the number of elements is divisible by
    // FUSION_BUFFER_ATOMIC_UNIT for improved performance
    int div = global_state_->local_size * FUSION_BUFFER_ATOMIC_UNIT;
    num_elements = ((num_elements + div - 1) / div) * div;
    buffer_len = num_elements * element_size;
  }

  // Split the elements into two groups: num_elements_per_rank*local_size,
  // and num_elements_remaining. Cross-node reduction for the first group
  // is done by all local_rank's in parallel, while for the second group
  // it it is only done by the root_rank. If the cluster is not
  // homogeneous first group is zero, and root_rank is 0.

  // Homogeneous case:
  // For the part of data divisible by local_size, perform NCCL
  // ReduceScatter - Parallelized MPI Allreduce - NCCL Allgather. For the
  // non-divisible part (if any), do NCCL Reduce (at rank local_size-1),
  // MPI Allreduce (across rank (local_size-1)'s), and NCCL Bcast

  int64_t num_elements_per_rank =
      global_state_->is_homogeneous
      ? num_elements / global_state_->local_size
      : 0;

  size_t buffer_len_per_rank = element_size * num_elements_per_rank;

  void* buffer_data_at_rank_offset =
      (uint8_t*)buffer_data +
      buffer_len_per_rank * global_state_->local_rank;

  int64_t num_elements_remaining =
      global_state_->is_homogeneous
      ? num_elements % global_state_->local_size
      : num_elements;

  size_t buffer_len_remaining = element_size * num_elements_remaining;

  void* buffer_data_remainder =
      (uint8_t*)buffer_data +
      buffer_len_per_rank * global_state_->local_size;

  void* fused_input_data_remainder =
      (uint8_t*)fused_input_data +
      buffer_len_per_rank * global_state_->local_size;

  int root_rank =
      global_state_->is_homogeneous ? global_state_->local_size - 1 : 0;
  bool is_root_rank = global_state_->local_rank == root_rank;

  int64_t total_num_elements =
      is_root_rank ? num_elements_per_rank + num_elements_remaining
                   : num_elements_per_rank;
  int64_t total_buffer_len =
      is_root_rank ? buffer_len_per_rank + buffer_len_remaining
                   : buffer_len_per_rank;

  auto& timeline = global_state_->timeline;
  if (num_elements_per_rank > 0) {
    NCCL_CHECK(entries, timeline, "ncclReduceScatter",
               ncclReduceScatter(fused_input_data,
                                 buffer_data_at_rank_offset,
                                 (size_t)num_elements_per_rank,
                                 GetNCCLDataType(first_entry.tensor),
                                 ncclSum, nccl_comm, stream))

    if (timeline.Initialized()) {
      RECORD_EVENT(entries, timeline, event_queue, NCCL_REDUCESCATTER, stream)
    }
  }

  if (num_elements_remaining > 0) {
    // Reduce the remaining data at local_size-1 to append to
    // existing buffer
    NCCL_CHECK(entries, timeline, "ncclReduce",
               ncclReduce(fused_input_data_remainder,
                          buffer_data_remainder,
                          (size_t)num_elements_remaining,
                          GetNCCLDataType(first_entry.tensor), ncclSum,
                          root_rank, nccl_comm, stream))

    if (timeline.Initialized()) {
      RECORD_EVENT(entries, timeline, event_queue, NCCL_REDUCE, stream)
    }
  }

  if (global_state_->is_homogeneous || is_root_rank) {
    // cudaHostAlloc is significantly slower than malloc.  Pre-allocating
    // a buffer is not safe since the tensor can be arbitrarily large.
    host_buffer = malloc(total_buffer_len);

    // Synchronize.
    WAIT_FOR_EVENTS(entries, timeline, event_queue)

    // According to https://docs.nvidia.com/cuda/cuda-runtime-api/
    // api-sync-behavior.html#api-sync-behavior__memcpy-async,
    // cudaMemcpyAsync is synchronous with respect to the host, so we
    // memcpy (effectively) synchronously to generate an accurate timeline
    timeline.ActivityStartAll(entries, MEMCPY_IN_HOST_BUFFER);
    CUDA_CHECK(entries, timeline, "cudaMemcpyAsync",
               cudaMemcpyAsync(host_buffer, buffer_data_at_rank_offset,
                               total_buffer_len, cudaMemcpyDeviceToHost,
                               stream))
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
    MPI_CHECK(entries, timeline, "MPI_Allreduce",
              MPI_Allreduce(MPI_IN_PLACE, host_buffer,
                            (int)total_num_elements,
                            global_state_->GetMPIDataType(first_entry.tensor),
                            first_entry.tensor->dtype() == HOROVOD_FLOAT16
                            ? global_state_->mpi_float16_sum
                            : MPI_SUM,
                            global_state_->cross_comm))
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, MEMCPY_OUT_HOST_BUFFER);
    CUDA_CHECK(entries, timeline, "cudaMemcpyAsync",
               cudaMemcpyAsync(buffer_data_at_rank_offset, host_buffer,
                               total_buffer_len, cudaMemcpyHostToDevice,
                               stream))
    timeline.ActivityEndAll(entries);
  }

  if (num_elements_per_rank > 0) {
    NCCL_CHECK(entries, timeline, "ncclAllGather",
               ncclAllGather(buffer_data_at_rank_offset, buffer_data,
                             (size_t)num_elements_per_rank,
                             GetNCCLDataType(first_entry.tensor),
                             nccl_comm, stream))

    if (timeline.Initialized()) {
      RECORD_EVENT(entries, timeline, event_queue, NCCL_ALLGATHER, stream)
    }
  }
  if (num_elements_remaining > 0) {
    NCCL_CHECK(entries, timeline, "ncclBcast",
               ncclBcast(buffer_data_remainder,
                         (size_t)num_elements_remaining,
                         GetNCCLDataType(first_entry.tensor), root_rank,
                         nccl_comm, stream))

    if (timeline.Initialized()) {
      RECORD_EVENT(entries, timeline, event_queue, NCCL_BCAST, stream)
    }
  }
}

std::vector<int32_t> HierarchicalAllreduce::GetDeviceMap(std::vector<int32_t>& devices) {
  std::vector<int32_t> nccl_device_map;
  for (int rank : global_state_->local_comm_ranks) {
    nccl_device_map.push_back(devices[rank]);
  }
  return nccl_device_map;
}

void HierarchicalAllreduce::SetCommStrategy(int& nccl_rank, int& nccl_size, MPI_Comm& nccl_id_bcast_comm) {
  nccl_rank = global_state_->local_rank;
  nccl_size = global_state_->local_size;
  nccl_id_bcast_comm = global_state_->local_comm;
}

} // namespace common
} // namespace horovod
