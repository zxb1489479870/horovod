//
// Created by Travis Addair on 2018-12-17.
//

#ifndef HOROVOD_GLOBAL_STATE_H
#define HOROVOD_GLOBAL_STATE_H

#include "fusion_buffer_manager.h"
#include "parameter_manager.h"
#include "timeline.h"

namespace horovod {
namespace common {

// The global state required for the MPI ops.
//
// MPI is a library that stores a lot of global per-program state and often
// requires running on a single thread. As a result, we have to have a single
// background thread responsible for all MPI operations, and communicate with
// that background thread through global state.
struct HorovodGlobalState {
  // An atomic boolean which is set to true when background thread is started.
  // This ensures that only one background thread is spawned.
  std::atomic_flag initialize_flag = ATOMIC_FLAG_INIT;

  // A mutex that needs to be used whenever MPI operations are done.
  std::mutex mutex;

  // Tensors waiting to be allreduced or allgathered.
  TensorTable tensor_table;

  // Queue of MPI requests waiting to be sent to the coordinator node.
  std::queue<MPIRequest> message_queue;

  // Background thread running MPI communication.
  std::thread background_thread;

  // Whether the background thread should shutdown.
  std::atomic_bool shut_down {false};

  // Whether Horovod should finalize MPI (only if it has initialized it).
  bool should_finalize = false;

  // Only exists on the coordinator node (rank zero). Maintains a count of
  // how many nodes are ready to allreduce every tensor (keyed by tensor
  // name) and time point when tensor started allreduce op.
  std::unique_ptr<MessageTable> message_table;

  // Time point when coordinator last checked for stalled tensors.
  std::chrono::steady_clock::time_point last_stall_check;

  // Flag indicating whether to perform stall tensor check.
  bool perform_stall_check = true;

  // Timeline writer.
  Timeline timeline;

  ParameterManager param_manager;

  // Encapsulates the fusion buffers, handles resizing and auto-tuning of buffer size.
  FusionBufferManager fusion_buffer;

  // Time point when last cycle started.
  std::chrono::steady_clock::time_point last_cycle_start;

  // Whether MPI_Init has been completed on the background thread.
  std::atomic_bool initialization_done {false};

  // The MPI rank, local rank, size, local size, flag indicating whether MPI
  // multi-threading is supported, ranks from which the MPI communicator will
  // be made and the communicator itself.
  int rank = 0;
  int local_rank = 0;
  int cross_rank = 0;
  int size = 1;
  int local_size = 1;
  int cross_size = 1;
  bool mpi_threads_supported = false;
  bool is_homogeneous = false;
  std::vector<int> ranks;

  // COMM_WORLD ranks of processes running on this node.
  std::vector<int> local_comm_ranks;

  // Numbers of ranks running per node
  std::vector<int> local_sizes;

  // MPI custom data type for float16.
  MPI_Datatype mpi_float16_t;
  MPI_Op mpi_float16_sum;

  // Private MPI communicator for Horovod to ensure no collisions with other
  // threads using MPI.
  MPI_Comm mpi_comm;

  // Node-local communicator.
  MPI_Comm local_comm;

  // Cross-node communicator for hierarchical allreduce.
  MPI_Comm cross_comm;

  // MPI Window used for shared memory allgather
  MPI_Win window;

  // Pointer to shared buffer for allgather
  void* shared_buffer = nullptr;

  // Current shared buffer size
  int64_t shared_buffer_size = 0;

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
#if HAVE_CUDA
  std::unordered_map<int, cudaStream_t> streams;
#endif
#if HAVE_NCCL
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;
#endif

  // Will be set to true after initialization when ddl is used
  bool ddl_initialized = false;
  int32_t ddl_local_device_id = 0;

// We reuse CUDA events as it appears that their creation carries non-zero cost.
#if HAVE_CUDA
  std::unordered_map<int, std::queue<cudaEvent_t>> cuda_events;
  std::mutex cuda_events_mutex;
#endif

  ~HorovodGlobalState() {
    // Make sure that the destructor of the background thread is safe to
    // call. If a thread is still joinable (not detached or complete) its
    // destructor cannot be called.
    if (background_thread.joinable()) {
      shut_down = true;
      background_thread.join();
    }
  }

  MPI_Datatype GetMPIDataType(const std::shared_ptr<Tensor> tensor) {
    switch (tensor->dtype()) {
      case HOROVOD_UINT8:
        return MPI_UINT8_T;
      case HOROVOD_INT8:
        return MPI_INT8_T;
      case HOROVOD_UINT16:
        return MPI_UINT16_T;
      case HOROVOD_INT16:
        return MPI_INT16_T;
      case HOROVOD_INT32:
        return MPI_INT32_T;
      case HOROVOD_INT64:
        return MPI_INT64_T;
      case HOROVOD_FLOAT16:
        return mpi_float16_t;
      case HOROVOD_FLOAT32:
        return MPI_FLOAT;
      case HOROVOD_FLOAT64:
        return MPI_DOUBLE;
      case HOROVOD_BOOL:
        return MPI_C_BOOL;
      default:
        throw std::logic_error("Type " + MPIDataType_Name(tensor->dtype()) +
                               " is not supported in MPI mode.");
    }
  }
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_GLOBAL_STATE_H
