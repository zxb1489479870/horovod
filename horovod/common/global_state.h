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

  // Pointer to shared buffer for allgather
  void* shared_buffer = nullptr;

  // Current shared buffer size
  int64_t shared_buffer_size = 0;

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
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_GLOBAL_STATE_H
