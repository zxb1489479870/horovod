//
// Created by Travis Addair on 2018-12-14.
//

#ifndef HOROVOD_MPI_OPERATIONS_H
#define HOROVOD_MPI_OPERATIONS_H

#include <iostream>

#include "mpi.h"

#include "common.h"
#include "global_state.h"

namespace horovod {
namespace common {

#define MPI_CHECK(entries, timeline, op_name, op)                              \
  {                                                                            \
    auto mpi_result = (op);                                                    \
    if (mpi_result != MPI_SUCCESS) {                                           \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(                                       \
            std::string(op_name) + " failed, see MPI output for details."));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

class MPIAllreduce {
public:
  MPIAllreduce(HorovodGlobalState* global_state);
  virtual void Allreduce(std::vector<TensorTableEntry>& entries, std::vector<int32_t>& devices);

protected:
  virtual void MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                    std::vector<TensorTableEntry>& entries);
  virtual void MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                     std::vector<TensorTableEntry>& entries);
  virtual void StreamSynchronize(std::vector<TensorTableEntry>& entries);

  HorovodGlobalState* global_state_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_OPERATIONS_H
