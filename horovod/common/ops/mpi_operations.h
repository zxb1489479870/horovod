//
// Created by Travis Addair on 2018-12-14.
//

#ifndef HOROVOD_MPI_OPERATIONS_H
#define HOROVOD_MPI_OPERATIONS_H

#include <iostream>
#include <queue>

#include "mpi.h"

#include "common.h"
#include "communication_context.h"
#include "global_state.h"

namespace horovod {
namespace common {

class MPIContext : public CommunicationContext {
public:
  void Allreduce(const void* buffer_data, int64_t num_elements,
                 TensorTableEntry& first_entry, const void* sendbuff=nullptr,
                 Communicator comm=Communicator::GLOBAL) override;

  void Broadcast(const void* buffer_data, int64_t num_elements,
                 DataType dtype, int root_rank,
                 Communicator comm=Communicator::GLOBAL) override;

  void Barrier(Communicator comm=Communicator::GLOBAL) override;

  void GetTypeSize(DataType dtype, int* out) override;

  inline std::string AllreduceActivity() const {
    return MPI_ALLREDUCE;
  }

private:
  MPI_Datatype GetMPIDataType(std::shared_ptr<Tensor> tensor);
  MPI_Datatype GetMPIDataType(DataType dtype);

  MPI_Comm GetMPICommunicator(Communicator comm);

  // Queue of MPI requests waiting to be sent to the coordinator node.
  std::queue<MPIRequest> message_queue;

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
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_MPI_OPERATIONS_H
