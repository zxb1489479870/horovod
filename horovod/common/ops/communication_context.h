//
// Created by Travis Addair on 2018-12-18.
//

#ifndef HOROVOD_COMMUNICATION_CONTEXT_H
#define HOROVOD_COMMUNICATION_CONTEXT_H

#include "../message/message.h"

namespace horovod {
namespace common {

class CommunicationContext {
public:
  enum Communicator {
    GLOBAL = 0,
    LOCAL = 1,
    CROSS = 2
  };

  inline std::string CommunicatorName(Communicator comm) {
    switch (comm) {
      case GLOBAL:
        return "global";
      case LOCAL:
        return "local";
      case CROSS:
        return "cross";
      default:
        return "<unknown>";
    }
  }

  virtual void Allreduce(const void* buffer_data, int64_t num_elements,
                         TensorTableEntry& first_entry, const void* sendbuff=nullptr,
                         Communicator comm=Communicator::GLOBAL) = 0;

  virtual void Allgatherv(const void *sendbuf, int sendcount, DataType sendtype,
                          void *recvbuf, const int recvcounts[],
                          const int displs[], DataType recvtype,
                          Communicator comm=Communicator::GLOBAL) = 0;

  virtual void Broadcast(const void* buffer_data, int64_t num_elements,
                         DataType dtype, int root_rank,
                         Communicator comm=Communicator::GLOBAL) = 0;

  virtual void Barrier(Communicator comm=Communicator::GLOBAL) = 0;

  virtual void AllocateSharedBuffer(int64_t window_size, int element_size, void* baseptr,
                                    Communicator comm=Communicator::GLOBAL) = 0;

  virtual void FreeSharedBuffer() = 0;

  virtual void QuerySharedBuffer(int rank, void* baseptr) = 0;

  virtual void GetTypeSize(DataType dtype, int* out) = 0;

  virtual std::string AllreduceActivity() const = 0;

  virtual std::string BroadcastActivity() const = 0;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_COMMUNICATION_CONTEXT_H
