//
// Created by Travis Addair on 2018-12-14.
//

#ifndef HOROVOD_DDL_OPERATIONS_H
#define HOROVOD_DDL_OPERATIONS_H

#include <ddl.hpp>

#include "cuda_operations.h"

namespace horovod {
namespace common {

class DDLAllreduce : public CUDACustomAllreduce {
public:
  DDLAllreduce(CUDAContext* cuda_context,
               CommunicationContext* comm_context,
               HorovodGlobalState* global_state);

protected:
  void InitComm(std::vector<TensorTableEntry>& entries, std::vector<int32_t>& devices) override;
  void CustomAllreduce(std::vector<TensorTableEntry>& entries,
                       cudaStream_t& stream, std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                       const void* fused_input_data, void* buffer_data,
                       int64_t& num_elements, size_t& buffer_len) override;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_DDL_OPERATIONS_H
