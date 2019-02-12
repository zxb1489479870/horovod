//
// Created by Travis Addair on 2018-12-14.
//

#ifndef HOROVOD_NCCL_OPERATIONS_H
#define HOROVOD_NCCL_OPERATIONS_H

#include <nccl.h>

#include "cuda_operations.h"

namespace horovod {
namespace common {

struct NCCLContext {
  std::unordered_map<std::vector<int32_t>, ncclComm_t> nccl_comms;

  void ErrorCheck(std::string op_name, ncclResult_t nccl_result);
};

class NCCLAllreduce : public CUDACustomAllreduce {
public:
  NCCLAllreduce(NCCLContext* nccl_context, CUDAContext* cuda_context,
                CommunicationContext* comm_context, HorovodGlobalState* global_state);

protected:
  void InitComm(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices) override;
  void CustomAllreduce(std::vector<TensorTableEntry>& entries,
                       cudaStream_t& stream, std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                       const void* fused_input_data, void* buffer_data,
                       int64_t& num_elements, size_t& buffer_len) override;

private:
  virtual const std::vector<int32_t> GetDeviceMap(const std::vector<int32_t>& devices);
  virtual void SetCommStrategy(int& nccl_rank, int& nccl_size,
                               CommunicationContext::Communicator& nccl_id_bcast_comm);

  NCCLContext* nccl_context_;
};

class HierarchicalAllreduce : public NCCLAllreduce {
public:
  HierarchicalAllreduce(NCCLContext* nccl_context, CUDAContext* cuda_context, HorovodGlobalState* global_state);

protected:
  void CustomAllreduce(std::vector<TensorTableEntry>& entries,
                       cudaStream_t& stream, std::queue<std::pair<std::string, cudaEvent_t>>& event_queue,
                       const void* fused_input_data, void* buffer_data,
                       int64_t& num_elements, size_t& buffer_len) override;

private:
  const std::vector<int32_t> GetDeviceMap(const std::vector<int32_t>& devices) override;
  void SetCommStrategy(int& nccl_rank, int& nccl_size,
                       CommunicationContext::Communicator& nccl_id_bcast_comm) override;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_NCCL_OPERATIONS_H
