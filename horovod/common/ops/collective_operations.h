//
// Created by Travis Addair on 2018-12-18.
//

#ifndef HOROVOD_COLLECTIVE_OPERATIONS_H
#define HOROVOD_COLLECTIVE_OPERATIONS_H

#include <iostream>

#include "common.h"
#include "communication_context.h"
#include "global_state.h"

namespace horovod {
namespace common {

class AllreduceOp {
public:
  AllreduceOp(CommunicationContext* comm_context, HorovodGlobalState* global_state);
  virtual void Allreduce(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices);

protected:
  virtual void MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                    std::vector<TensorTableEntry>& entries);
  virtual void MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                     std::vector<TensorTableEntry>& entries);
  virtual void StreamSynchronize(std::vector<TensorTableEntry>& entries);

  CommunicationContext* comm_context_;
  HorovodGlobalState* global_state_;
};

class AllgatherOp {
public:
  AllgatherOp(CommunicationContext* comm_context, HorovodGlobalState* global_state);
  virtual void Allgather(std::vector<TensorTableEntry>& entries, const std::vector<int64_t>& tensor_sizes);

protected:
  virtual void DoAllgather(std::vector<TensorTableEntry>& entries, int* recvcounts, int* displcmnts,
                           int64_t** entry_component_offsets, int64_t** entry_component_sizes,
                           int64_t total_size, int element_size);

  CommunicationContext* comm_context_;
  HorovodGlobalState* global_state_;
};

class HierarchicalAllgather : public AllgatherOp {
public:
  HierarchicalAllgather(CommunicationContext* comm_context,
                        HorovodGlobalState* global_state);

  virtual void DoAllgather(std::vector<TensorTableEntry>& entries, int* recvcounts, int* displcmnts,
                           int64_t** entry_component_offsets, int64_t** entry_component_sizes,
                           int64_t total_size, int element_size) override;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_COLLECTIVE_OPERATIONS_H
