//
// Created by Travis Addair on 2019-02-07.
//

#include "operation_manager.h"

namespace horovod {
namespace common {

OperationManager::OperationManager(ParameterManager* param_manager,
                                   std::shared_ptr<AllreduceOp> allreduce_op,
                                   std::shared_ptr<AllgatherOp> allgather_op,
                                   std::shared_ptr<BroadcastOp> broadcast_op,
                                   std::shared_ptr<AllreduceOp> hierarchical_allreduce_op,
                                   std::shared_ptr<AllgatherOp> hierarchical_allgather_op)
                                   : param_manager_(param_manager),
                                     allreduce_op_(allreduce_op),
                                     allgather_op_(allgather_op),
                                     hierarchical_allreduce_op_(hierarchical_allreduce_op),
                                     hierarchical_allgather_op_(hierarchical_allgather_op) {}

std::shared_ptr<AllreduceOp> OperationManager::GetAllreduceOp() const {
  if (param_manager_->HierarchicalAllreduce() && hierarchical_allreduce_op_ != nullptr) {
    return hierarchical_allreduce_op_;
  }
  return allreduce_op_;
}

std::shared_ptr<AllgatherOp> OperationManager::GetAllgatherOp() const {
  if (param_manager_->HierarchicalAllgather() && hierarchical_allgather_op_ != nullptr) {
    return hierarchical_allgather_op_;
  }
  return allgather_op_;
}

std::shared_ptr<BroadcastOp> OperationManager::GetBroadcastOp() const {
  return broadcast_op_;
}

} // namespace common
} // namespace horovod
