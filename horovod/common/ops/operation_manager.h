//
// Created by Travis Addair on 2019-02-07.
//

#ifndef HOROVOD_OPERATION_MANAGER_H
#define HOROVOD_OPERATION_MANAGER_H

#include "../parameter_manager.h"
#include "collective_operations.h"

namespace horovod {
namespace common {

class OperationManager {
public:
  OperationManager(ParameterManager* param_manager,
                   std::shared_ptr<AllreduceOp> allreduce_op,
                   std::shared_ptr<AllgatherOp> allgather_op,
                   std::shared_ptr<BroadcastOp> broadcast_op,
                   std::shared_ptr<AllreduceOp> hierarchical_allreduce_op,
                   std::shared_ptr<AllgatherOp> hierarchical_allgather_op);
  virtual ~OperationManager()=default;

  std::shared_ptr<AllreduceOp> GetAllreduceOp() const;

  std::shared_ptr<AllgatherOp> GetAllgatherOp() const;

  std::shared_ptr<BroadcastOp> GetBroadcastOp() const;

private:
  ParameterManager* param_manager_;

  std::shared_ptr<AllreduceOp> allreduce_op_;
  std::shared_ptr<AllgatherOp> allgather_op_;
  std::shared_ptr<BroadcastOp> broadcast_op_;

  std::shared_ptr<AllreduceOp> hierarchical_allreduce_op_;
  std::shared_ptr<AllgatherOp> hierarchical_allgather_op_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_OPERATION_MANAGER_H
