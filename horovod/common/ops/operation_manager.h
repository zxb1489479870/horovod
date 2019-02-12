// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

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
                   std::shared_ptr<ErrorOp> error_op,
                   std::shared_ptr<AllreduceOp> hierarchical_allreduce_op,
                   std::shared_ptr<AllgatherOp> hierarchical_allgather_op);
  virtual ~OperationManager()=default;

  std::shared_ptr<AllreduceOp> GetAllreduceOp() const;

  std::shared_ptr<AllgatherOp> GetAllgatherOp() const;

  std::shared_ptr<BroadcastOp> GetBroadcastOp() const;

  std::shared_ptr<ErrorOp> GetErrorOp() const;

  std::shared_ptr<HorovodOp> GetOp(HorovodResponse& response) const;

private:
  ParameterManager* param_manager_;

  std::shared_ptr<AllreduceOp> allreduce_op_;
  std::shared_ptr<AllgatherOp> allgather_op_;
  std::shared_ptr<BroadcastOp> broadcast_op_;
  std::shared_ptr<ErrorOp> error_op_;

  std::shared_ptr<AllreduceOp> hierarchical_allreduce_op_;
  std::shared_ptr<AllgatherOp> hierarchical_allgather_op_;
};

} // namespace common
} // namespace horovod

#endif //HOROVOD_OPERATION_MANAGER_H
