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

#include "operation_manager.h"

namespace horovod {
namespace common {

OperationManager::OperationManager(ParameterManager* param_manager,
                                   std::shared_ptr<AllreduceOp> allreduce_op,
                                   std::shared_ptr<AllgatherOp> allgather_op,
                                   std::shared_ptr<BroadcastOp> broadcast_op,
                                   std::shared_ptr<ErrorOp> error_op,
                                   std::shared_ptr<AllreduceOp> hierarchical_allreduce_op,
                                   std::shared_ptr<AllgatherOp> hierarchical_allgather_op)
                                   : param_manager_(param_manager),
                                     allreduce_op_(allreduce_op),
                                     allgather_op_(allgather_op),
                                     broadcast_op_(broadcast_op),
                                     error_op_(error_op),
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

std::shared_ptr<ErrorOp> OperationManager::GetErrorOp() const {
  return error_op_;
}

std::shared_ptr<HorovodOp> OperationManager::GetOp(HorovodResponse& response) const {
  if (response.response_type() == HorovodResponse::ALLGATHER) {
    return GetAllgatherOp();
  } else if (response.response_type() == HorovodResponse::ALLREDUCE) {
    return GetAllreduceOp();
  } else if (response.response_type() == HorovodResponse::BROADCAST) {
    return GetBroadcastOp();
  } else if (response.response_type() == HorovodResponse::ERROR) {
    return GetErrorOp();
  } else {
    throw std::logic_error("No operation found for response type provided");
  }
}

} // namespace common
} // namespace horovod
