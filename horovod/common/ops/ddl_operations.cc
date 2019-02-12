//
// Created by Travis Addair on 2018-12-14.
//

#include "ddl_operations.h"

namespace horovod {
namespace common {

#define DDL_CHECK(entries, timeline, op_name, op)                              \
  {                                                                            \
    auto ddl_result = (op);                                                    \
    if (ddl_result != DDL_SUCCESS) {                                           \
      for (auto& e : (entries)) {                                              \
        timeline.End(e.tensor_name, nullptr);                                  \
        e.callback(Status::UnknownError(std::string(op_name) + " failed."));   \
      }                                                                        \
      return;                                                                  \
    }                                                                          \
  }

#define OP_ERROR(entries, timeline, error_message)                             \
  {                                                                            \
    for (auto& e : (entries)) {                                                \
      timeline.End(e.tensor_name, nullptr);                                    \
      e.callback(Status::UnknownError(error_message));                         \
    }                                                                          \
    return;                                                                    \
  }

DDL_Type GetDDLDataType(const std::shared_ptr<Tensor> tensor) {
  switch (tensor->dtype()) {
    case HOROVOD_FLOAT32:
      return DDL_TYPE_FLOAT;
    default:
      throw std::logic_error("Type " + DataType_Name(tensor->dtype()) +
                             " is not supported in DDL mode.");
  }
}

DDLAllreduce::DDLAllreduce(CUDAContext* cuda_context,
                           CommunicationContext* comm_context,
                           HorovodGlobalState* global_state) :
                           CUDACustomAllreduce(cuda_context, comm_context, global_state) {}

void DDLAllreduce::InitComm(std::vector<TensorTableEntry>& entries, const std::vector<int32_t>& devices) {
  auto& timeline = global_state_->timeline;
  if (!global_state_->ddl_initialized) {
    // Initialize DDL
    auto ddl_options = std::getenv("DDL_OPTIONS");
    if (ddl_options == nullptr) {
      OP_ERROR(entries, timeline,
               "DDL_OPTIONS env variable needs to be set to use DDL.")
    }

    auto& first_entry = entries[0];
    DDL_CHECK(entries, timeline, "ddl_init", ddl_init(ddl_options))
    global_state_->ddl_initialized = true;
    global_state_->ddl_local_device_id = first_entry.device;
  } else if (global_state_->ddl_local_device_id != first_entry.device) {
    OP_ERROR(entries, timeline,
             "DDL does not support more than one GPU device per process.")
  }
}

void DDLAllreduce::CustomAllreduce(std::vector<TensorTableEntry>& entries,
                                   cudaStream_t& stream, std::queue<std::pair<std::string,
                                   cudaEvent_t>>& event_queue,
                                   const void* fused_input_data, void* buffer_data,
                                   int64_t& num_elements, size_t& buffer_len) {
  // Synchronize.
  auto& timeline = global_state_->timeline;
  WAIT_FOR_EVENTS(entries, timeline, event_queue)
  DDL_Type ddl_data_type;
  try {
    auto& first_entry = entries[0];
    ddl_data_type = GetDDLDataType(first_entry.tensor);
  } catch (const std::logic_error& ex) {
    OP_ERROR(entries, timeline, ex.what())
  }
  DDL_CHECK(entries, timeline, "ddl_allreduce",
            ddl_allreduce(buffer_data, (size_t)num_elements, ddl_data_type,
                          DDL_OP_SUM))
}

} // namespace common
} // namespace horovod
