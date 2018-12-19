//
// Created by Travis Addair on 2018-12-14.
//

#include "collective_operations.h"

namespace horovod {
namespace common {

AllreduceOp::AllreduceOp(CommunicationContext* comm_context, HorovodGlobalState* global_state)
    : comm_context_(comm_context), global_state_(global_state) {}

void AllreduceOp::Allreduce(std::vector<TensorTableEntry>& entries, std::vector<int32_t>& devices) {
  auto& first_entry = entries[0];
  auto& timeline = global_state_->timeline;

  if (entries.size() > 1) {
    // Access the fusion buffer.
    auto &buffer = global_state_->fusion_buffer.GetBuffer(
        first_entry.device, first_entry.context->framework());
    auto buffer_data = buffer->AccessData(first_entry.context);

    // Copy memory into the fusion buffer.
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    int64_t offset = 0;
    for (auto &e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyInFusionBuffer(buffer_data_at_offset, e, entries);
      offset += e.tensor->size();
    }

    StreamSynchronize(entries);
    timeline.ActivityEndAll(entries);

    timeline.ActivityStartAll(entries, comm_context_->AllreduceActivity());
    int64_t num_elements = 0;
    for (auto& e : entries) {
      num_elements += e.tensor->shape().num_elements();
    }
    comm_context_->Allreduce(buffer_data, num_elements, first_entry);
    timeline.ActivityEndAll(entries);

    // Copy memory out of the fusion buffer.
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyOutFusionBuffer(buffer_data_at_offset, e, entries);
      offset += e.tensor->size();
    }

    StreamSynchronize(entries);
    timeline.ActivityEndAll(entries);
  } else {
    auto& e = first_entry;
    timeline.ActivityStartAll(entries, comm_context_->AllreduceActivity());
    const void* sendbuff = e.tensor->data() == e.output->data() ? nullptr : e.tensor->data();
    comm_context_->Allreduce((void*)e.output->data(), e.tensor->shape().num_elements(), first_entry, sendbuff);
    timeline.ActivityEndAll(entries);
  }

  for (auto& e : entries) {
    timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());
  }
}

void AllreduceOp::MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                        std::vector<TensorTableEntry>& entries) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t)e.tensor->size());
}

void AllreduceOp::MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                         std::vector<TensorTableEntry>& entries) {
  std::memcpy((void*)e.output->data(), buffer_data_at_offset,
              (size_t)e.tensor->size());
}

void AllreduceOp::StreamSynchronize(std::vector<TensorTableEntry>& entries) {
}

} // namespace common
} // namespace horovod
