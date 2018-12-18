//
// Created by Travis Addair on 2018-12-14.
//

#include "mpi_operations.h"

namespace horovod {
namespace common {

MPIAllreduce::MPIAllreduce(horovod::common::HorovodGlobalState* global_state)
: global_state_(global_state) {}

void MPIAllreduce::Allreduce(std::vector<TensorTableEntry>& entries, std::vector<int32_t>& devices) {
  auto& first_entry = entries[0];

  if (entries.size() > 1) {
    // Access the fusion buffer.
    auto &buffer = global_state_->fusion_buffer.GetBuffer(
        first_entry.device, first_entry.context->framework());
    auto buffer_data = buffer->AccessData(first_entry.context);

    // Copy memory into the fusion buffer.
    global_state_->timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    int64_t offset = 0;
    for (auto &e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyInFusionBuffer(buffer_data_at_offset, e, entries);
      offset += e.tensor->size();
    }

    StreamSynchronize(entries);
    global_state_->timeline.ActivityEndAll(entries);

    global_state_->timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
    int64_t num_elements = 0;
    for (auto& e : entries) {
      num_elements += e.tensor->shape().num_elements();
    }
    MPI_CHECK(entries, global_state_->timeline, "MPI_Allreduce",
              MPI_Allreduce(MPI_IN_PLACE, (void*)buffer_data,
                            (int)num_elements,
                            global_state_->GetMPIDataType(first_entry.tensor),
                            first_entry.tensor->dtype() == HOROVOD_FLOAT16
                            ? global_state_->mpi_float16_sum
                            : MPI_SUM,
                            global_state_->mpi_comm))
    global_state_->timeline.ActivityEndAll(entries);

    // Copy memory out of the fusion buffer.
    global_state_->timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    offset = 0;
    for (auto& e : entries) {
      void* buffer_data_at_offset = (uint8_t*) buffer_data + offset;
      MemcpyOutFusionBuffer(buffer_data_at_offset, e, entries);
      offset += e.tensor->size();
    }

    StreamSynchronize(entries);
    global_state_->timeline.ActivityEndAll(entries);
  } else {
    auto& e = first_entry;
    global_state_->timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
    const void* sendbuf = e.tensor->data() == e.output->data()
                          ? MPI_IN_PLACE
                          : e.tensor->data();
    MPI_CHECK(entries, global_state_->timeline, "MPI_Allreduce",
              MPI_Allreduce(sendbuf, (void*)e.output->data(),
                            (int)e.tensor->shape().num_elements(),
                            global_state_->GetMPIDataType(e.tensor),
                            first_entry.tensor->dtype() == HOROVOD_FLOAT16
                            ? global_state_->mpi_float16_sum
                            : MPI_SUM,
                            global_state_->mpi_comm))
    global_state_->timeline.ActivityEndAll(entries);
  }

  for (auto& e : entries) {
    global_state_->timeline.End(e.tensor_name, e.output);
    e.callback(Status::OK());
  }
}

void MPIAllreduce::MemcpyInFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                        std::vector<TensorTableEntry>& entries) {
  std::memcpy(buffer_data_at_offset, e.tensor->data(),
              (size_t)e.tensor->size());
}

void MPIAllreduce::MemcpyOutFusionBuffer(void* buffer_data_at_offset, TensorTableEntry& e,
                                         std::vector<TensorTableEntry>& entries) {
  std::memcpy((void*)e.output->data(), buffer_data_at_offset,
              (size_t)e.tensor->size());
}

void MPIAllreduce::StreamSynchronize(std::vector<TensorTableEntry>& entries) {
}

} // namespace common
} // namespace horovod
