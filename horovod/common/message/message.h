//
// Created by Travis Addair on 2018-12-19.
//

#ifndef HOROVOD_MESSAGE_H
#define HOROVOD_MESSAGE_H

#include <iostream>

namespace horovod {
namespace common {

enum DataType {
  HOROVOD_UINT8 = 0,
  HOROVOD_INT8 = 1,
  HOROVOD_UINT16 = 2,
  HOROVOD_INT16 = 3,
  HOROVOD_INT32 = 4,
  HOROVOD_INT64 = 5,
  HOROVOD_FLOAT16 = 6,
  HOROVOD_FLOAT32 = 7,
  HOROVOD_FLOAT64 = 8,
  HOROVOD_BOOL = 9,
  HOROVOD_BYTE = 10
};

const std::string& DataType_Name(DataType value) {
  switch (value) {
    case HOROVOD_UINT8:
      static const std::string uint8("uint8");
      return uint8;
    case HOROVOD_INT8:
      static const std::string int8("int8");
      return int8;
    case HOROVOD_UINT16:
      static const std::string uint16("uint16");
      return uint16;
    case HOROVOD_INT16:
      static const std::string int16("int16");
      return int16;
    case HOROVOD_INT32:
      static const std::string int32("int32");
      return int32;
    case HOROVOD_INT64:
      static const std::string int64("int64");
      return int64;
    case HOROVOD_FLOAT16:
      static const std::string float16("float16");
      return float16;
    case HOROVOD_FLOAT32:
      static const std::string float32("float32");
      return float32;
    case HOROVOD_FLOAT64:
      static const std::string float64("float64");
      return float64;
    case HOROVOD_BOOL:
      static const std::string bool_("bool");
      return bool_;
    case HOROVOD_BYTE:
      static const std::string byte_("byte");
      return byte_;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

class HorovodRequest {
public:
  enum RequestType {
    ALLREDUCE = 0,
    ALLGATHER = 1,
    BROADCAST = 2
  };

  static const std::string& RequestType_Name(RequestType value) {
    switch (value) {
      case RequestType::ALLREDUCE:
        static const std::string allreduce("ALLREDUCE");
        return allreduce;
      case RequestType::ALLGATHER:
        static const std::string allgather("ALLGATHER");
        return allgather;
      case RequestType::BROADCAST:
        static const std::string broadcast("BROADCAST");
        return broadcast;
      default:
        static const std::string unknown("<unknown>");
        return unknown;
    }
  }
};

class HorovodResponse {
public:
  enum ResponseType {
    ALLREDUCE = 0,
    ALLGATHER = 1,
    BROADCAST = 2,
    ERROR = 3
  };

  static const std::string& ResponseType_Name(ResponseType value) {
    switch (value) {
      case ResponseType::ALLREDUCE:
        static const std::string allreduce("ALLREDUCE");
        return allreduce;
      case ResponseType::ALLGATHER:
        static const std::string allgather("ALLGATHER");
        return allgather;
      case ResponseType::BROADCAST:
        static const std::string broadcast("BROADCAST");
        return broadcast;
      case ResponseType::ERROR:
        static const std::string error("ERROR");
        return error;
      default:
        static const std::string unknown("<unknown>");
        return unknown;
    }
  }
};


} // namespace common
} // namespace horovod

#endif //HOROVOD_MESSAGE_H
