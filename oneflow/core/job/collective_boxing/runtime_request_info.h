/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_RUNTIME_REQUEST_INFO_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_RUNTIME_REQUEST_INFO_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

namespace boxing {

namespace collective {

struct RuntimeRequestInfo {
  const void* send_buff;
  void* recv_buff;
  std::function<void(const Maybe<void>&)> callback;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_RUNTIME_REQUEST_INFO_H_
