/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if !defined(ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS)
#warning                                                                                                               \
    "impl/tensor_api/arch/datamove/l0c_to_ub/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_UB_ROUTING_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_UB_ROUTING_H

#include "impl/tensor_api/arch/datamove/common/l0c2out_utils.h"
#include "impl/tensor_api/arch/datamove/l0c_to_ub/npu_arch_3510/data_copy.h"

namespace AscendC {
namespace Te {

class CopyL0C2UBIgnore {
public:
    template <const CopyL0C2UBTrait& trait, typename... Args>
    __aicore__ inline static void Run(const Args&... args)
    {}
};

template <typename dstTPos, typename srcTpos, uint32_t Version>
struct CopyL0C2UBTensor2Tensor {
    using type = CopyL0C2UBIgnore;
};

template <>
struct CopyL0C2UBTensor2Tensor<Location::UB, Location::L0C, ArchVersion::V3510> {
    using type = DataCopyL0C2UB3510;
};

template <typename dstTPos, typename srcTpos, uint32_t Version>
struct CopyL0C2UBVectorQuantTensor2Tensor {
    using type = CopyL0C2UBIgnore;
};

template <>
struct CopyL0C2UBVectorQuantTensor2Tensor<Location::UB, Location::L0C, ArchVersion::V3510> {
    using type = DataCopyL0C2UBVectorQuant3510;
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_L0C_TO_UB_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif
