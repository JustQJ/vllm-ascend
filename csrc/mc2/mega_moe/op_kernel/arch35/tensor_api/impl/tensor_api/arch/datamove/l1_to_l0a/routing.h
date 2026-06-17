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
    "impl/tensor_api/arch/datamove/l1_to_l0a/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file routing.h
 * \brief
 */

#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0A_ROUTING_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0A_ROUTING_H

#include "impl/tensor_api/arch/datamove/l1_to_l0a/npu_arch_3510/nz2nz.h"
#include "impl/tensor_api/arch/datamove/l1_to_l0a/npu_arch_3510/zn2nz.h"
#include "impl/tensor_api/arch/datamove/l1_to_l0a/npu_arch_3510/zn2nzb8b4.h"
#include "impl/tensor_api/arch/datamove/l1_to_l0a/npu_arch_3510/nz2nz_with_coord.h"
#include "impl/tensor_api/arch/datamove/l1_to_l0a/npu_arch_3510/zn2nz_with_coord.h"
#include "impl/tensor_api/arch/datamove/l1_to_l0a/npu_arch_3510/zn2nzb8b4_with_coord.h"

namespace AscendC {
namespace Te {

class CopyL12L0AIgnore {
public:
    template <const CopyL12L0ATrait& trait, typename ...Args>
    __aicore__ inline void static Run(const Args&... args) {
        static_assert(Std::is_same_v<Args..., void>, "The data format is not supported in CopyL12L0A.");
    }
};

template <typename dstPos, typename srcPos, uint32_t Version, typename DstLayoutPattern, typename SrcLayoutPattern, typename CopyMode>
struct CopyL12L0ATensor2Tensor {
    using type = CopyL12L0AIgnore;
};

template <>
struct CopyL12L0ATensor2Tensor<Location::L0A, Location::L1, ArchVersion::V3510, NZLayoutPtn, NZLayoutPtn, CopyMode::NORMAL>
{
    using type = LoadDataL12L0ANZ2NZ3510;
};

template <>
struct CopyL12L0ATensor2Tensor<Location::L0A, Location::L1, ArchVersion::V3510, NZLayoutPtn, NZLayoutPtn, CopyMode::NORMAL_COORD>
{
    using type = LoadDataL12L0ANZ2NZWithCoord3510;
};

template <>
struct CopyL12L0ATensor2Tensor<Location::L0A, Location::L1, ArchVersion::V3510, NZLayoutPtn, ZNLayoutPtn, CopyMode::TRANS>
{
    using type = LoadDataL12L0AZN2NZ3510;
};

template <>
struct CopyL12L0ATensor2Tensor<Location::L0A, Location::L1, ArchVersion::V3510, NZLayoutPtn, ZNLayoutPtn, CopyMode::TRANS_COORD>
{
    using type = LoadDataL12L0AZN2NZWithCoord3510;
};

template <>
struct CopyL12L0ATensor2Tensor<Location::L0A, Location::L1, ArchVersion::V3510, NZLayoutPtn, ZNLayoutPtn, CopyMode::TRANS_B8B4>
{
    using type = LoadDataL12L0AZN2NZB8B43510;
};

template <>
struct CopyL12L0ATensor2Tensor<Location::L0A, Location::L1, ArchVersion::V3510, NZLayoutPtn, ZNLayoutPtn, CopyMode::TRANS_B8B4_COORD>
{
    using type = LoadDataL12L0AZN2NZB8B4WithCoord3510;
};
} // namespace Te
} // namespace AscendC
#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_L1_TO_L0A_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

