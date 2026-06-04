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
    "impl/tensor_api/arch/datamove/gm_to_l1/routing.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file routing.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_L1_ROUTING_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_L1_ROUTING_H

#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/dn2nz.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/dn2zn.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/nd2nd.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/nd2nz.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/nd2zn.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/nz2nz.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/zn2zn.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/scalea_nd2zz.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/scalea_dn2zz.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/scalea_zz2zz.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/scaleb_nd2nn.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/scaleb_dn2nn.h"
#include "impl/tensor_api/arch/datamove/gm_to_l1/npu_arch_3510/gm_to_l1/scaleb_nn2nn.h"

namespace AscendC {
namespace Te {

class CopyGM2L1Ignore {
public:
    template <const CopyGM2L1Trait& trait, typename... Args>
    __aicore__ inline static void Run(const Args&... args)
    {
        static_assert(Std::is_same_v<Args..., void>, "The data format is not supported.");
    }
};

template <typename dstPos, typename srcPos, uint32_t Version, typename DstLayoutPtn, typename SrcLayoutPtn>
struct CopyGM2L1Tensor2Tensor {
    using type = CopyGM2L1Ignore;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, NDExtLayoutPtn, NDExtLayoutPtn> {
    using type = CopyGmToCbufAlignV2ND;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, NZLayoutPtn, NDExtLayoutPtn> {
    using type = CopyGmToCbufMultiND2Nz;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, ZNLayoutPtn, NDExtLayoutPtn> {
    using type = CopyGmToCbufMultiND2Zn;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, NZLayoutPtn, DNExtLayoutPtn> {
    using type = CopyGmToCbufMultiDN2Nz;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, ZNLayoutPtn, DNExtLayoutPtn> {
    using type = CopyGmToCbufMultiDN2Zn;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, NZLayoutPtn, NZLayoutPtn> {
    using type = CopyGmToCbufAlignV2NZ;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, ZNLayoutPtn, ZNLayoutPtn> {
    using type = CopyGmToCbufAlignV2ZN;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, ZZLayoutPtn, ScaleANDLayoutPtn> {
    using type = CopyGmToCbufScaleAND2Zz;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, ZZLayoutPtn, ScaleADNLayoutPtn> {
    using type = CopyGmToCbufScaleADN2Zz;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, ZZLayoutPtn, ZZLayoutPtn> {
    using type = CopyGmToCbufScaleAZz2Zz;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, NNLayoutPtn, ScaleBNDLayoutPtn> {
    using type = CopyGmToCbufScaleBND2Nn;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, NNLayoutPtn, ScaleBDNLayoutPtn> {
    using type = CopyGmToCbufScaleBDN2Nn;
};

template <>
struct CopyGM2L1Tensor2Tensor<Location::L1, Location::GM, ArchVersion::V3510, NNLayoutPtn, NNLayoutPtn> {
    using type = CopyGmToCbufScaleBNn2Nn;
};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_GM_TO_L1_ROUTING_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

