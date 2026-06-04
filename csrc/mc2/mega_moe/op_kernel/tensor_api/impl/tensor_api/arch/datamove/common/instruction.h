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
    "impl/tensor_api/arch/datamove/common/instruction.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file instruction.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ARCH_DATAMOVE_COMMON_INSTRUCTION_H
#define IMPL_TENSOR_API_ARCH_DATAMOVE_COMMON_INSTRUCTION_H

#include "impl/tensor_api/arch/datamove/common/l0c2out_utils.h"

namespace AscendC {
namespace Te {

class SetRegisterInstr {
public:
    __aicore__ inline static void SetRegister(uint64_t quant, uint32_t ndNum, uint32_t dstNDStride,
                                              uint32_t srcNDStride)
    {
        SetQuantPre(quant);
        SetLoop3Para<uint64_t>(ndNum, dstNDStride, srcNDStride);
    }

    __aicore__ inline static void SetRegister(uint64_t quant, uint32_t dnNum, uint32_t dstDNStride,
                                              uint32_t srcNZMatrixStride, uint32_t srcNZC0Stride)
    {
        SetQuantPre(quant);
        SetLoop3Para<uint64_t>(dnNum, dstDNStride, srcNZMatrixStride);
        SetChannelPara<uint64_t>(srcNZC0Stride);
    }

    __aicore__ inline static void SetRegister(uint32_t ndNum, uint32_t dstNDStride, uint32_t srcNDStride)
    { SetLoop3Para<uint64_t>(ndNum, dstNDStride, srcNDStride); }

    __aicore__ inline static void SetRegister(uint32_t dnNum, uint32_t dstDNStride, uint32_t srcNZMatrixStride,
                                              uint32_t srcNZC0Stride)
    {
        SetLoop3Para<uint64_t>(dnNum, dstDNStride, srcNZMatrixStride);
        SetChannelPara<uint64_t>(srcNZC0Stride);
    }

private:
    static constexpr uint32_t SHIFT_LOOP3_DST_STRIDE = 32;
    static constexpr uint32_t SHIFT_LOOP3_SRC_MATRIX = 16;
    static constexpr uint32_t SHIFT_CHANNEL_C0_STRIDE = 48;

    __aicore__ inline static void SetQuantPre(uint64_t quant)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            asc_set_l0c_copy_prequant(quant);
        }
    }

    template <typename T>
    __aicore__ inline static void SetLoop3Para(uint32_t num, uint32_t dstStride, uint32_t srcStride)
    {
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            asc_set_l0c2gm_nz2nd(static_cast<T>(num), static_cast<T>(srcStride), static_cast<T>(dstStride));
        }
    }

    template <typename T>
    __aicore__ inline static void SetChannelPara(uint32_t srcNZC0Stride)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
            T channelPara = 0;
            channelPara |= static_cast<T>(srcNZC0Stride) << SHIFT_CHANNEL_C0_STRIDE;
            asc_set_l0c2gm_channel_para(channelPara);
        }
    }
};

__aicore__ inline auto AllocFbTempBuf(const uint16_t& /* calNSize */)
{
    if ASCEND_IS_AIV {
        return 0UL;
    }
    uint64_t deqTensorTempBuf = 0;
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
        deqTensorTempBuf = reinterpret_cast<uint64_t>(asc_get_phy_buf_addr(0));
    }
    return deqTensorTempBuf;
}

template <typename T>
__aicore__ inline void SetFpc(const __fbuf__ T* deqTensorTempBuf)
{
    if ASCEND_IS_AIV {
        return;
    }
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
        uint64_t deqTensorAddr = (reinterpret_cast<uint64_t>(deqTensorTempBuf) >> 7) << 8;
        asc_set_l0c_copy_prequant(deqTensorAddr);
    }
}

__aicore__ inline void InsertSync()
{
    if ASCEND_IS_AIV {
        return;
    }
    if constexpr (CURRENT_ARCH_VERSION == ArchVersion::V3510) {
        asc_sync_pipe(PIPE_FIX);
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ARCH_DATAMOVE_COMMON_INSTRUCTION_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif