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
    "impl/tensor_api/atom/mmad_atom_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
* \file mmad_atom_impl.h
* \brief
*/
#ifndef IMPL_TENSOR_API_DETAIL_ATOM_MAD_ATOM_IMPL_H
#define IMPL_TENSOR_API_DETAIL_ATOM_MAD_ATOM_IMPL_H

#include "impl/tensor_api/atom/compute/mmad.h"
#include "impl/tensor_api/atom/mmad_traits_impl.h"

namespace AscendC {
namespace Te {

template <typename... Args>
struct MmadAtom;

template <typename MmaOperation>
struct MmadAtom<MmaOperation> : MmadAtom<MmadTraits<MmaOperation>> {};

template <typename MmaOperation, typename... Args>
struct MmadAtom<MmadTraits<MmaOperation, Args...>> : MmadTraits<MmaOperation, Args...>
{
    using MmaTraitType = MmadTraits<MmaOperation, Args...>;
    using TraitType = typename MmaTraitType::TraitType;
    static constexpr const TraitType defaultTrait = MmaTraitType::defaultTrait;

    template <const TraitType& traits = defaultTrait, typename... Params>
    __aicore__ inline void Call(const Params& ...params) const {
        MmaTraitType::template MmadUnpack<traits>(params...);
    }

    template <typename... TraitsArgs>
    __aicore__ inline auto with(TraitsArgs&&... args) const {
        auto traits = MmaTraitType::with(static_cast<TraitsArgs&&>(args)...);
        return MmadAtom<decltype(traits)>{traits};
    }
};

}
}

#endif // IMPL_TENSOR_API_DETAIL_ATOM_MAD_ATOM_IMPL_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif