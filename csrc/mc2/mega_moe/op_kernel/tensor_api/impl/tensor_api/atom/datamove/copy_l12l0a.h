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
    "impl/tensor_api/atom/datamove/copy_l12l0a.h is an internal header file and must not be used directly. Functions or variables defined in this file maybe removed in the future. Please use "#include "tensor_api/tensor.h"" and use public functions or variables defined in interface headers files."
#define ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#define UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif

/*!
 * \file copy_l12l0a.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_L12L0A_H
#define IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_L12L0A_H

#include "impl/tensor_api/arch/datamove/l1_to_l0a/copy.h"
#include "impl/tensor_api/atom/copy_traits_impl.h"

namespace AscendC {
namespace Te {

template <typename Traits>
struct CopyTraits<CopyL12L0A, Traits> : public CopyTraits<CopyL12L0A, Traits, CopyL12L0A, CopyL12L0ATraitDefault> {};

template <>
struct CopyTraits<CopyL12L0A> : public CopyTraits<CopyL12L0A, CopyL12L0ATraitDefault> {};

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_ATOM_CUBE_DATAMOVE_COPY_L12L0A_H

#if defined(UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC)
#undef ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS
#undef UNDEF_ASCENDC_TENSOR_API_INCLUDE_COMPILER_INTERNAL_HEADERS_ASCENDC
#endif