/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the Apache License, Version 2.0.
 *
 * Host-side CeilDiv / CeilAlign for workspace_info.h etc.
 * Device side uses CANN SDK built-in Ops::Base::CeilDiv/Align from math_util.h.
 */

#ifndef MEGA_MOE_CEIL_UTILS_H
#define MEGA_MOE_CEIL_UTILS_H

#if !defined(__CCE_AICORE__) && !defined(__CCE_AIVECTOR__)
// Host-side only: CANN SDK math_util.h not available for g++

#include <cstdint>

namespace Ops {
namespace Base {

template <typename T>
inline T CeilDiv(T a, T b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

template <typename T>
inline T CeilAlign(T a, T b)
{
    return (b == 0) ? 0 : ((a + b - 1) / b) * b;
}

} // namespace Base
} // namespace Ops

#endif // !__CCE_AICORE__

#endif // MEGA_MOE_CEIL_UTILS_H
