#ifndef MEGA_MOE_KERNEL_MATH_UTIL_H
#define MEGA_MOE_KERNEL_MATH_UTIL_H

#if defined(__ASCEND_C__) || defined(__CCE_KT_TEST__)
#define MEGA_MOE_HOST_DEVICE __aicore__
#else
#define MEGA_MOE_HOST_DEVICE
#endif

namespace Ops { namespace Base {

template <typename T>
MEGA_MOE_HOST_DEVICE static constexpr T CeilAlign(T num, T align) {
    return ((num + align - 1) / align) * align;
}
template <typename T>
MEGA_MOE_HOST_DEVICE static constexpr T CeilDiv(T num, T div) {
    return (num + div - 1) / div;
}
template <typename T>
MEGA_MOE_HOST_DEVICE static constexpr T FloorAlign(T num, T align) {
    return (num / align) * align;
}
template <typename T>
MEGA_MOE_HOST_DEVICE static constexpr T FloorDiv(T num, T div) {
    return num / div;
}

} }  // namespace Ops::Base

#endif
