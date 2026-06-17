#ifndef MEGA_MOE_KERNEL_MATH_UTIL_H
#define MEGA_MOE_KERNEL_MATH_UTIL_H

namespace Ops { namespace Base {
template <typename T>
static constexpr T CeilAlign(T num, T align) {
    return ((num + align - 1) / align) * align;
}
template <typename T>
static constexpr T CeilDiv(T num, T div) {
    return (num + div - 1) / div;
}
template <typename T>
static constexpr T FloorAlign(T num, T align) {
    return (num / align) * align;
}
template <typename T>
static constexpr T FloorDiv(T num, T div) {
    return num / div;
}
} }  // namespace Ops::Base

#endif
