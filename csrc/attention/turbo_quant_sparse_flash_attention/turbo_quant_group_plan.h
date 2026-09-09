// SPDX-License-Identifier: Apache-2.0
#ifndef TURBO_QUANT_GROUP_PLAN_H
#define TURBO_QUANT_GROUP_PLAN_H

#include <cstdint>

// Shared by the device planner and the dependency-free CPU regression tests.
#ifndef TQ_GROUP_INLINE
#define TQ_GROUP_INLINE inline
#endif

namespace tq_group {
constexpr uint32_t MAX_GROUP = 4;
constexpr uint32_t MAX_TOPK = 2048;
constexpr uint32_t HASH_SIZE = 8192; // load factor <= 3/8 at the quad cap
constexpr uint32_t QUAD_CAP = 3072;
constexpr uint32_t PAIR_CAP = 2560;
constexpr uint32_t DESC_WORDS = 8; // one aligned 32-byte descriptor per query
constexpr uint32_t GROUP_HEADS = 16;
constexpr uint32_t MAX_GROUP_M = MAX_GROUP * GROUP_HEADS;
constexpr uint64_t MAX_METADATA_BYTES = 128ULL * 1024 * 1024;
enum Kind : int32_t { SKIP = 0, GROUP = 1, LEGACY = 2, ZERO = 3 };
enum Field : uint32_t { KIND = 0, SIZE = 1, BATCH = 2, LOCAL_START = 3, LENGTH = 4 };

// Storage owns the hash table and union output for ONE attempt. A rejected
// parent is discarded before rebuilding its children; no truncated union runs.
template <typename Storage>
TQ_GROUP_INLINE bool BuildUnion(Storage &s, uint32_t first, uint32_t count, uint32_t k,
                               int64_t kvLen, int64_t qLen, uint32_t localStart,
                               uint32_t sparseMode, uint32_t cap, uint32_t &length)
{
    s.ResetHash();
    length = 0;
    for (uint32_t q = 0; q < count; ++q) {
        const uint32_t bit = 1U << q;
        const int64_t visibleEnd = sparseMode == 3 ? kvLen - qLen + localStart + q + 1 : kvLen;
        for (uint32_t j = 0; j < k; ++j) {
            const int32_t id = s.ReadIndex((first + q) * k + j);
            // Match the public contract: -1 is trailing padding. Bad positive
            // IDs and future positions are never used for physical addressing.
            if (id < 0 || id >= kvLen || id >= visibleEnd) {
                continue;
            }
            uint32_t slot = (static_cast<uint32_t>(id) * 2654435761U) & (HASH_SIZE - 1);
            bool inserted = false;
            for (uint32_t probe = 0; probe < HASH_SIZE; ++probe) {
                const int32_t previous = s.HashKey(slot);
                if (previous == id) {
                    const uint32_t pos = s.HashPosition(slot);
                    const uint32_t owner = s.Owner(pos);
                    if (owner & bit) {
                        return false; // duplicate in one query: preserve multiplicity via legacy
                    }
                    s.SetOwner(pos, owner | bit);
                    inserted = true;
                    break;
                }
                if (previous < 0) {
                    if (length == cap) {
                        return false; // test BEFORE writing cap+1
                    }
                    s.SetHash(slot, id, length);
                    s.SetId(length, id);
                    s.SetOwner(length, bit);
                    ++length;
                    inserted = true;
                    break;
                }
                slot = (slot + 1) & (HASH_SIZE - 1);
            }
            if (!inserted) {
                return false;
            }
        }
    }
    return true;
}

// The runtime processes disjoint windows of <=4 queries. No global prefix sum,
// atomic task queue, or data-dependent host dispatch is needed.
template <typename Storage>
TQ_GROUP_INLINE void PlanWindow(Storage &s, uint32_t count, uint32_t k, int64_t kvLen,
                               int64_t qLen, uint32_t localStart, uint32_t sparseMode,
                               uint32_t quadCap = QUAD_CAP, uint32_t pairCap = PAIR_CAP)
{
    for (uint32_t q = 0; q < count; ++q) {
        s.Describe(q, SKIP, 0, 0);
    }
    if (kvLen == 0) {
        for (uint32_t q = 0; q < count; ++q) {
            s.Describe(q, ZERO, 1, 0);
        }
        return;
    }
    uint32_t first = 0;
    while (first < count) {
        const uint32_t remaining = count - first;
        uint32_t group = remaining == MAX_GROUP ? MAX_GROUP : remaining >= 2 ? 2 : 1;
        uint32_t length = 0;
        while (group > 1) {
            const uint32_t configuredCap = group == MAX_GROUP ? quadCap : pairCap;
            const uint32_t cap = configuredCap < group * k ? configuredCap : group * k;
            if (BuildUnion(s, first, group, k, kvLen, qLen, localStart + first, sparseMode, cap, length)) {
                s.Publish(first, length); // copy into [(windowStart+first)*K, ...)
                s.Describe(first, length == 0 ? ZERO : GROUP, group, length);
                break;
            }
            group /= 2;
        }
        if (group == 1) {
            const int64_t visibleEnd = sparseMode == 3 ? kvLen - qLen + localStart + first + 1 : kvLen;
            bool hasValid = false;
            for (uint32_t j = 0; j < k; ++j) {
                const int32_t id = s.ReadIndex(first * k + j);
                hasValid |= id >= 0 && id < kvLen && id < visibleEnd;
            }
            s.Describe(first, hasValid ? LEGACY : ZERO, 1, 0);
        }
        first += group;
    }
}
} // namespace tq_group
#endif
