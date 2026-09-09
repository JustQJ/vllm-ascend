// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"
#define TQ_GROUP_INLINE __aicore__ inline
#include "../../turbo_quant_sparse_flash_attention/turbo_quant_group_plan.h"
using namespace AscendC;

// Functional prototype: bulk GM->UB loads and bounded scalar integer hashing.
// This avoids float sorting of large token IDs, but hashing is NOT performance tuned.
class GroupPrepareStorage {
public:
    TPipe pipe;
    TBuf<> rawBuffer, hashKeyBuffer, hashPositionBuffer, idBuffer, ownerBuffer, descBuffer;
    LocalTensor<int32_t> raw, hashKeys, hashPositions, ids, descriptors;
    LocalTensor<uint8_t> owners;
    GlobalTensor<int32_t> indicesGm, qGm, kvGm, descGm, idsGm;
    GlobalTensor<uint8_t> ownersGm;
    uint32_t k = 0, windowStart = 0, batch = 0, localStart = 0;

    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR q, GM_ADDR kv, GM_ADDR desc,
                                GM_ADDR unionIds, GM_ADDR owner)
    {
        pipe.InitBuffer(rawBuffer, tq_group::MAX_GROUP * tq_group::MAX_TOPK * sizeof(int32_t));
        pipe.InitBuffer(hashKeyBuffer, tq_group::HASH_SIZE * sizeof(int32_t));
        pipe.InitBuffer(hashPositionBuffer, tq_group::HASH_SIZE * sizeof(int32_t));
        pipe.InitBuffer(idBuffer, tq_group::QUAD_CAP * sizeof(int32_t));
        pipe.InitBuffer(ownerBuffer, tq_group::QUAD_CAP);
        pipe.InitBuffer(descBuffer, tq_group::MAX_GROUP * tq_group::DESC_WORDS * sizeof(int32_t));
        raw = rawBuffer.Get<int32_t>();
        hashKeys = hashKeyBuffer.Get<int32_t>();
        hashPositions = hashPositionBuffer.Get<int32_t>();
        ids = idBuffer.Get<int32_t>();
        owners = ownerBuffer.Get<uint8_t>();
        descriptors = descBuffer.Get<int32_t>();
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices);
        qGm.SetGlobalBuffer((__gm__ int32_t *)q);
        kvGm.SetGlobalBuffer((__gm__ int32_t *)kv);
        descGm.SetGlobalBuffer((__gm__ int32_t *)desc);
        idsGm.SetGlobalBuffer((__gm__ int32_t *)unionIds);
        ownersGm.SetGlobalBuffer((__gm__ uint8_t *)owner);
    }
    __aicore__ inline void ResetHash()
    {
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        Duplicate(hashKeys, int32_t(-1), tq_group::HASH_SIZE);
        SetFlag<HardEvent::V_S>(0);
        WaitFlag<HardEvent::V_S>(0);
    }
    __aicore__ inline int32_t ReadIndex(uint32_t i) { return raw.GetValue(i); }
    __aicore__ inline int32_t HashKey(uint32_t i) { return hashKeys.GetValue(i); }
    __aicore__ inline uint32_t HashPosition(uint32_t i) { return hashPositions.GetValue(i); }
    __aicore__ inline uint32_t Owner(uint32_t i) { return owners.GetValue(i); }
    __aicore__ inline void SetHash(uint32_t i, int32_t id, uint32_t pos)
    {
        hashKeys.SetValue(i, id);
        hashPositions.SetValue(i, pos);
    }
    __aicore__ inline void SetId(uint32_t i, int32_t id) { ids.SetValue(i, id); }
    __aicore__ inline void SetOwner(uint32_t i, uint32_t owner) { owners.SetValue(i, owner); }
    __aicore__ inline void Describe(uint32_t q, int32_t kind, uint32_t count, uint32_t length)
    {
        const uint32_t base = q * tq_group::DESC_WORDS;
        for (uint32_t i = 0; i < tq_group::DESC_WORDS; ++i) {
            descriptors.SetValue(base + i, 0);
        }
        descriptors.SetValue(base + tq_group::KIND, kind);
        descriptors.SetValue(base + tq_group::SIZE, count);
        descriptors.SetValue(base + tq_group::BATCH, batch);
        descriptors.SetValue(base + tq_group::LOCAL_START, localStart + q);
        descriptors.SetValue(base + tq_group::LENGTH, length);
    }
    __aicore__ inline void Publish(uint32_t first, uint32_t length)
    {
        if (length == 0) {
            return;
        }
        SetFlag<HardEvent::S_MTE3>(0);
        WaitFlag<HardEvent::S_MTE3>(0);
        const uint64_t offset = uint64_t(windowStart + first) * k;
        DataCopyExtParams idsCopy{1, static_cast<uint32_t>(length * sizeof(int32_t)), 0, 0, 0};
        DataCopyExtParams ownerCopy{1, length, 0, 0, 0};
        DataCopyPad(idsGm[offset], ids, idsCopy);
        DataCopyPad(ownersGm[offset], owners, ownerCopy);
        // A child attempt may reuse IDs/owners immediately.
        SetFlag<HardEvent::MTE3_S>(0);
        WaitFlag<HardEvent::MTE3_S>(0);
    }
    __aicore__ inline void Run(uint32_t tokens, uint32_t batches, uint32_t topk, uint32_t mode)
    {
        k = topk;
        // Each sequence's <=4-token windows are assigned exactly once. Every
        // core computes the same small prefix of window counts from device data.
        uint32_t start = 0;
        uint64_t ordinal = 0;
        for (batch = 0; batch < batches; ++batch) {
            const int32_t endValue = qGm.GetValue(batch);
            // Invalid cumulative lengths are outside the original op contract.
            if (endValue < int64_t(start) || endValue > tokens) {
                return;
            }
            const uint32_t end = endValue;
            const uint32_t qLen = end - start;
            for (localStart = 0; localStart < qLen; localStart += tq_group::MAX_GROUP, ++ordinal) {
                if (ordinal % GetBlockNum() != GetBlockIdx()) {
                    continue;
                }
                windowStart = start + localStart;
                const uint32_t count = (qLen - localStart < tq_group::MAX_GROUP ? qLen - localStart : tq_group::MAX_GROUP);
                DataCopyExtParams copy{1, static_cast<uint32_t>(count * k * sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
                DataCopyPad(raw, indicesGm[uint64_t(windowStart) * k], copy, pad);
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                tq_group::PlanWindow(*this, count, k, kvGm.GetValue(batch), qLen, localStart, mode);
                SetFlag<HardEvent::S_MTE3>(0);
                WaitFlag<HardEvent::S_MTE3>(0);
                DataCopy(descGm[uint64_t(windowStart) * tq_group::DESC_WORDS], descriptors,
                         count * tq_group::DESC_WORDS);
                SetFlag<HardEvent::MTE3_S>(0);
                WaitFlag<HardEvent::MTE3_S>(0);
            }
            start = end;
        }
        // Capacity-only trailing rows are explicitly zero tasks, refreshed on replay.
        for (uint32_t t = start + GetBlockIdx(); t < tokens; t += GetBlockNum()) {
            localStart = 0;
            Describe(0, tq_group::ZERO, 1, 0);
            descriptors.SetValue(tq_group::BATCH, -1);
            SetFlag<HardEvent::S_MTE3>(0);
            WaitFlag<HardEvent::S_MTE3>(0);
            DataCopy(descGm[uint64_t(t) * tq_group::DESC_WORDS], descriptors, tq_group::DESC_WORDS);
            SetFlag<HardEvent::MTE3_S>(0);
            WaitFlag<HardEvent::MTE3_S>(0);
        }
    }
};

extern "C" __global__ __aicore__ void turbo_quant_group_prepare(
    GM_ADDR indices, GM_ADDR q, GM_ADDR kv, GM_ADDR desc, GM_ADDR unionIds,
    GM_ADDR owners, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    GroupPrepareStorage op;
    op.Init(indices, q, kv, desc, unionIds, owners);
    op.Run(data.tokens, data.batches, data.topk, data.sparseMode);
}
