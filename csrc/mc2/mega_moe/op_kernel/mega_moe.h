/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe.h
 * \brief
 */

#ifndef MEGA_MOE_H
#define MEGA_MOE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mc2_kernel_utils.h"
#include "kernel_operator_list_tensor_intf.h"
#include "mega_moe_base.h"
#include "mega_moe_workspace_info.h"
#include "block_epilogue_swiglu_mx_quant.h"
#include "mega_moe_impl.h"


using namespace AscendC;

namespace MegaMoeImpl {
// SyncFuncStatic event IDs — avoid 0/1 which may collide with AscendC EVENT_ID0/EVENT_ID1
constexpr int32_t SYNC_ID2 = 2;

using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;
using BlockOffset = Shape<int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t,
                            int64_t, int64_t>;

#define TemplateMegaMoeTypeClass typename XType, typename OutputType, typename TopkWeightsType, int32_t QuantMode
#define TemplateMegaMoeTypeFunc XType, OutputType, TopkWeightsType, QuantMode

template<int32_t QuantMode>
struct QuantTraits {
    using OutType = fp8_e4m3fn_t;  // 默认
};
template<>
struct QuantTraits<E5M2_QUANT> {
    using OutType = fp8_e5m2_t;
};
template<>
struct QuantTraits<E2M1_QUANT> {
    using OutType = fp4x2_e2m1_t;
};

template <TemplateMegaMoeTypeClass>
class MegaMoe {
public:
    using QuantOutType = typename QuantTraits<QuantMode>::OutType;
    using ActivationType = typename std::conditional<
        Std::IsSame<QuantOutType, fp4x2_e2m1_t>::value,
        uint8_t,
        QuantOutType
    >::type;
    using QuantScaleOutType = typename std::conditional<(QuantMode >= E5M2_QUANT), fp8_e8m0_t, float>::type;
    __aicore__ inline MegaMoe() {};
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights,
        GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2,
        GM_ADDR scales, GM_ADDR yOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM,
        MegaMoeTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void FirstBuffInit();
    __aicore__ inline void SecondBuffInit();
    __aicore__ inline void ResetWorkSpaceFlagList();
    __aicore__ inline void GetCumsumForMMAIV();
    __aicore__ inline void CrossRankSyncCntPerExpert();
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx);
    __aicore__ inline bool UpdateGroupParams(uint32_t expertIdx);
    __aicore__ inline void UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex);
    __aicore__ inline void ArgSortExpandedRowIdx();
    __aicore__ inline void ResetTokenPerExpert();
    __aicore__ inline void Unpermute();
    __aicore__ inline void EndSync();
    __aicore__ inline void EndGMM2Sync();
    __aicore__ inline void CrossRankSyncInWorldSize();
    __aicore__ inline void CopyGMToGMPerToken(GlobalTensor<ActivationType> dst,
                GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<ActivationType> src,
                int32_t rows, int32_t hiddenSize, int32_t& pingpongId);

    __aicore__ inline void TilingByCore(int32_t totalLen, int32_t &coreLen,
        int32_t& coreOffset, int32_t align = ALIGN_32)
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum() * 2; // 取到vec核总数
        int32_t lenPerCore = Ops::Base::CeilDiv(static_cast<uint32_t>(totalLen), static_cast<uint32_t>(coreNum));
        int32_t lenPerCoreAlign = Ops::Base::CeilAlign(static_cast<uint32_t>(lenPerCore), static_cast<uint32_t>(align));
        coreLen = lenPerCoreAlign;
        coreOffset = coreIdx * lenPerCoreAlign;
        if (coreOffset + coreLen >= totalLen) {
            coreLen = totalLen - coreOffset;
        }
        if (coreOffset >= totalLen) {
            coreLen = 0;
        }
    }

    __aicore__ inline int64_t TokenPerExpertLayout(int64_t dim0, int64_t dim1, int64_t dim2)
    {
        return dim0 * perRankStridesInTokenPerExpert_ + dim1 * expertPerRank_ + dim2;
    }

    __aicore__ inline void GmSignalWait(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) != cmp_val) {
                return;
            }
        } while (true);
    }

    __aicore__ inline void GmSignalWaitBarrier(__gm__ int32_t *sig_addr, int32_t cmp_val)
    {
        do {
            if (ReadGmByPassDCache(sig_addr) == cmp_val) {
                return;
            }
        } while (true);
    }

    uint32_t crossRankSyncAddr_ = 0;
    uint32_t crossRankSyncSize_ = 0;
    LocalTensor<int32_t> prevSumBuf_;
    LocalTensor<int32_t> cumsumTmpTensor_;
    LocalTensor<ActivationType> copyTempTensor_;
    LocalTensor<int32_t> resetFlagTensor_;
    LocalTensor<bfloat16_t> dataResBf16_;
    LocalTensor<int32_t> resetTokenTensor_;
    LocalTensor<float> dataResFp32_;

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    Params params_{};
    TupleShape problemShape_{};
    BlockOffset baseOffset_{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    GM_ADDR tokenPerExpertCurRankAddr_;

    GlobalTensor<int32_t> preSumBeforeRankWorkSpace_;
    GlobalTensor<int32_t> tokenPerExpertWin_;
    GlobalTensor<int32_t> cumsumInWorkSpace_;
    GlobalTensor<int32_t> groupFlagListWorkSpace_;
    GlobalTensor<int32_t> groupFlagListGm_;
    GlobalTensor<int32_t> groupFlagListGm2_;
    GlobalTensor<int32_t> groupList_;
    GlobalTensor<int32_t> expertTokenNumsOut_;
    LocalTensor<float> topKWeightsTensor_;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t expertPerRank_ = 0;
    uint32_t groupListType_ = 0;
    uint64_t maxOutputSize_ = 0;
    int32_t prevSum_ = 0;
    int32_t dispatchRows_ = 0;
    int32_t vecSetSyncCom_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t dispatchBlockNumPerEP_ = 2;
    int32_t perRankStridesInTokenPerExpert_ = 0;
    // Per-expert stride into ptrFlagDispatchToGmm1 (in int32 slots); one slot per wave (L1_TILE_M rows).
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum() * 2;
    uint32_t blockIdx_ = GetBlockIdx() / GetTaskRation();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t subBlockIdx_ = GetSubBlockIdx();

    uint64_t preOffset_ = 0;
    uint16_t gmm2PingPongIdx_ = 0;

    using BlockEpilogue = BlockEpilogueSwigluMxQuant<QuantOutType, bfloat16_t,
        QuantScaleOutType, QuantScaleOutType, true>;
    BlockEpilogue epilogueOp_;
};

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Init(
    GM_ADDR context, GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
    GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR yOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, MegaMoeTilingData *tilingData)
{
    m_ = tilingData->bs;
    k_ = tilingData->h;
    topK_ = tilingData->topK;
    worldSize_ = tilingData->epWorldSize;
    expertPerRank_ = tilingData->expertPerRank;
    dispatchBlockNumPerEP_ = tilingData->blockNumPerEP;
    dispatchRows_ = tilingData->dispatchRows;
    groupListType_ = tilingData->groupListType;
    maxOutputSize_ = tilingData->maxOutputSize;
    // Same allocation as the workspace constructor (see WorkspaceInfo::WorkspaceInfo).
    maxWavesPerExpert_ = static_cast<int32_t>(Ops::Base::CeilDiv(
        static_cast<int64_t>(maxOutputSize_), DISPATCH_WAVE_TILE_M));
    dispatchFlagSlotsPerExpert_ = static_cast<int32_t>(Ops::Base::CeilAlign(
        static_cast<int64_t>(maxWavesPerExpert_), static_cast<int64_t>(INT_CACHELINE)));
    Get<N_VALUE>(problemShape_) = tilingData->hiddenDim;
    Get<K_VALUE>(problemShape_) = k_;
    mc2Context_ = reinterpret_cast<__gm__ Mc2MoeContext*>(context);
    rankId_ = mc2Context_->epRankId;
    for (int i = 0; i < worldSize_; i++) {
        winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer[i];
    }
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    params_.bGmAddr = GetTensorAddr(0, weight1);
    params_.b2GmAddr = GetTensorAddr(0, weight2);
    params_.bScaleGmAddr = GetTensorAddr(0, weightScales1);
    params_.b2ScaleGmAddr = GetTensorAddr(0, weightScales2);

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    params_.workspaceInfo = WorkspaceInfo(workspaceGM, tilingData);
    params_.peermemInfo = PeermemInfo(winRankAddr_[rankId_], tilingData);
    params_.tilingData = tilingData;
    preSumBeforeRankWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrSumBeforeRank);
    tokenPerExpertWin_.SetGlobalBuffer((__gm__ int32_t*)params_.peermemInfo.ptrTokenPerExpert);
    cumsumInWorkSpace_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(params_.workspaceInfo.ptrcumsumMM));
    expertTokenNumsOut_.SetGlobalBuffer((__gm__ int32_t*)params_.expertTokenNumsOutGmAddr);
    groupList_ = cumsumInWorkSpace_[(worldSize_ - 1) * expertPerRank_];
    groupFlagListWorkSpace_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    epilogueOp_.Init({params_.workspaceInfo.ptrA2, params_.workspaceInfo.ptrA2Scale,
        params_.workspaceInfo.ptrFlagSwiGluToGmm2, nullptr, nullptr, nullptr, ALIGN_256, ALIGN_256});
    perRankStridesInTokenPerExpert_ = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    tokenPerExpertCurRankAddr_ = params_.peermemInfo.ptrTokenPerExpert +
        TokenPerExpertLayout(rankId_, 0, 0) * sizeof(int32_t);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::FirstBuffInit()
{
    uint32_t syncBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    uint32_t prevSumBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128))
        * sizeof(int32_t) * expertPerRank_;
    uint32_t getCumsumBufAlign = (AlignUp(worldSize_ * expertPerRank_, ALIGN_128)) * worldSize_ * sizeof(int32_t);
    uint32_t copyTempBufAlign = BUFFER_ALIGN * sizeof(ActivationType);
    // Total int32 flag entries to reset: SwiGluToGmm2 region (expertPerRank * INT_CACHELINE)
    // + DispatchToGmm1 region (expertPerRank * dispatchFlagSlotsPerExpert_).
    int32_t totalFlagInt32 = static_cast<int32_t>(expertPerRank_) *
        (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_);
    uint32_t maxResetBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(totalFlagInt32 * sizeof(int32_t)), static_cast<uint32_t>(ALIGN_32));

    uint32_t addr = 0;
    // [0] crossRankSync: 跨rank同步临时buffer，存放tokenPerExpert拷贝及加偏移值
    // 大小 = CeilAlign(worldSize * expertPerRank, 128) * sizeof(int32_t)，按128对齐
    crossRankSyncAddr_ = addr;
    crossRankSyncSize_ = syncBufAlign / sizeof(int32_t);
    addr += syncBufAlign;

    // [1] prevSumBuf: 每个expert在本rank之前的累加token数前缀和
    // 大小 = CeilAlign(worldSize * expertPerRank, 128) * sizeof(int32_t) * expertPerRank
    // 每个dstRank需要expertPerRank个int32，共worldSize组，128列对齐后乘expertPerRank
    prevSumBuf_ = LocalTensor<int32_t>(TPosition::VECCALC, addr, prevSumBufAlign / sizeof(int32_t));
    addr += prevSumBufAlign;
    // [2] cumsumTmpTensor: cumsum计算临时buffer， 存放各rank的tokenPerExpert并做前缀累加
    // 大小 = CeilAlign(worldSize * expertPerRank, 128) * worldSize * sizeof(int32_t)
    // 每个rank一行expertPerRank列，worldSize行，128列对齐
    cumsumTmpTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, addr, getCumsumBufAlign / sizeof(int32_t));
    addr += getCumsumBufAlign;
    // [3] copyTempTensor: Dispatch阶段GM→GM中转buffer， pingpong双缓冲用
    // 大小 = BUFFER_ALIGN * sizeof(ActivationType)，固定BUFFER_ALIGN大小
    copyTempTensor_ = LocalTensor<ActivationType>(TPosition::VECCALC, addr, copyTempBufAlign / sizeof(ActivationType));
    addr += copyTempBufAlign;
    // [4] resetFlagTensor: groupFlagList清零临时buffer
    // 大小 = CeilAlign(expertPerRank * (INT_CACHELINE + dispatchFlagSlots) * sizeof(int32_t), 32)
    // SwiGluToGmm2 + DispatchToGmm1 两个区域的总flag条目数，32对齐
    resetFlagTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, addr, maxResetBufAlign / sizeof(int32_t));
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::SecondBuffInit()
{
    uint32_t dataResBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(UNPERMUTE_LIST_NUM * k_ * sizeof(bfloat16_t)), static_cast<uint32_t>(ALIGN_32));
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
        static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128)) * sizeof(int32_t);
    uint32_t dataResFp32BufAlign = dataResBufAlign * 2;
    uint32_t topKWeightsBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    uint32_t tempBufAlign = Ops::Base::CeilAlign(
        static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), uint32_t(ALIGN_32));

    uint32_t addr = 0;
    // [0] dataResBf16: Unpermute阶段bf16结果buffer，存放加权和后的输出行
    // size = CeilAlign(UNPERMUTE_LIST_NUM * k * sizeof(bfloat16_t), 32)
    // UNPERMUTE_LIST_NUM行每行k个bf16，32对齐
    dataResBf16_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, addr, dataResBufAlign / sizeof(bfloat16_t));
    addr += dataResBufAlign;
    // [1] resetTokenTensor: ResetTokenPerExpert清零tokenPerExpert win区用的临时buffer
    // size = worldSize * CeilAlign(worldSize * expertPerRank, 128) * sizeof(int32_t)
    // 需清零的总int32条目数（跨所有rank的所有expert），128列对齐
    resetTokenTensor_ = LocalTensor<int32_t>(TPosition::VECCALC, addr, num / sizeof(int32_t));
    addr += num;
    // [2] dataResFp32: Unpermute阶段fp32累加buffer，bf16→fp32计算加权和
    // size = dataResBufAlign * 2（bf16大小的2倍，因fp32占2字节宽度）
    dataResFp32_ = LocalTensor<float>(TPosition::VECCALC, addr, dataResFp32BufAlign / sizeof(float));
    addr += dataResFp32BufAlign;
    // [3] topKWeightsTensor: topK权重buffer，存放m*topK个权重值用于Unpermute加权
    // size = CeilAlign(m * topK * sizeof(float), 32)
    topKWeightsTensor_ = LocalTensor<float>(TPosition::VECCALC, addr, topKWeightsBufAlign / sizeof(float));
    addr += topKWeightsBufAlign;

    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        GlobalTensor<float> topKWeightsGlobalTensor_;
        topKWeightsGlobalTensor_.SetGlobalBuffer((__gm__ float*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(float)), 0U, 0U, 0U};
        DataCopyPadExtParams<float> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(topKWeightsTensor_, topKWeightsGlobalTensor_, copyParams, copyPadParams);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        LocalTensor<bfloat16_t> tempLocal(TPosition::VECCALC, addr, tempBufAlign / sizeof(bfloat16_t));
        GlobalTensor<bfloat16_t> topkWeightsGlobalTensor;
        topkWeightsGlobalTensor.SetGlobalBuffer((__gm__ bfloat16_t*)params_.probsGmAddr);
        DataCopyExtParams copyParams = {1U, static_cast<uint32_t>(m_ * topK_ * sizeof(bfloat16_t)), 0U, 0U, 0U};
        DataCopyPadExtParams<bfloat16_t> copyPadParams{false, 0U, 0U, 0U};
        DataCopyPad(tempLocal, topkWeightsGlobalTensor, copyParams, copyPadParams);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_ID2>();
        Cast(topKWeightsTensor_, tempLocal, AscendC::RoundMode::CAST_NONE, m_ * topK_);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetWorkSpaceFlagList()
{
    // workSpace groupFlagList清零（含 dispatch->gmm1 wave-grain 槽位区）
    groupFlagListGm_.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2);
    if constexpr(g_coreType == AIV) {
        int32_t flagNum = static_cast<int32_t>(expertPerRank_) *
            (static_cast<int32_t>(INT_CACHELINE) + dispatchFlagSlotsPerExpert_);
        int32_t coreLen, coreOffset;
        TilingByCore(flagNum, coreLen, coreOffset);
        if (coreLen != 0) {
            Duplicate(resetFlagTensor_, 0, coreLen);
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_ID2>();
            DataCopy(groupFlagListGm_[coreOffset], resetFlagTensor_, coreLen);
        }
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::GetCumsumForMMAIV()
{
    uint32_t expertPerRankAligned = (expertPerRank_ + 8 - 1) / 8 * 8;
    LocalTensor<int32_t> tmpTensor = cumsumTmpTensor_;
    DataCopyPad(tmpTensor, tokenPerExpertWin_[rankId_ * expertPerRank_],
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)),
         static_cast<uint16_t>((AlignUp(worldSize_ * expertPerRank_, ALIGN_128)
             - expertPerRank_) * sizeof(int32_t)), 0},
        {});
    SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_ID2>();

    for (uint32_t i = 1; i < worldSize_; ++i) {
        Add(tmpTensor[i * expertPerRankAligned], tmpTensor[i * expertPerRankAligned],
            tmpTensor[(i - 1) * expertPerRankAligned], expertPerRank_);
        PipeBarrier<PIPE_V>();
    }
    SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_ID2>();
    DataCopyPad(cumsumInWorkSpace_, tmpTensor,
        {static_cast<uint16_t>(worldSize_), static_cast<uint16_t>((expertPerRank_) * sizeof(int32_t)), 0, 0});
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ExpertTokenNumCopyOut()
{
    int32_t copyNum = Ops::Base::CeilAlign(static_cast<uint32_t>(expertPerRank_), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<int32_t> tmpCopyBuffer(TPosition::VECCALC, crossRankSyncAddr_, crossRankSyncSize_);
    DataCopy(tmpCopyBuffer, groupList_, copyNum);
    SyncFuncStatic<AscendC::HardEvent::MTE2_MTE3, SYNC_ID2>();
    DataCopy(expertTokenNumsOut_, tmpCopyBuffer, copyNum);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EndSync()
{
    if (vecSetSyncCom_ == 0) {
        return;
    }
    if constexpr(g_coreType == AIC) {
        WaitForVector();
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::EndGMM2Sync()
{
    if constexpr(g_coreType == AIV) {
        return;
    }
    if (vecSetSyncCom_ <= 0) {
        return;
    } else if (vecSetSyncCom_ == 1) {
        WaitForVector();
    } else {
        WaitForVector(gmm2PingPongIdx_);
        WaitForVector(1 - gmm2PingPongIdx_);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncCntPerExpert()
{
    if constexpr(g_coreType == AIC) {
        return;
    }
    uint32_t numPerCore = Ops::Base::CeilAlign(static_cast<uint32_t>(worldSize_ * expertPerRank_),
            static_cast<uint32_t>(ALIGN_128));
    LocalTensor<int32_t> tmpBuffer(TPosition::VECCALC, crossRankSyncAddr_, crossRankSyncSize_);
    LocalTensor<int32_t> prevSumBuf = prevSumBuf_;
    // 给对端发送
    uint64_t offsetInDstWinAddr = perRankStridesInTokenPerExpert_ * sizeof(int32_t) * rankId_ + PEERMEM_DATA_OFFSET;
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        if (dstRankIdx == rankId_) {
            continue;
        }
        GlobalTensor<int32_t> srcAddress;
        srcAddress.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(tokenPerExpertCurRankAddr_));
        GlobalTensor<int32_t> dstAddress;
        __gm__ void* dstPeermemPtr = GetRankWinAddrWithOffset(dstRankIdx, offsetInDstWinAddr);
        dstAddress.SetGlobalBuffer((__gm__ int32_t*)dstPeermemPtr);
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_ID2>();
        DataCopy(tmpBuffer, srcAddress, numPerCore);
        SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_ID2>();
        Adds(tmpBuffer, tmpBuffer, 0x800000, numPerCore);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_ID2>();
        DataCopy(dstAddress, tmpBuffer, numPerCore);
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_ID2>();
    }
    // 本端接收
    int32_t intPer512 = ALIGN_512 / sizeof(int);
    for (int32_t dstRankIdx = aivCoreIdx_; dstRankIdx < worldSize_; dstRankIdx += blockAivNum_) {
        if (dstRankIdx != rankId_) {
            for (int32_t checkIdx = 0; checkIdx < numPerCore; checkIdx += intPer512) {
                __gm__ int32_t* syncCheck = reinterpret_cast<__gm__ int32_t*>(params_.peermemInfo.ptrTokenPerExpert)
            + TokenPerExpertLayout(dstRankIdx, 0, checkIdx);
                GmSignalWait(syncCheck, 0);
            }
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], numPerCore);
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_ID2>();
            Adds(tmpBuffer, tmpBuffer, -0x800000, numPerCore);
            PipeBarrier<PIPE_V>();
            SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_ID2>();
            DataCopy(tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], tmpBuffer, numPerCore);
        } else {
            DataCopy(tmpBuffer, tokenPerExpertWin_[TokenPerExpertLayout(dstRankIdx, 0, 0)], numPerCore);
            SyncFuncStatic<AscendC::HardEvent::MTE2_V, SYNC_ID2>();
        }
        PipeBarrier<PIPE_ALL>();
        int32_t tmpSum = 0;
        int32_t j = 0;
        for (int32_t i = 0; i < (rankId_ + 1) * expertPerRank_; i++) {
            if (i >= rankId_ * expertPerRank_) {
                prevSumBuf(j) = tmpSum;
                j++;
            }
            tmpSum += tmpBuffer(i);
        }
        SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_ID2>();
        DataCopyPad(preSumBeforeRankWorkSpace_[dstRankIdx * expertPerRank_], prevSumBuf,
                    DataCopyParams{1, static_cast<uint16_t>(expertPerRank_ * sizeof(int32_t)), 0, 0});
    }
    SyncAll<true>();
    if (aivCoreIdx_ == 0) {
        GetCumsumForMMAIV();
    }
    if (blockIdx_ < worldSize_ * dispatchBlockNumPerEP_) {
        prevSum_ = preSumBeforeRankWorkSpace_(blockIdx_ / dispatchBlockNumPerEP_ * expertPerRank_);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CopyGMToGMPerToken(
    GlobalTensor<ActivationType> dst, GlobalTensor<QuantScaleOutType> dstScale, GlobalTensor<ActivationType> src,
    int32_t rows, int32_t hiddenSize, int32_t& pingpongId)
{
    constexpr int32_t BufferNum = 2;
    LocalTensor<ActivationType> tmpTensor1 = copyTempTensor_;
    LocalTensor<ActivationType> tmpTensor2 = tmpTensor1[96 * 1024];
    uint32_t scaleLen = Ops::Base::CeilDiv(static_cast<int32_t>(k_),
        static_cast<int32_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    uint32_t copyInNum = hiddenSize + scaleLen;
    auto processCount = Ops::Base::CeilDiv(rows, dispatchRows_);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        pingpongId = (pingpongId + 1) % BufferNum;
        TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        LocalTensor<ActivationType> buf = pingpongId == 0 ? tmpTensor1 : tmpTensor2;
        LocalTensor<QuantScaleOutType> bufScale = buf[hiddenSize].template ReinterpretCast<QuantScaleOutType>();

        auto inputOffset = processIndex * dispatchRows_ * copyInNum;

        int32_t rowNum = dispatchRows_;
        if (processIndex == processCount - 1) {
            rowNum = rows - processIndex * dispatchRows_;
        }
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        int64_t dataLen = rowNum * copyInNum;
        DataCopy(buf, src[inputOffset], dataLen);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        auto outputOffset = processIndex * dispatchRows_ * hiddenSize;
        DataCopyPad(dst[outputOffset], buf,
            {static_cast<uint16_t>(rowNum), static_cast<uint16_t>(hiddenSize),
            static_cast<uint16_t>(scaleLen / ALIGN_32), 0, 0});
        DataCopyPad(dstScale[processIndex * dispatchRows_ * scaleLen], bufScale,
            {static_cast<uint16_t>(rowNum), static_cast<uint16_t>(scaleLen),
             static_cast<uint16_t>(hiddenSize / ALIGN_32), 0, 0});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
    }
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::DispatchTokens(GMMAddrInfo &gmmAddrInfo, uint32_t expertIdx)
{
    int64_t widthA; // H 长度
    if constexpr(QuantMode == E2M1_QUANT) {
        widthA = Get<K_VALUE>(problemShape_) / FP4_NUM;
    } else {
        widthA = Get<K_VALUE>(problemShape_);
    }
    int64_t widthAScale = Ops::Base::CeilDiv(Get<K_VALUE>(problemShape_),
                            static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    int32_t pingpongId = 0;
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    // wave-grain dispatch flag: 每个 expert 拥有 maxWavesPerExpert_ 个槽位，
    // 每个槽位累计该 wave (L1_TILE_M rows) 实际收到的行数。
    // 当槽位值达到 min(L1_TILE_M, m - waveIdx * L1_TILE_M) 时，gmm1 可以早启动该 wave。
    constexpr int32_t L1_TILE_M_I32 = static_cast<int32_t>(MegaMoeImpl::L1_TILE_M);
    for (uint32_t idx = blockIdx_; idx < worldSize_ * dispatchBlockNumPerEP_; idx += blockNum_) {
        uint32_t dstRankIdx = idx / dispatchBlockNumPerEP_;
        uint32_t aivIdxInGroup = idx % dispatchBlockNumPerEP_;
        uint32_t rowStartInDst = (dstRankIdx == 0 ? 0 :
            cumsumInWorkSpace_((dstRankIdx - 1) * expertPerRank_ + expertIdx));
        int32_t rowsThisCore = 0;
        int32_t rowDstOffsetPerCore = 0;
        if (rowStartInDst < maxOutputSize_) {
            int32_t rowsCopyCnt = tokenPerExpertWin_(TokenPerExpertLayout(dstRankIdx, rankId_, expertIdx));
            if (rowStartInDst + rowsCopyCnt > maxOutputSize_) {
                rowsCopyCnt = maxOutputSize_ - rowStartInDst;
            }
            int32_t rowSrcOffset = prevSum_;
            prevSum_ += rowsCopyCnt;
            int32_t rowsPerCore = Ops::Base::CeilDiv(rowsCopyCnt, static_cast<int32_t>(dispatchBlockNumPerEP_));
            rowDstOffsetPerCore = rowStartInDst + aivIdxInGroup * rowsPerCore;
            int32_t rowSrcOffsetPerCore = rowSrcOffset + aivIdxInGroup * rowsPerCore;
            if (rowDstOffsetPerCore + rowsPerCore > rowStartInDst + rowsCopyCnt) {
                rowsPerCore = rowStartInDst + rowsCopyCnt - rowDstOffsetPerCore;
            }
            if (rowsPerCore > 0) {
                GlobalTensor <ActivationType> gmRemoteA, aGlobalTensor;
                GlobalTensor <QuantScaleOutType> aScaleGlobalTensor;
                int64_t gmOffsetA = rowDstOffsetPerCore * widthA;
                int64_t gmOffsetAScale = rowDstOffsetPerCore * widthAScale;
                int64_t gmOffsetPeer = rowSrcOffsetPerCore * (widthA + widthAScale);
                gmRemoteA.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(GetRankWinAddrWithOffset(dstRankIdx,
                    PEERMEM_DATA_OFFSET + perRankStridesInTokenPerExpert_ * worldSize_ * sizeof(uint32_t))));
                aGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType*>(gmmAddrInfo.aGlobal));
                aScaleGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ QuantScaleOutType *>(
                    gmmAddrInfo.aScaleGlobal));
                CopyGMToGMPerToken(aGlobalTensor[gmOffsetA], aScaleGlobalTensor[gmOffsetAScale],
                    gmRemoteA[gmOffsetPeer], rowsPerCore, widthA, pingpongId);
                rowsThisCore = rowsPerCore;
            }
        }
        if (rowsThisCore > 0) {
            // drain MTE3 后再按 wave 边界做 AtomicAdd
            SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_ID2>();
            int32_t rowEnd = rowDstOffsetPerCore + rowsThisCore;
            int32_t waveLo = rowDstOffsetPerCore / L1_TILE_M_I32;
            int32_t waveHi = (rowEnd - 1) / L1_TILE_M_I32;
            __gm__ int32_t* flagBase = (__gm__ int32_t*)(groupFlagListGm2_.GetPhyAddr());
            for (int32_t w = waveLo; w <= waveHi; ++w) {
                int32_t waveStartRow = w * L1_TILE_M_I32;
                int32_t waveEndRow = waveStartRow + L1_TILE_M_I32;
                int32_t lo = rowDstOffsetPerCore > waveStartRow ? rowDstOffsetPerCore : waveStartRow;
                int32_t hi = rowEnd < waveEndRow ? rowEnd : waveEndRow;
                AtomicAdd(flagBase + w, int32_t(hi - lo));
            }
        }
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline bool MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGroupParams(uint32_t expertIdx)
{
    // baseOffset_ is 0 when groupIdx = 0
    if (expertIdx != 0) {
        uint64_t m = Get<M_VALUE>(problemShape_);
        uint64_t n = Get<N_VALUE>(problemShape_);
        uint64_t k = Get<K_VALUE>(problemShape_);
        if constexpr (QuantMode == E2M1_QUANT) {
            Get<IDX_A_OFFSET>(baseOffset_) += (m * k) / FP4_NUM;
            Get<IDX_B_OFFSET>(baseOffset_) += (n * k) / FP4_NUM;
        } else {
            Get<IDX_A_OFFSET>(baseOffset_) += (m * k);
            Get<IDX_B_OFFSET>(baseOffset_) += (n * k);
        }
        // only splitM
        auto scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) += m * scaleK;
        Get<IDX_B_SCALE_OFFSET>(baseOffset_) += n * scaleK;
        Get<IDX_C_OFFSET>(baseOffset_) += m * n / SWIGLU_N_HALF;
        Get<IDX_C_SCALE_OFFSET>(baseOffset_) +=
            m * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_FLAG_OFFSET>(baseOffset_) += 1;
        if constexpr (QuantMode == E2M1_QUANT) {
            Get<IDX_B2_OFFSET>(baseOffset_) += (k * n / SWIGLU_N_HALF) / FP4_NUM;
        } else {
            Get<IDX_B2_OFFSET>(baseOffset_) += (k * n / SWIGLU_N_HALF);
        }
        Get<IDX_B2_SCALE_OFFSET>(baseOffset_) +=
            k * Ops::Base::CeilDiv(n / SWIGLU_N_HALF, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        Get<IDX_Y2_OFFSET>(baseOffset_) += m * k;
        Get<IDX_M_OFFSET>(baseOffset_) += m;
    }

    int32_t splitValue = 0;
    if (groupListType_ == 0) {
        int32_t offset = static_cast<int32_t>(groupList_.GetValue(expertIdx));
        splitValue = offset - preOffset_;
        preOffset_ = offset;
    } else {
        splitValue = static_cast<uint64_t>(groupList_.GetValue(expertIdx));
    }
    Get<M_VALUE>(problemShape_) = splitValue;
    // split_m，when m=0, skip
    if (Get<M_VALUE>(problemShape_) == 0) {
        return false;
    }
    return true;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::UpdateGlobalBuffer(GMMAddrInfo &gmmAddrInfo, uint32_t gmmIndex)
{
    if (gmmIndex == 0) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA + Get<IDX_A_OFFSET>(baseOffset_) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrAScale +
        Get<IDX_A_SCALE_OFFSET>(baseOffset_) * sizeof(QuantScaleOutType);
        if constexpr(g_coreType == AIC) {
            gmmAddrInfo.bGlobal = params_.bGmAddr + Get<IDX_B_OFFSET>(baseOffset_) * sizeof(ActivationType);
            gmmAddrInfo.bScaleGlobal = params_.bScaleGmAddr + Get<IDX_B_SCALE_OFFSET>(baseOffset_) *
                                        sizeof(QuantScaleOutType);
        } else {
            AscendC::Coord<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> vecBaseOffset{
                Get<IDX_C_OFFSET>(baseOffset_), Get<IDX_C_SCALE_OFFSET>(baseOffset_),
                Get<IDX_FLAG_OFFSET>(baseOffset_), 0L, 0L, 0L};
            epilogueOp_.UpdateGlobalAddr(vecBaseOffset);
        }
    } else if constexpr(g_coreType == AIC) {
        gmmAddrInfo.aGlobal = params_.workspaceInfo.ptrA2 + Get<IDX_C_OFFSET>(baseOffset_) * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal = params_.workspaceInfo.ptrA2Scale + Get<IDX_C_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = params_.b2GmAddr + Get<IDX_B2_OFFSET>(baseOffset_) * sizeof(ActivationType);
        gmmAddrInfo.bScaleGlobal = params_.b2ScaleGmAddr + Get<IDX_B2_SCALE_OFFSET>(baseOffset_) *
                                    sizeof(QuantScaleOutType);
    }
    gmmAddrInfo.groupFlagList = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagSwiGluToGmm2 +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * INT_CACHELINE;
    // wave-grain: per-expert stride is dispatchFlagSlotsPerExpert_ (one slot per L1_TILE_M wave).
    gmmAddrInfo.groupFlagList2 = (__gm__ int32_t*)params_.workspaceInfo.ptrFlagDispatchToGmm1 +
                                Get<IDX_FLAG_OFFSET>(baseOffset_) * dispatchFlagSlotsPerExpert_;
    groupFlagListGm_.SetGlobalBuffer(gmmAddrInfo.groupFlagList);
    groupFlagListGm2_.SetGlobalBuffer(gmmAddrInfo.groupFlagList2);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ArgSortExpandedRowIdx()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_ * topK_, coreLen, coreOffset, 1);
    __gm__ int32_t *src = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdx;
    __gm__ int32_t *dst = (__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather;
    Simt::VF_CALL<ArgsortSimt>(Simt::Dim3{MoeInitRoutingV3::SIMT_THREAD_NUM, 1, 1}, coreLen, coreOffset, src, dst);
    PipeBarrier<PIPE_ALL>();
    DataSyncBarrier<MemDsbT::DDR>();
    GlobalTensor<int32_t> rowIndexGatherGm;
    rowIndexGatherGm.SetGlobalBuffer((__gm__ int32_t *)params_.workspaceInfo.expandedRowIdxGather);
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(rowIndexGatherGm);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::ResetTokenPerExpert()
{
    if (aivCoreIdx_ != blockAivNum_ - 1) {
        return;
    }
    int32_t num = worldSize_ * Ops::Base::CeilAlign(
            static_cast<uint32_t>(worldSize_ * expertPerRank_), static_cast<uint32_t>(ALIGN_128));
    SyncFuncStatic<AscendC::HardEvent::MTE3_V, SYNC_ID2>();
    LocalTensor<int32_t> tmp = resetTokenTensor_;
    Duplicate(tmp, 0, num);
    PipeBarrier<PIPE_ALL>(); // 保证前侧combine做完了再清理
    DataCopy(tokenPerExpertWin_, tmp, num);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Unpermute()
{
    int32_t coreLen, coreOffset;
    TilingByCore(m_, coreLen, coreOffset, 1);
    GlobalTensor<int32_t> expandedRowIdx;
    expandedRowIdx.SetGlobalBuffer((__gm__ int32_t*)params_.workspaceInfo.expandedRowIdxGather);
    GlobalTensor<bfloat16_t> expandedX;
    expandedX.SetGlobalBuffer((__gm__ bfloat16_t*)params_.peermemInfo.ptrD);
    GlobalTensor<bfloat16_t> output;
    output.SetGlobalBuffer((__gm__ bfloat16_t*)params_.y2GmAddr);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    for (int32_t tokenIdx = coreOffset; tokenIdx < coreLen + coreOffset; tokenIdx++) {
        SyncFuncStatic<AscendC::HardEvent::MTE3_MTE2, SYNC_ID2>();
        LocalTensor<bfloat16_t> dataResBf16 = dataResBf16_;
        LocalTensor<bfloat16_t> dataIn0Bf16 = dataResBf16[k_];
        LocalTensor<bfloat16_t> dataIn1Bf16 = dataResBf16[k_ * 2];
        LocalTensor<float> dataResFp32 = dataResFp32_;
        LocalTensor<float> dataIn0Fp32 = dataResFp32[k_];
        LocalTensor<float> dataIn1Fp32 = dataResFp32[k_ * 2];
        for (int32_t expId = 0; expId < topK_; ++expId) {
            int32_t rowIdx = expandedRowIdx(tokenIdx * topK_ + expId);
            float expScale = topKWeightsTensor_.GetValue(tokenIdx * topK_ + expId);
            auto event = (expId % 2 == 0) ? EVENT_ID0 : EVENT_ID1;
            auto dataInBf16 = (expId % 2 == 0) ? dataIn0Bf16 : dataIn1Bf16;
            auto dataInFp32 = (expId % 2 == 0) ? dataIn0Fp32 : dataIn1Fp32;
            WaitFlag<AscendC::HardEvent::V_MTE2>(event);
            DataCopy(dataInBf16, expandedX[rowIdx * k_], k_);
            SetFlag<AscendC::HardEvent::MTE2_V>(event);
            WaitFlag<AscendC::HardEvent::MTE2_V>(event);
            SetFlag<AscendC::HardEvent::S_V>(event);
            WaitFlag<AscendC::HardEvent::S_V>(event);
            // bf16 -> fp32
            Cast(dataInFp32, dataInBf16, AscendC::RoundMode::CAST_NONE, k_);
            PipeBarrier<PIPE_V>();
            if (expId == 0) {
                Muls(dataResFp32, dataInFp32, expScale, k_);
            } else {
                Muls(dataInFp32, dataInFp32, expScale, k_);
                PipeBarrier<PIPE_V>();
                Add(dataResFp32, dataResFp32, dataInFp32, k_);
                PipeBarrier<PIPE_V>();
            }
            SetFlag<AscendC::HardEvent::V_MTE2>(event);
        }
        // fp32 -> bf16
        Cast(dataResBf16, dataResFp32, AscendC::RoundMode::CAST_RINT, k_);
        SyncFuncStatic<AscendC::HardEvent::V_MTE3, SYNC_ID2>();
        DataCopy(output[tokenIdx * k_], dataResBf16, k_);
    }
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::CrossRankSyncInWorldSize()
{
    __gm__ int32_t* syncRank = (__gm__ int32_t*)(params_.peermemInfo.ptrBase);
    __gm__ int32_t* syncCount = (__gm__ int32_t*)(params_.peermemInfo.ptrBase + 48 * 1024);
    int count = *(syncCount + aivCoreIdx_ * 16) + 1;
    for (int i = aivCoreIdx_; i < worldSize_; i += blockAivNum_) {
        __gm__ int32_t* syncRemoteAddr = (__gm__ int32_t*)(winRankAddr_[i]) + rankId_ * 16;
        WriteGmByPassDCache(syncRemoteAddr, count);
        auto syncCheck = syncRank + i * 16;
        GmSignalWaitBarrier(syncCheck, count);
    }
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    *(syncCount + aivCoreIdx_ * 16) = count;
}

template <TemplateMegaMoeTypeClass>
__aicore__ inline void MegaMoe<TemplateMegaMoeTypeFunc>::Process()
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    // 1. moe_init_routing_v3 + 量化
    moe_init_routing_v3<bfloat16_t, QuantOutType>(params_.aGmAddr, params_.expertIdxGmAddr, nullptr, nullptr,
            params_.peermemInfo.ptrA0, params_.workspaceInfo.expandedRowIdx, tokenPerExpertCurRankAddr_, nullptr,
            params_.workspaceInfo.ptrA, &params_.tilingData->moeInitRoutingTilingData,
            params_.tilingData->moeInitRoutingTilingData.tilingKey);
    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();

    // 2. buffer申请 & workspace Init
    FirstBuffInit();
    ResetWorkSpaceFlagList();

    // 3. cnt值全ep域内的发送以及本端接收整理
    CrossRankSyncCntPerExpert();
    if (subBlockIdx_ == 0) {
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll<false>();

    // 4. for循环按照本卡专家id进行Dispatch，内部gmm1SwigluQuant按照基本块进行流水
    GMMAddrInfo gmmAddrInfo;
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 0);
        if constexpr(g_coreType == AscendC::AIV) {
            if (subBlockIdx_ == 1) {
                DispatchTokens(gmmAddrInfo, expertIdx);
            }
        }

        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmulSwigluQuant<
                QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                epilogueOp_, params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_);
        }
    }
    EndSync();
    if constexpr(g_coreType == AscendC::AIV) {
        if (aivCoreIdx_ == 1) {
            ExpertTokenNumCopyOut();
        }
        if (subBlockIdx_ == 1) {
            PipeBarrier<PIPE_ALL>();
        }
    }

    // 5. for循环按照本卡专家id进行gmm2 & combine，内部gmm2与combine之间按照基本块进行流水
    preOffset_ = 0;
    vecSetSyncCom_ = 0;
    baseOffset_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (uint32_t expertIdx = 0; expertIdx < expertPerRank_; expertIdx++) {
        if (!UpdateGroupParams(expertIdx)) {
            continue;
        }
        UpdateGlobalBuffer(gmmAddrInfo, 1);
        if (subBlockIdx_ == 0) {
            MegaMoeImpl::GroupMatmul2<QuantOutType, QuantOutType, bfloat16_t, QuantScaleOutType, QuantScaleOutType>(
                params_, problemShape_, gmmAddrInfo, startBlockIdx_, vecSetSyncCom_, expertIdx, gmm2PingPongIdx_,
                rankId_);
        }
    }
    EndGMM2Sync();
    SyncAll<true>();

    // 6. unpermute & 清理win区上放置token的位置
    if constexpr(g_coreType == AIV) {
        ArgSortExpandedRowIdx();
        SecondBuffInit();
        ResetTokenPerExpert();
        CrossRankSyncInWorldSize(); // 全卡软同步，确认combine send完成
        Unpermute();
        PipeBarrier<PIPE_ALL>();
    }
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

}   // namespace MegaMoeImpl
#endif