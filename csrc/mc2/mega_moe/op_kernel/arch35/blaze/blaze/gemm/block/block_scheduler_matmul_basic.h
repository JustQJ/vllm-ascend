/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file block_scheduler_matmul_basic.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/common_utils.h"

namespace Blaze {
namespace Gemm {
namespace Block {
constexpr uint16_t A_FULL_LOAD_MODE = 1;
constexpr uint16_t B_FULL_LOAD_MODE = 2;
constexpr int64_t FP32_K_SWITCH_THRESHOLD = 268435456; // 1024 * 32 * 8192
constexpr int64_t FP32_SPLIT_K_THRESHOLD1 = 1024;
constexpr int64_t FP32_SPLIT_K_THRESHOLD2 = 8192;

template <class ProblemShape_, int64_t FullLoadMode_ = 0>
class BlockSchedulerMatmulBasic {
public:
    int64_t mTileNum_{0};
    int64_t nTileNum_{0};
    int64_t kTileNum_{0};
    int64_t blockIdx_{0};
    int64_t perCoreBlockNum_{0};
    int64_t blockNum_{0};
    int64_t batch_{0};
    int64_t innerBatch_{0};
    int64_t k_{0};
    int64_t tailL1M_{0};
    int64_t tailL1N_{0};
    int64_t mTailCnt_{1};
    int64_t nTailCnt_{1};
    int64_t tailCnt_{1};
    int64_t tileNum_{1};
    int64_t mainWindow_{1};
    int64_t mainRow_{1};
    int64_t tailWindow_{1};
    int64_t mTileIdx_{1};
    int64_t nTileIdx_{1};
    int64_t splitSingleKIdx_{0};
    int64_t lastTileIdx_{-1};
    int64_t nSplitOffset_{0};
    int64_t mSplitOffset_{0};
    bool isSlice_{false};
    bool isNdFormat_{true};
    bool isFp32_{false};
    bool isSplitSingleK_{false};
    int64_t blkK_{0};
    int64_t splitSingleKRound_{0};
    int64_t splitSingleK_{0};
    int64_t splitSingleKTail_{0};
    int64_t mL1_{0};
    int64_t nL1_{0};
    int64_t kL1_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t baseK_{0};
    uint8_t isHf32_{0};
    uint8_t l1BuferNum_{0};
    uint8_t l0cDB_{1};
    uint8_t ubDB_{1};
    L2CacheMode l2CacheDisable_{L2CacheMode::L2_CACHE_DEFAULT};
    int64_t sliceM_{1};
    int64_t srcNdStride_{1};
    int64_t mL1NormCnt_{0};
    int64_t mL1TailSplitCnt_{1};
    int64_t mL1TailMain_{0};
    int64_t mL1TailLast_{0};
    int64_t nL1NormCnt_{0};
    int64_t nL1TailSplitCnt_{1};
    int64_t nL1TailMain_{0};
    int64_t nL1TailLast_{0};

    static constexpr uint64_t WINDOW_LEN = 4UL;
    static constexpr uint64_t BLOCK_SIZE_16 = 16UL;
    static constexpr uint64_t BLOCK_SIZE_32 = 32UL;
    using BlockShape = Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockL1L0Shape = Shape<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;
    static constexpr int64_t FullLoadMode = FullLoadMode_;

    struct Params {
        uint32_t mL1 = 0;
        uint32_t nL1 = 0;
        uint32_t kL1 = 0;
        uint32_t baseM = 0;
        uint32_t baseN = 0;
        uint32_t baseK = 0;
        uint32_t mTailCnt = 0;
        uint32_t nTailCnt = 0;
        uint32_t mBaseTailSplitCnt = 1;
        uint32_t nBaseTailSplitCnt = 1;
        uint32_t mTailMain = 1;
        uint32_t nTailMain = 1;
        uint8_t isHf32 = 0;
        uint8_t l1BufferNum = 0;
        uint8_t l0cDB = 1;                                          // 默认不开db为1
        uint8_t ubDB = 1;                                           // ub默认不开db为1
        L2CacheMode l2CacheDisable = L2CacheMode::L2_CACHE_DEFAULT; // L2Cache默认使能
        uint32_t sliceM;                                            // 非连续场景m轴
        uint32_t srcNdStride;                                       // 非连续场景m轴stride
        uint32_t innerBatch = 1;                                    // 非连续transpose场景内轴batch值
    };

public:
    __aicore__ inline BlockSchedulerMatmulBasic(
        const ProblemShape& shape, int64_t blockIdx, int64_t blockNum, const Params& params, bool isFp32 = false,
        bool isNdFormat = true)
        : blockIdx_(blockIdx), blockNum_(blockNum), isFp32_(isFp32), isNdFormat_(isNdFormat)
    {
        k_ = AscendC::Te::Get<2>(shape);
        batch_ = AscendC::Std::max(AscendC::Te::Get<3>(shape), 1L);
        innerBatch_ = params.innerBatch;
        mL1_ = params.mL1;
        nL1_ = params.nL1;
        kL1_ = params.kL1;
        baseM_ = params.baseM;
        baseN_ = params.baseN;
        baseK_ = params.baseK;
        isHf32_ = params.isHf32;
        l1BuferNum_ = params.l1BufferNum;
        l0cDB_ = params.l0cDB;
        ubDB_ = params.ubDB;
        int64_t m = AscendC::Te::Get<0>(shape);
        int64_t n = AscendC::Te::Get<1>(shape);
        mTileNum_ = CeilDiv(static_cast<uint32_t>(m), params.mL1);
        nTileNum_ = CeilDiv(static_cast<uint32_t>(n), params.nL1);
        kTileNum_ = CeilDiv(static_cast<uint32_t>(k_), params.kL1);
        perCoreBlockNum_ = GetPerBlockNum(blockNum_, mTileNum_, nTileNum_, batch_);
        tileNum_ = mTileNum_ * nTileNum_;
        int64_t tailTileNum = tileNum_ % blockNum_;
        mL1TailSplitCnt_ = params.mBaseTailSplitCnt;
        nL1TailSplitCnt_ = params.nBaseTailSplitCnt;
        mL1NormCnt_ = mTileNum_ - mL1TailSplitCnt_;
        nL1NormCnt_ = nTileNum_ - nL1TailSplitCnt_;
        tailL1M_ = m - mL1NormCnt_ * params.mL1;
        tailL1N_ = n - nL1NormCnt_ * params.nL1;
        mL1TailMain_ = mL1TailSplitCnt_ == 1 ? tailL1M_ : params.mTailMain;
        mL1TailLast_ = tailL1M_ - (mL1TailSplitCnt_ - 1) * mL1TailMain_;
        nL1TailMain_ = nL1TailSplitCnt_ == 1 ? tailL1N_ : params.nTailMain;
        nL1TailLast_ = tailL1N_ - (nL1TailSplitCnt_ - 1) * nL1TailMain_;
        l2CacheDisable_ = params.l2CacheDisable;
        sliceM_ = params.sliceM;
        srcNdStride_ = params.srcNdStride;
        isSlice_ = srcNdStride_ != 1 && sliceM_ != 0;
        blkK_ = k_;
        int64_t fp32SplitKThreshold = k_ > FP32_K_SWITCH_THRESHOLD ? FP32_SPLIT_K_THRESHOLD2 : FP32_SPLIT_K_THRESHOLD1;
        // 连续且非全载场景切K
        if (!isSlice_ && isFp32_ && !isHf32_ && isNdFormat_ && k_ > fp32SplitKThreshold && FullLoadMode_ == 0) {
            isSplitSingleK_ = true;
            splitSingleK_ = fp32SplitKThreshold;
            if (k_ % fp32SplitKThreshold == 0) {
                splitSingleKRound_ = k_ / fp32SplitKThreshold;
                splitSingleKTail_ = fp32SplitKThreshold;
            } else {
                splitSingleKRound_ = CeilDiv(k_, fp32SplitKThreshold) - 1;
                splitSingleKTail_ = k_ % splitSingleK_ + splitSingleK_;
            }
        }
        if (batch_ == 1) {
            mTailCnt_ = params.mTailCnt;
            nTailCnt_ = params.nTailCnt;
            int64_t mTailSplit = CeilDiv(mL1TailLast_, mTailCnt_);
            int64_t nTailSplit = CeilDiv(nL1TailLast_, nTailCnt_);
            mTailCnt_ = CeilDiv(mL1TailLast_, mTailSplit);
            nTailCnt_ = CeilDiv(nL1TailLast_, nTailSplit);
            tailCnt_ = mTailCnt_ * nTailCnt_;
            tileNum_ += (tailCnt_ - 1) * tailTileNum;
        }
        mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
        mainRow_ = mTileNum_ / mainWindow_ - 1;
        tailWindow_ = mTileNum_ - mainRow_ * mainWindow_;
    }

    __aicore__ inline void DisableSplitSingleK()
    {
        isSplitSingleK_ = false;
    }

    __aicore__ inline int64_t GetTileNum()
    {
        return tileNum_ * batch_;
    }

    __aicore__ inline bool Gethf32Flag()
    {
        return isHf32_ > 0;
    }

    __aicore__ inline uint64_t GetL1BuferNum_()
    {
        return static_cast<uint64_t>(l1BuferNum_);
    }

    __aicore__ inline bool GetL0cDB()
    {
        return l0cDB_ > 1;
    }

    __aicore__ inline bool GetUbDB()
    {
        return ubDB_ > 1;
    }

    __aicore__ inline bool GetAL2CacheDisable()
    {
        return (
            l2CacheDisable_ == L2CacheMode::ALL_L2_CACHE_DISABLE || l2CacheDisable_ == L2CacheMode::A_L2_CACHE_DISABLE);
    }

    __aicore__ inline bool GetBL2CacheDisable()
    {
        return (
            l2CacheDisable_ == L2CacheMode::ALL_L2_CACHE_DISABLE || l2CacheDisable_ == L2CacheMode::B_L2_CACHE_DISABLE);
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t> GetNonContinuousParams()
    {
        return {sliceM_, srcNdStride_, innerBatch_};
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTailParams()
    {
        return {mL1NormCnt_, mL1TailMain_, nL1NormCnt_, nL1TailMain_};
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL1Shape()
    {
        return {mL1_, nL1_, kL1_, 1};
    }

    __aicore__ inline Shape<int64_t, int64_t, int64_t, int64_t> GetTileL0Shape()
    {
        return {baseM_, baseN_, baseK_, 1};
    }

    __aicore__ inline int64_t GetBlockNum(ProblemShape shape, int64_t blockNum)
    {
        int64_t tilingBlockNum = 0;
        if (tileNum_ * batch_ < blockNum) {
            tilingBlockNum = tileNum_ * batch_;
        } else {
            tilingBlockNum = blockNum;
        }
        return tilingBlockNum;
    }

    template <bool TransB_ = false, class B_T>
    __aicore__ inline BlockL1L0Shape GetBlockShape(
        int64_t tileIdx, int64_t mOffset = 0, int64_t nOffset = 0, int64_t kOffset = 0)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t blkM = mL1_;
        int64_t blkN = nL1_;
        int64_t nAlignSize;
        if constexpr (TransB_) {
            nAlignSize = BLOCK_SIZE_16;
        } else {
            nAlignSize = BLOCK_SIZE_32 / sizeof(B_T);
        }
        if (nTileIdx_ >= nL1NormCnt_) {
            blkN = nTileIdx_ == (nTileNum_ - 1) ? nL1TailLast_ : nL1TailMain_;
        }
        if (mTileIdx_ >= mL1NormCnt_) {
            blkM = mTileIdx_ == (mTileNum_ - 1) ? mL1TailLast_ : mL1TailMain_;
        }
        if (isSplitSingleK_) {
            splitSingleKIdx_ = CeilDiv(kOffset, splitSingleK_);
            blkK_ = splitSingleKIdx_ == (splitSingleKRound_ - 1) ? splitSingleKTail_ : splitSingleK_;
        }
        int64_t mL0 = blkM;
        int64_t nL0 = blkN;
        if (tileIdx / blockNum_ != (perCoreBlockNum_ - 1) || tailCnt_ == 1) {
            // mL1, nL1, k, batch, mL0, nL0
            mL0 = AscendC::Std::min(AscendC::Std::min(baseM_, blkM), blkM - mOffset);
            nL0 = AscendC::Std::min(AscendC::Std::min(baseN_, blkN), blkN - nOffset);
            return {blkM, blkN, blkK_, batch_, mL0, nL0};
        }
        // SplitM and SplitN
        int64_t splitBlkM = CeilDiv(blkM, mTailCnt_);
        int64_t splitBlkN = CeilDiv(blkN, nTailCnt_);
        if (!isNdFormat_) {
            splitBlkN = CeilAlign(splitBlkN, nAlignSize);
            nTailCnt_ = CeilDiv(blkN, splitBlkN);
        }
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        mSplitOffset_ = mSplitIdx * splitBlkM;
        nSplitOffset_ = nSplitIdx * splitBlkN;
        if (mSplitOffset_ >= blkM || nSplitOffset_ >= blkN) {
            return {0, 0, blkK_, batch_, 0, 0};
        }
        splitBlkM = AscendC::Std::min(blkM - mSplitOffset_, splitBlkM);
        splitBlkN = AscendC::Std::min(blkN - nSplitOffset_, splitBlkN);
        mL0 = AscendC::Std::min(AscendC::Std::min(baseM_, splitBlkM), splitBlkM - mOffset);
        nL0 = AscendC::Std::min(AscendC::Std::min(baseN_, splitBlkN), splitBlkN - nOffset);
        return {splitBlkM, splitBlkN, blkK_, batch_, mL0, nL0};
    }

    __aicore__ inline BlockCoord GetBlockCoord(int tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t batchIdx = 0;
        if (batch_ > 1) {
            batchIdx = tileIdx / tileNum_;
        }

        int64_t mOffset = mTileIdx_ * mL1_ + mSplitOffset_;
        int64_t nOffset = nTileIdx_ * nL1_ + nSplitOffset_;
        int64_t ndNum = mL1_ > sliceM_ ? mL1_ / sliceM_ : 1;
        // 非连续计算m方向上的起点坐标
        int64_t mOffsetNonContiguous = mTileIdx_ * (ndNum * (srcNdStride_ / k_)) + mSplitOffset_;
        if (mTileIdx_ > mL1NormCnt_) {
            mOffset = mL1NormCnt_ * mL1_ + (mTileIdx_ - mL1NormCnt_) * mL1TailMain_ + mSplitOffset_;
        }
        if (nTileIdx_ > nL1NormCnt_) {
            nOffset = nL1NormCnt_ * nL1_ + (nTileIdx_ - nL1NormCnt_) * nL1TailMain_ + nSplitOffset_;
        }
        // 当前不切k, 使用kOffset传递非连续场景mOffset
        return {mOffset, nOffset, mOffsetNonContiguous, batchIdx};
    }

    __aicore__ inline BlockCoord GetSplitKBlockCoord(int tileIdx)
    {
        UpdateMNTileIdx(tileIdx);
        int64_t batchIdx = 0;
        if (batch_ > 1) {
            batchIdx = tileIdx / tileNum_;
        }
        int64_t mOffset = mTileIdx_ * mL1_ + mSplitOffset_;
        int64_t nOffset = nTileIdx_ * nL1_ + nSplitOffset_;
        int64_t kOffset = splitSingleKIdx_ * splitSingleK_;
        if (mTileIdx_ > mL1NormCnt_) {
            mOffset = mL1NormCnt_ * mL1_ + (mTileIdx_ - mL1NormCnt_) * mL1TailMain_ + mSplitOffset_;
        }
        if (nTileIdx_ > nL1NormCnt_) {
            nOffset = nL1NormCnt_ * nL1_ + (nTileIdx_ - nL1NormCnt_) * nL1TailMain_ + nSplitOffset_;
        }
        // 连续场景切K
        return {mOffset, nOffset, kOffset, batchIdx};
    }

    __aicore__ inline Shape<int64_t, int64_t> GetSplitOffset()
    {
        return {mSplitOffset_, nSplitOffset_};
    }

private:
    __aicore__ inline void UpdateMNTileIdx(int64_t tmpIdx)
    {
        if (lastTileIdx_ == tmpIdx) {
            return;
        }
        lastTileIdx_ = tmpIdx;

        int64_t tileIdx = tmpIdx % tileNum_;
        if (tileIdx / blockNum_ == (perCoreBlockNum_ - 1) && tailCnt_ > 1) {
            tileIdx = (perCoreBlockNum_ - 1) * blockNum_ + blockIdx_ / tailCnt_;
        }
        int64_t rowIdx = tileIdx / nTileNum_ / mainWindow_;
        if (rowIdx < mainRow_) {
            mTileIdx_ = rowIdx * mainWindow_ + tileIdx % mainWindow_;
            nTileIdx_ = (tileIdx / mainWindow_) % nTileNum_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIndex = tileIdx - mainRow_ * mainWindow_ * nTileNum_;
            mTileIdx_ = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
            nTileIdx_ = (tailIndex / tailWindow_) % nTileNum_;
        }
        if (rowIdx % 2 != 0) { // 2: mode 2 means even row, need reverse scan
            nTileIdx_ = nTileNum_ - 1 - nTileIdx_;
        }
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze