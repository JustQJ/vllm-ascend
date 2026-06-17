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
 * \file block_scheduler_qbmm.h
 * \brief
 */

#pragma once

#include "blaze/gemm/utils/layout_utils.h"
#include "blaze/gemm/utils/common_utils.h"
#include "tensor_api/tensor.h"

namespace Blaze {
namespace Gemm {
namespace Block {

template <class ProblemShape_, uint64_t FullLoadMode_, class LayoutA_, class LayoutB_, class AType_>
class BlockSchedulerQuantBatchMatmulV3 {
public:
    int64_t m_{0};
    int64_t n_{0};
    int64_t k_{0};
    int64_t baseM_{0};
    int64_t baseN_{0};
    int64_t mCnt_{0};
    int64_t nCnt_{0};
    int64_t totalCnt_{0};
    int64_t mBaseNormCnt_{0};
    int64_t nBaseNormCnt_{0};
    int64_t mBaseTailMain_{0};
    int64_t nBaseTailMain_{0};
    int64_t mBaseTailLast_{0};
    int64_t nBaseTailLast_{0};
    int64_t mCoreNum_{0};
    int64_t mTailCoreNum_{0};
    int64_t blockIdx_{AscendC::GetBlockIdx() / AscendC::GetTaskRation()};
    int64_t blockNum_{AscendC::GetBlockNum()};
    int64_t startBlockIdx_{0};
    int64_t endBlockIdx_{0};
    int64_t roundIdx_{0};
    int64_t round_{0};
    int64_t mTailTile_{1};     // init value must be 1
    int64_t nTailTile_{1};     // init value must be 1
    int64_t totalTailTile_{1}; // init value must be 1
    int64_t mSplitAddrOffset_{0};
    int64_t nSplitAddrOffset_{0};
    int64_t mainRow_{0};

    using BlockShape = AscendC::Te::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Te::Coord<int64_t, int64_t, int64_t, int64_t>;
    using ProblemShape = ProblemShape_;
    using AType = AType_;

    constexpr static bool transA = IsTrans<LayoutA_>::value;
    constexpr static bool transB = IsTrans<LayoutB_>::value;
    constexpr static int64_t C0_SIZE = IsFp4<AType>() ? C0_SIZE_B4 : C0_SIZE_B8;
    constexpr static int64_t WINDOW_LEN = 4;

    struct Params {
        int64_t baseM;
        int64_t baseN;
        int64_t mTailTile;
        int64_t nTailTile;
        int64_t mBaseTailSplitCnt;
        int64_t nBaseTailSplitCnt;
        int64_t mTailMain;
        int64_t nTailMain;
    };

public:
    __aicore__ inline BlockSchedulerQuantBatchMatmulV3(const ProblemShape& shape, const Params& params)
    {
        m_ = AscendC::Te::Get<MNK_M>(shape);
        n_ = AscendC::Te::Get<MNK_N>(shape);
        k_ = AscendC::Te::Get<MNK_K>(shape);
        baseM_ = static_cast<int64_t>(params.baseM);
        baseN_ = static_cast<int64_t>(params.baseN);
        mCnt_ = Blaze::Gemm::CeilDiv(m_, baseM_);
        nCnt_ = Blaze::Gemm::CeilDiv(n_, baseN_);
        totalCnt_ = mCnt_ * nCnt_;
        mCoreNum_ = Blaze::Gemm::Min(WINDOW_LEN, mCnt_);
        if (mCoreNum_ != 0) {
            mainRow_ = mCnt_ / mCoreNum_ - 1;
        }
        mTailCoreNum_ = mCnt_ - mCoreNum_ * mainRow_;
        endBlockIdx_ = (totalCnt_ - 1) % blockNum_;
        round_ = Blaze::Gemm::CeilDiv(totalCnt_, blockNum_);
        if (blockIdx_ > endBlockIdx_) {
            round_ -= 1;
        }
        if constexpr (!transA) {
            mBaseNormCnt_ = mCnt_ - params.mBaseTailSplitCnt;
            int64_t mMergeSize = m_ - mBaseNormCnt_ * baseM_;
            mBaseTailMain_ = params.mBaseTailSplitCnt == 1 ? mMergeSize : params.mTailMain;
            mBaseTailLast_ = mMergeSize - (params.mBaseTailSplitCnt - 1) * mBaseTailMain_;
        } else {
            mBaseTailMain_ = m_ - (mCnt_ - 1) * baseM_;
        }
        if constexpr (transB) {
            nBaseNormCnt_ = nCnt_ - params.nBaseTailSplitCnt;
            int64_t nMergeSize = n_ - nBaseNormCnt_ * baseN_;
            nBaseTailMain_ = params.nBaseTailSplitCnt == 1 ? nMergeSize : params.nTailMain;
            nBaseTailLast_ = nMergeSize - (params.nBaseTailSplitCnt - 1) * nBaseTailMain_;
        } else {
            nBaseTailMain_ = n_ - (nCnt_ - 1) * baseN_;
        }
    }

    __aicore__ inline void UpdateTailTile(uint32_t mTailTile, uint32_t nTailTile)
    {
        mTailTile_ = mTailTile;
        nTailTile_ = nTailTile;
        totalTailTile_ = mTailTile * nTailTile;
        uint64_t tailOriCnt = AscendC::Std::min(totalCnt_, endBlockIdx_ + 1);
        int64_t newEndBlockIdx = endBlockIdx_ + tailOriCnt * (totalTailTile_ - 1);
        if (blockIdx_ > endBlockIdx_ && blockIdx_ <= newEndBlockIdx) {
            round_ += 1;
        }
        if (blockIdx_ > newEndBlockIdx) {
            mTailTile_ = 1;
            nTailTile_ = 1;
            totalTailTile_ = 1;
        }
        endBlockIdx_ = newEndBlockIdx;
    }

    __aicore__ inline int64_t GetTotalCnt()
    {
        return totalCnt_;
    }

    __aicore__ inline int64_t GetEndBlockIdx()
    {
        return endBlockIdx_;
    }

    /**
     * @brief Round the input value up to the smallest power of two.
     *
     * Modifies the input value in place so that it becomes the smallest
     * power of two greater than or equal to its original value.
     * This implementation uses a bit-smearing technique and assumes
     * the input value is in the range [1, 256].
     *
     * @param inputValue  Input value to be rounded up.
     */
    __aicore__ inline void CeilPowerOfTwo(int64_t& inputValue)
    {
        inputValue--;
        inputValue |= inputValue >> 1; // Propagate the highest set bit to the right by 1 position,ensuring the most
                                       // significant bit and its adjacent lower bit are set.
        inputValue |= inputValue >> 2; // Continue propagating the highest set bit by 2 positions, expanding the
                                       // contiguous range of set bits below the MSB to 3 bits.
        inputValue |= inputValue >> 4; // Further propagate the highest set bit by 4 positions, resulting in all bits
                                       // below the MSB (up to 7 positions) being set.
        inputValue++;
    }

    __aicore__ inline void CalSingleCoreShapeByCoord(
        int64_t& singleCoreM, int64_t& singleCoreN, const BlockCoord& blockCoord)
    {
        if constexpr (!transA) {
            if (AscendC::Te::Get<MNK_M>(blockCoord) >= mBaseNormCnt_) {
                singleCoreM = AscendC::Te::Get<MNK_M>(blockCoord) < mCnt_ - 1 ? mBaseTailMain_ : mBaseTailLast_;
            }
        } else {
            if (AscendC::Te::Get<MNK_M>(blockCoord) == mCnt_ - 1) {
                singleCoreM = mBaseTailMain_;
            }
        }
        if constexpr (transB) {
            if (AscendC::Te::Get<MNK_N>(blockCoord) >= nBaseNormCnt_) {
                singleCoreN = AscendC::Te::Get<MNK_N>(blockCoord) < nCnt_ - 1 ? nBaseTailMain_ : nBaseTailLast_;
            }
        } else {
            if (AscendC::Te::Get<MNK_N>(blockCoord) == nCnt_ - 1) {
                singleCoreN = nBaseTailMain_;
            }
        }
    }

    template <QuantMode aQuantMode, QuantMode bQuantMode, bool weightNz = false>
    __aicore__ inline BlockShape GetBlockShape(BlockCoord blockCoord)
    {
        int64_t singleCoreM = baseM_;
        int64_t singleCoreN = baseN_;
        CalSingleCoreShapeByCoord(singleCoreM, singleCoreN, blockCoord);

        if (totalTailTile_ == 1 || roundIdx_ < round_) {
            return {singleCoreM, singleCoreN, 0, 0};
        }

        int64_t singleCoreMSplit = Blaze::Gemm::CeilDiv(singleCoreM, mTailTile_);
        int64_t singleCoreNSplit = Blaze::Gemm::CeilDiv(singleCoreN, nTailTile_);
        if constexpr (IsFp4<AType>() && transA) {
            singleCoreMSplit = (singleCoreMSplit + 1) & ~1;
        }
        if constexpr (IsFp4<AType>() && !transB) {
            singleCoreNSplit = (singleCoreNSplit + 1) & ~1;
        }
        if constexpr ((aQuantMode == QuantMode::PERGROUP_MODE || aQuantMode == QuantMode::PERBLOCK_MODE) && transA) {
            singleCoreMSplit = PER_BLOCK_SIZE << (singleCoreMSplit > PER_BLOCK_SIZE);
        } else if constexpr (aQuantMode == QuantMode::PERBLOCK_MODE) {
            CeilPowerOfTwo(singleCoreMSplit);
        }
        if constexpr (bQuantMode == QuantMode::PERBLOCK_MODE) {
            if constexpr (!transB) { // (k, n)
                singleCoreNSplit = PER_BLOCK_SIZE << (singleCoreNSplit > PER_BLOCK_SIZE);
            } else {
                CeilPowerOfTwo(singleCoreNSplit);
            }
        }

        if constexpr (weightNz) {
            if constexpr (!transB) {
                singleCoreNSplit = Blaze::Gemm::CeilAlign(singleCoreNSplit, static_cast<int64_t>(C0_SIZE));
            } else {
                singleCoreNSplit = Blaze::Gemm::CeilAlign(singleCoreNSplit, static_cast<int64_t>(BLOCK_CUBE));
            }
        }

        int64_t mSplitIdx = (blockIdx_ % totalTailTile_) % mTailTile_;
        int64_t nSplitIdx = 0;
        if constexpr (FullLoadMode_ == A_FULL_LOAD_MODE) {
            nSplitIdx = blockIdx_ / mCnt_ % nTailTile_;
        } else {
            nSplitIdx = (blockIdx_ % totalTailTile_) / mTailTile_;
        }
        mSplitAddrOffset_ = mSplitIdx * singleCoreMSplit;
        nSplitAddrOffset_ = nSplitIdx * singleCoreNSplit;
        if (mSplitAddrOffset_ >= singleCoreM || nSplitAddrOffset_ >= singleCoreN) {
            return {0, 0, 0, 0};
        }
        singleCoreM = Blaze::Gemm::Min(singleCoreM - mSplitAddrOffset_, singleCoreMSplit);
        singleCoreN = Blaze::Gemm::Min(singleCoreN - nSplitAddrOffset_, singleCoreNSplit);
        return {singleCoreM, singleCoreN, mSplitAddrOffset_, nSplitAddrOffset_};
    }

    __aicore__ inline AscendC::Std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> GetLoadBalanceInfo()
    {
        return {
            static_cast<uint32_t>(mBaseNormCnt_), static_cast<uint32_t>(mBaseTailMain_),
            static_cast<uint32_t>(nBaseNormCnt_), static_cast<uint32_t>(nBaseTailMain_)};
    }

    __aicore__ inline void UpdateNextBatchBlockRoundParams()
    {
        startBlockIdx_ = endBlockIdx_ + 1 == blockNum_ ? 0 : (endBlockIdx_ + 1);
        endBlockIdx_ = (totalCnt_ + startBlockIdx_ - 1) % blockNum_;

        roundIdx_ = 0;
        round_ = Blaze::Gemm::CeilDiv(totalCnt_, blockNum_);
        if (startBlockIdx_ > endBlockIdx_ && (blockIdx_ > endBlockIdx_ && blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        } else if (startBlockIdx_ <= endBlockIdx_ && (blockIdx_ > endBlockIdx_ || blockIdx_ < startBlockIdx_)) {
            round_ -= 1;
        }
    }

    __aicore__ inline bool GetTileIdx(BlockCoord& blockCoord)
    {
        if (roundIdx_ >= round_) {
            return false;
        }

        int64_t blockCoordM = AscendC::Te::Get<IDX_M_TILEIDX>(blockCoord);
        int64_t blockCoordN = AscendC::Te::Get<IDX_N_TILEIDX>(blockCoord);

        int64_t newBlockIdx = (roundIdx_ == round_ - 1) ? blockIdx_ / totalTailTile_ : blockIdx_;
        int64_t tileIdx = newBlockIdx + roundIdx_ * blockNum_;
        if constexpr (FullLoadMode_ == A_FULL_LOAD_MODE) {
            blockCoordM = blockIdx_ % mCnt_;
            int64_t curNTailTile = (roundIdx_ == round_ - 1) ? nTailTile_ : 1;
            blockCoordN = roundIdx_ * blockNum_ / mCnt_ % nCnt_ + blockIdx_ / mCnt_ / curNTailTile;
            roundIdx_++;
            blockCoord = BlockCoord{
                blockCoordM, blockCoordN, AscendC::Te::Get<IDX_M_TAIL_SPLIT_TILEIDX>(blockCoord),
                AscendC::Te::Get<IDX_N_TAIL_SPLIT_TILEIDX>(blockCoord)};
            return true;
        }
        if (blockIdx_ < startBlockIdx_) {
            tileIdx += blockNum_ - startBlockIdx_;
        } else if (endBlockIdx_ + 1 >= totalTailTile_ * totalCnt_) {
            tileIdx -= startBlockIdx_ / totalTailTile_;
        } else {
            tileIdx -= startBlockIdx_;
        }
        int64_t rowIdx = tileIdx / nCnt_ / mCoreNum_;
        if (rowIdx < mainRow_) {
            blockCoordM = rowIdx * mCoreNum_ + tileIdx % mCoreNum_;
            blockCoordN = (tileIdx / mCoreNum_) % nCnt_;
        } else {
            rowIdx = mainRow_;
            int64_t tailIdx = tileIdx - mainRow_ * mCoreNum_ * nCnt_;
            blockCoordM = mainRow_ * mCoreNum_ + tailIdx % mTailCoreNum_;
            blockCoordN = (tailIdx / mTailCoreNum_) % nCnt_;
        }
        if (rowIdx & 1) {
            blockCoordN = nCnt_ - 1 - blockCoordN;
        }
        roundIdx_++;
        blockCoord = BlockCoord{
            blockCoordM, blockCoordN, AscendC::Te::Get<IDX_M_TAIL_SPLIT_TILEIDX>(blockCoord),
            AscendC::Te::Get<IDX_N_TAIL_SPLIT_TILEIDX>(blockCoord)};
        return true;
    }

    __aicore__ inline void GetTileCoord(const BlockCoord& blockCoord, int64_t& mPos, int64_t& nPos)
    {
        auto mTileIdx = AscendC::Te::Get<MNK_M>(blockCoord);
        auto nTileIdx = AscendC::Te::Get<MNK_N>(blockCoord);
        mPos = mTileIdx * baseM_ + mSplitAddrOffset_;
        nPos = nTileIdx * baseN_ + nSplitAddrOffset_;
        if constexpr (!transA) {
            if (mTileIdx > mBaseNormCnt_) {
                mPos -= (mTileIdx - mBaseNormCnt_) * (baseM_ - mBaseTailMain_);
            }
        }
        if constexpr (transB) {
            if (nTileIdx > nBaseNormCnt_) {
                nPos -= (nTileIdx - nBaseNormCnt_) * (baseN_ - nBaseTailMain_);
            }
        }
    }
};

} // namespace Block
} // namespace Gemm
} // namespace Blaze
