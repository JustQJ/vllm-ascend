// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.

#include "kernel_operator.h"
#include "ngram_spec_decode_tiling.h"

constexpr int32_t ELEM_SIZE = sizeof(int32_t);
constexpr uint32_t SAFE_CHUNK = 8192u;

// GM -> UB copy (DataCopyPad, handles unaligned GM addresses)
#define COPY_GM_TO_UB(dst, src, src_offset, count, T)                       \
    do {                                                                    \
        if ((count) > 0) {                                                  \
            constexpr uint32_t __align_elem = 8u;                           \
            uint32_t __c = static_cast<uint32_t>(count);                    \
            uint32_t __aligned = ((__c + __align_elem - 1u) / __align_elem) \
                                 * __align_elem;                            \
            uint8_t __pad = static_cast<uint8_t>(__aligned - __c);          \
            AscendC::DataCopyExtParams __p{                                 \
                1,                                                          \
                static_cast<uint32_t>(__c * sizeof(T)),                     \
                0, 0, 0};                                                   \
            AscendC::DataCopyPadExtParams<T> __pp{true, 0, __pad, -1};      \
            AscendC::DataCopyPad((dst), (src)[(src_offset)], __p, __pp);    \
        }                                                                   \
    } while (0)

// UB -> GM copy (reverse direction)
#define COPY_UB_TO_GM(dst, dst_offset, src, src_offset, count, T)           \
    do {                                                                    \
        if ((count) > 0) {                                                  \
            constexpr uint32_t __store_max = 16383u;                        \
            uint32_t __c = static_cast<uint32_t>(count);                    \
            for (uint32_t __off = 0; __off < __c; __off += __store_max) {   \
                uint32_t __chunk = (__off + __store_max <= __c)             \
                                    ? __store_max                           \
                                    : (__c - __off);                        \
                AscendC::DataCopyParams __p{                                \
                    1, static_cast<uint16_t>(__chunk * sizeof(T)), 0, 0};   \
                AscendC::DataCopyPad(                                       \
                    (dst)[(dst_offset) + __off],                            \
                    (src)[(src_offset) + __off], __p);                      \
            }                                                               \
        }                                                                   \
    } while (0)

class KernelNgramSpecDecode {
public:
    __aicore__ inline KernelNgramSpecDecode() {}

    __aicore__ inline void Init(
        GM_ADDR token_ids_gm, GM_ADDR num_tokens_gm, GM_ADDR sampled_gm,
        GM_ADDR discard_gm, GM_ADDR next_tokens_gm, GM_ADDR draft_tokens_gm,
        GM_ADDR num_valid_gm, GM_ADDR workspace, GM_ADDR tiling)
    {
        REGISTER_TILING_DEFAULT(NgramSpecDecodeTilingData);
        GET_TILING_DATA_WITH_STRUCT(NgramSpecDecodeTilingData, tilingData, tiling);

        batch_size = tilingData.ngramInfo.batchSize;
        max_seq_len = tilingData.ngramInfo.maxSeqLen;
        max_new_tokens = tilingData.ngramInfo.maxNewTokens;
        vocab_size_val = tilingData.ngramInfo.vocabSize;
        min_n_val = tilingData.ngramInfo.minN;
        max_n_val = tilingData.ngramInfo.maxN;
        k_val = tilingData.ngramInfo.k;
        former_num = tilingData.ngramInfo.formerNum;
        rows_per_core = tilingData.ngramInfo.rowsPerCore;
        tail_rows = tilingData.ngramInfo.tailRows;
        block_rows = tilingData.ngramInfo.blockRows;

        int32_t align_elems = 32 / ELEM_SIZE;
        max_seq_len_align = ((max_seq_len + align_elems - 1) / align_elems) * align_elems;
        max_new_tokens_align = ((max_new_tokens + align_elems - 1) / align_elems) * align_elems;
        k_align = ((k_val + align_elems - 1) / align_elems) * align_elems;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx < static_cast<uint32_t>(former_num)) {
            my_rows = static_cast<uint32_t>(rows_per_core) + 1;
            row_offset = (static_cast<uint32_t>(rows_per_core) + 1) * blockIdx;
        } else {
            my_rows = static_cast<uint32_t>(rows_per_core);
            row_offset = static_cast<uint32_t>(rows_per_core + 1)
                       * static_cast<uint32_t>(former_num)
                       + my_rows * (blockIdx - static_cast<uint32_t>(former_num));
        }

        tokenGm.SetGlobalBuffer((__gm__ int32_t *)token_ids_gm,
            static_cast<uint64_t>(batch_size) * max_seq_len);
        numTokensGm.SetGlobalBuffer((__gm__ int32_t *)num_tokens_gm,
            static_cast<uint64_t>(batch_size));
        sampledGm.SetGlobalBuffer((__gm__ int32_t *)sampled_gm,
            static_cast<uint64_t>(batch_size) * max_new_tokens);
        discardGm.SetGlobalBuffer((__gm__ int32_t *)discard_gm,
            static_cast<uint64_t>(batch_size));
        nextTokensGm.SetGlobalBuffer((__gm__ int32_t *)next_tokens_gm,
            static_cast<uint64_t>(batch_size));
        draftTokensGm.SetGlobalBuffer((__gm__ int32_t *)draft_tokens_gm,
            static_cast<uint64_t>(batch_size) * k_val);
        numValidGm.SetGlobalBuffer((__gm__ int32_t *)num_valid_gm,
            static_cast<uint64_t>(batch_size));

        // buffer sizes
        uint32_t mnta_u = static_cast<uint32_t>(max_new_tokens_align);
        uint32_t ka_u = static_cast<uint32_t>(k_align);
        uint32_t my_rows_align = ((my_rows + 7u) / 8u) * 8u;

        // Compute buffer sizes: for short runtime nt≤SAFE_CHUNK need nt, but
        // at Init time we only know max_seq_len. Safe: always allocate at
        // most (SAFE_CHUNK + max_n) to bound UB usage.
        uint32_t chunk_ub = SAFE_CHUNK + static_cast<uint32_t>(max_n_val);
        uint32_t chunk_ub_align = ((chunk_ub + 7u) / 8u) * 8u;
        uint32_t token_buf_size = chunk_ub_align * ELEM_SIZE;

        // input queues (VECIN)
        pipe.InitBuffer(tokenInQue, 2, token_buf_size);  // depth=2 double buffer
        pipe.InitBuffer(sampledInQue, 1, my_rows * mnta_u * ELEM_SIZE);
        pipe.InitBuffer(numTokensInQue, 1, my_rows_align * ELEM_SIZE);
        pipe.InitBuffer(discardInQue, 1, my_rows_align * ELEM_SIZE);
        pipe.InitBuffer(suffixInQue, 1, static_cast<uint32_t>(max_n_val) * ELEM_SIZE);

        // output queues (VECOUT)
        pipe.InitBuffer(nextOutQue, 1, my_rows_align * ELEM_SIZE);
        pipe.InitBuffer(draftOutQue, 1, my_rows * ka_u * ELEM_SIZE);
        pipe.InitBuffer(numValidOutQue, 1, my_rows_align * ELEM_SIZE);

        // VECCALC scratch buffers
        pipe.InitBuffer(ngramCalcBuf, token_buf_size);
        pipe.InitBuffer(ngramTempBuf, token_buf_size);
        pipe.InitBuffer(ngramGatherBuf, token_buf_size);
        // maskBuf: 2x to hold two independent mask results before Or
        uint32_t mask_bytes = static_cast<uint32_t>(max_new_tokens_align) / 8;
        pipe.InitBuffer(maskCalcBuf, 2 * mask_bytes);

        // worst-case reduce: chunk size for long path, max_seq_len for short path
        uint32_t reduce_count = SAFE_CHUNK;
        uint32_t reduce_tmp_elems = CalcReduceMinTmpSize(reduce_count, ELEM_SIZE);
        uint32_t reduce_tmp_bytes = ((reduce_tmp_elems * ELEM_SIZE + 31) / 32) * 32;
        pipe.InitBuffer(ngramReduceBuf, reduce_tmp_bytes);
    }

    __aicore__ inline void Process()
    {
        CopyInMetadata();
        auto sampledLocal = sampledInQue.DeQue<int32_t>();
        auto numTokensLocal = numTokensInQue.DeQue<int32_t>();
        auto discardLocal = discardInQue.DeQue<int32_t>();
        auto nextLocal = nextOutQue.AllocTensor<int32_t>();
        auto draftLocal = draftOutQue.AllocTensor<int32_t>();
        auto numValidLocal = numValidOutQue.AllocTensor<int32_t>();

        // Phase A: validate all rows, write sampled → tokenGm, collect nt[]
        int32_t nt_vals[256];  // max rows per core is bounded by tiling
        for (uint32_t r = 0; r < my_rows; ++r) {
            nt_vals[r] = ValidateTokens(r, row_offset + r,
                sampledLocal, numTokensLocal, discardLocal, nextLocal);
        }

        // Sync all Phase A MTE3 writes before Phase B reads from tokenGm
        AscendC::TQueSync<PIPE_MTE3, PIPE_S> sync_a;
        auto event = GetTPipePtr()->AllocEventID<AscendC::HardEvent::MTE3_S>();
        sync_a.SetFlag(event);
        sync_a.WaitFlag(event);
        GetTPipePtr()->ReleaseEventID<AscendC::HardEvent::MTE3_S>(event);

        // Phase B: interleaved ngram with double-buffer token load.
        // Preload next row BEFORE computing current row: MTE2 overlaps V.
        // Short rows → load full sequence; long rows → load first chunk only.
        int32_t preloaded_row = -1;
        for (uint32_t r = 0; r < my_rows; ++r) {
            uint32_t global_row = row_offset + r;
            int32_t nt = nt_vals[r];
            bool is_short = (nt > 0 && nt <= static_cast<int32_t>(SAFE_CHUNK));

            if (nt <= 0 || nt < min_n_val) {
                uint32_t doff = r * static_cast<uint32_t>(k_align);
                AscendC::Duplicate(draftLocal[doff], static_cast<int32_t>(-1),
                                   static_cast<uint32_t>(k_align));
                numValidLocal.SetValue(r, 0);
                continue;
            }

            // Was this row preloaded by the previous iteration?
            bool this_row_preloaded = (preloaded_row == static_cast<int32_t>(r));

            // Only SHORT rows can preload the next row:
            // - Short row needs 1 buffer for single-pass V compute
            // - Other buffer is free for next row's MTE2 preload
            // - Long row uses BOTH buffers for internal chunk double-buffering
            preloaded_row = -1;
            if (is_short && r + 1 < my_rows && nt_vals[r + 1] >= min_n_val) {
                uint32_t next_gr = row_offset + r + 1;
                int32_t next_nt = nt_vals[r + 1];
                if (next_nt <= static_cast<int32_t>(SAFE_CHUNK)) {
                    LaunchPreloadNextRow(next_gr, next_nt);
                } else {
                    LaunchPreloadNextLongFirstChunk(next_gr, next_nt);
                }
                preloaded_row = static_cast<int32_t>(r + 1);
            }

            AscendC::LocalTensor<int32_t> tokenLocal;
            int32_t best_match_pos = -1;
            int32_t best_ngram_len = 0;

            // Stage 4: draft
            uint32_t doff = r * static_cast<uint32_t>(k_align);
            int32_t draft_load = 0;
            AscendC::Duplicate(draftLocal[doff], static_cast<int32_t>(-1),
                               static_cast<uint32_t>(k_align));

            if (is_short) {
                // ---- Short row: copy draft from tokenLocal (already in UB) ----
                if (this_row_preloaded) {
                    tokenLocal = tokenInQue.DeQue<int32_t>();
                } else {
                    tokenLocal = LoadOneRowToken(global_row, nt);
                }
                NgramMatchRowShort(nt, tokenLocal,
                                   best_match_pos, best_ngram_len);
                if (best_match_pos >= 0) {
                    int32_t draft_start = best_match_pos + best_ngram_len;
                    int32_t avail = nt - draft_start;
                    draft_load = (avail < k_val) ? avail : k_val;
                    // Scalar copy from tokenLocal (k_val ≤ 5, no alignment concern)
                    for (int32_t j = 0; j < draft_load; ++j) {
                        draftLocal.SetValue(doff + static_cast<uint32_t>(j),
                            tokenLocal.GetValue(static_cast<uint32_t>(draft_start + j)));
                    }
                }
                tokenInQue.FreeTensor(tokenLocal);

            } else {
                // ---- Long row ----
                int32_t next_gr = (r + 1 < my_rows) ? static_cast<int32_t>(row_offset + r + 1) : -1;
                int32_t next_nt = (r + 1 < my_rows) ? nt_vals[r + 1] : 0;
                int32_t preloaded_out = -1;
                NgramMatchRowLong(global_row, nt, this_row_preloaded,
                                  next_gr, next_nt, preloaded_out,
                                  best_match_pos, best_ngram_len);
                if (preloaded_out >= 0) {
                    preloaded_row = static_cast<int32_t>(r + 1);
                }
                if (best_match_pos >= 0) {
                    int32_t avail = nt - (best_match_pos + best_ngram_len);
                    draft_load = (avail < k_val) ? avail : k_val;
                    if (draft_load > 0) {
                        uint64_t gmRow = static_cast<uint64_t>(global_row) *
                                         static_cast<uint32_t>(max_seq_len);
                        COPY_GM_TO_UB(draftLocal[doff], tokenGm,
                                      gmRow + best_match_pos + best_ngram_len,
                                      draft_load, int32_t);
                    }
                }
            }
            numValidLocal.SetValue(r, draft_load > 0 ? draft_load : 0);
        }

        // Cleanup any unused preloaded token
        if (preloaded_row >= 0) {
            auto unusedToken = tokenInQue.DeQue<int32_t>();
            tokenInQue.FreeTensor(unusedToken);
        }

        sampledInQue.FreeTensor(sampledLocal);
        numTokensInQue.FreeTensor(numTokensLocal);
        discardInQue.FreeTensor(discardLocal);
        nextOutQue.EnQue(nextLocal);
        draftOutQue.EnQue(draftLocal);
        numValidOutQue.EnQue(numValidLocal);
        CopyOutMetadata();
    }

private:
    // ---------------------------------------------------------------
    // GM -> UB (MTE2)
    // ---------------------------------------------------------------
    __aicore__ inline void CopyInMetadata()
    {
        uint32_t mnta_u = static_cast<uint32_t>(max_new_tokens_align);

        // sampled: all rows, one DataCopyPad per row via repeat
        auto sampledTensor = sampledInQue.AllocTensor<int32_t>();
        uint32_t srcRowBytes = static_cast<uint32_t>(max_new_tokens) * ELEM_SIZE;
        AscendC::DataCopyExtParams sampledParams{
            static_cast<uint16_t>(my_rows), srcRowBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> padParams{
            true, 0, static_cast<uint8_t>(mnta_u - max_new_tokens), 0};
        AscendC::DataCopyPad(
            sampledTensor,
            sampledGm[static_cast<uint64_t>(row_offset) * max_new_tokens],
            sampledParams, padParams);
        sampledInQue.EnQue(sampledTensor);

        // numTokens
        auto numTokensTensor = numTokensInQue.AllocTensor<int32_t>();
        uint32_t metaBytes = static_cast<uint32_t>(my_rows) * ELEM_SIZE;
        AscendC::DataCopyExtParams metaParams{1, metaBytes, 0, metaBytes, 0};
        AscendC::DataCopyPadExtParams<int32_t> noPadT{false, 0, 0, 0};
        AscendC::DataCopyPad(numTokensTensor, numTokensGm[row_offset], metaParams, noPadT);
        numTokensInQue.EnQue(numTokensTensor);

        // discard
        auto discardTensor = discardInQue.AllocTensor<int32_t>();
        AscendC::DataCopyPad(discardTensor, discardGm[row_offset], metaParams, noPadT);
        discardInQue.EnQue(discardTensor);
    }

    // ---------------------------------------------------------------
    // UB -> GM (MTE3)
    // ---------------------------------------------------------------
    __aicore__ inline void CopyOutMetadata()
    {
        auto nextLocal = nextOutQue.DeQue<int32_t>();
        auto draftLocal = draftOutQue.DeQue<int32_t>();
        auto numValidLocal = numValidOutQue.DeQue<int32_t>();

        uint16_t metaBytes16 = static_cast<uint16_t>(my_rows) * ELEM_SIZE;
        AscendC::DataCopyParams nextParams{1, metaBytes16, 0, 0};
        AscendC::DataCopyPad(nextTokensGm[row_offset], nextLocal, nextParams);
        AscendC::DataCopyPad(numValidGm[row_offset], numValidLocal, nextParams);

        // draft: per-row copies handling UB padding (k_align) -> GM compact (k_val)
        uint32_t ka_u = static_cast<uint32_t>(k_align);
        for (uint32_t r = 0; r < my_rows; ++r) {
            uint32_t kBytes = static_cast<uint32_t>(k_val) * ELEM_SIZE;
            AscendC::DataCopyExtParams rowParams{1, kBytes, 0, 0, 0};
            AscendC::DataCopyPad(
                draftTokensGm[static_cast<uint64_t>(row_offset + r) * k_val],
                draftLocal[r * ka_u], rowParams);
        }

        nextOutQue.FreeTensor(nextLocal);
        draftOutQue.FreeTensor(draftLocal);
        numValidOutQue.FreeTensor(numValidLocal);
    }

    // ---------------------------------------------------------------
    // Phase A: validate sampled + compute nextToken + write to tokenGm
    // Returns nt = seq_len + valid_count
    // ---------------------------------------------------------------
    __aicore__ inline int32_t ValidateTokens(
        uint32_t local_idx, uint32_t global_row,
        AscendC::LocalTensor<int32_t>& sampledLocal,
        AscendC::LocalTensor<int32_t>& numTokensLocal,
        AscendC::LocalTensor<int32_t>& discardLocal,
        AscendC::LocalTensor<int32_t>& nextLocal)
    {
        uint32_t msl = static_cast<uint32_t>(max_seq_len);
        uint32_t mnta_u = static_cast<uint32_t>(max_new_tokens_align);
        uint64_t gmRow = static_cast<uint64_t>(global_row) * msl;
        uint32_t soff = local_idx * mnta_u;
        int32_t seq_len = numTokensLocal.GetValue(local_idx);
        int32_t discard = discardLocal.GetValue(local_idx);
        int32_t valid_count = 0;

        // Stage 1: validate sampled tokens
        if (discard != 0) {
            AscendC::Duplicate(sampledLocal[soff], static_cast<int32_t>(-1),
                               static_cast<uint32_t>(max_new_tokens_align));
        } else {
            uint32_t align_n = static_cast<uint32_t>(max_new_tokens_align);
            uint32_t mbytes = align_n / 8;
            auto mask1 = maskCalcBuf.Get<uint8_t>(0);
            auto mask2 = maskCalcBuf.Get<uint8_t>(mbytes);

            AscendC::CompareScalar<int32_t, uint8_t>(
                mask1, sampledLocal[soff], static_cast<int32_t>(0),
                AscendC::CMPMODE::LT, align_n);
            AscendC::CompareScalar<int32_t, uint8_t>(
                mask2, sampledLocal[soff], static_cast<int32_t>(vocab_size_val),
                AscendC::CMPMODE::GE, align_n);
            AscendC::Or<uint8_t>(mask1, mask1, mask2, mbytes);

            int32_t first_invalid = -1;
            for (uint32_t mb = 0; mb < mbytes; ++mb) {
                uint8_t byte = mask1.GetValue(mb);
                if (byte) {
                    first_invalid = static_cast<int32_t>(mb * 8 + __builtin_ctz(byte));
                    break;
                }
            }
            if (first_invalid >= 0) {
                valid_count = first_invalid;
            } else {
                valid_count = static_cast<int32_t>(align_n);
            }
        }

        int32_t avail_space = max_seq_len - seq_len;
        if (avail_space < 0) avail_space = 0;
        if (valid_count > avail_space) valid_count = avail_space;
        int32_t nt = seq_len + valid_count;

        // Stage 1.5: nextToken
        if (valid_count > 0) {
            nextLocal.SetValue(local_idx,
                sampledLocal.GetValue(soff + static_cast<uint32_t>(valid_count - 1)));
        } else {
            int32_t bp = (nt > 0) ? (nt - 1) : 0;
            nextLocal.SetValue(local_idx, tokenGm.GetValue(gmRow + bp));
        }

        // Stage 2: write valid sampled tokens → tokenGm
        if (valid_count > 0) {
            COPY_UB_TO_GM(tokenGm, static_cast<uint64_t>(gmRow) + seq_len,
                          sampledLocal, soff, valid_count, int32_t);
        }
        return nt;
    }

    // ---------------------------------------------------------------
    // Load entire sequence token for one row (Alloc→COPY→EnQue→DeQue)
    // ---------------------------------------------------------------
    __aicore__ inline AscendC::LocalTensor<int32_t> LoadOneRowToken(
        uint32_t global_row, int32_t nt)
    {
        auto t = tokenInQue.AllocTensor<int32_t>();
        uint64_t gmRow = static_cast<uint64_t>(global_row) *
                         static_cast<uint32_t>(max_seq_len);
        COPY_GM_TO_UB(t, tokenGm, gmRow, nt, int32_t);
        tokenInQue.EnQue(t);
        return tokenInQue.DeQue<int32_t>();
    }

    // ---------------------------------------------------------------
    // Launch MTE2 preload for next short row (Alloc→COPY→EnQue, no DeQue)
    // ---------------------------------------------------------------
    __aicore__ inline void LaunchPreloadNextRow(uint32_t global_row, int32_t nt)
    {
        auto t = tokenInQue.AllocTensor<int32_t>();
        uint64_t gmRow = static_cast<uint64_t>(global_row) *
                         static_cast<uint32_t>(max_seq_len);
        COPY_GM_TO_UB(t, tokenGm, gmRow, nt, int32_t);
        tokenInQue.EnQue(t);
    }

    // Preload next long row's FIRST chunk only (≤ SAFE_CHUNK + max_n_val)
    __aicore__ inline void LaunchPreloadNextLongFirstChunk(uint32_t global_row, int32_t nt)
    {
        int32_t load_cnt = SAFE_CHUNK + max_n_val;
        if (load_cnt > nt) load_cnt = nt;
        auto t = tokenInQue.AllocTensor<int32_t>();
        uint64_t gmRow = static_cast<uint64_t>(global_row) *
                         static_cast<uint32_t>(max_seq_len);
        COPY_GM_TO_UB(t, tokenGm, gmRow, load_cnt, int32_t);
        tokenInQue.EnQue(t);
    }

    // ---------------------------------------------------------------
    // Stage 3: ngram matching with double-buffered chunk load (long path)
    // ---------------------------------------------------------------
    // ---------------------------------------------------------------
    // Stage 3 short path: single-pass ngram on preloaded token
    // ---------------------------------------------------------------
    __aicore__ inline void NgramMatchRowShort(
        int32_t nt,
        AscendC::LocalTensor<int32_t>& tokenLocal,
        int32_t& best_match_pos, int32_t& best_ngram_len)
    {
        best_match_pos = -1;
        best_ngram_len = 0;

        // suffix = tokenLocal[nt - max_n .. nt - 1], accessed as tokenLocal[nt - n]
        auto ngramResult = ngramCalcBuf.Get<int32_t>();
        auto ngramTemp   = ngramTempBuf.Get<int32_t>();
        auto ngramTempF  = ngramTempBuf.Get<float>();
        auto ngramGather = ngramGatherBuf.Get<int32_t>();
        auto ngramReduce = ngramReduceBuf.Get<float>();

        if (max_n_val > 1) {
            AscendC::Arange<int32_t>(ngramGather,
                static_cast<int32_t>(sizeof(int32_t)),
                static_cast<int32_t>(sizeof(int32_t)), nt);
        }
        for (int32_t n = 1; n <= max_n_val; ++n) {
            int32_t valid_len = nt - n;
            if (valid_len <= 0) break;
            if (n > 1) {
                AscendC::Gather<int32_t>(ngramTemp, ngramResult,
                    ngramGather.ReinterpretCast<uint32_t>(), 0, valid_len);
            }
            AscendC::Adds<int32_t>(ngramResult, tokenLocal,
                -tokenLocal.GetValue(static_cast<uint32_t>(nt - n)),
                valid_len);
            if (n > 1) {
                AscendC::Or<uint16_t>(ngramResult.ReinterpretCast<uint16_t>(),
                    ngramResult.ReinterpretCast<uint16_t>(),
                    ngramTemp.ReinterpretCast<uint16_t>(),
                    static_cast<uint32_t>(valid_len * 2));
            }
            if (n < min_n_val) continue;
            AscendC::Cast<float, int32_t>(ngramTempF, ngramResult,
                AscendC::RoundMode::CAST_CEIL, nt - n);
            AscendC::Abs<float>(ngramTempF, ngramTempF, nt - n);
            AscendC::ReduceMin<float>(ngramReduce, ngramTempF, ngramReduce,
                static_cast<uint32_t>(nt - n), true);
            float min_val_f = ngramReduce.GetValue(0);
            if (min_val_f == 0.0f) {
                float min_idx_f = ngramReduce.GetValue(1);
                best_match_pos = static_cast<int32_t>(
                    *reinterpret_cast<uint32_t*>(&min_idx_f));
                best_ngram_len = n;
                if (n == max_n_val) break;
            } else { break; }
        }
    }

    // ---------------------------------------------------------------
    // Stage 3 long path: double-buffered chunk loop, frees all internally
    // ---------------------------------------------------------------
    __aicore__ inline void NgramMatchRowLong(
        uint32_t global_row, int32_t nt, bool first_chunk_preloaded,
        int32_t next_global_row, int32_t next_nt,
        int32_t& preloaded_row_out,
        int32_t& best_match_pos, int32_t& best_ngram_len)
    {
        best_match_pos = -1;
        best_ngram_len = 0;

        int32_t suffix_gm_start = nt - max_n_val;
        if (suffix_gm_start < 0) suffix_gm_start = 0;
        int32_t suffix_load = max_n_val;
        if (suffix_gm_start + suffix_load > nt)
            suffix_load = nt - suffix_gm_start;

        uint64_t gmRow = static_cast<uint64_t>(global_row) *
                         static_cast<uint32_t>(max_seq_len);

        auto suffixTensor = suffixInQue.AllocTensor<int32_t>();
        COPY_GM_TO_UB(suffixTensor, tokenGm, gmRow + suffix_gm_start,
                      suffix_load, int32_t);
        suffixInQue.EnQue(suffixTensor);
        auto suffixLocal = suffixInQue.DeQue<int32_t>();

        auto ngramResult = ngramCalcBuf.Get<int32_t>();
        auto ngramTemp   = ngramTempBuf.Get<int32_t>();
        auto ngramTempF  = ngramTempBuf.Get<float>();
        auto ngramGather = ngramGatherBuf.Get<int32_t>();
        auto ngramReduce = ngramReduceBuf.Get<float>();

        uint32_t max_gather = SAFE_CHUNK + static_cast<uint32_t>(max_n_val);
        if (max_n_val > 1) {
            AscendC::Arange<int32_t>(ngramGather,
                static_cast<int32_t>(sizeof(int32_t)),
                static_cast<int32_t>(sizeof(int32_t)),
                static_cast<int32_t>(max_gather));
        }

        preloaded_row_out = -1;
        bool has_next_row = (next_nt >= min_n_val);
        bool next_is_short = (next_nt > 0 && next_nt <= static_cast<int32_t>(SAFE_CHUNK));
        bool found_global_max = false;
        int32_t search_limit = nt - min_n_val;

        // Load first chunk (use preloaded if previous row launched it)
        AscendC::LocalTensor<int32_t> tokenLocal;
        int32_t cs = 0;
        int32_t cc = (SAFE_CHUNK <= search_limit) ? SAFE_CHUNK : search_limit;
        int32_t lc = cc + max_n_val;
        if (cs + lc > nt) lc = nt - cs;
        if (first_chunk_preloaded) {
            tokenLocal = tokenInQue.DeQue<int32_t>();
        } else {
            auto t0 = tokenInQue.AllocTensor<int32_t>();
            COPY_GM_TO_UB(t0, tokenGm, gmRow + cs, lc, int32_t);
            tokenInQue.EnQue(t0);
            tokenLocal = tokenInQue.DeQue<int32_t>();
        }
        int32_t chunk_start = cs, chunk_count = cc, load_count = lc;
        bool has_preloaded = false;
        bool preloaded_next_row = false;

        while (!found_global_max) {
            // Launch next chunk preload (MTE2, overlaps V).
            // On the last chunk: preload the NEXT ROW instead.
            int32_t next_cs = chunk_start + SAFE_CHUNK;
            bool has_next = (next_cs < search_limit);
            if (has_next) {
                int32_t next_cc = (next_cs + SAFE_CHUNK <= search_limit)
                                ? SAFE_CHUNK : (search_limit - next_cs);
                int32_t next_lc = next_cc + max_n_val;
                if (next_cs + next_lc > nt) next_lc = nt - next_cs;
                auto tn = tokenInQue.AllocTensor<int32_t>();
                COPY_GM_TO_UB(tn, tokenGm, gmRow + next_cs, next_lc, int32_t);
                tokenInQue.EnQue(tn);
                has_preloaded = true;
            } else if (has_next_row) {
                // Last chunk of this row: preload next row's first data
                uint32_t next_gr = static_cast<uint32_t>(next_global_row);
                if (next_is_short) {
                    auto tn = tokenInQue.AllocTensor<int32_t>();
                    uint64_t nr = static_cast<uint64_t>(next_gr) *
                                  static_cast<uint32_t>(max_seq_len);
                    COPY_GM_TO_UB(tn, tokenGm, nr, next_nt, int32_t);
                    tokenInQue.EnQue(tn);
                } else {
                    int32_t ld = SAFE_CHUNK + max_n_val;
                    if (ld > next_nt) ld = next_nt;
                    auto tn = tokenInQue.AllocTensor<int32_t>();
                    uint64_t nr = static_cast<uint64_t>(next_gr) *
                                  static_cast<uint32_t>(max_seq_len);
                    COPY_GM_TO_UB(tn, tokenGm, nr, ld, int32_t);
                    tokenInQue.EnQue(tn);
                }
                preloaded_next_row = true;
                preloaded_row_out = 1;  // signal to caller that next row is preloaded
            }

            // Compute current chunk (V)
            for (int32_t n = 1; n <= max_n_val; ++n) {
                int32_t valid_len = load_count - n;
                if (valid_len <= 0) break;
                if (n > 1) {
                    AscendC::Gather<int32_t>(ngramTemp, ngramResult,
                        ngramGather.ReinterpretCast<uint32_t>(), 0, valid_len);
                }
                AscendC::Adds<int32_t>(ngramResult, tokenLocal,
                    -suffixLocal.GetValue(static_cast<uint32_t>(suffix_load - n)),
                    valid_len);
                if (n > 1) {
                    AscendC::Or<uint16_t>(ngramResult.ReinterpretCast<uint16_t>(),
                        ngramResult.ReinterpretCast<uint16_t>(),
                        ngramTemp.ReinterpretCast<uint16_t>(),
                        static_cast<uint32_t>(valid_len * 2));
                }
                if (n < min_n_val) continue;
                int32_t cc2 = (chunk_start + chunk_count <= nt - n)
                            ? chunk_count : (nt - n - chunk_start);
                if (cc2 <= 0) break;
                AscendC::Cast<float, int32_t>(ngramTempF, ngramResult,
                    AscendC::RoundMode::CAST_CEIL, cc2);
                AscendC::Abs<float>(ngramTempF, ngramTempF, cc2);
                AscendC::ReduceMin<float>(ngramReduce, ngramTempF, ngramReduce,
                    static_cast<uint32_t>(cc2), true);
                float min_val_f = ngramReduce.GetValue(0);
                if (min_val_f == 0.0f) {
                    if (n > best_ngram_len) {
                        float min_idx_f = ngramReduce.GetValue(1);
                        uint32_t pos_u = *reinterpret_cast<uint32_t*>(&min_idx_f);
                        best_match_pos = chunk_start + static_cast<int32_t>(pos_u);
                        best_ngram_len = n;
                        if (n == max_n_val) { found_global_max = true; break; }
                    }
                } else { break; }
            }

            if (found_global_max || !has_next) break;

            // Swap: free current, DeQue next (blocks until MTE2 done)
            tokenInQue.FreeTensor(tokenLocal);
            tokenLocal = tokenInQue.DeQue<int32_t>();
            has_preloaded = false;
            chunk_start = next_cs;
            chunk_count = (chunk_start + SAFE_CHUNK <= search_limit)
                        ? SAFE_CHUNK : (search_limit - chunk_start);
            load_count = chunk_count + max_n_val;
            if (chunk_start + load_count > nt) load_count = nt - chunk_start;
        }

        // Drain unused preloaded chunk (skip next-row preload — leave in queue)
        if (has_preloaded && !preloaded_next_row) {
            auto extra = tokenInQue.DeQue<int32_t>();
            tokenInQue.FreeTensor(extra);
        }
        tokenInQue.FreeTensor(tokenLocal);
        suffixInQue.FreeTensor(suffixLocal);
    }

    // ---------------------------------------------------------------
    // Helper: estimate ReduceMin temp buffer size
    // ---------------------------------------------------------------
    __aicore__ inline uint32_t CalcReduceMinTmpSize(uint32_t count, uint32_t typeSize)
    {
        uint32_t elementsPerBlock  = 32 / typeSize;
        uint32_t elementsPerRepeat = 256 / typeSize;

        auto RoundUp = [](uint32_t x, uint32_t unit) -> uint32_t {
            return (x + unit - 1) / unit;
        };

        uint32_t firstMaxRepeat   = RoundUp(count, elementsPerRepeat);
        uint32_t iter1OutputCount = firstMaxRepeat * 2;
        uint32_t iter2AlignStart  = RoundUp(iter1OutputCount, elementsPerBlock) * elementsPerBlock;
        uint32_t iter2OutputCount = RoundUp(iter1OutputCount, elementsPerRepeat) * 2;
        uint32_t iter3AlignStart  = RoundUp(iter2OutputCount, elementsPerBlock) * elementsPerBlock;
        uint32_t iter3OutputCount = RoundUp(iter2OutputCount, elementsPerRepeat) * 2;
        uint32_t iter3AlignEnd    = RoundUp(iter3OutputCount, elementsPerBlock) * elementsPerBlock;

        return iter2AlignStart + iter3AlignStart + iter3AlignEnd;
    }

private:
    AscendC::TPipe pipe;

    // VECIN input queues
    AscendC::TQue<AscendC::TPosition::VECIN, 2> tokenInQue;  // depth=2 double buffer
    AscendC::TQue<AscendC::TPosition::VECIN, 1> sampledInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> numTokensInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> discardInQue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> suffixInQue;

    // VECOUT output queues
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> nextOutQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> draftOutQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> numValidOutQue;

    // VECCALC scratch
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramCalcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramTempBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramGatherBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ngramReduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> maskCalcBuf;

    // Global tensors
    AscendC::GlobalTensor<int32_t> tokenGm;
    AscendC::GlobalTensor<int32_t> numTokensGm;
    AscendC::GlobalTensor<int32_t> sampledGm;
    AscendC::GlobalTensor<int32_t> discardGm;
    AscendC::GlobalTensor<int32_t> nextTokensGm;
    AscendC::GlobalTensor<int32_t> draftTokensGm;
    AscendC::GlobalTensor<int32_t> numValidGm;

    // Tiling parameters
    int32_t batch_size;
    int32_t max_seq_len;
    int32_t max_seq_len_align;
    int32_t max_new_tokens;
    int32_t max_new_tokens_align;
    int32_t k_val;
    int32_t k_align;
    int32_t vocab_size_val;
    int32_t min_n_val;
    int32_t max_n_val;
    int32_t former_num;
    int32_t rows_per_core;
    int32_t tail_rows;
    int32_t block_rows;
    uint32_t my_rows;
    uint32_t row_offset;
};

extern "C" __global__ __aicore__ void ngram_spec_decode(
    GM_ADDR token_ids, GM_ADDR num_tokens, GM_ADDR sampled,
    GM_ADDR discard, GM_ADDR next_tokens, GM_ADDR draft_tokens,
    GM_ADDR num_valid, GM_ADDR workspace, GM_ADDR tiling)
{
    KernelNgramSpecDecode op;
    op.Init(token_ids, num_tokens, sampled, discard, next_tokens,
            draft_tokens, num_valid, workspace, tiling);
    op.Process();
}
