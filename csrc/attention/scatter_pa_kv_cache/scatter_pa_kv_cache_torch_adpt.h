/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SCATTER_PA_KV_CACHE_TORCH_ADPT_H_
#define SCATTER_PA_KV_CACHE_TORCH_ADPT_H_

namespace vllm_ascend {

void npu_scatter_pa_kv_cache(
    const at::Tensor& key,
    at::Tensor& key_cache,
    const at::Tensor& slot_mapping,
    const at::Tensor& value,
    at::Tensor& value_cache,
    const c10::optional<at::Tensor>& compress_lens = c10::nullopt,
    const c10::optional<at::Tensor>& compress_seq_offset = c10::nullopt,
    const c10::optional<at::Tensor>& seq_lens = c10::nullopt,
    c10::string_view cache_mode = "Norm",
    c10::string_view scatter_mode = "None",
    c10::optional<at::IntArrayRef> strides = c10::nullopt,
    c10::optional<at::IntArrayRef> offsets = c10::nullopt)
{
    TORCH_CHECK(key.defined(), "key must be defined");
    TORCH_CHECK(key_cache.defined(), "key_cache must be defined");
    TORCH_CHECK(slot_mapping.defined(), "slot_mapping must be defined");

    std::string cache_mode_str(cache_mode);
    std::string scatter_mode_str(scatter_mode);
    char* cache_mode_ptr = const_cast<char*>(cache_mode_str.c_str());
    char* scatter_mode_ptr = const_cast<char*>(scatter_mode_str.c_str());

    EXEC_NPU_CMD(
        aclnnScatterPaKvCache,
        key,
        key_cache,
        slot_mapping,
        value,
        value_cache,
        compress_lens,
        compress_seq_offset,
        seq_lens,
        cache_mode_ptr,
        scatter_mode_ptr,
        strides,
        offsets);
}

}  // namespace vllm_ascend
#endif
