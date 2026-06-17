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
#ifndef MEGA_MOE_TORCH_ADPT_H
#define MEGA_MOE_TORCH_ADPT_H

namespace vllm_ascend {

std::tuple<at::Tensor, at::Tensor> npu_mega_moe(
    const at::Tensor &context,
    const at::Tensor &x,
    const at::Tensor &topk_ids,
    const at::Tensor &topk_weights,
    const at::TensorList &weight1,
    const at::TensorList &weight2,
    int64_t moe_expert_num,
    int64_t ep_world_size,
    int64_t ccl_buffer_size,
    const c10::optional<at::TensorList> &weight_scales1,
    const c10::optional<at::TensorList> &weight_scales2,
    const c10::optional<at::TensorList> &bias1,
    const c10::optional<at::TensorList> &bias2,
    const c10::optional<at::Tensor> &x_active_mask,
    int64_t max_recv_token_num,
    int64_t dispatch_quant_mode,
    int64_t combine_quant_mode,
    c10::string_view comm_alg,
    int64_t num_max_tokens_per_rank,
    c10::string_view activation,
    c10::optional<float> activation_clamp,
    c10::optional<int64_t> dispatch_quant_out_dtype,
    c10::optional<int64_t> weight1_type,
    c10::optional<int64_t> weight2_type)
{
    auto x_shape = x.sizes();
    int bs = x_shape[0];
    int h = x_shape[1];

    at::Tensor y_out = at::empty({bs, h}, x.options());
    auto opts = x.options().dtype(at::kInt);
    int64_t num_local_experts = moe_expert_num / ep_world_size;
    at::Tensor expert_token_nums_out = at::empty({num_local_experts}, opts);

    // Resolve optional params to concrete values for the ACLNN API.
    float activation_clamp_f = activation_clamp.has_value()
        ? activation_clamp.value()
        : std::numeric_limits<float>::max();
    int64_t dispatch_quant_out_dtype_val = dispatch_quant_out_dtype.has_value()
        ? dispatch_quant_out_dtype.value()
        : static_cast<int64_t>(28);  // ge::DT_UNDEFINED

    std::string comm_alg_str(comm_alg);
    char *comm_alg_ptr = comm_alg_str.data();
    std::string activation_str(activation);
    char *activation_ptr = activation_str.data();

    // Note: weight1_type and weight2_type are accepted for API compatibility
    // with ops-transformer but are not passed to aclnnMegaMoe — the ACLNN
    // operator derives tensor data types from the input tensors directly.
    (void)weight1_type;
    (void)weight2_type;

    EXEC_NPU_CMD(aclnnMegaMoe,
        context, x, topk_ids, topk_weights,
        weight1, weight2,
        weight_scales1, weight_scales2,
        bias1, bias2,
        x_active_mask,
        moe_expert_num, ep_world_size, ccl_buffer_size,
        max_recv_token_num, dispatch_quant_mode, dispatch_quant_out_dtype_val,
        combine_quant_mode, comm_alg_ptr,
        num_max_tokens_per_rank, activation_ptr, activation_clamp_f,
        y_out, expert_token_nums_out);

    return {y_out, expert_token_nums_out};
}

}
#endif
