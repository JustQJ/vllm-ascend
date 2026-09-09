// SPDX-License-Identifier: Apache-2.0
#include "register/op_def_registry.h"
namespace ops {
class TurboQuantGroupPrepare : public OpDef {
public:
    explicit TurboQuantGroupPrepare(const char *name) : OpDef(name)
    {
        for (const char *input : {"sparse_indices", "actual_seq_lengths_query", "actual_seq_lengths_kv"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND}).AutoContiguous();
        }
        this->Output("descriptors").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("union_ids").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND});
        this->Output("owners").ParamType(REQUIRED).DataType({ge::DT_UINT8}).Format({ge::FORMAT_ND});
        this->Attr("sparse_mode").AttrType(REQUIRED).Int(3);
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true).DynamicFormatFlag(false)
            .DynamicRankSupportFlag(false).DynamicShapeSupportFlag(true).NeedCheckSupportFlag(false);
        this->AICore().AddConfig("ascend910b", config);
        this->AICore().AddConfig("ascend910_93", config);
    }
};
OP_ADD(TurboQuantGroupPrepare);
} // namespace ops
