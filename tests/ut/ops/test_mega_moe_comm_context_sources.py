from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MEGA_MOE_DIR = REPO_ROOT / "csrc" / "mc2" / "mega_moe"
MEGA_MOE_PY = REPO_ROOT / "vllm_ascend" / "ops" / "mega_moe.py"


def test_comm_context_sources_use_vllm_binding_with_ops_transformer_helpers():
    aclnn_common = MEGA_MOE_DIR / "aclnn_common.h"
    hccl_common = (MEGA_MOE_DIR / "hccl_common.h").read_text()
    comm_context = (MEGA_MOE_DIR / "comm_context.cpp").read_text()

    assert aclnn_common.exists()
    assert '#include "aclnn_common.h"' in hccl_common
    assert "GetOpApiFuncAddrInLib" in hccl_common
    assert '#include "comm_context.h"' in comm_context
    assert "namespace vllm_ascend" in comm_context
    assert "namespace op_api" not in comm_context
    assert "PYBIND11_MODULE" not in comm_context


def test_backend_resolution_is_owned_by_cpp_auto_mode():
    comm_context = (MEGA_MOE_DIR / "comm_context.cpp").read_text()
    mega_moe_py = MEGA_MOE_PY.read_text()

    assert 'backend == "auto"' in comm_context
    assert '"Ascend950"' in comm_context
    assert '"Ascend910B"' in comm_context
    assert '"Ascend910_93"' in comm_context
    assert "'auto', 'kfc' or 'channel'" in comm_context
    assert 'self.group_name, self.ep_world_size, "auto"' in mega_moe_py
    assert "backend = _resolve_backend()" not in mega_moe_py
