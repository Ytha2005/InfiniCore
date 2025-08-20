import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # n, m, group_size
    (1, 64, 64),        # 小尺寸测试
    (16, 2048, 64),     # 常见尺寸
    (32, 4096, 128),    # 较大尺寸
    (64, 8192, 256),    # 大尺寸测试
]

# 缩放因子数据类型
_SCALE_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]
# 输出数据类型
_OUTPUT_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]

# Form the test cases by appending each element of _SCALE_DTYPES to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (scale_dtype,) for test_case in _TEST_CASES_ for scale_dtype in _SCALE_DTYPES
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 2e-3, "rtol": 2e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def awq_dequantize_pytorch(qweight, zeros, scales, group_size):
    """PyTorch 实现的 AWQ 反量化参考实现"""
    bits = 4
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    
    # 解包权重
    iweights = torch.bitwise_right_shift(qweight[:, :, None], shifts[None, None, :]).to(torch.int8)
    iweights = iweights.view(iweights.shape[0], -1)
    
    # 解包零点
    zeros_unpacked = torch.bitwise_right_shift(zeros[:, :, None], shifts[None, None, :]).to(torch.int8)
    zeros_unpacked = zeros_unpacked.view(zeros.shape[0], -1)
    
    # 应用 AWQ 重新排序
    def reverse_awq_order(t):
        AWQ_REVERSE_ORDER = [0, 4, 1, 5, 2, 6, 3, 7]
        reverse_order_tensor = torch.arange(
            t.shape[-1],
            dtype=torch.int32,
            device=t.device,
        )
        reverse_order_tensor = reverse_order_tensor.view(-1, 32 // bits)
        reverse_order_tensor = reverse_order_tensor[:, AWQ_REVERSE_ORDER]
        reverse_order_tensor = reverse_order_tensor.view(-1)
        
        t = t[:, reverse_order_tensor] & 0xF
        return t
    
    zeros_unpacked = reverse_awq_order(zeros_unpacked)
    iweights = reverse_awq_order(iweights)
    
    # 确保只有低4位有效
    iweights = torch.bitwise_and(iweights, (2**bits) - 1)
    zeros_unpacked = torch.bitwise_and(zeros_unpacked, (2**bits) - 1)
    
    # 扩展缩放因子和零点
    scales_expanded = scales.repeat_interleave(group_size, dim=0)
    zeros_expanded = zeros_unpacked.repeat_interleave(group_size, dim=0)
    
    # 计算反量化值
    return (iweights - zeros_expanded) * scales_expanded

def test(
    handle,
    device,
    n,
    m,
    group_size,
    scale_dtype=InfiniDtype.F16,
    output_dtype=InfiniDtype.F16,
    sync=None,
):
    """测试 AWQ 反量化操作"""
    print(
        f"Testing AWQ_Dequantize on {InfiniDeviceNames[device]} with n:{n} m:{m} group_size:{group_size}"
        f" scale_dtype:{InfiniDtypeNames[scale_dtype]} output_dtype:{InfiniDtypeNames[output_dtype]}"
    )
    
    # 计算打包后的维度
    m_packed = m // 8
    zeros_n = n // group_size
    
    # 创建输入张量
    qweight = TestTensor((n, m_packed), None, InfiniDtype.I32, device, mode="random_int", int_range=(0, 2**32-1))
    zeros = TestTensor((zeros_n, m_packed), None, InfiniDtype.I32, device, mode="random_int", int_range=(0, 2**32-1))
    scales = TestTensor((zeros_n, m), None, scale_dtype, device, scale=0.01)
    
    # 创建输出张量
    y = TestTensor((n, m), None, output_dtype, device, mode="zeros")
    
    # 使用 PyTorch 实现计算参考结果
    y_ref = awq_dequantize_pytorch(
        qweight.torch_tensor(), 
        zeros.torch_tensor(), 
        scales.torch_tensor(), 
        group_size
    ).to(y.torch_dtype())
    
    if sync is not None:
        sync()
    
    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    
    check_error(
        LIBINFINIOP.infiniopCreateAWQDequantizeDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            qweight.descriptor,
            zeros.descriptor,
            scales.descriptor,
            group_size,
        )
    )
    
    # 获取工作空间大小
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAWQDequantizeWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)
    
    # 执行 AWQ 反量化
    def lib_awq_dequantize():
        check_error(
            LIBINFINIOP.infiniopAWQDequantize(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                qweight.data(),
                zeros.data(),
                scales.data(),
                None,
            )
        )
    
    lib_awq_dequantize()
    
    # 比较结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, output_dtype)
    if DEBUG:
        debug(y.actual_tensor(), y_ref, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y_ref, atol=atol, rtol=rtol)
    
    # 性能分析
    if PROFILE:
        profile_operation("PyTorch", lambda: awq_dequantize_pytorch(
            qweight.torch_tensor(), zeros.torch_tensor(), scales.torch_tensor(), group_size
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_awq_dequantize(), device, NUM_PRERUN, NUM_ITERATIONS)
    
    # 清理资源
    check_error(LIBINFINIOP.infiniopDestroyAWQDequantizeDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    
    # 配置测试选项
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    # 执行测试
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _OUTPUT_DTYPES)
    
    print("\033[92mAWQ Dequantize test passed!\033[0m")