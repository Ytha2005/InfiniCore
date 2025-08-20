#include "../../../devices/nvidia/nvidia_common.cuh"
#include "awq_dequantize_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

// 解包零点核函数的启动模板
template <unsigned int BLOCK_SIZE, typename Tzero>
INFINIOP_CUDA_KERNEL unpackZerosKernel(
    Tzero *__restrict__ unpacked_zeros,
    const int32_t *__restrict__ qzeros,
    int zeros_n,
    int zeros_m_packed,
    int zeros_m) {
    unpackZerosBlock<BLOCK_SIZE, Tzero>(unpacked_zeros, qzeros, zeros_n, zeros_m_packed, zeros_m);
}

// 主反量化核函数的启动模板
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tzero, typename Tscale>
INFINIOP_CUDA_KERNEL awqDequantizeKernel(
    Tdata *__restrict__ y,
    const int32_t *__restrict__ qweight,
    const Tzero *__restrict__ zeros,
    const Tscale *__restrict__ scales,
    int n,
    int m_packed,
    int m,
    int group_size) {
    awqDequantizeBlock<BLOCK_SIZE, Tdata, Tzero, Tscale>(y, qweight, zeros, scales, n, m_packed, m, group_size);
}

namespace op::awq_dequantize::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t zeros_desc,
    infiniopTensorDescriptor_t scales_desc,
    int group_size) {
    auto result = AWQDequantizeInfo::create(y_desc, qweight_desc, zeros_desc, scales_desc, group_size);
    CHECK_RESULT(result);
    auto info = result.take();

    // 检查张量步长：我们要求最后一维是连续的
    if (info.qweight_strides[1] != 1 || info.zeros_strides[1] != 1 || info.scales_strides[1] != 1 || info.y_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

// 启动解包零点核函数
template <unsigned int BLOCK_SIZE, typename Tzero>
infiniStatus_t launchUnpackZerosKernel(
    int zeros_n, int zeros_m_packed, int zeros_m,
    void *unpacked_zeros,
    const void *qzeros,
    cudaStream_t cuda_stream) {

    dim3 grid_size((zeros_m_packed + BLOCK_SIZE - 1) / BLOCK_SIZE, (zeros_n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    unpackZerosKernel<BLOCK_SIZE, Tzero><<<grid_size, block_size, 0, cuda_stream>>>(
        reinterpret_cast<Tzero *>(unpacked_zeros),
        reinterpret_cast<const int32_t *>(qzeros),
        zeros_n, zeros_m_packed, zeros_m);

    return INFINI_STATUS_SUCCESS;
}

// 启动主反量化核函数
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tzero, typename Tscale>
infiniStatus_t launchAWQDequantizeKernel(
    int n, int m_packed, int m, int group_size,
    void *y,
    const void *qweight,
    const void *zeros,
    const void *scales,
    cudaStream_t cuda_stream) {

    dim3 grid_size((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);

    awqDequantizeKernel<BLOCK_SIZE, Tdata, Tzero, Tscale><<<grid_size, block_size, 0, cuda_stream>>>(
        reinterpret_cast<Tdata *>(y),
        reinterpret_cast<const int32_t *>(qweight),
        reinterpret_cast<const Tzero *>(zeros),
        reinterpret_cast<const Tscale *>(scales),
        n, m_packed, m, group_size);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *qweight, const void *zeros, const void *scales,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 获取信息
    int n = _info.n;
    int m = _info.m;
    int m_packed = _info.m_packed;
    int zeros_n = _info.zeros_n;
    int zeros_m = _info.zeros_m;
    int zeros_m_packed = _info.zeros_m_packed;
    int group_size = _info.group_size;

    // 解包零点需要的工作空间：zeros_n * zeros_m * sizeof(Tzero)
    void *unpacked_zeros = workspace;
    auto zero_type = _info.zero_type;

    // 根据块大小启动解包零点核函数
    unsigned int block_size = _opaque->internal->maxThreadsPerBlock();
    if (block_size == CUDA_BLOCK_SIZE_1024) {
        if (zero_type == INFINI_DTYPE_I8) {
            CHECK_STATUS(launchUnpackZerosKernel<CUDA_BLOCK_SIZE_1024, int8_t>(zeros_n, zeros_m_packed, zeros_m, unpacked_zeros, zeros, cuda_stream));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (block_size == CUDA_BLOCK_SIZE_512) {
        if (zero_type == INFINI_DTYPE_I8) {
            CHECK_STATUS(launchUnpackZerosKernel<CUDA_BLOCK_SIZE_512, int8_t>(zeros_n, zeros_m_packed, zeros_m, unpacked_zeros, zeros, cuda_stream));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (block_size == CUDA_BLOCK_SIZE_4096) {
        if (zero_type == INFINI_DTYPE_I8) {
            CHECK_STATUS(launchUnpackZerosKernel<CUDA_BLOCK_SIZE_4096, int8_t>(zeros_n, zeros_m_packed, zeros_m, unpacked_zeros, zeros, cuda_stream));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    // 第二步：执行主反量化
    auto data_type = _info.data_type;
    auto scale_type = _info.scale_type;

    // 根据数据类型组合启动主反量化核函数
    if (block_size == CUDA_BLOCK_SIZE_1024) {
        if (data_type == INFINI_DTYPE_F16 && scale_type == INFINI_DTYPE_F16) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_1024, half, int8_t, half>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else if (data_type == INFINI_DTYPE_F16 && scale_type == INFINI_DTYPE_F32) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_1024, half, int8_t, float>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else if (data_type == INFINI_DTYPE_F32 && scale_type == INFINI_DTYPE_F32) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_1024, float, int8_t, float>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (block_size == CUDA_BLOCK_SIZE_512) {
        if (data_type == INFINI_DTYPE_F16 && scale_type == INFINI_DTYPE_F16) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_512, half, int8_t, half>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else if (data_type == INFINI_DTYPE_F16 && scale_type == INFINI_DTYPE_F32) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_512, half, int8_t, float>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else if (data_type == INFINI_DTYPE_F32 && scale_type == INFINI_DTYPE_F32) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_512, float, int8_t, float>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (block_size == CUDA_BLOCK_SIZE_4096) {
        if (data_type == INFINI_DTYPE_F16 && scale_type == INFINI_DTYPE_F16) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_4096, half, int8_t, half>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else if (data_type == INFINI_DTYPE_F16 && scale_type == INFINI_DTYPE_F32) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_4096, half, int8_t, float>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else if (data_type == INFINI_DTYPE_F32 && scale_type == INFINI_DTYPE_F32) {
            CHECK_STATUS(launchAWQDequantizeKernel<CUDA_BLOCK_SIZE_4096, float, int8_t, float>(n, m_packed, m, group_size, y, qweight, unpacked_zeros, scales, cuda_stream));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::awq_dequantize::nvidia