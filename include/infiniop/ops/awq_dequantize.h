#ifndef __INFINIOP_AWQ_DEQUANTIZE_API_H__
#define __INFINIOP_AWQ_DEQUANTIZE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAWQDequantizeDescriptor_t;

__C __export infiniStatus_t infiniopCreateAWQDequantizeDescriptor(
    infiniopHandle_t handle,
    infiniopAWQDequantizeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t qweight_desc,
    infiniopTensorDescriptor_t zeros_desc,
    infiniopTensorDescriptor_t scales_desc,
    int group_size);

__C __export infiniStatus_t infiniopGetAWQDequantizeWorkspaceSize(
    infiniopAWQDequantizeDescriptor_t desc, 
    size_t *size);

__C __export infiniStatus_t infiniopAWQDequantize(
    infiniopAWQDequantizeDescriptor_t desc, 
    void *workspace, 
    size_t workspace_size,
    void *y, 
    const void *qweight, 
    const void *zeros, 
    const void *scales, 
    void *stream);

__C __export infiniStatus_t infiniopDestroyAWQDequantizeDescriptor(
    infiniopAWQDequantizeDescriptor_t desc);

#endif // __INFINIOP_AWQ_DEQUANTIZE_API_H__