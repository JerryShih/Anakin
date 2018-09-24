#include "bmkernel_base.h"
#include "bm_config.h"
#include <stdio.h>
#include <stdio.h>
#include "bm_common.h"
/**
 * bmkernel_func is the user entry to BMKERNEL just like "main" to some applications.
 * 
 * \param args - Pointer to arguments that user sends from host.
 *               op - Flag to determine the operation type.             
 */
#define DIV_CEIL(a, b) (((a) - 1) /(b) + 1)
#define EU_NUM_ALIGN(n) (DIV_CEIL(n, EU_NUM) * EU_NUM)
#define EU_OFFSET_ALIGN(offset) (DIV_CEIL(offset, EU_NUM * sizeof(float)) \
    * EU_NUM * sizeof(float))
#define __MIN(a, b)  ((a) < (b) ? (a) :(b))

int bm_conv_fwd_test(bm_api_conv_forward conv_param)
{
    // Unpack parameters
    u64 ifmap_offset_global = conv_param.src_offset_global;
    u64 ofmap_offset_global = conv_param.dst_offset_global;
    u64 weight_offset_global = conv_param.weight_offset_global;
    u64 bias_offset_global = conv_param.bias_offset_global;
    int input_n = conv_param.input_n;
    int input_c = conv_param.input_c;
    int input_h = conv_param.input_h;
    int input_w = conv_param.input_w;
    int groups = conv_param.groups;
    int output_c = conv_param.output_c;
    int kh = conv_param.kh;
    int kw = conv_param.kw;
    int dh = conv_param.dh;
    int dw = conv_param.dw;
    int pad_h = conv_param.pad_h;
    int pad_w = conv_param.pad_w;
    int stride_h = conv_param.stride_h;
    int stride_w = conv_param.stride_w;
    int using_bias = conv_param.using_bias;
    int result_add = conv_param.result_add;
    int icsecs = conv_param.icsecs;
    int ocsecs = conv_param.ocsecs;
    int nsecs = conv_param.nsecs;
    int hsecs = conv_param.hsecs;

    BM_ATOMIC_RESULT bm_res = BM_ATOMIC_SUCCESS;
    const int start_npu_idx = 0;

    int c_per_npu_work = (input_c - 1) / NPU_NUM + 1;
    u32 tensor_local_size = EU_NUM_ALIGN(input_h * input_w) * sizeof(float)
                * c_per_npu_work * input_n;

    u32 src_offset_local = 0;
    u32 dst_local_offset = EU_OFFSET_ALIGN(src_offset_local +
                                             tensor_local_size);

    int local_stride_n = c_per_npu_work * EU_NUM_ALIGN(input_h * input_w);
    int local_stride_c = EU_NUM_ALIGN(input_h * input_w);
    int local_stride_h = input_h;
    int global_stride_n = input_c * input_h * input_w;
    int global_stride_c = input_h * input_w;
    int global_stride_h = input_w;

    bm_res = bm_atomic_tensor_stride_move(
                                start_npu_idx, src_local_offset,
                                ifmap_offset_global, input_n, input_c, input_h,
                                input_w, local_stride_n, local_stride_c,
                                local_stride_h, global_stride_n,
                                global_stride_c, global_stride_h, DMA_G2L,
                                DMA_F32, false);

    if (bm_res != BM_ATOMIC_SUCCESS) {
        printf("bm_atomic_tensor_stride_move failed.\n");
        return -1;
    }

    bm_res = bm_atomic_arithmetic_tensor_add_tensor(
                                start_npu_idx, 
                                src_local_offset,
                                dst_local_offset,
                                input_n, input_c, input_h, input_w,
                                input_n, input_c, input_h, input_w,
                                input_n, input_c, input_h, input_w);

    if (bm_res != BM_ATOMIC_SUCCESS) {
        printf("bm_atomic_tensor_stride_move failed.\n");
        return -1;
    }

    bm_res = bm_atomic_tensor_stride_move(
                                start_npu_idx, dst_local_offset,
                                ofmap_offset_global, input_n, input_c, input_h,
                                input_w, global_stride_n, global_stride_c,
                                global_stride_h, local_stride_n, local_stride_c,
                                local_stride_h, DMA_L2G, DMA_F32, false);
    if (bm_res != BM_ATOMIC_SUCCESS) {
        printf("bm_atomic_tensor_stride_move failed.\n");
        return -1;
    }

    bm_res = bm_atomic_wait_all_task_complete();
    if (bm_res != BM_ATOMIC_SUCCESS) {
        printf("bm_atomic_wait_all_task_complete failed.\n");
        return -1;
    }
    return 0;
}

int bmkernel_func(void *args)
{
    bmkernel_api_base* param = (bmkernel_api_base *)args;
    bm_api_conv_forward* api = (bm_api_conv_forward *)param->opParam;
    return bm_conv_fwd_test(*api);
}