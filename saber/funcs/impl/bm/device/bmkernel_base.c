#include "bmkernel_base.h"
#include "bm_common.h"
#include "bm_config.h"
#include <stdio.h>
#include "firmware_core_kernel.h"
#include "atomic_dma_gen_cmd.h"
#include "atomic_tensor_arithmetic_gen_cmd.h"
#include "atomic_md_linear_gen_cmd.h"

/**
 * bmkernel_func is the user entry to BMKERNEL just like "main" to some applications.
 * 
 * \param args - Pointer to arguments that user sends from host.
 *               op - Flag to determine the operation type.             
 */

int bm_conv_fwd_test(bm_api_conv_forward conv_param)
{
    /*const int start_npu_idx = 0;
    u32 src_offset_local = 0;
    u64 ifmap_offset_global = conv_param.src_offset_global;
    int input_n = conv_param.input_n;
    int input_c = conv_param.input_c;
    int input_h = conv_param.input_h;
    int input_w = conv_param.input_w;
    BM_ATOMIC_RESULT bm_res = bm_atomic_tensor_compact_move(
                                start_npu_idx, src_offset_local, ifmap_offset_global, 
                                input_n, input_c, input_h, input_w, 
                                DMA_G2L, false, false);
    printf("bm_atomic_tensor_stride_move： "+bm_res);

    bm_res = bm_atomic_wait_all_task_complete();
    if (bm_res != BM_ATOMIC_SUCCESS) {
        printf("bm_atomic_wait_all_task_complete failed.\n");
        return -1;
    }*/
    return 0;
}

int bmkernel_func(void *args)
{
    bmkernel_api_base* param = (bmkernel_api_base *)args;
    bm_api_conv_forward* api = (bm_api_conv_forward *)param->opParam;
    return bm_conv_fwd_test(*api);
}