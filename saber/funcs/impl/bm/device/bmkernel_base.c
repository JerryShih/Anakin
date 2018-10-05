#include "bmkernel_base.h"
#include "bm_config.h"
#include "bmk_conv.c"
#include "bmk_pooling.c"
#include "bmk_conv_relu.c"
#include <stdio.h>
/**
 * bmkernel_func is the user entry to BMKERNEL just like "main" to some applications.
 * 
 * \param args - Pointer to arguments that user sends from host.
 *               op - Flag to determine the operation type.             
 */

int bmkernel_func(void *args)
{
    bmkernel_api_base* param = (bmkernel_api_base *)args;
    switch (param->op) {
        case ACTIVATION: {
            // bm_activation_fwd(param)
            return 0;
        }
        case CONV: {
            bm_api_conv_forward* api = (bm_api_conv_forward *)param->opParam;
            return bm_conv_fwd(*api);
        }
        case CONV_RELU: {
            bm_api_conv_forward* api = (bm_api_conv_forward *)param->opParam;
            return bm_conv_relu_fwd(*api);
        }
        case POOLING: {
            bm_api_pooling_forward* api = (bm_api_pooling_forward *)param->opParam;
            return bm_pooling_fwd(*api);
        }
        default: {
            printf("op %d is not supported by BM yet.\n", param->op);
            return -1;
        }
    }
}
