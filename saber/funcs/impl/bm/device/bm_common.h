#ifndef ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_COMMON_H
#define ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kernel_param{
    int g;
    int oc;
    int ic;
    int h;
    int w;
} bm_kernel_param_t;

typedef struct bm_conv_param{
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dilation_h;
    int dilation_w;
    bool result_add;
} bm_conv_param_t;

typedef struct conv_secs_info{
    int ocsecs;
    int icsecs;
    int nsecs;
    int hsecs;
} conv_secs_info_t;


#ifdef __cplusplus
}
#endif
#endif //ANAKIN_SABER_FUNCS_IMPL_BM_DEVICE_BM_COMMON_H