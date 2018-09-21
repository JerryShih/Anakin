#include <stdio.h>
#include "bm_common.h"

int bm_conv_fwd(bm_api_conv_forward conv_param)
{
    // Unpack parameters
    u64 ifmap_offset_global = conv_param.ifmap_offset_global;
    u64 ofmap_offset_global = conv_param.ofmap_offset_global;
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

    int kh_ext = dh * (kh - 1) + 1;
    int kw_ext = dw * (kw - 1) + 1;
    int output_h = (input_h + 2 * pad_h - kh_ext) / stride_h + 1;
    int output_w = (input_w + 2 * pad_w - kw_ext) / stride_w + 1;

    int ic = input_c / groups;
    int oc = output_c / groups;
    int ic_per_NPU = ceiling_func_shift(ic, NPU_SHIFT);
    int oc_per_NPU = ceiling_func_shift(oc, NPU_SHIFT);
    int bias_offset_local = 0;
    int bias_tensor_size = oc_per_NPU * FLOAT_SIZE;
    int weight_offset_local = bias_offset_local + bias_tensor_size;
    int weight_group_offset = oc * ic * kh * kw;
    int weight_tensor_size = ic * oc_per_NPU * kh * kw * FLOAT_SIZE;
    int weight_capacity = addr_EU_align(weight_tensor_size + bias_tensor_size);
    int ifmap_group_offset = ic * input_h * input_w;
    int ofmap_group_offset = oc * output_h * output_w;
    int global_ifmap_Nstride = ifmap_group_offset * groups;
    int global_ofmap_Nstride = ofmap_group_offset * groups;
    int nslice = input_n, ocslice = oc, icslice = ic, hslice = output_h;
    nslice = input_n / nsecs;
    int n_residual = input_n - nslice * nsecs;
    hslice = output_h / hsecs;
    int h_residual = output_h - hslice * hsecs;
    icslice = ic / icsecs;
    int ic_residual = ic - icslice * icsecs;
    ocslice = oc / ocsecs;
    int oc_residual = oc - ocslice * ocsecs;
    int bias_group_offset = oc;
    int max_icslice = icslice + (ic_residual > 0);
    int max_ic_per_NPU = ceiling_func_shift(max_icslice, NPU_SHIFT);
    int max_ocslice = ocslice + (oc_residual > 0);
    int max_oc_per_NPU = ceiling_func_shift(max_ocslice, NPU_SHIFT);

    for (int ig = 0; ig < groups; ig++){
        int ocend = 0;
        for (int ocidx = 0; ocidx < ocsecs; ocidx++){
            int ocstart = ocend;
            int cur_ocslice = ocslice + (oc_residual > ocidx);
            ocend = ocstart + cur_ocslice;
            oc_per_NPU = ceiling_func_shift(cur_ocslice, NPU_SHIFT);
            if (using_bias){
                bm_res = bm_atomic_tensor_compact_move(
                    start_npu_idx,
                    bias_offset_local,
                    bias_offset_global + (ig * bias_group_offset + ocstart) * FLOAT_SIZE,
                    1, // n
                    cur_ocslice, // c 
                    1, // h
                    1, // w
                    DMA_G2L, // direction
                    false, // transpose
                    false // add results
                );
                if (bm_res != BM_ATOMIC_SUCCESS) {
                    printf("bm_atomic_tensor_compact_move failed.\n");
                    return -1;
                }
            }
            weight_capacity = max_icslice * oc_per_NPU * kh * kw * FLOAT_SIZE;
            int ofmap_offset_local = addr_EU_align(weight_capacity + weight_offset_local);
            int nend = 0;
            for (int nidx = 0; nidx < nsecs; nidx++){
                int nstart = nend;
                int sec_len_n = nslice + (nidx < n_residual);
                nend = nstart + sec_len_n;
                int o_hb = 0;
                for (int hidx = 0; hidx < hsecs; hidx++){
                    int o_ht = o_hb;
                    int o_h = hslice + (h_residual > hidx);
                    o_hb = o_ht + o_h;
                    int i_ht = bm_max(o_ht * stride_h - pad_h, 0);
                    int pad_h_t = 0;
                    if (i_ht == 0){
                        pad_h_t = pad_h - o_ht * stride_h;
                    }
                    int i_hb = bm_min(o_hb * stride_h + kh_ext - 1 - pad_h, input_h);
                    int pad_h_b = 0;
                    if (i_hb == input_h){
                        pad_h_b = o_hb * stride_h + kh_ext - 1 - pad_h - input_h;
                    }
                    int i_h = i_hb - i_ht;
                    int ifmap_align_size = get_neuron_csize_local(i_h, input_w);
                    int ifmap_tensor_size = sec_len_n * max_ic_per_NPU * ifmap_align_size;
                    int ofmap_align_size = get_neuron_csize_local(o_h, output_w);
                    int ofmap_tensor_size = sec_len_n * max_oc_per_NPU * ofmap_align_size;
                    int ifmap_offset_local = ofmap_offset_local + ofmap_tensor_size;
                    int offset_local_end = ifmap_offset_local + ifmap_tensor_size;
                    if (offset_local_end > LOCAL_MEM_SIZE) {
                        printf("local memory not enough.\n");
                        return -1;
                    }
                    if (result_add){
                        u64 shift = nstart * global_ofmap_Nstride + ig * ofmap_group_offset +
                                    (ocstart * output_h + o_ht) * output_w;
                        int local_cstride = get_cstride_local(o_h, output_w);
                        bm_res = bm_atomic_tensor_stride_move(
                            start_npu_idx,
                            ofmap_offset_local, 
                            ofmap_offset_global + shift * FLOAT_SIZE,
                            sec_len_n, // n
                            cur_ocslice, // c
                            o_h, // h
                            output_w, //w
                            oc_per_NPU * local_cstride, // dst_stride_n
                            local_cstride, // dst_stride_c
                            output_w, // dst_stride_h
                            global_ofmap_Nstride, // src_stride_n
                            output_h * output_w, // src_stride_c
                            output_w, // src_stride_h
                            DMA_G2L, // direction
                            DMA_F32, // format
                            false // transpose
                        );
                        if (bm_res != BM_ATOMIC_SUCCESS) {
                            printf("bm_atomic_tensor_stride_move failed.\n");
                            return -1;
                        }
                    }
                    int icend = 0;
                    for (int icidx = 0; icidx < icsecs; icidx++){
                        int icstart = icend;
                        int cur_icslice = icslice + (ic_residual > icidx);
                        icend = icstart + cur_icslice;
                        ic_per_NPU = ceiling_func_shift(cur_icslice, NPU_SHIFT);
                        u64 shift = (ocstart * ic + icstart) * kh * kw + ig * weight_group_offset;
                        if ((icsecs != 1) || (nidx == 0 && hidx == 0)){
                            bm_res = bm_atomic_tensor_stride_move(
                                start_npu_idx,
                                weight_offset_local, 
                                weight_offset_global + shift * FLOAT_SIZE,
                                1, // n
                                cur_ocslice, // c
                                cur_icslice, // h
                                kh * kw, // w
                                0, // dst_stride_n
                                cur_icslice * kh * kw, // dst_stride_c
                                kh * kw, // dst_stride_h
                                0, // src_stride_n
                                ic * kh * kw, // src_stride_c
                                kh * kw, // src_stride_h
                                DMA_G2L, // direction
                                DMA_F32, // format
                                false // transpose
                            );
                            if (bm_res != BM_ATOMIC_SUCCESS) {
                                printf("bm_atomic_tensor_stride_move failed.\n");
                                return -1;
                            }
                        }
                        shift = nstart * global_ifmap_Nstride + ig * ifmap_group_offset +
                                (icstart * input_h + i_ht) * input_w;
                        int local_cstride = get_cstride_local(i_h, input_w);
                        bm_res = bm_atomic_tensor_stride_move(
                            start_npu_idx,
                            ifmap_offset_local, 
                            ifmap_offset_global + shift * FLOAT_SIZE,
                            sec_len_n, // n
                            cur_icslice, // c
                            i_h, // h
                            input_w, // w
                            ic_per_NPU * local_cstride, // dst_stride_n
                            local_cstride, // dst_stride_c
                            input_w, // dst_stride_h
                            global_ifmap_Nstride, // src_stride_n
                            input_h * input_w, // src_stride_c
                            input_w, // src_stride_h
                            DMA_G2L, // direction
                            DMA_F32, // format
                            false // transpose
                        );
                        if (bm_res != BM_ATOMIC_SUCCESS) {
                            printf("bm_atomic_tensor_stride_move failed.\n");
                            return -1;
                        }

                        /*local_shape_t ifshape, ofshape;
                        ifshape.n = sec_len_n;
                        ifshape.c = cur_icslice;
                        ifshape.h = i_h;
                        ifshape.w = input_w;
                        ofshape.c = cur_ocslice;
                        ofshape.h = o_h;
                        ofshape.w = output_w;*/
                        
                        bm_res = bm_atomic_conv_kernel_stride(
                            start_npu_idx,
                            LOCAL_MEM_START_ADDR | ofmap_offset_local, // ouput local offset
                            LOCAL_MEM_START_ADDR | ifmap_offset_local, // input local offset
                            LOCAL_MEM_START_ADDR | weight_offset_local, // weight
                            LOCAL_MEM_START_ADDR | bias_offset_local, // bias
                            sec_len_n, // output n
                            cur_ocslice, // output c
                            cur_icslice, // input c
                            i_h, // input h
                            input_w, // input w
                            kh, // kernel h
                            kw, // kernel w
                            kh * kw, // kernel stride n
                            cur_icslice * kh * kw, // kernel stride c
                            kw, // kernel stride h
                            dh, // dilation h
                            dw, // dilation w
                            pad_h_t, // pad top
                            pad_h_b, // pad bottom
                            pad_w, // pad left
                            pad_w, // pad right
                            stride_h, // stride h
                            stride_w, // stride w
                            icidx == icsecs - 1 ? using_bias : 0, // using bias
                            result_add || icidx > 0 // add result
                        );
                        if (bm_res != BM_ATOMIC_SUCCESS) {
                            printf("bm_atomic_conv_kernel_stride failed.\n");
                            return -1;
                        }
                    }
                    u64 shift = nstart * global_ofmap_Nstride + ig * ofmap_group_offset +
                                (ocstart * output_h + o_ht) * output_w;
                    int local_cstride = get_cstride_local(o_h, output_w);
                    
                    bm_res = bm_atomic_tensor_stride_move(
                        start_npu_idx,
                        ofmap_offset_local, 
                        ofmap_offset_global + shift * FLOAT_SIZE,
                        sec_len_n, // n
                        cur_ocslice, // c
                        o_h, // h
                        output_w, // w
                        global_ofmap_Nstride, // dst_stride_n
                        output_h * output_w, // dst_stride_c
                        output_w, // dst_stride_h
                        oc_per_NPU * local_cstride, // src_stride_n
                        local_cstride, // src_stride_c
                        output_w, // src_stride_h
                        DMA_L2G, // direction
                        DMA_F32, // format
                        false // transpose
                    );
                    if (bm_res != BM_ATOMIC_SUCCESS) {
                        printf("bm_atomic_tensor_stride_move failed.\n");
                        return -1;
                    }
                }
            }
        }
    }

    bm_res = bm_atomic_wait_all_task_complete();
    if (bm_res != BM_ATOMIC_SUCCESS) {
        printf("bm_atomic_wait_all_task_complete failed.\n");
        return -1;
    }
    return 0;
}