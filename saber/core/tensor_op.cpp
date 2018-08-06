#include "tensor_op.h"
#include <random>

namespace anakin {

namespace saber {

template <typename Dtype>
void fill_tensor_host_const_impl(Dtype* dio, Dtype value, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = value;
    }
}

template <typename TargetType>
void fill_tensor_const(Tensor<TargetType>& tensor, float value, typename Tensor<TargetType>::API::stream_t stream) {

    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case AK_UINT8: fill_tensor_host_const_impl((unsigned char*)dio, static_cast<unsigned char>(value), size); break;
        case AK_INT8: fill_tensor_host_const_impl((char*)dio, static_cast<char>(value), size); break;
        case AK_INT16: fill_tensor_host_const_impl((short*)dio, static_cast<short>(value), size); break;
        case AK_UINT16: fill_tensor_host_const_impl((unsigned short*)dio, static_cast<unsigned short>(value), size); break;
        case AK_HALF: fill_tensor_host_const_impl((short*)dio, static_cast<short>(value), size); break;
        case AK_UINT32: fill_tensor_host_const_impl((unsigned int*)dio, static_cast<unsigned int>(value), size); break;
        case AK_INT32: fill_tensor_host_const_impl((int*)dio, static_cast<int>(value), size); break;
        case AK_FLOAT: fill_tensor_host_const_impl((float*)dio, static_cast<float>(value), size); break;
        case AK_DOUBLE: fill_tensor_host_const_impl((double*)dio, static_cast<double>(value), size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template <typename Dtype>
void fill_tensor_host_rand_impl(Dtype* dio, long long size) {
    for (long long i = 0; i < size; ++i) {
        dio[i] = static_cast<Dtype>(rand());
    }
}

template <typename TargetType>
void fill_tensor_rand(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {
    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case AK_UINT8: fill_tensor_host_rand_impl((unsigned char*)dio, size); break;
        case AK_INT8: fill_tensor_host_rand_impl((char*)dio, size); break;
        case AK_INT16: fill_tensor_host_rand_impl((short*)dio, size); break;
        case AK_UINT16: fill_tensor_host_rand_impl((unsigned short*)dio, size); break;
        case AK_UINT32: fill_tensor_host_rand_impl((unsigned int*)dio, size); break;
        case AK_INT32: fill_tensor_host_rand_impl((int*)dio, size); break;
        case AK_HALF: fill_tensor_host_rand_impl((short*)dio, size); break;
        case AK_FLOAT: fill_tensor_host_rand_impl((float*)dio, size); break;
        case AK_DOUBLE: fill_tensor_host_rand_impl((double*)dio, size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template <typename Dtype>
void fill_tensor_host_rand_impl2(Dtype* dio, Dtype vstart, Dtype vend, long long size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);
    for (long long i = 0; i < size; ++i) {
        Dtype random_num = static_cast<Dtype>(vstart + (vend - vstart) * dis(gen));
        dio[i] = random_num;
    }
}

template <typename TargetType>
void fill_tensor_rand(Tensor<TargetType>& tensor, float vstart, float vend, \
    typename Tensor<TargetType>::API::stream_t stream) {

    long long size = tensor.size();
    void* dio = tensor.mutable_data();
    DataType type = tensor.get_dtype();
    switch (type){
        case AK_UINT8: fill_tensor_host_rand_impl2((unsigned char*)dio, static_cast<unsigned char>(vstart),
                                                   static_cast<unsigned char>(vend), size); break;
        case AK_INT8: fill_tensor_host_rand_impl2((char*)dio, static_cast<char>(vstart), static_cast<char>(vend), size); break;
        case AK_INT16: fill_tensor_host_rand_impl2((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size); break;
        case AK_UINT16: fill_tensor_host_rand_impl2((unsigned short*)dio, static_cast<unsigned short>(vstart),
                                                    static_cast<unsigned short>(vend), size); break;
        case AK_UINT32: fill_tensor_host_rand_impl2((unsigned int*)dio, static_cast<unsigned int>(vstart),
                                                    static_cast<unsigned int>(vend), size); break;
        case AK_INT32: fill_tensor_host_rand_impl2((int*)dio, static_cast<int>(vstart), static_cast<int>(vend), size); break;
        case AK_HALF: fill_tensor_host_rand_impl2((short*)dio, static_cast<short>(vstart), static_cast<short>(vend), size); break;
        case AK_FLOAT: fill_tensor_host_rand_impl2((float*)dio, static_cast<float>(vstart), static_cast<float>(vend), size); break;
        case AK_DOUBLE: fill_tensor_host_rand_impl2((double*)dio, static_cast<double>(vstart), static_cast<double>(vend), size); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
}

template <typename Dtype>
void print_tensor_host_impl(const Dtype* din, long long size, int width) {
    for (int i = 0; i < size; ++i) {
        printf("%.6f ", static_cast<float>(din[i]));
        if ((i + 1) % width == 0) {
            printf("\n");
        }
    }
    printf("\n");
}

template <typename TargetType>
void print_tensor(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    LOG(INFO) << "host tensor data:" << tensor.size();
    const void* data_ptr = tensor.data();
    long long size = tensor.size();
    int width = tensor.width();
    DataType type = tensor.get_dtype();
    switch(type) {
        case AK_UINT8: print_tensor_host_impl((const unsigned char*)data_ptr, size, width); break;
        case AK_INT8: print_tensor_host_impl((const char*)data_ptr, size, width); break;
        case AK_UINT16: print_tensor_host_impl((const unsigned short*)data_ptr, size, width); break;
        case AK_INT16: print_tensor_host_impl((const short*)data_ptr, size, width); break;
        case AK_UINT32: print_tensor_host_impl((const unsigned int*)data_ptr, size, width); break;
        case AK_INT32: print_tensor_host_impl((const int*)data_ptr, size, width); break;
        case AK_FLOAT: print_tensor_host_impl((const float*)data_ptr, size, width); break;
        case AK_DOUBLE: print_tensor_host_impl((const double*)data_ptr, size, width); break;
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
    printf("\n");
}

template <typename TargetType>
void print_tensor_valid(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    LOG(INFO) << "host tensor data:" << tensor.valid_size();
    const void* data_ptr = (const void*)((const char*)tensor.data() + tensor.data_offset() * type_length(tensor.get_dtype()));
    long long size = tensor.valid_size();
    int width = tensor.width();
    DataType type = tensor.get_dtype();
    if (tensor.is_continue_mem()) {
        switch(type) {
            case AK_UINT8: print_tensor_host_impl((const unsigned char*)data_ptr, size, width); break;
            case AK_INT8: print_tensor_host_impl((const char*)data_ptr, size, width); break;
            case AK_UINT16: print_tensor_host_impl((const unsigned short*)data_ptr, size, width); break;
            case AK_INT16: print_tensor_host_impl((const short*)data_ptr, size, width); break;
            case AK_UINT32: print_tensor_host_impl((const unsigned int*)data_ptr, size, width); break;
            case AK_INT32: print_tensor_host_impl((const int*)data_ptr, size, width); break;
            case AK_FLOAT: print_tensor_host_impl((const float*)data_ptr, size, width); break;
            case AK_DOUBLE: print_tensor_host_impl((const double*)data_ptr, size, width); break;
            default: LOG(FATAL) << "data type: " << type << " is unsupported now";
        }
        printf("\n");
    } else {
        Tensor<TargetType> tvalid(tensor.valid_shape());
        tvalid.copy_from(tensor);
        print_tensor<TargetType>(tvalid, stream);
    }

}

template <typename Dtype>
void tensor_cmp_host(const Dtype* src1, const Dtype* src2, \
                     int size, double& max_ratio, double& max_diff) {

    const double eps = 1e-6f;
    max_diff = fabs(src1[0] - src2[0]);
    max_ratio = fabs(2.0 * max_diff / (src1[0] + src2[0] + eps));

    for (int i = 1; i < size; ++i) {
        double diff = fabs(src1[i] - src2[i]);

        if (max_diff < diff) {
            max_diff = diff;
            max_ratio = fabs(2.0 * max_diff / (src1[i] + src2[i] + eps));
            //LOG(INFO) << "compare two src1: "<< src1[i] << " src2: "<< src2[i] << "i = "<< i << " max_ratio: " << max_ratio ;
        }
    }
}

template <typename Dtype>
double tensor_mean_value_host_impl(const Dtype* din, long long size) {
    double sum = 0.0;
    for (long long i = 0; i < size; ++i) {
        sum += din[i];
    }
    return sum / size;
}

template <typename TargetType>
double tensor_mean_value(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    const void* data_ptr = tensor.data();
    long long size = tensor.size();
    DataType type = tensor.get_dtype();
    switch (type) {
        case AK_UINT8: return tensor_mean_value_host_impl((const unsigned char*)data_ptr, size);
        case AK_INT8: return tensor_mean_value_host_impl((const char*)data_ptr, size);
        case AK_UINT16: return tensor_mean_value_host_impl((const unsigned short*)data_ptr, size);
        case AK_INT16: return tensor_mean_value_host_impl((const short*)data_ptr, size);
        case AK_UINT32: return tensor_mean_value_host_impl((const unsigned int*)data_ptr, size);
        case AK_INT32: return tensor_mean_value_host_impl((const int*)data_ptr, size);
        case AK_FLOAT: return tensor_mean_value_host_impl((const float*)data_ptr, size);
        case AK_DOUBLE: return tensor_mean_value_host_impl((const double*)data_ptr, size);
        default: LOG(FATAL) << "data type: " << type << " is unsupported now";
    }
    return 0.0;
}

template <typename TargetType>
double tensor_mean_value_valid(Tensor<TargetType>& tensor, typename Tensor<TargetType>::API::stream_t stream) {

    const void* data_ptr = (const void*)((const char*)tensor.data() + tensor.data_offset() * type_length(tensor.get_dtype()));
    long long size = tensor.valid_size();
    DataType type = tensor.get_dtype();

    if (tensor.is_continue_mem()) {
        switch (type) {
            case AK_UINT8: return tensor_mean_value_host_impl((const unsigned char*)data_ptr, size);
            case AK_INT8: return tensor_mean_value_host_impl((const char*)data_ptr, size);
            case AK_UINT16: return tensor_mean_value_host_impl((const unsigned short*)data_ptr, size);
            case AK_INT16: return tensor_mean_value_host_impl((const short*)data_ptr, size);
            case AK_UINT32: return tensor_mean_value_host_impl((const unsigned int*)data_ptr, size);
            case AK_INT32: return tensor_mean_value_host_impl((const int*)data_ptr, size);
            case AK_FLOAT: return tensor_mean_value_host_impl((const float*)data_ptr, size);
            case AK_DOUBLE: return tensor_mean_value_host_impl((const double*)data_ptr, size);
            default: LOG(FATAL) << "data type: " << type << " is unsupported now";
        }
    } else {
        Tensor<TargetType> tvalid(tensor.valid_shape());
        tvalid.copy_from(tensor);
        return tensor_mean_value<TargetType>(tvalid, stream);
    }

    return 0.0;
}


#define FILL_TENSOR_HOST(target) \
    template void fill_tensor_const<target>(Tensor<target>& tensor, float value, typename Tensor<target>::API::stream_t stream); \
    template void fill_tensor_rand<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template void fill_tensor_rand<target>(Tensor<target>& tensor, float vstart, float vend, typename Tensor<target>::API::stream_t stream); \
    template void print_tensor<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template void print_tensor_valid<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template double tensor_mean_value<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream); \
    template double tensor_mean_value_valid<target>(Tensor<target>& tensor, typename Tensor<target>::API::stream_t stream);

#if defined(BUILD_LITE) || defined(USE_X86_PLACE) || defined(USE_AMD) || defined(USE_CUDA) || defined(USE_BM)
FILL_TENSOR_HOST(X86)
#endif

#ifdef USE_CUDA
FILL_TENSOR_HOST(NVHX86)
#endif

#ifdef USE_ARM_PLACE
FILL_TENSOR_HOST(ARM)
#endif

#ifdef USE_ARM_PLACE
FILL_TENSOR_HOST(BM)
#endif

template void tensor_cmp_host<float>(const float* src1, const float* src2, \
                                     int size, double& max_ratio, double& max_diff);
template void tensor_cmp_host<char>(const char* src1, const char* src2, int size, \
                                    double& max_ratio, double& max_diff);

#ifdef USE_BM

        template<>
void fill_tensor_device_rand<Tensor<BM, AK_BM, NCHW>>(Tensor<BM, AK_BM, NCHW>& tensor, \
    typename Tensor<BM, AK_BM, NCHW>::API::stream_t stream) {

    float *host_mem_input = new float[tensor.size()];
    for (int i = 0; i < tensor.size(); ++i) {
        host_mem_input[i] = static_cast<float>(rand());
    }

    bm_device_mem_t* device_data_ptr = tensor.mutable_data();
    BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

    delete [] host_mem_input;
}

void fill_tensor_device_rand(Tensor<BM, AK_BM, NCHW>& tensor, float vstart, \
    float vend, typename Tensor<BM, AK_BM, NCHW>::API::stream_t stream = NULL){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0, 1.f);

    float *host_mem_input = new float[tensor.size()];
    for (int i = 0; i < tensor.size(); ++i) {
        float random_num = vstart + (vend - vstart) * dis(gen);
        host_mem_input[i] = random_num;
    }

    bm_device_mem_t* device_data_ptr = tensor.mutable_data();
    BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

    delete [] host_mem_input;
}

void fill_tensor_device_const(Tensor<BM, AK_BM, NCHW>& tensor, float value, \
    typename Tensor<BM, AK_BM, NCHW>::API::stream_t stream = NULL){

    float *host_mem_input = new float[tensor.size()];
    for (int i = 0; i < tensor.size(); ++i) {
        host_mem_input[i] = value;
    }

    bm_device_mem_t* device_data_ptr = tensor.mutable_data();
    BMDNN_CHECK(bm_memcpy_s2d(get_bm_handle(), *device_data_ptr, bm_mem_from_system(host_mem_input)));

    delete [] host_mem_input;
}

template <>
void print_tensor_device<Tensor<BM, AK_BM, NCHW>>(Tensor<BM, AK_BM, NCHW>& tensor,  \
    typename Tensor<BM, AK_BM, NCHW>::API::stream_t stream) {

    LOG(INFO) << "BM device tensor data:" << tensor.size();

    /*
    const bm_device_mem_t* device_data_ptr = tensor.data();
    unsigned long long gaddr = bm_mem_get_device_addr(*device_data_ptr);
    bm_flush(get_bm_handle());
    float* device_data = (float*)bm_get_global_addr(gaddr);

    for (int i = 0; i < tensor.size(); ++i) {
        printf("%.2f ", device_data[i]);

        if ((i + 1) % (4 * tensor.width()) == 0) {
            printf("\n");
        }
    }*/

    float *host_mem = new float[tensor.size()];
    auto* device_data_ptr = const_cast<bm_device_mem_t *>(tensor.data());
    bm_memcpy_d2s(get_bm_handle(), bm_mem_from_system(host_mem), *device_data_ptr);

    for (int i = 0; i < tensor.size(); ++i) {
        printf("%.2f\t", host_mem[i]);

        if ((i + 1) % tensor.width() == 0){
            printf("\n");
        }
    }
    printf("\n");

    delete [] host_mem;
}

#endif

} //namespace saber

} //namespace anakin
