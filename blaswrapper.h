#ifndef BLAS_WRAPPER
#define BLAS_WRAPER

#ifndef  CUDA_ARRAY_H
 #error <blaswrapper.h> must be included via <cudaArray.h>
#endif

namespace cuda_array
{
    // y+=a*x
    template <int N_rank>
    inline void cuarray_Saxpy(cublasHandle_t handler,
                              cuArray<float,N> y,
                              cuArray<float,N> x,
                              float& coeff)
    {
        cublasSaxpy(handler, y.size(), coeff,
                            x.data(), 1, y.data(), 1);
    }

}// namespace 
