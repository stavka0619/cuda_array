#ifndef CUARRAYSLICE
#define CUARRAYSLICE

#ifndef  CUDA_ARRAY_H
 #error <slice.cc> must be included via <cudaArray.h>
#endif

namespace cuda_array
{
    template<typename P_numtype, int N_rank>
    void Array<T_numtype, N_rank>::slice(int rank, Range r)
    {
        int first = r.first(rank);
        int last  = r.last(rank);
        length_[rank] = (last - first) / stride + 1;
        int offset = first * stride_[rank];
        data_ += offset;
    }
}
