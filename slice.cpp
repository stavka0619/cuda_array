#ifndef CUARRAYSLICE
#define CUARRAYSLICE

#ifndef  CUDA_ARRAY_H
 #error <slice.cc> must be included via <cudaArray.h>
#endif

// slicing
cuArray(cuArray<T_numtype, N_rank>& array, Range r0)
{
    reference(array);
    slice(0, r0);
}

cuArray(cuArray<T_numtype, N_rank>& array, Range r0, Range r1)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
}

cuArray(cuArray<T_numtype, N_rank>& array, Range r0, Range r1, Range r2)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
}

cuArray(cuArray<T_numtype, N_rank>& array, Range r0, Range r1, Range r2,
        Range r3)
{
    reference(array);
    slice(0, r0);
    slice(1, r1);
    slice(2, r2);
    slice(3, r3);
}

void slice(int rank, Range r)
{
    int first = r.first();
    int last  = r.last(length_(rank));
    length_[rank] = (last - first) + 1;
    int offset = first * stride_[rank];
    data_ += offset;
}


#endif
