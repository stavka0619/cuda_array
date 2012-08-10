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

template<int N_rank2, typename R0, typename R1>
cuArray(cuArray<T_numtype,N_rank2>& array, R0 r0, R1 r1)
{
    deviceMemoryBlockReference<T_numtype>::changeBlock(array);
    int setRank = 0;
    slice(setRank, r0, array, 0);
    slice(setRank, r1, array, 1);
}


template<int N_rank2, typename R0, typename R1, typename R2>
cuArray(cuArray<T_numtype,N_rank2>& array, R0 r0, R1 r1, R2 r2)
{
    deviceMemoryBlockReference<T_numtype>::changeBlock(array);
    int setRank = 0;
    slice(setRank, r0, array, 0);
    slice(setRank, r1, array, 1);
    slice(setRank, r2, array, 2);
}

template<int N_rank2, typename R0, typename R1, typename R2, typename R3>
cuArray(cuArray<T_numtype,N_rank2>& array, R0 r0, R1 r1, R2 r2, R3 r3)
{
    deviceMemoryBlockReference<T_numtype>::changeBlock(array);
    int setRank = 0;
    slice(setRank, r0, array, 0);
    slice(setRank, r1, array, 1);
    slice(setRank, r2, array, 2);
    slice(setRank, r3, array, 3);
 
}

void slice(int rank, Range r)
{
    int first = r.first();
    int last  = r.last(length_(rank)-1);
    length_[rank] = last - first+1;
    int offset = first * stride_[rank];
    data_ += offset;
}

 template<int N_rank2>
void slice(int& setRank, Range r, cuArray<T_numtype,N_rank2>& array, int sourceRank)
{
    length_[setRank] = array.length(sourceRank);
    stride_[setRank] = array.stride(sourceRank);
    slice(setRank, r);
    ++setRank;
}

template<int N_rank2>
void slice(int&, int i, cuArray<T_numtype,N_rank2>& array, int sourceRank)
{
    data_ += i * array.stride(sourceRank);
}

#endif
