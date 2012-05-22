#ifndef CUARRAY_RESIZE
#define CUARRAY_RESIZE

#ifndef  CUDA_ARRAY_H
 #error <resize.cpp> must be included via <cudaArray.h>
#endif

namespace cuda_array
{
template<typename T_numtype, int N_rank>
void cuArray<T_numtype, N_rank>::resize(int extent0)
{
    if (extent0 != length_[0])
    {
        length_[0] = extent0;
        setupStorage(0);
    }
}

template<typename T_numtype, int N_rank>
void cuArray<T_numtype, N_rank>::resize(int extent0, int extent1)
{
    if ((extent0 != length_[0]) || (extent1 != length_[1]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        setupStorage(1);
    }
}

template<typename T_numtype, int N_rank>
void cuArray<T_numtype, N_rank>::resize(int extent0, int extent1,
    int extent2)
{
    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        setupStorage(2);
    }
}


template<typename T_numtype, int N_rank>
void cuArray<T_numtype, N_rank>::resize(int extent0, int extent1,
                                      int extent2, int extent3)
{
    if ((extent0 != length_[0]) || (extent1 != length_[1])
        || (extent2 != length_[2]) || (extent3 != length_[3]))
    {
        length_[0] = extent0;
        length_[1] = extent1;
        length_[2] = extent2;
        length_[3] = extent3;
        setupStorage(3);
    }
}

template<typename T_numtype, int N_rank>
void cuArray<T_numtype, N_rank>::resize(Range r0)
{
	length_[0] = r0.length();
    setupStorage(0);
}

template<typename T_numtype, int N_rank>
void cuArray<T_numtype, N_rank>::resize(Range r0, Range r1)
{
	length_[0] = r0.length();
	length_[1] = r1.length();
	setupStorage(1);
}

template<typename T_numtype, int N_rank>
void cuArray<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2)
{ 
	length_[0] = r0.length();
	length_[1] = r1.length();
	length_[2] = r2.length();
	setupStorage(2);
}

template<typename T_numtype, int N_rank>
void cuArray<T_numtype, N_rank>::resize(Range r0, Range r1, Range r2,
                                      Range r3)
{
	length_[0] = r0.length();
	length_[1] = r1.length();
	length_[2] = r2.length();
	length_[3] = r3.length();
	setupStorage(3);
} 

}// namespace end


#endif  //CUARRAY_RESIZE
