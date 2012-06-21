#ifndef ASSIGNMENT
#define ASSIGNMENT

namespace cuda_array
{
    const int THREADS = 512;
    
    // forward declaration
    template <class L, class R>
    __global__ void assign(L dest, R expr)
    {
        const int tid = blockIdx.x*blockDim.x + threadIdx.x;
        if (tid < dest.numElements() )
        {
            dest[tid] = expr[tid];
        }
    }
    
    template <class L, class R>
    __global__ void update_add(L dest, R expr)
    {
        const int tid = blockIdx.x*blockDim.x + threadIdx.x;
        if (tid < dest.numElements() )
        {
            dest[tid] += expr[tid];
        }
    }
    
    template <class L, class R>
    __global__ void update_sub(L dest, R expr)
    {
        const int tid = blockIdx.x*blockDim.x + threadIdx.x;
        if (tid < dest.numElements() )
        {
            dest[tid] -= expr[tid];
        }
    }

    template <class L, class R>
    __global__ void update_mul(L dest, R expr)
    {
        const int tid = blockIdx.x*blockDim.x + threadIdx.x;
        if (tid < dest.numElements() )
        {
            dest[tid] *= expr[tid];
        }
    }
    
    template<typename T_type >
    class  ExprLiteral
    {
        T_type value;
    public:
        typedef T_type T;
        ExprLiteral(T_type v):value(v) 
            { }
        __device__ T operator[] (size_t index) const
            {
                return value;
            }
    };
}   //namespace cuda_array

#endif //ASSIGNMENT
