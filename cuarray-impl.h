/***************************************************************************
 * Xiayu Wang
 * x9wang@ucsd.edu
 * 
 *
 ***************************************************************************/
#ifndef CUARRAY_IMPL
#define CUARRAY_IMPL

#include <device_memblock.h>
#include <idxvector.h>
#include <range.h>
#include <string>
#include <assignment.h>

namespace cuda_array
{

    
    template<typename T_numtype, int N_rank>
    class cuArray : public deviceMemoryBlockReference<T_numtype> 
    {
    public:
        typedef T_numtype T;
    private:
        typedef deviceMemoryBlockReference<T_numtype> Memblock;
        using Memblock::data_;
        using Memblock::changeToNullBlock;
        using Memblock::numReferences;
        
    protected:
        IdxVector<int, N_rank> length_;
        IdxVector<int, N_rank> stride_;

        inline void computeStrides()  //helper function
            {
                int current_stride = 1;
                for (int n=0; n<N_rank; n++)
                {
                    stride_[n] = current_stride;
                    current_stride *= length_[n];
                }
            }

        inline void setupStorage(int lastRank)  //helper function
            {
                for (int i=lastRank + 1; i < N_rank; ++i)
                    length_[i] = length_[lastRank];
                computeStrides();
                int numElem = product(length_);
                if (numElem==0)
                    deviceMemoryBlockReference<T_numtype>::changeToNullBlock();
                else
                    deviceMemoryBlockReference<T_numtype>::newBlock(numElem);
            }
        

    public:
        typedef IdxVector<int, N_rank>  T_index;
        typedef cuArray<T_numtype, N_rank> T_array;
        
        cuArray()
            {
                length_=0;
                stride_=0;
            }

        explicit  cuArray(int length0)
            {
                length_[0] = length0;
                setupStorage(0);
            }

        cuArray(int length0, int length1)
            {
                length_[0] = length0;
                length_[1] = length1;
                setupStorage(1);
            }

        cuArray(int length0, int length1, int length2)
            {
                length_[0] = length0;
                length_[1] = length1;
                length_[2] = length2;
                setupStorage(2);
            }

        cuArray(int length0, int length1, int length2, int length3)
            {
                length_[0] = length0;
                length_[1] = length1;
                length_[2] = length2;
                length_[3] = length3;
                setupStorage(3);
            }

        cuArray(T_numtype*  data, IdxVector<int, N_rank> shape)
            : deviceMemoryBlockReference<T_numtype>(product(shape), data, neverDeleteData)
            {
                length_ = shape;
                computeStrides();
            }

        cuArray(T_numtype*  data, IdxVector<int, N_rank> shape,
                preexistingMemoryPolicy deletionPolicy)
            : deviceMemoryBlockReference<T_numtype>(product(shape), data, deletionPolicy)
            {
                length_ = shape;
                computeStrides();
                if (deletionPolicy == duplicateData)
                    reference(copy());
            }


        cuArray(const IdxVector<int, N_rank>& extent)
            {
                length_ = extent;
                setupStorage(N_rank - 1);
            }

        // copy constructor
        cuArray(const cuArray<T_numtype, N_rank>& array)
            : deviceMemoryBlockReference<T_numtype>()
            {
                reference(const_cast<T_array&>(array));
            }

#include <slice.cpp>

        //////////////////////////////////////////////
        // Member functions
        //////////////////////////////////////////////

        __host__ __device__ int cols() const
            { return length_[1]; }

        int columns() const
            { return length_[1]; }

        T_array copy() const
            {
                int siz = numElements();
                if (siz)
                {
                    cuArray<T_numtype, N_rank> temp(length_);
                    cutilSafeCall( cudaMemcpy(temp.data_, data_, sizeof(T_numtype)*siz,
                                              cudaMemcpyDeviceToDevice) );
                    return temp;
                }
                else
                {
                    return *this;
                }
            }

        T_array copyAsync(cudaStream_t copyStream) const
            {
                int siz = numElements();
                if (siz)
                {
                    cuArray<T_numtype, N_rank> temp(length_);
                    cutilSafeCall( cudaMemcpyAsync(temp.data_, data_,
                                                   sizeof(T_numtype)*siz,
                                                   cudaMemcpyDeviceToDevice,
                                                   copyStream) );
                    return temp;
                }
                else
                {
                    return *this;
                }
            }
                
        void copyfromHost(T_numtype* host_ptr)
            {
                cutilSafeCall( cudaMemcpy(data_, host_ptr,
                                          sizeof(T_numtype)*numElements(), cudaMemcpyHostToDevice));
            }
        
        void copytoHost(T_numtype* host_ptr)
            {
                cutilSafeCall( cudaMemcpy(host_ptr, data_,
                                          sizeof(T_numtype)*numElements(),
                                          cudaMemcpyDeviceToHost));
            }

        void copyfromHostAsync(T_numtype* pinned_host_ptr, cudaStream_t stream)
            {
                cutilSafeCall( cudaMemcpyAsync(data_, pinned_host_ptr,
                                               sizeof(T_numtype)*numElements(),
                                               cudaMemcpyHostToDevice, stream));
            }
        
        void copytoHostAsync(T_numtype* pinned_host_ptr, cudaStream_t stream)
            {
                cutilSafeCall( cudaMemcpy(pinned_host_ptr, data_,
                                          sizeof(T_numtype)*numElements(),
                                          cudaMemcpyDeviceToHost, stream));
            }

        __host__ __device__ const T_numtype*      data() const
            { return data_; }

        __host__ __device__ T_numtype*            data() 
            { return data_; }


        __host__ __device__ int depth() const
            { return length_[2]; }

        __host__ __device__ int dimensions() const
            { return N_rank; }

        // void dump_hdf5(std::string filename, string fieldname) const;
        

        const IdxVector<int,N_rank>&  extent() const
            { return length_; }

        void free() 
            {
                changeToNullBlock();
                length_ = 0;
            }

        __host__ __device__ int length(int rank) const
            { return length_[rank]; }
        
        const IdxVector<int, N_rank>& length()
            const { return length_; }

        __host__ __device__ int numElements() const
            { return product(length_); }

        __host__ __device__ IdxVector<int, N_rank> position(int index) const
            {
                IdxVector<int, N_rank> pos;
                int current_slice = index;
                for (int rank=N_rank-1; rank>=0; rank--)
                {
                    pos[rank] = current_slice/stride_[rank];
                    current_slice -= pos[rank]*stride_[rank];
                }
                return pos;
            }

        int  rank() const
            { return N_rank; }

        void reference(const T_array& cuarray)
            {
                length_ = cuarray.length_;
                stride_ = cuarray.stride_;
                deviceMemoryBlockReference<T_numtype>::changeBlock(const_cast<T_array&>( cuarray));
            }
        
        // resize cuArray, implemented in resize.cpp
        void resize(int extent);
        void resize(int extent1, int extent2);
        void resize(int extent1, int extent2,
                    int extent3);
        void resize(int extent1, int extent2,
                    int extent3, int extent4);
        void resize(Range r1);
        void resize(Range r1, Range r2);
        void resize(Range r1, Range r2, Range r3);
        void resize(Range r1, Range r2, Range r3, Range r4);
         
        __host__ __device__ int  rows() const
            { return length_[0]; }
 
        __host__ __device__ const IdxVector<int, N_rank>&    shape() const
            { return length_; }

        __host__ __device__ int size() const
            { return numElements(); }

        __host__ __device__ int stride(int rank) const
            { return stride_[rank]; }

        //////////////////////////////////////////////
        // Subscripting operators
        //////////////////////////////////////////////
        __host__ __device__ const T_numtype&  operator[](int i0) const
            { 
                return data_[i0]; 
            }
        
        __host__ __device__ T_numtype&  operator[](int i0)
            { 
                return data_[i0]; 
            }

        template<int N_rank2>
        __host__ __device__ const T_numtype&  operator()(const IdxVector<int,N_rank2>& index) const
            {
                return data_[dot(index, stride_)];
            }

        template<int N_rank2>
        __host__ __device__ T_numtype&  operator()(const IdxVector<int,N_rank2>& index) 
            {
                return data_[dot(index, stride_)];
            }

        __host__ __device__ const T_numtype&  operator()(int i0) const
            { 
                return data_[i0 * stride_[0]]; 
            }

        __host__ __device__ T_numtype&  operator()(int i0) 
            {
                return data_[i0 * stride_[0]];
            }

        __host__ __device__ const T_numtype&  operator()(int i0, int i1) const
            { 
                return data_[i0 * stride_[0] + i1 * stride_[1]];
            }

        __host__ __device__ T_numtype&  operator()(int i0, int i1)
            {
                return data_[i0 * stride_[0] + i1 * stride_[1]];
            }

        const T_numtype&  operator()(int i0, int i1, int i2) const
            {
                return data_[i0 * stride_[0] + i1 * stride_[1]
                             + i2 * stride_[2]];
            }

        T_numtype&  operator()(int i0, int i1, int i2) 
            {
                return data_[i0* stride_[0] + i1 * stride_[1]
                             + i2 * stride_[2]];
            }

        const T_numtype&  operator()(int i0, int i1, int i2, int i3) const
            {
                return data_[i0 * stride_[0] + i1 * stride_[1]
                             + i2 * stride_[2] + i3 * stride_[3]];
            }

        T_numtype&  operator()(int i0, int i1, int i2, int i3)
            {
                return data_[i0 * stride_[0] + i1 * stride_[1]
                             + i2 * stride_[2] + i3 * stride_[3]];
            }

        template<class Expr>
        T_array& operator = (Expr exp)
            {
                int BLOCKS = numElements()/THREADS+1;
                assign<<<BLOCKS, THREADS>>>(*this, exp);
                return *this;
            }

        T_array& operator = (T_array& rhs)
            {
                cutilSafeCall(cudaMemcpy(data_, rhs.data_,
                                         sizeof(T_numtype)*numElements(),
                                         cudaMemcpyDeviceToDevice) );
                return *this;
            }

        template<class Expr>
        T_array& operator += (Expr exp)
            {
                int BLOCKS = numElements()/THREADS+1;
                update_add<<<BLOCKS/BLOCK_UNIT_SIZE+1, THREADS>>>(*this, exp);
                return *this;
            }
        
        template<class Expr>
        T_array& operator -= (Expr exp)
            {
                int BLOCKS = numElements()/THREADS+1;
                update_sub<<<BLOCKS, THREADS>>>(*this, exp);
                return *this;
            }
        
#define DEFINE_ASSIGN_CONSTANT(name, op)                                \
        T_array& op (T_numtype rhs)                            \
            {                                                           \
                int BLOCKS = numElements()/THREADS+1;                   \
                name<<<BLOCKS, THREADS>>>(*this, ExprLiteral<T_numtype>(rhs) ); \
                return *this;                                           \
            }                                                           \
        
        DEFINE_ASSIGN_CONSTANT(assign, operator =)
        DEFINE_ASSIGN_CONSTANT(update_add, operator +=)
        DEFINE_ASSIGN_CONSTANT(update_sub, operator -=)
        DEFINE_ASSIGN_CONSTANT(update_mul, operator *=)
    };


/*******************
 * Global Functions
 ***************** */



    template <typename T_numtype,int N_rank>
    void swap(cuArray<T_numtype,N_rank>& a, cuArray<T_numtype,N_rank>& b) {
        cuArray<T_numtype,N_rank> c(a);
        a.reference(b);
        b.reference(c);
    }

} // namespace end

#include <expr.h>
#include <operators.h>
#include <where.h>
#include <resize.cpp>

#endif // CUARRAY_IMPL
