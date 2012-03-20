/***************************************************************************
 * blitz/array-impl.h    Definition of the Array<P_numtype, N_rank> class
 *
 * $Id: array-impl.h,v 1.25 2005/10/13 23:46:43 julianc Exp $
 *
 ***************************************************************************/
#ifndef CUARRAY_IMPL
#define CUARRAY_IMPL

#include <device_memblock.h>
#include <idxvector.h>

#include <string>

namespace cuda_array
{

    template<typename T_numtype, int N_rank>
    class cuArray : public deviceMemoryBlockReference<T_numtype> 
    {
    private:
        typedef deviceMemoryBlockReference<T_numtype> T_base;
        using T_base::data_;
        using T_base::changeToNullBlock;
        using T_base::numReferences;
        
    protected:
        IdxVector<int, N_rank> length_;
        IdxVector<int, N_rank> stride_;

        inline void computeStrides()  //helper function
            {
                int current_stride = 1;
                for (int n=0; n<N_rank;++n)
                {
                    stride_[n] = current_stride;
                    current_stride *= length_[n];
                }
            }

        inline void setupStorage(int lastRank)
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

        cuArray(const cuArray<T_numtype, N_rank>& array)
            : deviceMemoryBlockReference<T_numtype>()
            {
                reference(const_cast<T_array&>(array));
            }

        //////////////////////////////////////////////
        // Member functions
        //////////////////////////////////////////////

        int cols() const
            { return length_[1]; }

        int columns() const
            { return length_[1]; }

        T_array copy() const;

        void copyfromHost(T_numtype* host_ptr)
            {
                cutilSafeCall( cudaMemcpy(data_, host_ptr, sizeof(T_numtype)*numElements(), cudaMemcpyHostToDevice));
            }
        
        void copytoHost(T_numtype* host_ptr)
            {
                cutilSafeCall( cudaMemcpy(host_ptr, data_, sizeof(T_numtype)*numElements(), cudaMemcpyDeviceToHost));
            }

        const T_numtype*      data() const
            { return data_; }

        T_numtype*            data() 
            { return data_; }


        int depth() const
            { return length_[2]; }

        int dimensions() const
            { return N_rank; }

        // void dump_hdf5(std::string filename, string fieldname) const;
        

        int  extent(int rank) const
            { return length_[rank]; }

        const IdxVector<int,N_rank>&  extent() const
            { return length_; }

        void free() 
            {
                changeToNullBlock();
                length_ = 0;
            }

        int length(int rank) const
            { return length_[rank]; }
        
        const IdxVector<int, N_rank>& length()
            const { return length_; }

        void makeUnique();

        int numElements() const
            { return product(length_); }


        int  rank() const
            { return N_rank; }

        void reference(const T_array&);

        void resize(int extent);
        void resize(int extent1, int extent2);
        void resize(int extent1, int extent2, int extent3);
        void resize(const IdxVector<int,N_rank>&);
         
        int  rows() const
            { return length_[0]; }
 
        const IdxVector<int, N_rank>&    shape() const
            { return length_; }

        int size() const
            { return numElements(); }

        int stride(int rank) const
            { return stride_[rank]; }

        //////////////////////////////////////////////
        // Subscripting operators
        //////////////////////////////////////////////

        template<int N_rank2>
        const T_numtype&  operator()(const IdxVector<int,N_rank2>& index) const
            {
                return data_[dot(index, stride_)];
            }

        template<int N_rank2>
        T_numtype&  operator()(const IdxVector<int,N_rank2>& index) 
            {
                return data_[dot(index, stride_)];
            }

        const T_numtype&  operator()(IdxVector<int,1> index) const
            {
                return data_[index[0] * stride_[0]];
            }

        T_numtype& operator()(IdxVector<int,1> index)
            {
                return data_[index[0] * stride_[0]];
            }

        const T_numtype&  operator()(IdxVector<int,2> index) const
            {
                return data_[index[0] * stride_[0] + index[1] * stride_[1]];
            }

        T_numtype& operator()(IdxVector<int,2> index)
            {
                return data_[index[0] * stride_[0] + index[1] * stride_[1]];
            }

        const T_numtype&  operator()(IdxVector<int,3> index) const
            {
                return data_[index[0] * stride_[0] + index[1] * stride_[1]
                             + index[2] * stride_[2]];
            }

        T_numtype& operator()(IdxVector<int,3> index)
            {
                return data_[index[0] * stride_[0] + index[1] * stride_[1]
                             + index[2] * stride_[2]];
            }

        const T_numtype&  operator()(const IdxVector<int,4>& index) const
            {
                return data_[index[0] * stride_[0] + index[1] * stride_[1]
                             + index[2] * stride_[2] + index[3] * stride_[3]];
            }

        T_numtype& operator()(const IdxVector<int,4>& index)
            {
                return data_[index[0] * stride_[0] + index[1] * stride_[1]
                             + index[2] * stride_[2] + index[3] * stride_[3]];
            }


        const T_numtype&  operator()(int i0) const
            { 
                return data_[i0 * stride_[0]]; 
            }

        T_numtype&  operator()(int i0) 
            {
                return data_[i0 * stride_[0]];
            }

        const T_numtype&  operator()(int i0, int i1) const
            { 
                return data_[i0 * stride_[0] + i1 * stride_[1]];
            }

        T_numtype&  operator()(int i0, int i1)
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
                return data_[i0 * stride_[0] + i1 * stride_[1]
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



        T_array& operator=(T_numtype);
        T_array& initialize(T_numtype);
        T_array& operator+=(T_numtype);
        T_array& operator-=(T_numtype);
        T_array& operator*=(T_numtype);
        T_array& operator/=(T_numtype);

        // Array operands
        T_array& operator=(const cuArray<T_numtype,N_rank>&);

        template<typename P_numtype2> 
        T_array& operator=(const cuArray<P_numtype2,N_rank>&);
        template<typename P_numtype2>
        T_array& operator+=(const cuArray<P_numtype2,N_rank>&);
        template<typename P_numtype2>
        T_array& operator-=(const cuArray<P_numtype2,N_rank>&);
        template<typename P_numtype2>
        T_array& operator*=(const cuArray<P_numtype2,N_rank>&);
        template<typename P_numtype2>
        T_array& operator/=(const cuArray<P_numtype2,N_rank>&);

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

}

#endif // CUARRAY_IMPL