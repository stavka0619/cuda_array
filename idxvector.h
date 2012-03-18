/***************************************************************************
 * Helper class for index
 ***************************************************************************/

#ifndef IDXVEC_H
#define IDXVEC_H

#include <stddef.h>

namespace cuda_array
{

    template<typename T_numtype, int N_length>
    class IdxVector {

    private:
        T_numtype data_[N_length];
        
    public:
        typedef IdxVector<T_numtype, N_length> T_vector;
        
        IdxVector()  { }
        ~IdxVector() { }

        inline IdxVector(const IdxVector<T_numtype,N_length>& x)
            {
                for (int i=0; i < N_length; ++i)
                    data_[i] = x.data_[i];
            }

        inline IdxVector(const T_numtype initValue)
            {
                for (int i=0; i < N_length; ++i)
                    data_[i] = initValue;
            }

        inline IdxVector(const T_numtype x[]) {
            memcpy(data_,x,N_length*sizeof(T_numtype));
        }

        IdxVector(T_numtype x0, T_numtype x1)
            {
                data_[0] = x0;
                data_[1] = x1;
            }
        IdxVector(T_numtype x0, T_numtype x1, T_numtype x2)
            {
                data_[0] = x0;
                data_[1] = x1;
                data_[2] = x2;
            }

        IdxVector(T_numtype x0, T_numtype x1, T_numtype x2,
                  T_numtype x3)
            {
                data_[0] = x0;
                data_[1] = x1;
                data_[2] = x2;
                data_[3] = x3;
            }

        T_numtype*  data()
            { return data_; }

        const T_numtype *  data() const
            { return data_; }


        int length() const
            { return N_length; }


        const T_numtype& operator()(size_t i) const
            {
                return data_[i];
            }

        T_numtype&  operator()(size_t i)
            { 
                return data_[i];
            }

        const T_numtype& operator[](size_t i) const
            {
                return data_[i];
            }

        T_numtype&  operator[](size_t i)
            {
                return data_[i];
            }

        //////////////////////////////////////////////
        // Assignment operators
        //////////////////////////////////////////////

//        T_vector& operator=(const T_numtype);
        T_vector& operator+=(const T_numtype);
        T_vector& operator-=(const T_numtype);
        T_vector& operator*=(const T_numtype);
        T_vector& operator/=(const T_numtype);

        template<typename T_numtype2> 
        T_vector& operator=(const IdxVector<T_numtype2, N_length> &);
        template<typename T_numtype2>
        T_vector& operator+=(const IdxVector<T_numtype2, N_length> &);
        template<typename T_numtype2>
        T_vector& operator-=(const IdxVector<T_numtype2, N_length> &);
        template<typename T_numtype2>
        T_vector& operator*=(const IdxVector<T_numtype2, N_length> &);
        template<typename T_numtype2>
        T_vector& operator/=(const IdxVector<T_numtype2, N_length> &);

    };


    template<typename T>
    class IdxVector<T,0> {
    };

//////////////////////////////////////////////
// Global functions
//////////////////////////////////////////////

    template<typename T_numtype, int N_length>
    T_numtype dot(const IdxVector<T_numtype, N_length>& a, 
                  const IdxVector<T_numtype, N_length>& b)
    {
        T_numtype sum=a[0]*b[0];
        for (int i=1; i < N_length; ++i)
            sum += a[i]*b[i];
        return sum;
    }

    template<typename T_numtype, int N_length>
    T_numtype product(const IdxVector<T_numtype, N_length>& vec)
    {
        T_numtype prod=vec[0];
        for (int i=1; i < N_length; ++i)
            prod *= vec[i];
        return prod;
    }

    template<typename T_numtype, int N_length>
    T_numtype sum(const IdxVector<T_numtype, N_length>& vec)
    {
        T_numtype sum=vec[0];
        for (int i=1; i < N_length; ++i)
            sum += vec[i];
        return sum;
    }
}

#endif

