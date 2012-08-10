#ifndef EXPR_H
#define EXPR_H

#ifndef  CUDA_ARRAY_H
 #error <expr.h> must be included via <cudaArray.h>
#endif

// #include <boost/type_traits/is_base_of.hpp>

namespace cuda_array
{
    template<class Expr>
    class cuArrayExpr
    {
        Expr& iter;
    public:
        typedef typename Expr::T T;
        cuArrayExpr(Expr& exp):
            iter(exp) { }
        __device__ T operator[] (size_t index) const
            {
                return iter[index];
            }
    };

    template<class Left_expr, class Op, class Right_expr >
    class  cuArrayBinExpr
    {
        Left_expr lhs;
        Right_expr rhs;
    public:
        typedef typename Left_expr::T T;
        cuArrayBinExpr(Left_expr l, Right_expr r)
            : lhs(l), rhs(r) { }
        __device__ typename Op::T operator[] (size_t index) const
            {
                return Op::apply(lhs[index], rhs[index] );
            }
    };

    
    template<typename T_type, int N_rank >
    class  ExprIdentity
    {
        cuArray<T_type,N_rank>& array;
    public:
        typedef T_type T;
        ExprIdentity(cuArray<T_type,N_rank>& ar): array(ar) 
            { }
        __device__ T operator[] (size_t index) const
            {
                return array[index];
            }
    };

    template<class offset, int N> struct offset_at;
    template<class offset>
    struct offset_at<offset, 0>
    {
        static const int value = offset::x1;
    };
    
    template<class offset>
    struct offset_at<offset, 1>
    {
        static const int value = offset::x2;
    };
    
    template<class offset>
    struct offset_at<offset, 2>
    {
        static const int value = offset::x3;
    };
    
    template<class offset>
    struct offset_at<offset, 3>
    {
        static const int value = offset::x4;
    };

    template<int X1=0, int X2=0, int X3=0, int X4=0>
    struct Offset
    {
        typedef Offset<X1, X2, X3, X4> type;
        static const int x1 = X1;
        static const int x2 = X2;
        static const int x3 = X3;
        static const int x4 = X4;
    };
    
    template<class offset, int N_rank>
    __device__ void apply_offset(IdxVector<int,N_rank> &pos)
    {
        pos[N_rank-1] += offset_at<offset,N_rank-1>::value;
        apply_offset<offset>(reinterpret_cast<IdxVector<int,N_rank-1>& >(pos) );
    }
                                
    template<class offset>
    __device__ void apply_offset(IdxVector<int,1> &pos)
    {
        pos[0] += offset_at<offset, 0>::value;
    }

    template<class offset, typename T_type, int N_rank>
    class  ExprIdentityShift: public offset
    {
        cuArray<T_type,N_rank> array;
    public:
        typedef T_type T;
        ExprIdentityShift(cuArray<T_type,N_rank> ar):
            array(ar)  
            {
            }
        __device__ T operator[] (int index) const
            {
                int isInboundary = 1;
                IdxVector<int,N_rank> pos( array.position(index) );
                apply_offset<offset, N_rank>(pos);
                for (int rank=0; rank<N_rank; rank++)
                {
                    isInboundary *= (pos[rank]>=0 && pos[rank]<array.length(rank) );
                }
                return isInboundary ? array(pos) : array[index]; // clamp boundary
            }
    };
    
    template<class offset, typename T, int N_rank>
    ExprIdentityShift<offset, T, N_rank>  shift(cuArray<T, N_rank> array) //, offset dummy)
    {
        ExprIdentityShift<offset, T, N_rank> temp(array);//, dummy);
        
        return  temp;
    }
        
    // cuArrayExpr<ExprIdentityShift<float, 2> > shift(cuArray<float, 2> array, IdxVector<int, 2> sft)
    // {
    //     typedef ExprIdentityShift<float, 2> ExprShift;
    //     return  cuArrayExpr<ExprShift> (ExprShift(array, sft) );
    // }
    // whether T is expression
    // template < typename T > struct IsExpressionHelper
    // {
    //     enum { value = boost::is_base_of <ArrayExpr ,T >::value && !boost::is_base_of<T, ArrayExpr >::value };
    //     typedef typename SelectType<value , TrueType , FalseType >::Type Type ;
    // };

    // template < typename T > struct IsExpression : public IsExpressionHelper <T >:: Type
    // {
    //     enum { value = IsExpressionHelper <T>:: value };
    //     typedef typename IsExpressionHelper <T>:: Type Type ;
    // };


} //namespace cuda_array

#endif //EXPR_H
