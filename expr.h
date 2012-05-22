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

    template<typename T_type, int N_rank >
    class  ExprIdentity
    {
        cuArray<T_type,N_rank>& array;
    public:
        typedef T_type T;
        ExprIdentity(cuArray<T_type,N_rank> ar): array(ar) 
            { }
        __device__ T operator[] (size_t index) const
            {
                return array[index];
            }
    };

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
