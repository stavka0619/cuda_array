#ifndef BLAS_EXPR
#define BLAS_EXPR

#ifndef  CUDA_ARRAY_H
 #error <expr.h> must be included via <cudaArray.h>
#endif

#include <boost/type_traits/is_base_of.hpp>

namespace cuda_array
{
    // whether T is expression
    template < typename T > struct IsExpressionHelper
    {
        enum { value = boost::is_base_of <ArrayExpr ,T >::value && !boost::is_base_of<T, ArrayExpr >::value };
        typedef typename SelectType<value , TrueType , FalseType >::Type Type ;
    };

    template < typename T > struct IsExpression : public IsExpressionHelper <T >:: Type
    {
        enum { value = IsExpressionHelper <T>:: value };
        typedef typename IsExpressionHelper <T>:: Type Type ;
    };
    // Select type based on select
    template < bool Select, typename T1, typename T2>
    struct SelectType
    {
        typedef T1 Type ; // if Select==true, selected T1 .
    };
    template < typename T1, typename T2 >
    struct SelectType <false ,T1 ,T2 >
    {
        typedef T2 Type ; // if Select==false, selected T2 .
    };
    // math trait: to prevent invalid expression
    template < typename T1 , typename T2 >
    struct math_trait;  //default is invalid, compilation should stop here

    template < typename T1 , T1 > //only matrix*vector is valid operation
    struct math_trait<cuarray<T1,2>,cuarray<T1,1> >
    {
        typedef cuarray<T1,1> MultType ;
    }
        
    // main definition
        template<typename T_numtype>
        template<typename Left_expr<T_numtype>, typename Op,
                 typename Right_expr<T_numtype> >
        struct  ArrayExpr <T_numtype>
        {
            ArrayExpr(Left_expr const& l, Right_expr const& r)
                : lhs(l), rhs(r) { }
            __device__ T_numtype operator[] (size_t index) const
                {
                    return Op::apply(lhs[index], rhs[index]);
                }

            Left_expr lhs;
            Right_expr rhs;
        };
    

} //namespace cuda_array
