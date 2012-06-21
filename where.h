#ifndef WHERE_H
#define WHERE_H

#ifndef  CUDA_ARRAY_H
 #error <where.h> must be included via <cudaArray.h>
#endif

namespace cuda_array
{
    template<typename Condition, typename ifTrue, typename ifFalse>
    class cuArrayWhere
    {
    public:
        typedef Condition Expr1;
        typedef ifTrue Expr2;
        typedef ifFalse Expr3;
        typedef typename Expr2::T T;
        typedef typename Expr2::T T2;
        typedef typename Expr3::T T3;

        cuArrayWhere(const cuArrayWhere<Expr1,Expr2,Expr3>& a)
            : iter1_(a.iter1_), iter2_(a.iter2_), iter3_(a.iter3_)
            { }
        
        cuArrayWhere(Expr1 iter1, Expr2 iter2, Expr3 iter3)
            : iter1_(iter1), iter2_(iter2), iter3_(iter3)
            { }

        __device__ T2 operator[] (size_t index) const
            {
                return iter1_[index] ? iter2_[index] : iter3_[index];
            }
    private:
        Expr1 iter1_;
        Expr2 iter2_;
        Expr3 iter3_;
    };
        
    template<typename Derived, typename Base>
    struct IsBase
    {
        typedef Base T;
    };
    template<typename Derived>
    struct IsBase<Derived, typename Derived::T>
    {
        typedef ExprLiteral<typename Derived::T> T;
    };
    
    template<typename Condition, typename ifTrue, typename ifFalse>
    inline cuArrayWhere<Condition,typename IsBase<Condition,ifTrue>::T, typename IsBase<Condition,ifFalse>::T>
    where(Condition a, ifTrue b, ifFalse c)
    {
        typedef typename IsBase<Condition,ifTrue>::T NewIfTrue;
        typedef typename IsBase<Condition,ifFalse>::T NewIfFalse;
        return cuArrayWhere<Condition, NewIfTrue, NewIfFalse>(a, NewIfTrue(b), NewIfFalse(c) );
    }
    
} //namespace ends

#endif
