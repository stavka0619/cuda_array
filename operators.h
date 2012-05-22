#ifndef OPERATORS_H
#define OPERATORS_H

#ifndef  CUDA_ARRAY_H
 #error <operators.h> must be included via <cudaArray.h>
#endif

namespace cuda_array
{
#define DEFINE_UNARY_OP(name,op)                \
    template<typename T_numtype>                \
    struct name {                               \
        static inline T_numtype                 \
        apply(T_numtype a)                      \
            { return op a; }                    \
    };

    DEFINE_UNARY_OP(UnaryPlus,+)
    DEFINE_UNARY_OP(UnaryMinus,-)

#define DEFINE_BINARY_OP(name,op)               \
    template<typename T_numtype>                \
    struct name {                               \
        typedef  T_numtype T ;                  \
        __device__ static inline T_numtype  \
        apply(T_numtype a, T_numtype b)         \
            { return a op b; }                  \
    };

    DEFINE_BINARY_OP(Add,+)
    DEFINE_BINARY_OP(Subtract,-)
    DEFINE_BINARY_OP(Multiply,*)
    DEFINE_BINARY_OP(Divide,/)

    /* Binary operators that return a specified type */
    
#define DEFINE_BINARY_OP_RET(name,op,ret)       \
    template<typename T_numtype1, typename T_numtype2>   \
    struct name {                               \
        typedef  ret R_numtype ;                  \
        __device__ static R_numtype apply(T_numtype1 a, T_numtype2 b)       \
            { return a op b; }                  \
    };                                          \

    DEFINE_BINARY_OP_RET(Greater,>,bool)
    DEFINE_BINARY_OP_RET(Less,<,bool)
    DEFINE_BINARY_OP_RET(GreaterOrEqual,>=,bool)
    DEFINE_BINARY_OP_RET(LessOrEqual,<=,bool)
    DEFINE_BINARY_OP_RET(Equal,==,bool)
    DEFINE_BINARY_OP_RET(NotEqual,!=,bool)
    DEFINE_BINARY_OP_RET(LogicalAnd,&&,bool)
    DEFINE_BINARY_OP_RET(LogicalOr,||,bool)


    template<class Expr1, class Expr2>
    inline cuArrayBinExpr<Expr1, Add<typename Expr1::T>, Expr2 > 
    operator+ (Expr1 lhs, Expr2 rhs)
    {
        typedef typename Expr1::T T;
        typedef cuArrayBinExpr< Expr1, Add<typename Expr1::T>, Expr2 > ExprT;
        return ExprT (lhs, rhs);
    }
    
    //define the assignment operators =, +=, -=, ....
    //note that ASSIGNMENT ALWAYS INVOVLES COPYING
} //namespace cuda_array

#endif //OPERATORS_H
