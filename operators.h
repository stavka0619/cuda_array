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
    template<typename T_numtype>              \
    struct name {                               \
        typedef T_numtype T ;                   \
        __device__ static inline T_numtype      \
        apply(T_numtype a, T_numtype b)         \
            { return a op b; }                  \
    };

    DEFINE_BINARY_OP(Add,+)
    DEFINE_BINARY_OP(Substract,-)
    DEFINE_BINARY_OP(Multiply,*)
    DEFINE_BINARY_OP(Divide,/)

    /* Binary operators that return a specified type */
    
#define DEFINE_BINARY_OP_RET(name,op,ret)                               \
    template<typename T_numtype>                  \
    struct name {                                                       \
        typedef  ret R_numtype ;                                        \
        __device__ static R_numtype apply(T_numtype a, T_numtype b)   \
            { return a op b; }                                          \
    };                                                                  \

    DEFINE_BINARY_OP_RET(Greater,>,bool)
    DEFINE_BINARY_OP_RET(Less,<,bool)
    DEFINE_BINARY_OP_RET(GreaterEqual,>=,bool)
    DEFINE_BINARY_OP_RET(LessEqual,<=,bool)
    DEFINE_BINARY_OP_RET(Equal,==,bool)
    DEFINE_BINARY_OP_RET(NotEqual,!=,bool)
    DEFINE_BINARY_OP_RET(LogicalAnd,&&,bool)
    DEFINE_BINARY_OP_RET(LogicalOr,||,bool)

#define DEFINE_BINARY_EXPR_OP(name,op)                                  \
    template<class Expr1, class Expr2>                                  \
    inline cuArrayBinExpr<Expr1, op<typename Expr1::T>, Expr2 >        \
    name (Expr1 lhs, Expr2 rhs)                                         \
    {                                                                   \
        typedef typename Expr1::T T;                                    \
        typedef cuArrayBinExpr< Expr1, op<typename Expr1::T>, Expr2 > ExprT; \
        return ExprT(lhs, rhs);                                        \
    }                                                                   \
    
    DEFINE_BINARY_EXPR_OP(operator+, Add)    
    DEFINE_BINARY_EXPR_OP(operator-, Substract)    
    DEFINE_BINARY_EXPR_OP(operator*, Multiply)    
    DEFINE_BINARY_EXPR_OP(operator/, Divide)    
    DEFINE_BINARY_EXPR_OP(operator==, Equal)    
    DEFINE_BINARY_EXPR_OP(operator!=, NotEqual)    
    DEFINE_BINARY_EXPR_OP(operator>, Greater)    
    DEFINE_BINARY_EXPR_OP(operator<, Less)    
    DEFINE_BINARY_EXPR_OP(operator>=, GreaterEqual)    
    DEFINE_BINARY_EXPR_OP(operator<=, LessEqual)    

#define DEFINE_BINARY_EXPR_CONSTANT(name, op, type)                    \
    template<typename T, class Expr>                                               \
    inline cuArrayBinExpr<ExprLiteral<type>, Add<type>, cuArrayExpr<Expr> > \
    name (const type lhs, cuArrayExpr<Expr>& rhs)                       \
    {                                                                   \
    typedef cuArrayBinExpr<ExprLiteral<type>, op<type>, cuArrayExpr<Expr> > ExprT; \
        return ExprT (ExprLiteral<type>(lhs), rhs);                  \
    }                                                                  \
                                                                       \
    template<class Expr>                                               \
    inline cuArrayBinExpr<cuArrayExpr<Expr>, ExprLiteral<type>, Add<type> > \
    name (const  cuArrayExpr<Expr>& lhs, type rhs)                       \
    {                                                                   \
        typedef cuArrayBinExpr<cuArrayExpr<Expr>, op<type>, ExprLiteral<type> > ExprT; \
        return ExprT (lhs, ExprLiteral<type>(rhs));              \
    }                                                            \
    DEFINE_BINARY_EXPR_CONSTANT(operator+, Add, type)                \
    DEFINE_BINARY_EXPR_CONSTANT(operator-, Substract, type)          \
    DEFINE_BINARY_EXPR_CONSTANT(operator*, Multiply, type)           \
    DEFINE_BINARY_EXPR_CONSTANT(operator/, Divide, type)             \
    DEFINE_BINARY_EXPR_CONSTANT(operator==, Equal, type)             \
    DEFINE_BINARY_EXPR_CONSTANT(operator!=, NotEqual, type)          \
    DEFINE_BINARY_EXPR_CONSTANT(operator>, Greater, type)            \
    DEFINE_BINARY_EXPR_CONSTANT(operator<, Less, type)               \
    DEFINE_BINARY_EXPR_CONSTANT(operator>=, GreaterEqual, type)      \
    DEFINE_BINARY_EXPR_CONSTANT(operator<=, LessEqual, type)         \
    
} //namespace cuda_array

#endif //OPERATORS_H
