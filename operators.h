#ifndef OPERATORS
#define OPERATORS

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
        static inline T_numtype                 \
        apply(T_numtype a, T_numtype b)         \
            { return a op b; }                  \
    };

    DEFINE_BINARY_OP(Add,+)
    DEFINE_BINARY_OP(Subtract,-)
    DEFINE_BINARY_OP(Multiply,*)
    DEFINE_BINARY_OP(Divide,/)

    /* Binary operators that return a specified type */
    
#define DEFINE_BINARY_OP_RET(name,op,ret)       \
    template<typename T_numtype>                \
    struct name {                               \
        typedef ret R_numtype;                  \
        static inline R_numtype                 \
        apply(T_numtype1 a, T_numtype2 b)       \
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

} //namespace cuda_array
