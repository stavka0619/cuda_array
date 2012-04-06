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

    //define math operators
    template<typename Left_expr, typename Right_expr>
    inline ArrayExpr<L, Add, R> operator+ (L const& lhs, R const& rhs)
    {
        return ArrayExpr<Left_expr, Add, Right_expr> (lhs, rhs);
    }
    
    //define the assignment operators =, +=, -=, ....
    //note that ASSIGNMENT ALWAYS INVOVLES COPYING
    
    // very crude version of initilization
    template<typename T_numtype, int N_rank>
    void  Array<T_numtype,N_rank>::initialize(int x)
    {
        size_t count = sizeof(int)*numElements();
        cudaMemset (data_, x, count);
    }
    
    template<typename T_numtype, int N_rank> 
    inline Array<T_numtype, N_rank>&
    Array<P_numtype, N_rank>::operator=(const Array<P_numtype2,N_rank>& x)
    {
        (*this) = _bz_ArrayExpr<FastArrayIterator<P_numtype2, N_rank> >(x.beginFast());
        return *this;
    }

    template<typename T_numtype, int N_rank>
    inline Array<T_numtype, N_rank>&
    template<typename Left_expr, typename Op , typename Right_expr>
    Array<T_numtype, N_rank>::operator=(ArrayExpr<Left_expr, Op, Right_expr> expr)
    {
        assign<<<NBLOCKS, NTHREADS>>>(*this, expr);
        return *this;
    }

    template <typename T_numtype, N_rank>
    template<typename Left_expr<T_numtype>, typename Op<T_numtype>
             , typename Right_expr<T_numtype> >
    __global__ void assign( cuArray<T_numtype, N_rank> dest,
                 ArrayExpr<Left_expr, Op, Right_expr> expr)
    {
            const int tid = (blockIdx.y*NBLOCKX + blockIdx.x)*blockDim.x + threadIdx.x;
            if (tid < dest.numElements() )
            {
                dest[tid] = expr[tid];
            }
            
    }
    
} //namespace cuda_array
