#ifndef BLAS_EXPR
#define BLAS_EXPR

#ifndef  CUDA_ARRAY_H
 #error <blasexpr.h> must be included via <cudaArray.h>
#endif

namespace cuda_array
{
    template<typename Matrix_type, typename Vector_type>
    class MatVecMultExpr: public Vector< MatVecMultExpr<Matrix_type, Vector_type> >, private ArrayExpr
    {
    private: //shortcut 
        typedef typename Matrix_type::ResultType T_MatRes;
        typedef typename Vector_type::ResultType T_VecRes;
        typedef typename T_MatRes::T_Element T_MatEle;
        typedef typename T_VecRes::T_Element T_VecEle;

    public:
        typedef typename math_trait<T_MatRes, T_VecRes>::T_Mult ResultType;
        typedef const MatVecMultExpr& Composite_Type;
        typedef typename math_trait<T_MatRes, T_VecRes>::T_Mult::T_Element T_Element;
        typedef typename Matrix_type::Composite_Type Left;
        typedef typename select_type<is_expression<Vector_type>::value, const T_VecRes, const Vector_type>::Type Right;
        inline const T_Element operator [](size_t index) const;
        inline size_t size() const;
        template <typename T>
        inline bool is_aliased(const T* alias) const
            {
                return vector_.is_aliased(alias);
            }
    private:
        Left matrix_;
        Right vector_;
    };

    template <typename Vector_type1, typename Vector_type2>
    class VecVecAddExpr: public Vector<VecVecAddExpr<Vector_type1, Vector_type2> >, private ArrayExpr
    {
    private: //shortcut 
        typedef typename Vector_type1::ResultType T_VecRes1;
        typedef typename Vector_type2::ResultType T_VecRes2;
        typedef typename T_VecRes1::T_Element T_VecEle1;
        typedef typename T_VecRes2::T_Element T_VecEle2;

    public:
       typedef typename math_trait<T_VecRes1, T_VecRes2>::T_Add ResultType;
        typedef const ResultType Composite_Type;
        typedef typename math_trait<T_VecRes1, T_VecRes2>::T_Add::T_Element T_Element;
        typedef typename select_type<is_expression<Vector_type1>::value, const T_VecRes1, const Vector_type1>::Type Left;
        typedef typename select_type<is_expression<Vector_type2>::value, const T_VecRes2, const Vector_type2>::Type Right;
        inline const T_Element operator [](size_t index) const;
        inline size_t size() const;

    private:
        Left vector1_;
        Right vector2_;
    };


          
    
} //namespace cuda_array

#endif
