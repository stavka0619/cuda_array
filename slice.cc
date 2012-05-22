#ifndef CUARRAYSLICE
#define CUARRAYSLICE

#ifndef  CUDA_ARRAY_H
 #error <slice.cc> must be included via <blitz/array.h>
#endif

template<typename P_numtype, int N_rank>
void Array<P_numtype, N_rank>::slice(int rank, Range r)
{
    BZPRECONDITION((rank >= 0) && (rank < N_rank));

    int first = r.first(lbound(rank));
    int last  = r.last(ubound(rank));
    int stride = r.stride();

#ifdef BZ_DEBUG_SLICE
cout << "slice(" << rank << ", Range):" << endl
     << "first = " << first << " last = " << last << "stride = " << stride
     << endl << "length_[rank] = " << length_[rank] << endl;
#endif

    BZPRECHECK(
        ((first <= last) && (stride > 0)
         || (first >= last) && (stride < 0))
        && (first >= base(rank) && (first - base(rank)) < length_[rank])
        && (last >= base(rank) && (last - base(rank)) < length_[rank]),
        "Bad array slice: Range(" << first << ", " << last << ", "
        << stride << ").  Array is Range(" << lbound(rank) << ", "
        << ubound(rank) << ")");

    // Will the storage be non-contiguous?
    // (1) Slice in the minor dimension and the range does not span
    //     the entire index interval (NB: non-unit strides are possible)
    // (2) Slice in a middle dimension and the range is not Range::all()

    length_[rank] = (last - first) / stride + 1;

    // TV 20000312: added second term here, for testsuite/Josef-Wagenhuber
    int offset = (first - base(rank) * stride) * stride_[rank];

    data_ += offset;
    zeroOffset_ += offset;

    stride_[rank] *= stride;
    // JCC: adjust ascending flag if slicing with backwards Range
    if (stride<0)
        storage_.setAscendingFlag(rank, !isRankStoredAscending(rank));
}

}
