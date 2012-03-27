// Helper class Range

#ifndef RANGE
#define RANGE

namespace cuda_array
{
    
    class Range {

    public:
        Range()
            {
                first_ = 0;
                last_ = 0;
            }

        Range(const Range& r)
            {
                first_ = r.first_;
                last_ = r.last_;
            }

        explicit Range(int slicePosition)
            {
                first_ = slicePosition;
                last_ = slicePosition;
            }

        Range(int first, int last)
            : first_(first), last_(last)
            { 
            }

        int first() const
            { 
                return first_; 
            }

        int last() const
            {
                return last_;
            }

        unsigned length() const
            {
                return (last_ - first_)  + 1;
            }

        void setRange(int first, int last)
            {
                first_ = first;
                last_ = last;
            }

        static Range all() 
            { return Range(fromStart,toEnd); }

        // Operators
        Range operator-(int shift) const
            { 
                return Range(first_ - shift, last_ - shift); 
            }

        Range operator+(int shift) const
            { 
                return Range(first_ + shift, last_ + shift); 
            }

        int operator[](unsigned i) const
            {
                return first_ + i;
            }

        int operator()(unsigned i) const
            {
                return first_ + i;
            }

    private:
        int first_, last_;
    };

}

#endif // BZ_RANGE_H
