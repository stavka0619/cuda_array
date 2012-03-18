/***************************************************************************
 *
 ***************************************************************************/

#ifndef DEVICE_MEMBLOCK_H
#define DEVICE_MEMBLOCK_H

#include <cutil_inline.h>
#include <cstddef>// for size_t type

namespace cuda_array
{
    
    enum preexistingMemoryPolicy { 
        duplicateData, 
        deleteDataWhenDone, 
        neverDeleteData 
    };

    template<typename T_type> class deviceMemoryBlockReference;

    template<typename T_type>
    class deviceMemoryBlock
    {
        friend class deviceMemoryBlockReference<T_type>;

    private:
        T_type*  data_;
        volatile int references_;
        size_t  length_;
    protected:
        deviceMemoryBlock()
            {
                length_ = 0;
                data_ = 0;
                references_ = 0;
            }

        explicit deviceMemoryBlock(size_t items)
            {
                length_ = items;
                allocate(length_);
                references_ = 0;
            }

        deviceMemoryBlock(size_t length, T_type* data)
            {
                length_ = length;
                data_ = data;
                references_ = 0;
            }

        virtual ~deviceMemoryBlock()
            {
                if (data_) 
                {
                    deallocate();
                }
            }

        void addReference()
            { 
                ++references_; 
            }

        T_type*  data() 
            { 
                return data_; 
            }

        const T_type* data()      const
            { 
                return data_; 
            }

        size_t length()    const
            { 
                return length_; 
            }

        int removeReference()
            {
                int refcount = --references_;
                return refcount;
            }

        int references() const
            {
                int refcount = references_;
                return refcount;
            }
        
        inline void allocate(size_t length)
            {
                size_t blocksize = length * sizeof(T_type);
                cutilSafeCall( cudaMalloc((void**) &data_, blocksize ));
            }
        
        void deallocate()
            {
                cutilSafeCall(cudaFree(data_));
            }

    private:
        deviceMemoryBlock(const deviceMemoryBlock<T_type>&)
            { }

        void operator=(const deviceMemoryBlock<T_type>&)
            { }

    };

    template<typename T_type>
    class UnowneddeviceMemoryBlock : public deviceMemoryBlock<T_type> {
    public:
        UnowneddeviceMemoryBlock(size_t length, T_type* data)
            : deviceMemoryBlock<T_type>(length,data)
            {
                deviceMemoryBlock<T_type>::data() = 0;
            }

        virtual ~UnowneddeviceMemoryBlock()
            {
            }
    };

    template<typename T_type>
    class NulldeviceMemoryBlock : public deviceMemoryBlock<T_type> {
    public:
        NulldeviceMemoryBlock()
            { 
                deviceMemoryBlock<T_type>::addReference();        
            }

        virtual ~NulldeviceMemoryBlock()  
            { }
    };

    template<typename T_type>
    class deviceMemoryBlockReference {

    protected:
        T_type * data_;

    private:
        deviceMemoryBlock<T_type>* block_;
        NulldeviceMemoryBlock<T_type> nullBlock_; // NEED to add static here
    public:

        deviceMemoryBlockReference()
            {
                block_ = &nullBlock_;
                block_->addReference();
                data_ = 0;
            }

        deviceMemoryBlockReference(deviceMemoryBlockReference<T_type>& ref, size_t offset=0)
            {
                block_ = ref.block_;
                block_->addReference();
                data_ = ref.data_ + offset;
            }

        deviceMemoryBlockReference(size_t length, T_type* data, 
                                   preexistingMemoryPolicy deletionPolicy)
            {
                // Create a memory block using already allocated memory. 
                if ((deletionPolicy == neverDeleteData) 
                    || (deletionPolicy == duplicateData))
                    block_ = new UnowneddeviceMemoryBlock<T_type>(length, data);
                else if (deletionPolicy == deleteDataWhenDone)
                    block_ = new deviceMemoryBlock<T_type>(length, data);
                block_->addReference();
                data_ = data;
            }

        explicit deviceMemoryBlockReference(size_t items)
            {
                block_ = new deviceMemoryBlock<T_type>(items);
                block_->addReference();
                data_ = block_->data();
            }

        void blockRemoveReference()
            {
                int refcount = block_->removeReference();
                if ((refcount == 0) && (block_ != &nullBlock_))
                {
                    delete block_;
                }
            }

        ~deviceMemoryBlockReference()
            {
                blockRemoveReference();
            }

        int numReferences() const
            {
                return block_->references();
            }

    protected:
        void changeToNullBlock()
            {
                blockRemoveReference();
                block_ = &nullBlock_;
                block_->addReference();
                data_ = 0;
            }

        void changeBlock(deviceMemoryBlockReference<T_type>& ref, size_t offset=0)
            {
                blockRemoveReference();
                block_ = ref.block_;
                block_->addReference();
                data_ = ref.data_ + offset;
            }

        void newBlock(size_t blocksize)
            {
                blockRemoveReference();
                block_ = new deviceMemoryBlock<T_type>(blocksize);
                block_->addReference();
                data_ = block_->data();
            }

    private:
        void operator=(const deviceMemoryBlockReference<T_type>&)
            { }
    };
}

#endif // DEVICE_MEMBLOCK_H
