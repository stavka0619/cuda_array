#include "cudaArray.h"
#include "operators.h"
#include "cuda.h"
#include <iostream>
#include <boost/progress.hpp>
#include <cublas_v2.h>

using namespace std;
using namespace cuda_array;


__global__ void setValue(float * dest, int nx, int ny)
{
	const int tid = (blockIdx.y*1 + blockIdx.x)*blockDim.x + threadIdx.x;
        if (tid < nx*nx*ny)
        {
            float value = dest[tid]*2+0.5;
            dest[tid] = value;
        }
}

// __global__ void setValue(cuda_array::cuArray<float,2> a)
// {
// 	const int tid = (blockIdx.y*1 + blockIdx.x)*blockDim.y*threadIdx.x + threadIdx.y;
//     if (tid < a.numElements())
//         {
//             float value = tid;
//             a(threadIdx.x, threadIdx.y) = value;
//         }
// }

int main()
{
    int nx=512;
    int nz=120;
    cuArray<float,3> a(nx,nx,nz);
    cuArray<float,3> b(nx,nx,nz);
    cuArray<float,3> c(nx,nx,nz);
    cuArray<float,3> cc(10,10,10);
    int N=400;
    
    const int sz = 1000;//nx*nx*nz;
    
    float* aa = new float[sz];
    float* bb = new float[sz];
    for (int i=0;i<sz;i++)
    {
        aa[i]=i;
        bb[i]=i;
    }
    // a.copyfromHost(aa);
    // b.copyfromHost(aa);
    // c.copyfromHost(aa);
    // cc = 0.5f+1.0f*c;
    // cc *=2.0f;
    
    // cc = where(cc<=50.0f , 50.0f , -23.4f );    
    // cc = shift<Offset<-1,1,1> >(c );
    //  cudaDeviceSynchronize();
    // setValue<<<grid,threads>>>(a);
    // setValue<<<grid,threads>>>(b);
    // {
    //     cout<<"on CPU"<<endl;
    //     boost::progress_timer timer;
    //     for (int ii=0;ii<N;ii++)
    //         for (int jj=0;jj<sz;jj++)
    //             bb[jj] += aa[jj];
    // }
    // cout<<"computational time of the increment of an 512*512*120 array "<<endl;
    
    {
        c=0.0f;
        cout<<"template library: a += b"<<endl;
        boost::progress_timer timer;
        for (int ii=0;ii<N;ii++)
            c += a;
        cudaDeviceSynchronize();
    }

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    float one = 1;
    {
        c=0.0f;
        cout<<"on GPU with cuBLAS"<<endl;
        boost::progress_timer timer;
        for (int ii=0;ii<N;ii++)
        {
            cublasSaxpy(blas_handle, b.size(), &one, a.data(), 1, c.data(), 1); 
            //cublasSaxpy(blas_handle, b.size(), &one, b.data(), 1, c.data(), 1); 
        }
            cudaDeviceSynchronize();
    } 
    //  cc.copytoHost(bb);
    // for (int i=0;i<10;i++)
    // {
    //     for (int j=0;j<10;j++)
    //         cout<<bb[i*10+j]<<' ';
    //     cout<<endl;
    // }


    // delete [] aa;
    // delete [] bb;
    
    return 0;
}
