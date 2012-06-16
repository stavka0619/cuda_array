#include "cudaArray.h"
#include "operators.h"
#include "cuda.h"
#include <iostream>
#include <boost/progress.hpp>
#include <cublas_v2.h>

using namespace std;
using namespace cuda_array;


// __global__ void setValue(float * dest, int nx, int ny)
// {
// 	const int tid = (blockIdx.y*1 + blockIdx.x)*blockDim.x + threadIdx.x;
//         if (tid < nx*ny)
//         {
//             float value = dest[tid]*2+0.5;
//             dest[tid] = value;
//         }
// }

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
    cuArray<float,2> c(10,10);
    cuArray<float,2> cc(10,10);
    
    const int sz = 100;//nx*nx*nz;
    
    const int N = 1000;
    
    
    float* aa = new float[sz];
    float* bb = new float[sz];
    for (int i=0;i<sz;i++)
    {
        aa[i]=i;
        bb[i]=i;
    }
    // a.copyfromHost(aa);
    // b.copyfromHost(aa);
    c.copyfromHost(aa);
    //cc = where(c>50.0f,c,c-50.0f);    
    cc = shift(c, Offset<1,0>() );
     cudaDeviceSynchronize();
    // // setValue<<<grid,threads>>>(a);
    // // setValue<<<grid,threads>>>(b);
    // {
    //     cout<<"on CPU"<<endl;
    //     boost::progress_timer timer;
    //     for (int ii=0;ii<N;ii++)
    //         for (int jj=0;jj<sz;jj++)
    //             bb[jj] += aa[jj];
    // }
    // cout<<"computational time of the increment of an 512*512*120 array "<<endl;
    
    // {
    //     cout<<"template library: a += b"<<endl;
    //     boost::progress_timer timer;
    //     for (int ii=0;ii<N;ii++)
    //         c += a+b;
    //     cudaDeviceSynchronize();
    // }

    // cublasHandle_t blas_handle;
    // cublasCreate(&blas_handle);
    // float one = 1;
    // {
    //     cout<<"on GPU with cuBLAS"<<endl;
    //     boost::progress_timer timer;
    //     for (int ii=0;ii<N;ii++)
    //     {
    //         cublasSaxpy(blas_handle, b.size(), &one, a.data(), 1, c.data(), 1); 
    //         cublasSaxpy(blas_handle, b.size(), &one, b.data(), 1, c.data(), 1); 
    //     }
    //         cudaDeviceSynchronize();
    // }
    cc.copytoHost(bb);
    for (int i=0;i<10;i++)
    {
        for (int j=0;j<10;j++)
            cout<<bb[i*10+j]<<' ';
        cout<<endl;
    }


    delete [] aa;
    delete [] bb;
    
    return 0;
}
