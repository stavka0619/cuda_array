#include "cudaArray.h"
#include "cuda.h"
#include <iostream>
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

__global__ void setValue(cuda_array::cuArray<float,2> a)
{
	const int tid = (blockIdx.y*1 + blockIdx.x)*blockDim.y*threadIdx.x + threadIdx.y;
    if (tid < a.rows()*a.cols())
        {
            float value = tid+0.5;
            a(threadIdx.x, threadIdx.y) = value;
        }
}

int main()
{
    dim3 grid(1,1);
    dim3 threads(10,10,1);
    cuda_array::cuArray<float,2> a(10,10);
    float aa[100];
    for (int i=0;i<100;i++)
        aa[i]=i;
    a.copyfromHost(aa);
    cuda_array::cuArray<float,2> b(a,Range::all(),Range(3,7));
    setValue<<<grid,threads>>>(b);
    
    cudaThreadSynchronize();
    float bb[100];
    
      
    a.copytoHost(bb);
    
    for (int i=0;i<10;i++)
    {
        for (int j=0;j<10;j++)
            cout<<bb[i*10+j]<<' ';
        cout<<endl;
    }
    return 0;
}
