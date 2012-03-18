#include "cudaArray.h"
#include "cuda.h"
#include <iostream>
using namespace std;


__global__ void setValue(float * dest, int nx, int ny)
{
	const int tid = (blockIdx.y*1 + blockIdx.x)*blockDim.x + threadIdx.x;
        if (tid < nx*ny)
        {
            float value = tid+0.5;
            dest[tid] = value;
        }
}

int main()
{
    dim3 grid(1,1);
    dim3 threads(100,1);
    cuda_array::cuArray<float,2> a(10,10);
    setValue<<<grid,threads>>>(a.data(),a.rows(),a.cols());
    cudaThreadSynchronize();
    float bb[100];
    a.copytoHost(bb);
    int b=0;
    for (int i=0;i<10;i++)
    {
        for (int j=0;j<10;j++)
            cout<<bb[i*10+j]<<' ';
        cout<<endl;
    }
    
}
