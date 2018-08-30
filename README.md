#Introduction

It is a template library for CUDA runtime API. The purpose to develop this library is to free the users from doing routine job like memory management, array size verification and writing kernel function to do simple arithmetic operation, and therefore focus on the non-trivial kernel function that implements the core algorithms. Performance is the first priority of design of this library so that it can be used without any concern about the performance loss. Therefore, this page mentions the implementation mechanism in addition to the usage to help the user have some idea about what's going on under the cover.

#Get started

As the template techniques are heavily used in the implementation of this library, CUDA 4.0 is a must in order the code using it to be compiled. However, device with compute capability lower than 2.0 is supported, although those lower than 1.3 haven't been tested. The same as all the template libraries, just copy all the files to a directory where the compiler can find (with compile option `-I/path/to/library`) is enough. Add the following code to current source code enable all the features of the library:

#Include
```using namespace cuda_array; ```

#Constructors

The first step for almost all the CUDA programs is to allocate device memory. This is encapsulated in the `cuArray<T,N>` class, which is the centre of the library. Here the template parameters T indicates the numeric type to be stored, and although it can be a general type, only built-in types like `int`, `float` and `double` are supported, and support for `complex<T>` have not been added. The following examples use float as it is most widely used in CUDA programming. Template parameters N indicates the dimension of the array. As the data are stored in device memory continuously, the dimension does not really matter, but it can help when dealing with array subscripting. Currently the maximum dimension supported is 4, which is believe to be enough to most application. The default constructor create a cuArray without actually allocating any device memory, but it can be used as a handle for further operation
```
cuArray<float,3> d_array();
```
If the size is already known, it can be provided initially for each dimension, and `cudaMalloc()` is called to allocate the exact size of memory
```
cuArray<float,3> d_array(512,512,70);
```
The size can also be represented with internal type IdxVector, which can be provided to create a device array with the same as the other array:
```
IdxVector siz(512,512,70); 
cuArray<float,3> d_array(siz); 
cuArray<float,3> new_array(d_array.length());
```
The cuArray can be a reference to an existing cuArray, and at that time both of them are reference to the same memory chunk. It can also be the reference to a slice (or several slices) of the existing array
```
cuArray<float,3> new_array(d_array);
cuArray<float,3> new_array(d_array(Range::all(),Range::all(),10));
cuArray<float,3> new_array1(d_array(Range::all(),Range::all(),Range(5,10)));
```
However, if the slice are not stored continuously, the following code can compile but the following operation on it can lead to disastrous result. 
```
cuArray<float,3> new_array(d_array(10,Range::all(),Range::all()));
```
  
##release the resource
The cuArray class is also a reference counting smart pointer, therefore the device memory chunk can be released automatically when there is no handle connected to it. Never try to deallocate the device memory manually

#Member functions
The member function of `cuArray<T,N>` can be called from both host code or device code (kernel function). If some functions can only be called only from host side it will be noted in the description. 
*`void copyfromHost(T*)` get the current array from host memory. User must make sure the array at host side have the same size as device array. 
*`void copytoHost(T*)` copy the current array to host memory. User must make sure the array at host side have the same size as device array. 
`T* data()` return the raw pointer to the device memory represented 
`IdxVector<int,N> length(int dim)` return the vector containing the length of each dimension 
`int length(int dim)` return the length of dimension 
`dim IdxVector<int,N> position(int idx)` return the N-D coordinator of the idxth element in the current array. Equivalent to the `idx2sub()` in MATLAB. Result undefined if idx is out of boundary. 
`cuArray<T,N> reference()` return a cuArray object pointing to the same device memory chunk as the current. Reference count is also incremented. 
`void resize(int s1) void resize(int s1, int s2) void resize(int s1, int s3, int s3) void resize(int s1, int s3, int s3, int s4)` resize the memory chunk the current cuArray refer to. It is equivalent to free the current memory and allocate a new chunk of memory, thus what is in the new memory is undefined. 
`int size()` return the total number of elements in the array

#Indexing
In the kernel function, the device array can be indexed in two ways. `array(i)` or `array[i]` return the element at linear position i, and `array(i,j,k)` return the element at 3D position `(i,j,k)` based on the size of the current array. Note that boundary check is currently not included for the sake of performance, the user must be responsible for the validation of index to avoid segmentation fault.

#Expression
Previously, in order to do element-wise operation, the programmer need to write a separate kernel, no matter how trivial the operation is. In this library, with the C++ template support from CUDA 4.0, it is possible the implement the expression template technique so that the compiler can interpreter the native expression of cuArray to the corresponding kernel function without generating any temporary object, which is the main problem of naive operator overloading. Therefore, the performance of cuArray expression is the same as the hand-written code, but much easier to write and less error-prone The following sample code shows how the expression is used 
```
cuArray<float,3> a(10,10,10);
cuArray<float,3> b(10,10,10); 
cuArray<float,3> c(10,10,10); //...generate data for a and b 
c = a + b; 
c += 1.0f/a + 2.0f*a*b;
```
The compiler will generate a kernel function for the second expression equivalent to 
```
__global__ void expr(cuArray<float,3> c,cuArray<float,3> a,cuArray<float,3> b) { 
  const int tid=threadIdx.x+blockIdx.x*blockDim.x; 
  if (tid<c.size()) { 
    c[tid] = 1.0f/a[tid] + 2.0f*a[tid]*b[tid]; 
  } 
}
```
Note that the postfix f in the constant number, if it is not added the code won't compile as it is regarded as double type and doesn't match `cuArray<float>`. Even if it is possible to implement a type promote template to help it compile, it is still not recommended as the binary operation between the float and double types are very expensive on GPU.
  
##logistical operation

With expression template, it is possible to write: 
```
a=where(a>0.0f,a,0.0f); 
```
to set all the negative elements in a to zero. The compiler generate the following kernel for the code above: 
```
__global__ void expr_where(cuArray<float,3> a) { 
  const int tid=threadIdx.x+blockIdx.x*blockDim.x; 
  if (tid<a.size()) { 
    a[tid] = a[tid]>0.0f ? a[tid] : 0.0f 
  } 
}
```
In the where function, the parameters can be more complex expression `a=where(a>=b,a-b,b-a);`

##finite difference

Another common element-wise operation is finite difference, where we need to shift the current array the certain direction before subtract or be subtracted by itself, and which direction to shift depends whether we wanna do forward or backward difference. This library provides a tool to do that in the similar manner as above with template metaprogramming:
```
fdiff=shift<Offset<1,0,0> >(a)-a;//forward difference along x-direction 
bdiff=a-shift<Offset<0,0,-1> >(a);//back difference along z-direction 
laplacian=6*a-shift<Offset<1,0,0> >(a)-shift<Offset<-1,0,0> >(a) -shift<Offset<0,1,0> >(a)-shift<Offset<0,-1,0> >(a) -shift<Offset<0,0,1> >(a)-shift<Offset<0,0,-1> >(a);//3D Laplacian
```
Note that the shift has to be constant or can be determined at compile time. Also the shifted array is boundary checked, thus for index out of boundary, the value of unshifted array is return.
```
shifta=shift<Offset<-1,0,0> >(a); shifta(1,0,0); //return a(0,0,0)
shifta(0,0,0); //a(-1,0,0) is out of boundary, return a(0,0,0)
```
Finally, the shift function does not generate a copy of shifted array, instead it only provides a new method to index it. Therefore, to do finite difference in place will lead to undefined result 
```
a=shift<Offset<1,0,0> >(a)-a; //undefined 
```
During the test, it is found that such long expression above is not stable. Runtime error unspecified launch failure can occur occasionally. The reason is not very clear and is possibly due to too many recursions when the expression template is expanded. It is recommended to use the following short cut function to do finite difference and keep such expression short.
```
cuArray<float,3> float(array.length()); 
fdiff = forward_diff<0>(array); // forward difference along x-direction 
bdiff = back_diff<1>(array); // back difference along y-direction 
L = laplacian(array); // 3-D Laplacian
```
#Performance

As mentioned above, template techniques can help generate hand-written ocde without the loss of runtime performance. And the implementation can be further optimized. In the current release, the operation `a += b` can achieve comparable performance as `cublasSaxpy(blas_handle, a.size(), &one, a.data(), 1, b.data(), 1);`
