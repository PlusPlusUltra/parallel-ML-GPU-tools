#include <iostream>
#include <cmath>

int pos = 0;
long* tests;

// CUDA Kernel function (specified by __global__) to add the elements of two arrays
__global__ void add(int n, float **matr1, float **matr2, float **resMatr)
{
    /*
    // index of thread in the thread block
    int index = threadIdx.x;
    // number of threads in the block
    int stride = blockDim.x;
    long total = n*n*n*n;

    for (long i = index; i < total; i += stride)
    {
        long number = i;
        long column2 = number % n;
        number = number / n;
        long row2 = number % n;
        number = number / n;
        long column1 = number % n;
        number = number / n;
        long row1 = number;

        //resMatr[row1][column2] = (matr1[row1][column1] * matr2[row2][column2]) + resMatr[row1][column2];
        resMatr[row1][column2] = 1 + resMatr[row1][column2];
    }
    */
    long i = blockIdx.x*blockDim.x + threadIdx.x;
    long total = n*n*n*n;
    if (i < total)
    {
        long number = i;
        long column2 = number % n;
        number = number / n;
        long row2 = number % n;
        number = number / n;
        long column1 = number % n;
        number = number / n;
        long row1 = number;

        //resMatr[row1][column2] = (matr1[row1][column1] * matr2[row2][column2]) + resMatr[row1][column2];
        resMatr[row1][column2] = 1 + resMatr[row1][column2];
    }
}

int main(void)
{
    
    int N = 3; // 1M elements
    int total = N*N*N*N;

    // to allocate the space in Unified memory we use
    float **matr1, **matr2, **resMatr;
    cudaMallocManaged(&matr1, N * sizeof(float*));
    cudaMallocManaged(&matr2, N * sizeof(float*));
    cudaMallocManaged(&resMatr, N * sizeof(float*));
    cudaMallocManaged(&tests, total);

    for (int i = 0; i < N; i++)
    {
        cudaMallocManaged(&(matr1[i]), N * sizeof(float));
        cudaMallocManaged(&(matr2[i]), N * sizeof(float));
        cudaMallocManaged(&(resMatr[i]), N * sizeof(float));
    }

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
        matr1[i][j] = 1.0f;
        matr2[i][j] = 2.0f;
        resMatr[i][j] = 0.0f;
        }
    }

    // run kernel on 1M elements on the GPU
    // 2nd parameter in triple angle brackets is number of threads in thread block.
    // it has to be a multiple of 32
    add<<<1, 256>>>(N, matr1, matr2, resMatr, &pos, tests);

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    
    for (int i = 0; i < N; i++)
    {
        cudaFree(matr1[i]);
        cudaFree(matr2[i]);
    }
    // Free memory
    cudaFree(matr1);
    cudaFree(matr2);
    int sum = 0;

    

    return 0;
}