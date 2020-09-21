#include <iostream>
#include <cmath>


__global__ void add(int n, float **matr1, float **matr2, float **resMatr)
{
    
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

        
        if(column1 == row2 )atomicAdd(&resMatr[row1][column2], matr1[row1][column1] * matr2[row2][column2]);
        //realized too late that this hash function is not optimal. Now it works but needs a better hash function. I basically multiplied the complexity by a factor of n
    }
}

int main(void)
{
    
    int N = 2; // 1M elements
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
    add<<<1, 32>>>(N, matr1, matr2, resMatr);

    // Wait for the GPU to finish before accessing on host
    cudaDeviceSynchronize();

    
    for (int i = 0; i < N; i++)
    {
        cudaFree(matr1[i]);
        cudaFree(matr2[i]);
    }
    cudaFree(matr1);
    cudaFree(matr2);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << resMatr[i][j] << "\n";
        }
    }
    return 0;
}