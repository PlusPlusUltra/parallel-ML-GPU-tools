#include <iostream>
#include <cmath>


__global__ void add(int n, float **matr1, float **matr2, float **resMatr)
{
    
    long i = blockIdx.x*blockDim.x + threadIdx.x;
    long total = n*n*n;
    if (i < total)
    {
        long number = i;
        long common = number % n;
        number = number / n;
        long column2 = number % n;
        number = number / n;
        long row1 = number;
        atomicAdd(&resMatr[row1][column2], matr1[row1][common] * matr2[common][column2]);
    }
}


int main(void)
{
    
    int N = 2;
    int total = N*N*N;

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
    //the number of threads as of now should be equal to total, or N^3
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
/*
//alternative versions
//realized too late that this hash function is not optimal. Now it works but needs a better hash function. I basically multiplied the complexity by a factor of n
//I wanted each thread to manage a single scalar multiplication, but at this point it would be easier to just let each thread manage one row and one column
// the idea remains the same, but the hash function n^2 and not n^4
//actually it would be interesting to see which one is faster, I will develop them both and compare the results

__global__ void add(int n, float **matr1, float **matr2, float **resMatr)
{
    long total = n*n;
    if (idx < total)
    {
        long column 2 = idx % n;
        long row1 = number / n;
        for (int i = 0; int < n; i++)
        {
            resMatr[row1][column2] = resMatr[row1][column2] + (matr1[row1][i] * matr2[i][column2]);
        }
    }
}
__global__ void add(int n, float **matr1, float **matr2, float **resMatr)
{
    // this should work. Also it does not need to be synchronized since every cell of the output is managed by a single thread.
    // will compare the 2 functions ASAP
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
    }
}

*/
