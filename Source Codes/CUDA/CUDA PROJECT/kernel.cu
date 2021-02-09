#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include<chrono>
#include<ctime>
#include <stdio.h>
#include<vector>

// FUNCTION IDS
float* gpuDet(float* data, int s, int thread, int count);


// KERNELS
// Kernel for eliminating a column
__global__ void detKernel(float* u, int col, float* det, int size)
{
    int m = blockIdx.x;             // Index of matrix
    int first = m * size * size;    // First element of matrix in dev_u
    if (col == size - 1) {          // Handle last element
        det[m] *= u[first + size * size - 1];
        return;
    }
    int i = threadIdx.x + col + 1;  // Row
    int j = threadIdx.y + col;      // Column

    // Multiply by pivot
    det[m] *= u[first + col * size + col];

    // Calculate coefficient
    float z = u[first + i * size + col] / u[first + col * size + col];

    // Do the elimination
    u[first + i * size + j] -= z * u[first + col * size + j];
}

// Kernel for setting all determinants to 1 (Default)
__global__ void setKernel(float* det) {
    int i = threadIdx.x;
    det[i] = 1.f;
}

int main()
{
    /*  Initialize OPENMP
        add -Xcompiler -fopenmp to command line
        add "/openmp" to nvcc.profile
    */
#ifndef _OPENMP
    printf("OpenMP is not supported, sorry!\n");
    getchar();
    return 0;
#endif

    // Initialize Variables
    int i, j;                           // Iterators
    srand(time(NULL));                  // For making random numbers
    cudaError_t cudaStatus;             // Check cuda commands
    int devices;                        // Number of devices
    cudaGetDeviceCount(&devices);       // Set Number of devices
    int cores = omp_get_num_procs();    // Set number of cores
    int threads = cores * 2;            // Set number of threads
    int matrixNumbers = 200;            // Number of matrices to calculate
    int matrixDim = 32;                 // Size of matrix
    int differenSizes = 1;              // Variety of matrix size: e.g 2 for 32x32 and 64x64
    float* matrices;                    // All matrices in one place
    float* determinants;                // Determinants
    // Allocate Host Variables
    determinants = (float*)malloc(matrixNumbers * sizeof(float));
    matrices = (float*)malloc(sizeof(float) * matrixDim * matrixDim * matrixNumbers);

    // Fill matrices with random
    for (i = 0; i < matrixDim * matrixDim * matrixNumbers; i++) {
        matrices[i] = rand() % 10;
    }

    printf("Calculating determiants of %d matrices with size %d x %d\n", matrixNumbers, matrixDim, matrixDim);

    auto t_start = std::chrono::high_resolution_clock::now();
    // Each thread one device
#pragma omp parallel for num_threads(devices)
    for (i = 0; i < differenSizes; i++) {
        int threadId = omp_get_thread_num();
        determinants = gpuDet(matrices, matrixDim, threadId, matrixNumbers);
    }

    // Finish timing
    auto t_end = std::chrono::high_resolution_clock::now();

    // Print time
    double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    printf("%.2f miliseconds\n", elapsed_time_ms);
    system("PAUSE");
    return 0;
}

float* gpuDet(float* data, int size, int thread, int count) {
    int i;
    cudaError_t cudaStatus;
    cudaSetDevice(thread);
    float* dev_u = 0;
    float* dev_dets;
    float* dets = (float*)malloc(count * sizeof(float));

    // Allocate Device Memory
    cudaStatus = cudaMalloc((void**)&dev_u, count * size * size * sizeof(float));

    // Set Device Variables
    cudaStatus = cudaMallocHost((void**)&dev_dets, count * sizeof(float));
    cudaStatus = cudaMemcpy(dev_u, data, count * size * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaStatus = cudaMemset(dev_dets, 0, count * sizeof(float));
    setKernel << <1, count >> > (dev_dets);

    // Calculate determinant of a batch of matrices
    int b = 0;
    for (i = 0; i < size; i++) {
        if (i == size - 1) b++;
        dim3 threads(size - 1 - i + b, size - i, 1);
        detKernel << <count, threads >> > (dev_u, i, dev_dets, size);
    }

    // Copy back the answers
    cudaStatus = cudaMemcpy(dets, dev_dets, count * sizeof(float), cudaMemcpyDeviceToHost);

    // Free up
    cudaFree(dev_u);
    cudaFree(dev_dets);
    cudaStatus = cudaDeviceReset();
    return dets;
}