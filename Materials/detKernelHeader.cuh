#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include<chrono>
#include<ctime>
#include <stdio.h>
#include<vector>

float* gpuDet(float* data, int size, int device, int count);

__global__ void detKernel(float* u, int col, float* det, int size)
{
    int m = blockIdx.x;
    int first = m * size * size;
    if (col == size - 1) {
        det[m] *= u[first + size * size - 1];
        return;
    }
    int i = threadIdx.x + col + 1;
    int j = threadIdx.y + col;
    det[m] *= u[first + col * size + col];
    float z = u[first + i * size + col] / u[first + col * size + col];

    u[first + i * size + j] -= z * u[first + col * size + j];
}

__global__ void setKernel(float* det) {
    int i = threadIdx.x;
    det[i] = 1.f;
}

float* gpuDet(float* data, int size, int device, int count) {
    int i;
    cudaError_t cudaStatus;
    cudaSetDevice(device);
    float* dev_u = 0;
    float* dev_dets;
    float* dets = (float*)malloc(count * sizeof(float));
    auto t_start = std::chrono::high_resolution_clock::now();


    cudaStatus = cudaMalloc((void**)&dev_u, count * size * size * sizeof(float));

    cudaStatus = cudaMallocHost((void**)&dev_dets, count * sizeof(float));

    cudaStatus = cudaMemcpy(dev_u, data, count * size * size * sizeof(float), cudaMemcpyHostToDevice);

    cudaStatus = cudaMemset(dev_dets, 0, count * sizeof(float));
    setKernel << <1, count >> > (dev_dets);

    int b = 0;
    for (i = 0; i < size; i++) {
        if (i == size - 1) b++;
        dim3 threads(size - 1 - i + b, size - i, 1);
        detKernel << <count, threads >> > (dev_u, i, dev_dets, size);
    }

    cudaStatus = cudaMemcpy(dets, dev_dets, count * sizeof(float), cudaMemcpyDeviceToHost);





    cudaFree(dev_u);
    cudaFree(dev_dets);
    cudaStatus = cudaDeviceReset();

    return dets;
}