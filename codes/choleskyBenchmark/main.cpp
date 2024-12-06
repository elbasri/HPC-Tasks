#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "mkl.h"
#include "utils.h"
#include "helper_string.h" // Added helper_string.h
#include "helper_cuda.h"   // Added helper_cuda.h

// Error checking macros
#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
        { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#define CHECK_CUSOLVER(call) \
    { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) \
        { \
            std::cerr << "CUSOLVER Error: " << _cudaGetErrorEnum(status) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

// Generate a symmetric positive-definite matrix of size NxN
void generatePositiveDefiniteMatrix(int N, double* A) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    // A = A * A^T to ensure it is symmetric positive-definite
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, N, N, 1.0, A, N, A, N, 0.0, A, N);
}

// CPU Cholesky factorization using LAPACKE
void choleskyCPU(int N, double* A) {
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', N, A, N);
    if (info != 0) {
        std::cerr << "CPU Cholesky factorization failed at info = " << info << std::endl;
    }
}

// GPU Cholesky factorization using cuSOLVER
void choleskyGPU(int N, double* h_A) {
    cusolverDnHandle_t cusolverH;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    double* d_A;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeof(double) * N * N));
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeof(double) * N * N, cudaMemcpyHostToDevice));

    int lwork = 0;
    CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(cusolverH, CUBLAS_FILL_MODE_LOWER, N, d_A, N, &lwork));

    double* d_work;
    CHECK_CUDA(cudaMalloc((void**)&d_work, sizeof(double) * lwork));

    int* devInfo;
    CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));

    CHECK_CUSOLVER(cusolverDnDpotrf(cusolverH, CUBLAS_FILL_MODE_LOWER, N, d_A, N, d_work, lwork, devInfo));

    int info;
    CHECK_CUDA(cudaMemcpy(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::cerr << "GPU Cholesky factorization failed at info = " << info << std::endl;
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_work));
    CHECK_CUDA(cudaFree(devInfo));
    CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
}

int main(int argc, char** argv) {
    int N = 1024; // Size of the matrix
    double* A = new double[N * N];
    
    // Generate a symmetric positive-definite matrix
    generatePositiveDefiniteMatrix(N, A);

    // CPU Cholesky Factorization
    double startCPU = get_time_in_seconds();
    choleskyCPU(N, A);
    double endCPU = get_time_in_seconds();
    double timeCPU = endCPU - startCPU;
    std::cout << "CPU Cholesky factorization time: " << timeCPU << " seconds" << std::endl;

    // Regenerate the matrix since it has been factorized
    generatePositiveDefiniteMatrix(N, A);

    // GPU Cholesky Factorization
    double startGPU = get_time_in_seconds();
    choleskyGPU(N, A);
    double endGPU = get_time_in_seconds();
    double timeGPU = endGPU - startGPU;
    std::cout << "GPU Cholesky factorization time: " << timeGPU << " seconds" << std::endl;

    delete[] A;
    return 0;
}
