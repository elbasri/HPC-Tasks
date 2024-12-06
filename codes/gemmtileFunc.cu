#include <iostream>
#include <iomanip>
#include <fstream>
#include "mkl.h"
#include "cublas_v2.h"
#include "utils.h"

//#define SINGLE_PRECISION //Comment out to use double precision arithmetic
#define DOUBLE_PRECISION

#ifdef SINGLE_PRECISION
    #define elem_t float
    #define blasGemm cblas_sgemm 
    #define cublasGemm cublasSgemm
    #define cublasGemmBatched cublasSgemmBatched
#elif defined(DOUBLE_PRECISION)
    #define elem_t double
    #define blasGemm cblas_dgemm 
    #define cublasGemm cublasDgemm
    #define cublasGemmBatched cublasDgemmBatched
#endif

#ifndef TILE_M
#define TILE_M 64
#endif
#ifndef TILE_N
#define TILE_N 64
#endif

#ifndef NB_STREAMS
#define NB_STREAMS 16
#endif

#ifndef WARMUPS
#define WARMUPS 1
#endif
#ifndef ITERS
#define ITERS 10
#endif

#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) \
        { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

#define CHECK_CUBLAS_ERROR(call) \
    { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) \
        { \
            std::cerr << "CUBLAS Error at line " << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

void tileGemm(cublasHandle_t handle, int M, int N, int K, elem_t alpha,
              const elem_t* d_A, int lda, const elem_t* d_B, int ldb,
              elem_t beta, elem_t* d_C, int ldc, int tileM, int tileN) {
    // Loop over all tiles
    for (int i = 0; i < M; i += tileM) {
        for (int j = 0; j < N; j += tileN) {
            int currentTileM = std::min(tileM, M - i);
            int currentTileN = std::min(tileN, N - j);

            CHECK_CUBLAS_ERROR(cublasGemm(handle,
                                          CUBLAS_OP_N, CUBLAS_OP_N,
                                          currentTileM, currentTileN, K,
                                          &alpha,
                                          d_A + i, lda,
                                          d_B + j * ldb, ldb,
                                          &beta,
                                          d_C + i + j * ldc, ldc));
        }
    }
}

int main(int argc, char **argv)
{
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    cudaStream_t *streams;
    createStreams(NB_STREAMS, &streams);

    float *times = new float[2 * (ITERS + 1)];
    float *timesCPU = times;
    float *timesGPU = times + (ITERS + 1);

    elem_t *A, *B, *C, *C_gpu;
    elem_t *d_A, *d_B, *d_C; // Device pointers
    elem_t alpha = 1.0;
    elem_t beta = 0.0;

    // Open CSV files to write results
    std::ofstream csvFile("gemmtask3.csv");
    csvFile << "M,N,K,AverageTimeCPU(ms),AverageTimeGPU(ms),FLOPs,PerformanceCPU(GFLOP/s),PerformanceGPU(GFLOP/s),PerformanceProportion(time/MNK)" << std::endl;

    std::ofstream execPerfFile("execPerf.csv");
    execPerfFile << "M,N,K,tileM,tileN,AverageTimeGPU(ms),PerformanceGPU(GFLOP/s)" << std::endl;

    // Fixed parameters for M, N, K, but increase by 100 each iteration
    int M = 1024;
    int N = 1024;
    int K = 1024;

    // Initial execution before the loop
    // Allocate and initialize matrices A, B, C on CPU
    allocateMatrixCPU(M, K, &A);
    allocateMatrixCPU(K, N, &B);
    allocateMatrixCPU(M, N, &C);
    allocateMatrixCPU(M, N, &C_gpu);
    initMatrixRandomCPU(M, K, A);
    initMatrixRandomCPU(K, N, B);
    initMatrixCPU(M, N, C, static_cast<elem_t>(0.0));
    initMatrixCPU(M, N, C_gpu, static_cast<elem_t>(0.0));

    // Allocate memory on GPU using the updated functions
    allocateMatrixGPU(M, K, &d_A);
    allocateMatrixGPU(K, N, &d_B);
    allocateMatrixGPU(M, N, &d_C);

    // Copy data from CPU to GPU
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, A, M * K * sizeof(elem_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, B, K * N * sizeof(elem_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, C_gpu, M * N * sizeof(elem_t), cudaMemcpyHostToDevice));

    // CPU Execution
    struct timespec cpu_start, cpu_end;
    for (int i = 0; i < ITERS + 1; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &cpu_start);
        blasGemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
        clock_gettime(CLOCK_MONOTONIC, &cpu_end);
        timesCPU[i] = computeCPUTime(&cpu_start, &cpu_end);
    }

    // Compute and print average execution time on CPU (excluding the first run)
    float totalTimeCPU = 0.0;
    for (int i = 1; i < ITERS + 1; i++)
    {
        totalTimeCPU += timesCPU[i];
    }
    float avgTimeCPU = totalTimeCPU / ITERS;
    std::cout << "Initial Run - M: " << M << ", N: " << N << ", K: " << K << " - Average CPU execution time (excluding first): " << avgTimeCPU << " ms" << std::endl;

    // GPU Execution using CUDA Events with Tiling
    int tileM = TILE_M;
    int tileN = TILE_N;

    cudaEvent_t gpu_start, gpu_end;
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&gpu_end));

    float avgTimeGPU = 0.0;
    for (int i = 0; i < ITERS + 1; i++)
    {
        CHECK_CUDA_ERROR(cudaEventRecord(gpu_start, 0));
        tileGemm(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M, tileM, tileN);
        CHECK_CUDA_ERROR(cudaEventRecord(gpu_end, 0));
        CHECK_CUDA_ERROR(cudaEventSynchronize(gpu_end));

        float elapsedTime;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, gpu_start, gpu_end));
        timesGPU[i] = elapsedTime;
        if (i > 0) avgTimeGPU += elapsedTime;  // Exclude the first run
    }
    avgTimeGPU /= ITERS;

    // Print average GPU execution time (excluding the first run)
    std::cout << "Initial Run - Average GPU execution time with tiling (excluding first): " << avgTimeGPU << " ms" << std::endl;

    // Compute FLOPs (2 * M * N * K)
    float flops = 2.0f * M * N * K;

    // Compute performance in GFLOP/s for GPU
    float performanceGFLOPsGPU = (flops / 1.0e9) / (avgTimeGPU / 1.0e3);
    std::cout << "Initial Run - GPU Performance with tiling: " << performanceGFLOPsGPU << " GFLOP/s" << std::endl;

    // Write execution performance data to CSV file
    execPerfFile << M << "," << N << "," << K << "," << tileM << "," << tileN << "," << avgTimeGPU << "," << performanceGFLOPsGPU << std::endl;

    // Copy result back from GPU to CPU
    CHECK_CUDA_ERROR(cudaMemcpy(C_gpu, d_C, M * N * sizeof(elem_t), cudaMemcpyDeviceToHost));

    // Compare CPU and GPU results
    compareMatrices(M, N, C, C_gpu);

    // Free GPU memory using the updated functions
    freeMatrixGPU(M, K, d_A);
    freeMatrixGPU(K, N, d_B);
    freeMatrixGPU(M, N, d_C);

    // Free CPU memory
    freeMatrixCPU(M, K, A);
    freeMatrixCPU(K, N, B);
    freeMatrixCPU(M, N, C);
    freeMatrixCPU(M, N, C_gpu);

    // Destroy CUDA events
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(gpu_end));

    // Close CSV files
    csvFile.close();
    execPerfFile.close();
    destroyStreams(NB_STREAMS, streams);
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));

    delete[] times;

    return 0;
}
