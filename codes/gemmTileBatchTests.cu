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

void tileGemmBatch(cublasHandle_t handle, int M, int N, int K, elem_t alpha,
                   const elem_t* d_A, int lda, const elem_t* d_B, int ldb,
                   elem_t beta, elem_t* d_C, int ldc, int tileM, int tileN) {
    // Create pointers to device memory
    std::vector<const elem_t*> A_array;
    std::vector<const elem_t*> B_array;
    std::vector<elem_t*> C_array;

    // Calculate the number of batches
    int batchCount = 0;
    for (int i = 0; i < M; i += tileM) {
        for (int j = 0; j < N; j += tileN) {
            int currentTileM = std::min(tileM, M - i);
            int currentTileN = std::min(tileN, N - j);

            A_array.push_back(d_A + i);
            B_array.push_back(d_B + j * ldb);
            C_array.push_back(d_C + i + j * ldc);

            batchCount++;
        }
    }

    // Allocate arrays for device pointers
    const elem_t** d_A_array;
    const elem_t** d_B_array;
    elem_t** d_C_array;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_A_array, batchCount * sizeof(elem_t*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_B_array, batchCount * sizeof(elem_t*)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_C_array, batchCount * sizeof(elem_t*)));

    // Copy pointers to device memory
    CHECK_CUDA_ERROR(cudaMemcpy(d_A_array, A_array.data(), batchCount * sizeof(elem_t*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B_array, B_array.data(), batchCount * sizeof(elem_t*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C_array, C_array.data(), batchCount * sizeof(elem_t*), cudaMemcpyHostToDevice));

    // Perform batched GEMM
    CHECK_CUBLAS_ERROR(cublasGemmBatched(handle,
                                         CUBLAS_OP_N, CUBLAS_OP_N,
                                         tileM, tileN, K,
                                         &alpha,
                                         d_A_array, lda,
                                         d_B_array, ldb,
                                         &beta,
                                         d_C_array, ldc,
                                         batchCount));

    // Free allocated memory
    CHECK_CUDA_ERROR(cudaFree(d_A_array));
    CHECK_CUDA_ERROR(cudaFree(d_B_array));
    CHECK_CUDA_ERROR(cudaFree(d_C_array));
}

int main(int argc, char **argv)
{
    cublasHandle_t handle;
    CHECK_CUBLAS_ERROR(cublasCreate(&handle));

    float *times = new float[ITERS + 1];
    elem_t *A, *B, *C, *C_gpu;
    elem_t *d_A, *d_B, *d_C; // Device pointers
    elem_t alpha = 1.0;
    elem_t beta = 0.0;

    // Open CSV files to write results
    std::ofstream execPerfFile("execPerf.csv");
    execPerfFile << "M,N,K,tileM,tileN,TimeGPU(ms),PerformanceGPU(GFLOP/s)" << std::endl;

    // Loop over multiple parameters to test their effects
    std::vector<int> sizes = {512, 1024, 2048}; // Different sizes for M, N, K
    std::vector<int> tileSizes = {32, 64, 128}; // Different tile sizes for tileM and tileN

    for (int M : sizes) {
        for (int N : sizes) {
            for (int K : sizes) {
                for (int tileM : tileSizes) {
                    for (int tileN : tileSizes) {
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

                        // GPU Execution using CUDA Events with Tiling and Batch
                        cudaEvent_t gpu_start, gpu_end;
                        CHECK_CUDA_ERROR(cudaEventCreate(&gpu_start));
                        CHECK_CUDA_ERROR(cudaEventCreate(&gpu_end));

                        for (int i = 0; i < ITERS + 1; i++) {
                            CHECK_CUDA_ERROR(cudaEventRecord(gpu_start, 0));

                            // Use the tileGemmBatch function
                            tileGemmBatch(handle, M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M, tileM, tileN);

                            CHECK_CUDA_ERROR(cudaEventRecord(gpu_end, 0));
                            CHECK_CUDA_ERROR(cudaEventSynchronize(gpu_end));

                            float elapsedTime;
                            CHECK_CUDA_ERROR(cudaEventElapsedTime(&elapsedTime, gpu_start, gpu_end));
                            times[i] = elapsedTime;
                        }

                        // Take the average excluding the first run
                        float totalTimeGPU = 0.0;
                        for (int i = 1; i < ITERS + 1; i++) {
                            totalTimeGPU += times[i];
                        }
                        float avgTimeGPU = totalTimeGPU / ITERS;

                        // Compute FLOPs (2 * M * N * K)
                        float flops = 2.0f * M * N * K;

                        // Compute performance in GFLOP/s for GPU
                        float performanceGFLOPsGPU = (flops / 1.0e9) / (avgTimeGPU / 1.0e3);
                        std::cout << "M: " << M << ", N: " << N << ", K: " << K << ", tileM: " << tileM << ", tileN: " << tileN
                                  << " - GPU Execution time: " << avgTimeGPU << " ms, Performance: " << performanceGFLOPsGPU << " GFLOP/s" << std::endl;

                        // Write execution performance data to CSV file
                        execPerfFile << M << "," << N << "," << K << "," << tileM << "," << tileN << "," << avgTimeGPU << "," << performanceGFLOPsGPU << std::endl;

                        // Free GPU memory
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
                    }
                }
            }
        }
    }

    // Close CSV files
    execPerfFile.close();
    CHECK_CUBLAS_ERROR(cublasDestroy(handle));

    delete[] times;

    return 0;
}
