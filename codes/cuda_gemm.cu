#include <iostream>
#include <iomanip>
#include <fstream>
#include "mkl.h"
#include "utils.h"


//#define SINGLE_PRECISION //Comment out to use double precision arithmetic
#define DOUBLE_PRECISION

#ifdef SINGLE_PRECISION
	#define elem_t float
	#define blasGemm cblas_sgemm 
	#define cublasGemm cublasSgemm
#elif defined(DOUBLE_PRECISION)
	#define elem_t double
	#define blasGemm cblas_dgemm 
	#define cublasGemm cublasDgemm
#endif

#ifndef GEMM_M
#define GEMM_M 9000
#endif
#ifndef GEMM_N
#define GEMM_N 9000
#endif
#ifndef GEMM_K
#define GEMM_K 9000
#endif

#ifndef WARMUPS
#define WARMUPS 3
#endif
#ifndef ITERS
#define ITERS 10
#endif

// Kernel for GEMM - gemmV1
__global__ void gemmV1Kernel(int M, int N, int K, elem_t alpha, elem_t *A, elem_t *B, elem_t beta, elem_t *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        elem_t sum = 0.0;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

void gemmV1(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC) {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    gemmV1Kernel<<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
}




#define V2_TILE_M 32
#define V2_TILE_N 32
#define V2_TILE_K 16

// Kernel for GEMM - gemmV2 with shared memory
__global__ void gemmV2Kernel(int M, int N, int K, elem_t alpha, elem_t *A, elem_t *B, elem_t beta, elem_t *C) {
    // Define the tile size
    __shared__ elem_t As[V2_TILE_M][V2_TILE_K];
    __shared__ elem_t Bs[V2_TILE_K][V2_TILE_N];

    int row = blockIdx.y * V2_TILE_M + threadIdx.y;
    int col = blockIdx.x * V2_TILE_N + threadIdx.x;

    elem_t value = 0.0;

    for (int t = 0; t < (K + V2_TILE_K - 1) / V2_TILE_K; ++t) {
        // Load tiles into shared memory
        if (row < M && t * V2_TILE_K + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + t * V2_TILE_K + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < N && t * V2_TILE_K + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * V2_TILE_K + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Compute partial product for the tile
        for (int i = 0; i < V2_TILE_K; ++i) {
            value += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        // Synchronize to avoid race conditions
        __syncthreads();
    }

    // Write the result to the output matrix
    if (row < M && col < N) {
        C[row * N + col] = alpha * value + beta * C[row * N + col];
    }
}

void gemmV2(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC) {
    // Set up block and grid dimensions
    dim3 blockSize(V2_TILE_N, V2_TILE_M);
    dim3 gridSize((N + V2_TILE_N - 1) / V2_TILE_N, (M + V2_TILE_M - 1) / V2_TILE_M);

    // Launch the kernel
    gemmV2Kernel<<<gridSize, blockSize>>>(M, N, K, alpha, A, B, beta, C);
}


#define V3_TILE_M 64
#define V3_TILE_N 64
#define V3_TILE_K 8
#define V3_THREAD_M 2
#define V3_THREAD_N 2
 void gemmV3(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//write kernel with shared memory and higher arithmetic intensity
}

void runGemmV1(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC) {
    gemmV1(M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC);
}

void runGemmV2(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//call the gpu kernel
}
void runGemmV3(int M, int N, int K, elem_t alpha, elem_t *A, int ldA, elem_t *B, int ldB, elem_t beta, elem_t *C, int ldC)
{
	//call the gpu kernel
}

int main(int argc, char **argv)
{
    // Sawb CSV
    std::ofstream csvFile;
    csvFile.open("cuda_gemm_performance_iterations.csv");
    csvFile << "Iteration,Matrix Size (M),Matrix Size (N),Matrix Size (K),Version,Execution Time (ms),Performance (GFLOP/s)\n";

	for(int iter = 0; iter < 10; iter++){
		int scale_factor = iter + 1;
		int M = GEMM_M * scale_factor;
		int N = GEMM_N * scale_factor;
		int K = GEMM_K * scale_factor;

		float *times = new float[2 * ITERS];
		float *timesCPU = times;
		float *timesGPU = times + ITERS;

		elem_t *A, *B, *C, *Cgpu;

		// Allocate and initialize matrices
		allocateMatrixCPU(M, K, &A);
		allocateMatrixCPU(K, N, &B);
		allocateMatrixCPU(M, N, &C);
		initMatrixRandomCPU<elem_t>(M, K, A);
		initMatrixRandomCPU<elem_t>(K, N, B);
		initMatrixCPU<elem_t>(M, N, C, 0.0);

		elem_t *d_A, *d_B, *d_C;
		allocateMatrixGPU(M, K, &d_A);
		allocateMatrixGPU(K, N, &d_B);
		allocateMatrixGPU(M, N, &d_C);

		cudaMemcpy(d_A, A, sizeof(elem_t) * M * K, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, B, sizeof(elem_t) * K * N, cudaMemcpyHostToDevice);
		cudaMemcpy(d_C, C, sizeof(elem_t) * M * N, cudaMemcpyHostToDevice);

		elem_t alpha = 1.0;
		elem_t beta = 0.0;

		// CPU Benchmark
		struct timespec cpu_start, cpu_end;
		for (int i = 0; i < ITERS; i++)
		{
			clock_gettime(CLOCK_MONOTONIC, &cpu_start);
			blasGemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, M, B, K, beta, C, M);
			clock_gettime(CLOCK_MONOTONIC, &cpu_end);
			timesCPU[i] = computeCPUTime(&cpu_start, &cpu_end);
		}

		// GPU Benchmark
		for (int i = 0; i < WARMUPS; i++)
		{
			runGemmV2(M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
			cudaDeviceSynchronize();
		}

		cudaEvent_t gpu_start, gpu_end;
		cudaEventCreate(&gpu_start);
		cudaEventCreate(&gpu_end);
		for (int i = 0; i < ITERS; i++)
		{
			cudaEventRecord(gpu_start);
			runGemmV2(M, N, K, alpha, d_A, M, d_B, K, beta, d_C, M);
			cudaEventRecord(gpu_end);
			cudaDeviceSynchronize();
			cudaEventElapsedTime(&(timesGPU[i]), gpu_start, gpu_end);
		}
		cudaEventDestroy(gpu_start);
		cudaEventDestroy(gpu_end);

		// Calculate FLOPs
		float flops = 2 * (float)M * (float)N * (float)K;

		// Calculate average CPU performance
		float avg_cpu = 0.0;
		for (int i = 0; i < ITERS; i++)
			avg_cpu += timesCPU[i];
		avg_cpu = avg_cpu / (float)ITERS;

		float cpu_gflops = (flops / 1.0e9) / (avg_cpu / 1.0e3);

		std::cout << "Iteration " << iter + 1 << " - CPU ====\n";
		std::cout << "Execution time: " << avg_cpu << " ms.\n";
		std::cout << "Performance: " << cpu_gflops << " GFLOP/s.\n";

		// Calculate average GPU performance
		float avg_gpu = 0.0;
		for (int i = 0; i < ITERS; i++)
			avg_gpu += timesGPU[i];
		avg_gpu = avg_gpu / (float)ITERS;

		float gpu_gflops = (flops / 1.0e9) / (avg_gpu / 1.0e3);

		std::cout << "Iteration " << iter + 1 << " - GPU ====\n";
		std::cout << "Execution time: " << avg_gpu << " ms.\n";
		std::cout << "Performance: " << gpu_gflops << " GFLOP/s.\n";

		// Write results for this iteration to CSV
		csvFile << iter + 1 << "," << M << "," << N << "," << K << ",CPU," << avg_cpu << "," << cpu_gflops << "\n";
		csvFile << iter + 1 << "," << M << "," << N << "," << K << ",GPU," << avg_gpu << "," << gpu_gflops << "\n";
		//update hada kolla iter
		csvFile.flush();
		// Free resources
		allocateMatrixCPU(M, N, &Cgpu);
		cudaMemcpy(Cgpu, d_C, sizeof(elem_t) * M * N, cudaMemcpyDeviceToHost);
		compareMatrices(M, N, C, Cgpu);
		freeMatrixCPU(M, N, Cgpu);

		freeMatrixGPU(M, K, d_A);
		freeMatrixGPU(K, N, d_B);
		freeMatrixGPU(M, N, d_C);

		freeMatrixCPU(M, K, A);
		freeMatrixCPU(K, N, B);
		freeMatrixCPU(M, N, C);


		delete[] times;
	}
    // Sed CSV
    csvFile.close();

    return 0;
}
