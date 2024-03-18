#include <curand_kernel.h>

// C = AB
// A = (m,n), B = (n,k), C = (m,k)
__global__ void matrixMulKernel(float* A, float* B, float* C, int m, int n, int k, int divide) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum/divide;
    }
}

// A = (m,n)
// A^T = (n,m)
// Simple non-coalesced transpose
__global__ void matrixTransposeKernel(float* A, float* At, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row <  m && col < n) {
        At[col*m+row] = A[row*n+col];
    }
}

// A = (m,n), B=(m,n)
// A = A + lr*B
__global__ void matrixAddKernel(float* A, float* B, int m, int n, float lr=1) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        A[row*n+col] += lr*B[row*n+col];
    }
}

// a = (m,), b=(m,)
// a = a + lr*b
__global__ void vectorAddKernel(float* a, float* b, int m, float lr=1) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row <  m) {
        a[row] += lr*b[row];
    }
}

// Broadcast adding
// A(m,n) + b(m,1)
// C(m,n)
__global__ void matrixVecAddKernel(float* A, float* b, float* C, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
//    printf("Row %d, Col %d\n", row, col);
    if (row < m && col < n) {
        C[row * n + col] = A[row * n + col] + b[row];
//        printf("Row %d, Col %d, Value1 %f, Value2 %f\n", row, col, A[row * n + col], b[row]);
    }
}

// Sum along axis
// Input --> A(m,n)
// Output --> b(m,1)
__global__ void matrixAvgAlongAxisKernel(float* A, float* b, int m, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i];
        }
        b[row] = sum / n;
    }
}

// weights (m,n)
// between -1 to 1
__global__ void initializeWeightsKernel(float* weights, int m, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        curandState state;
        curand_init(444, row * n + col, 0, &state);
        weights[row * n + col] = 2 * curand_uniform(&state) - 1;
    }
}

// bias (m,1)
// between -1 to 1
__global__ void initializeBiasKernel(float* bias, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        curandState state;
        curand_init(0, row, 0, &state);
        bias[row] = 0; //2 * curand_uniform(&state) - 1;
    }
}

void matrixMultiply(float* A, float* B, float* C, int m, int n, int k, int divide=1){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((k + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    matrixMulKernel <<< gridSize, blockSize >>> (A, B, C, m, n, k, divide);
    cudaDeviceSynchronize();
}

void matrixTranspose(float* A, float* At, int m, int n){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    matrixTransposeKernel <<< gridSize, blockSize >>> (A, At, m, n);
    cudaDeviceSynchronize();
}

void matrixVectorAdd(float* A, float* b, float* C, int m, int n){
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    matrixVecAddKernel <<< gridSize, blockSize >>> (A, b, C, m, n);
    cudaDeviceSynchronize();
}

void matrixAvgAlongAxis(float* A, float* b, int m, int n){
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x);
    matrixAvgAlongAxisKernel <<< gridSize, blockSize >>> (A, b, m, n);
    cudaDeviceSynchronize();
}
