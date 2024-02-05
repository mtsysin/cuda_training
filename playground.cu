#include "stdio.h"

__global__ void square(float* d_out, float* d_in) {
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f*f*f;
}

int main(int argc, char** argv) {
    const int ARRAY_SIZE = 96;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate input array on the host
    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = float(i);
    }
    float h_out[ARRAY_SIZE];
    
    // declare gpu memory pointers
    float* d_in;
    float* d_out;

    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    // transfer the array to GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    //launch the kernel
    square<<<1, ARRAY_SIZE>>>(d_out, d_in);

    // Copy back to CPU
    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    // print out the resulting array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        printf("%f", h_out[i]);
        printf(((i % 4) != 3) ? "\t": "\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);

}

/*
We can create kernels with special descriptions:

Kernela<<<nBlocks, nThreads>>>
We can have a bunch of blocks with a bunch of threads in each of them.
threads can have shared memory within the same block but not in different blocks

Can also use 1d 2d 3d block and index dimaneions.

CUDA api:

Example: vector addition

The regular C: vecAdd

Host can talk to global memory. There is also memory in registers associated with each thread

cudaMalloc(Address, Size) -- allocate in global memory in GPU

*/

void vecAdd(float* h_A, float* h_B, float* h_C, int n) {
    int i; 
    for (i = 0 ; i < n; i++) {
        h_C[i] = h_A[i] + h_B[i];
    }
}

//Do it in cuda:

#include <cuda.h>

void vecAdd_CUDA(float* h_A, float* h_B, float* h_C, int n) {
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;  
    // alloocate device memory A B C
    // copy A and B
    // Launch kernel code
    // Copy C back from device to host
    // cudaMalloc((void**) &d_A, size);

    // Better way to allocate:
    cudaError_t err = cudaMalloc((void**) &d_A, size);
    if (err != cudaSuccess) {
        printf("%s in %s at line %d", 
        cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    cudaMalloc((void**) &d_B, size);
    cudaMalloc((void**) &d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Run
    vecAddKernel <<<ceil(n/256.0), 256>>> (d_A, d_B, d_C, n);
    // <<<num blocks, num threads in a block>>>

    // Another way:

    dim3 DimGrid((n-1)/256 + 1, 1, 1);
    dim3 DimBlock(256, 1, 1);
    vecAddKernel <<<DimGrid, DimBlock>>> (d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

}

// Can also use __device__, __host__
__global__
void vecAddKernel(float *A, float *B, float *C, int n) {
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    if (i < n) { // boundary check !!!
        C[i] = A[i] + B[i];
    }
}


// Multidimensional grids
// Considering a 62x76 picture (YxX)
// Use 16 x 16 blocks
// In memory, we use row-major layout index = row_idx * row_size + col indx; 

__global__ 
void PictureKernel(float *d_Pin, float * d_Pout, int n, int m) {
    // calculate teh row idx
    int Row = blockDim.y*blockIdx.y +  threadIdx.y;

    // calculate column idx
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((Row < m) && (Col < n)) {
        d_Pout[Row * n + Col] = 2.0 * d_Pin[Row * n + Col];
    }

    // The memoryt indices are gloabl, the blocc sizes don't determine the memory indices.
}

void cuda_PictureKernel(float* h_A, float* h_B, int m , int n) {
    // do processing
    float * d_Pin, *d_Pout;
    dim3 DimGrid((n-1)/16 + 1, (m-1)/16 + 1, 1);
    dim3 DimBlock(16, 16, 1);
    PictureKernel<<<DimGrid, DimBlock>>> (d_Pin, d_Pout, m, n);
}


// Matrix matrix multiplication

// SImple code in C

__host__
void MatMul(int m, int n, int k, float* A, float* B, float* C) {
    for (int row = 0; row < m; row++)
    for (int col = 0; col < k; col++) {
        float sum = 0;
        for (int i = 0; i < n; i++) {
            float a = A[row * n + i];
            float b = B[col + i * k];
            sum += a*b;
        }
        C[row*k + col] = sum;
    }
}

__global__
void KernelMatMul(int m, int n, int k,  float* d_A, float* d_B, float* d_C){
    int Row = blockDim.y * blockIdx.y +  threadIdx.y;
    int Col = blockDim.x * blockIdx.x + threadIdx.x;

    if ((Row < m) && (Col < k)) {
        float sum = 0.0;
        for (int i = 0; i < n; i++) {
            float a = d_A[Row * n + i];
            float b = d_B[Col + i * k];
            sum += a*b;
        }
        d_C[Row*k + Col] = sum;
    }

}

__global__
void ImprovedKernelMatMul(int m, int n, int k,  float* d_A, float* d_B, float* d_C){
    // Loading a tile, phased execution, briier synchronization
    // Example with shared memeory:
    const int TILE_WIDTH = 2;
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x; int bx = blockIdx.x;
    int ty = threadIdx.y; int by = blockIdx.y;

    // Get corresponding rows and coolumns from global memory:
    int Row = blockDim.y * by + ty;
    int Col = blockDim.x * bx + tx;
    float Cvalue = 0;

    // Outer loop over the A and B tils required to compute the C element
    for (int t = 0; t < n / TILE_WIDTH; ++t) {
        // Collaborative loading into the shared memory
        ds_A[ty][tx] = d_A[Row * n + t * TILE_WIDTH + tx]; // [Row][t * TILE_WIDTH + tx];
        ds_B[ty][tx] = d_A[(TILE_WIDTH * t + ty) * k + Col]; // [Row][t * TILE_WIDTH + tx];
        __syncthreads();

        // Do actual computations
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        }
        __syncthreads(); // Signals that we can overwrite the shared mamory 
                            //(Otherwise some fast thread might start the loop again and overwrite somethign)

    }

    d_C[Row * k + Col] = Cvalue;
}

void cuda_KernelMatMul(float* h_A, float* h_B, int m , int n, int k) {
    // do processing
    float * d_A, *d_B, *d_C;
    int TILE_WIDTH = 2;
    dim3 DimGrid((k-1)/TILE_WIDTH + 1, (m-1)/TILE_WIDTH + 1, 1);
    dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    KernelMatMul<<<DimGrid, DimBlock>>> (m, n, k, d_A, d_B, d_C);
}

// Week 1 done

// Thread scheduling.

// Declaing cuda varibales
__global__
void exapmleKernel() {
    int LocalVar; // register thread scope, thread lifetime
    __device__ __shared__ int SharedVar; //shared memory, block scope, block lifetime
    // There is a shared mempry ofr every streaming multiprocessor.
    __device__ int GlobalVar; // global memory, grid scope, application lifetime.
    __device__ __constant__ int ConstVar; // constant, grid scope and applicaiotn lifetime.
}
