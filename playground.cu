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
    for (int t = 0; t < (n-1) / TILE_WIDTH + 1; ++t) {
        // Collaborative loading into the shared memory
        // Add checks for non-even sizes
        ds_A[ty][tx] = ((Row < m) && (t * TILE_WIDTH + tx < n)) ? d_A[Row * n + t * TILE_WIDTH + tx] : 0; // [Row][t * TILE_WIDTH + tx];
        ds_B[ty][tx] = ((TILE_WIDTH * t + ty < n) && (Col < k)) ? d_A[(TILE_WIDTH * t + ty) * k + Col] : 0; // [Row][t * TILE_WIDTH + tx];
        __syncthreads();

        // Do actual computations
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += ds_A[ty][i] * ds_B[i][tx];
        } 
        __syncthreads(); // Signals that we can overwrite the shared mamory 
                            //(Otherwise some fast thread might start the loop again and overwrite somethign)
    }

    if ((Row < m) && (Col < k)) 
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


void runner() {
    // Way pf querying the device parameters.
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    cudaDeviceProp dev_prop;
    for (int i = 0; i < dev_count ; i++) {
        cudaGetDeviceProperties(&dev_prop, i);
        // dev_prop.maxThreadsPerBlock;
        // dev_prop.sharedMemoryPerBlock;
    }
}


// Week 3

// DRAM -- dynamically random access memory design.
// Memory coalescing
// good access pattern: A[(independednt of thread index) + thread index]

// Convolution

// 1D

__global__ void convolution_1D(float* N, float * M, float * P, int mask_width, int width) {
    // Really, it's correlation
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float Pvalue;

    int N_start_point = i - mask_width /2 ;
    for (int j = 0 ; j < mask_width; j++) {
        if (N_start_point + j >= 0 && N_start_point + j < width) {
    int N_start_point = i - mask_width /2 ;
            Pvalue += N[N_start_point + j] * M[j];
        }
    }
    P[i] = Pvalue;
}

__global__ void convolution_1D_tiled(float* N, float * M, float * P, int mask_width, int width) {
    // Really, it's correlation
    // Input tile is all the elements that are needed to compute output tiles
    const int O_TILE_WIDTH = 2;
    __shared__ float Ns[O_TILE_WIDTH];
    
    int index_o = blockIdx.x * blockDim.x + threadIdx.x;
    int index_i = index_o - mask_width /2;
    int tx = threadIdx.x;

    float output = 0.0;

    if ((index_i >= 0) && (index_i < width)) {
        Ns[tx] = N[index_i];
    } else {
        Ns[tx] = 0.0;
    }
    __syncthreads();

    if (threadIdx.x < O_TILE_WIDTH) { // only do if this thread actualy runs something
        output = 0.0;
        for (int j = 0; j < mask_width; ++j) { // j is the index in the mask
            output += M[j] * Ns[j + threadIdx.x]; // in the input array we need to shift the index by the thread index
                                                    //because we start relative to the ouput array 
        }
        P[index_o] = output;
    }
}

#define MASK_WIDTH 5
#define O_TILE_WIDTH 1020
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
// void conv_runner() {
//     dim3 dimBlock(BLOCK_WIDTH, 1, 1);
//     dim3 dimGrid((Width - 1) / O_TILE_WITH, 1, 1);
    
// }


// For 2d we try to pad each row to the multiple if DRAM burst size.
// For now, we'll work with a single channel image

typedef struct {
    int width;
    int height;
    int pitch;
    int channels;
    float* data;
} * wbImage_t;

void conv_runner_2d(wbImage_t img) {
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 dimGrid((img->width - 1) / O_TILE_WIDTH + 1, 
        (img->height - 1) / O_TILE_WIDTH + 1, 1);
}

__global__ void conviolution_2D_kernel(float *P, float *N,
                                        int height, int width, int channels,
                                        const float** __restrict__ M) {
    // CDUA devices provide constant memory whocse contents are aggressively cached.
    // Cached values are broadcast to all threads in a wrap
    // Effectively magnifies memory bandwith without consuming shared memory

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // figure out where this thread will write the output
    // of the convolution operation
    int row_o = blockIdx.y * O_TILE_WIDTH + ty;
    int col_o = blockIdx.x * O_TILE_WIDTH + tx;
    // Figure out where the element written in to the shared mamory will be
    int row_i = row_o - MASK_WIDTH;
    int col_i = col_o - MASK_WIDTH;

    // Loading the input element
    __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];
    if ((row_i >= 0) && (row_i < height) && 
        (col_i >= 0) && (col_i < height)) {
        Ns[ty][tx] = N[row_i * width + col_i]; // can also use pitch instead of width if needed; 
    } else {
        Ns[ty][tx] = 0.0;
    }
    __syncthreads();

    float output = 0.0;

    if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH) {
        for (int i = 0; i < MASK_WIDTH; i++)
        for (int j = 0; j < MASK_WIDTH; j++) {
            output += M[i][j] * Ns[i + ty][j + tx];
        }
    }

    if (row_o < height && col_o < width)
        P[row_o * width + col_o] = output;
}


// Week 4

// Parallel reduction pattern
// Work efficiency analysis 
// Partition and summarize
// something like MapReduce

// Basic reduction kernel

__global__ void reduction_kernel(float * input) {
    __shared__ float partialSum[2 * BLOCK_WIDTH];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    partialSum[t] = input[start + t];
    partialSum[t + blockDim.x] = input[start + t + blockDim.x];

    for (unsigned int stride = 1; stride < blockDim.x; stride*=2) {
        __syncthreads();
        if (t % stride == 0) {
            partialSum[2 * t] += partialSum[2 * t + stride];
        }
    }
}

// Better implementation keeping active threads consecutive and 
// reducting the number of active wraps

__global__ void better_reduction_kernel(float * input) {
    __shared__ float partialSum[2 * BLOCK_WIDTH];

    unsigned int t = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;
    partialSum[t] = input[start + t];
    partialSum[t + blockDim.x] = input[start + t + blockDim.x];

    for (unsigned int stride = blockDim.x; stride > 0; stride/=2) {
        __syncthreads();
        if (t < stride) {
            partialSum[t] += partialSum[2 * t + stride];
        } 
        // else {
        //     break;
        // }
    }
}

// Takeaway: always amake sure that the running threads are close to each other

// prefix sum
//http://developer.download.nvidia.com/compute/cuda/1_1/Website/projects/scan/doc/scan.pdf
//https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda