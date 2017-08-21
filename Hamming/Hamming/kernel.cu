#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define RANDOMIZE 0
#define COMPLEXITY 2
#define VERBOSE_LEVEL 0
#define COMPUTE_CPU 1
#define BLOCK_SIZE 1024

#define cudaCheckError(status, message) \
    if (status != cudaSuccess) { \
        fprintf(stderr, message); \
        goto Error; \
    }

cudaError_t processWithCuda(char* sequences, int* reduced, int* result, const unsigned int sequencesCount, const unsigned int charCount);

__device__ __host__ int bitcount(char c) {
    c -= (c >> 1) & 0x55;
    c = (c & 0x33) + ((c >> 2) & 0x33);
    c = (c + (c >> 4)) & 0x0f;
    return c & 0x7f;
}

__global__ void reduceKernel(char* g_idata, int* g_odata, const unsigned int len) {
    __shared__ int s_data[BLOCK_SIZE];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * (2 * BLOCK_SIZE) + tid;
    unsigned int gridSize = 2 * BLOCK_SIZE * gridDim.x;
    s_data[tid] = 0;

    while (idx < len) {
        s_data[tid] += bitcount(g_idata[idx]) + bitcount(g_idata[idx + BLOCK_SIZE]);
        idx += gridSize;
    }
    __syncthreads();

    if (BLOCK_SIZE >= 1024) { if (tid < 512) { s_data[tid] += s_data[tid + 512]; } __syncthreads(); }
    if (BLOCK_SIZE >=  512) { if (tid < 256) { s_data[tid] += s_data[tid + 256]; } __syncthreads(); }
    if (BLOCK_SIZE >=  256) { if (tid < 128) { s_data[tid] += s_data[tid + 128]; } __syncthreads(); }
    if (BLOCK_SIZE >=  128) { if (tid <  64) { s_data[tid] += s_data[tid +  64]; } __syncthreads(); }

    if (tid < 32) {
        if (BLOCK_SIZE >= 64) { s_data[tid] += s_data[tid + 32]; }
        if (BLOCK_SIZE >= 32) { s_data[tid] += s_data[tid + 16]; }
        if (BLOCK_SIZE >= 16) { s_data[tid] += s_data[tid +  8]; }
        if (BLOCK_SIZE >=  8) { s_data[tid] += s_data[tid +  4]; }
        if (BLOCK_SIZE >=  4) { s_data[tid] += s_data[tid +  2]; }
        if (BLOCK_SIZE >=  2) { s_data[tid] += s_data[tid +  1]; }
    }

    if (tid == 0) {
        g_odata[blockIdx.x] = s_data[0];
    }
}

__global__ void hammingKernel(char* g_idata, int* g_odata, const unsigned int i, const unsigned int j, const unsigned int count, const unsigned int len) {
    int pos = i * count + j;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len && g_odata[pos] <= 1) {
        char xor = g_idata[i * len + idx] ^ g_idata[j * len + idx];
        if (xor) {
            atomicAdd(&g_odata[pos], bitcount(xor));
        }
    }
}

int main() {
    cudaError_t cudaStatus;
    clock_t start, end;
    
    unsigned int i, j, k, pairs, limit;
    const unsigned int sequencesCount = 1 << 8;
    const unsigned int sequenceLength = 1 << 20;
    const unsigned int firstChar = sequenceLength % 8;
    const unsigned int charCount = sequenceLength / 8 + (firstChar ? 1 : 0);

    char* sequences = (char*)malloc(sequencesCount * charCount * sizeof(char));
    int* reduced = (int*)malloc(sequencesCount * sizeof(int));
    int* result = (int*)malloc(sequencesCount * sequencesCount * sizeof(int));

    srand(2137);

    // generate sequences
    printf("Generating %d sequences %d bits each starting...\n", sequencesCount, sequenceLength);
    if (RANDOMIZE != 0) {
        for (i = 0; i < sequencesCount; i++) {
            sequences[i * charCount] = rand() % (firstChar ? 1 << firstChar : 256);
            for (j = 1; j < charCount; j++) {
                sequences[i * charCount + j] = rand() % 256;
            }
        }
    }
    else {
        limit = COMPLEXITY == 1 ? sequencesCount - 1 : sequencesCount / 2;

        for (i = 0; i < limit; i++) {
            for (j = 0; j < charCount; j++) {
                sequences[i * charCount + j] = 0;
            }
        }

        for (i = limit; i < sequencesCount; i++) {
            for (j = 0; j < charCount - 1; j++) {
                sequences[i * charCount + j] = 0;
            }
            sequences[(i + 1) * charCount - 1] = 1;
        }
    }

    // print randomized sequences
    if (VERBOSE_LEVEL > 0) {
        for (i = 0; i < sequencesCount; i++) {
            printf("%2d: ", i);
            for (j = 0; j < charCount; j++) {
                for (k = 0; k < 8; k++) {
                    printf("%d", sequences[i * charCount + j] >> (7 - k) & 1);
                }
                printf(" ");
            }
            printf("\n");
        }
    }

    start = clock();
    printf("Generating finished\n\nComputing on GPU starting...\n");

    // copy data to GPU and perform calculations
    cudaStatus = processWithCuda(sequences, reduced, result, sequencesCount, charCount);
    cudaCheckError(cudaStatus, "processWithCuda failed!")

    end = clock();
    printf("Computing finished\nElapsed time (GPU): %f seconds\n\n", (float)(end - start) / CLOCKS_PER_SEC);

    cudaStatus = cudaDeviceReset();
    cudaCheckError(cudaStatus, "cudaDeviceReset failed!")

    // count pairs
    pairs = 0;
    for (i = 0; i < sequencesCount; i++) {
        for (j = i + 1; j < sequencesCount; j++) {
            if (result[i * sequencesCount + j] == 1) {
                if (VERBOSE_LEVEL > 1) {
                    printf("(%d,%d)\n", i, j);
                }
                pairs++;
            }
        }
    }
    printf("Found %d pairs with Hamming distance of 1\n", pairs);

    // print calculated distances
    if (VERBOSE_LEVEL > 1) {
        for (i = 0; i < sequencesCount; i++) {
            for (j = 0; j < sequencesCount; j++) {
                printf("%3d", result[i * sequencesCount + j]);
            }
            printf("\n");
        }

        printf("Sum reduce for each sequence:\n");
        for (i = 0; i < sequencesCount; i++) {
            printf("%d ", reduced[i]);
        }
        printf("\n");
    }

    // perform the same calculations on CPU
    if (COMPUTE_CPU != 0) {
        memset(reduced, 0, sequencesCount * sizeof(int));
        memset(result, 0, sequencesCount * sequencesCount * sizeof(int));

        start = clock();
        printf("\nComputing on CPU starting...\n");

        for (i = 0; i < sequencesCount; i++) {
            for (j = 0; j < charCount; j++) {
                reduced[i] += bitcount(sequences[i * charCount + j]);
            }
        }

        for (i = 0; i < sequencesCount; i++) {
            for (j = i + 1; j < sequencesCount; j++) {
                int dist = reduced[i] - reduced[j];
                if (dist == 1 || dist == -1) {
                    for (k = 0; k < charCount; k++) {
                        int idx = i * sequencesCount + j;
                        if (result[idx] <= 1) {
                            char xor = sequences[i * charCount + k] ^ sequences[j * charCount + k];
                            if (xor) {
                                result[idx] += bitcount(xor);
                            }
                        }
                    }
                }
            }
        }

        end = clock();
        printf("Computing finished\nElapsed time (CPU): %f seconds\n\n", (float)(end - start) / CLOCKS_PER_SEC);

        pairs = 0;
        for (i = 0; i < sequencesCount; i++) {
            for (j = i + 1; j < sequencesCount; j++) {
                if (result[i * sequencesCount + j] == 1) {
                    if (VERBOSE_LEVEL > 1) {
                        printf("(%d,%d)\n", i, j);
                    }
                    pairs++;
                }
            }
        }
        printf("Found %d pairs with Hamming distance of 1\n", pairs);

        // print calculated distances
        if (VERBOSE_LEVEL > 1) {
            for (i = 0; i < sequencesCount; i++) {
                for (j = 0; j < sequencesCount; j++) {
                    printf("%3d", result[i * sequencesCount + j]);
                }
                printf("\n");
            }

            printf("Sum reduce for each sequence:\n");
            for (i = 0; i < sequencesCount; i++) {
                printf("%d ", reduced[i]);
            }
            printf("\n");
        }
    }

Error:
    free(sequences);
    free(reduced);
    free(result);

    if (cudaStatus != cudaSuccess) {
        return 1;
    }

    return 0;
}

cudaError_t processWithCuda(char* sequences, int* reduced, int* result, const unsigned int sequencesCount, const unsigned int charCount) {
    cudaError_t cudaStatus;
    unsigned int i, j;
    char* dev_sequences = 0;
    int* dev_reduced = 0;
    int* dev_result = 0;
    int* host_reduced = 0;

    const unsigned int reduceBlockCount = charCount / (2 * BLOCK_SIZE) + (charCount % (2 * BLOCK_SIZE) ? 1 : 0);
    const unsigned int hammingBlockCount = (charCount + BLOCK_SIZE - 1) / BLOCK_SIZE;
    host_reduced = (int*)malloc(sequencesCount * reduceBlockCount * sizeof(int));

    cudaStatus = cudaSetDevice(0);
    cudaCheckError(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?")

    cudaStatus = cudaMalloc(&dev_result, sequencesCount * sequencesCount * sizeof(int));
    cudaCheckError(cudaStatus, "cudaMalloc failed!")

    cudaStatus = cudaMalloc(&dev_reduced, sequencesCount * reduceBlockCount * sizeof(int));
    cudaCheckError(cudaStatus, "cudaMalloc failed!")

    cudaStatus = cudaMalloc(&dev_sequences, sequencesCount * charCount * sizeof(char));
    cudaCheckError(cudaStatus, "cudaMalloc failed!")

    cudaStatus = cudaMemcpy(dev_sequences, sequences, sequencesCount * charCount * sizeof(char), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus, "cudaMemcpy failed!")

    for (i = 0; i < sequencesCount; i++) {
        reduceKernel<<<reduceBlockCount, BLOCK_SIZE>>>(&dev_sequences[i * charCount], &dev_reduced[i * reduceBlockCount], charCount);
    }

    cudaStatus = cudaGetLastError();
    cudaCheckError(cudaStatus, "reduceKernel launch failed!")

    cudaStatus = cudaDeviceSynchronize();
    cudaCheckError(cudaStatus, "cudaDeviceSynchronize failed after launching reduceKernel!")

    cudaStatus = cudaMemcpy(host_reduced, dev_reduced, sequencesCount * reduceBlockCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus, "cudaMemcpy failed!")

    for (i = 0; i < sequencesCount; i++) {
        reduced[i] = 0;
        for (j = 0; j < reduceBlockCount; j++) {
            reduced[i] += host_reduced[i * reduceBlockCount + j];
        }
    }

    for (i = 0; i < sequencesCount; i++) {
        for (j = i + 1; j < sequencesCount; j++) {
            int dist = reduced[i] - reduced[j];
            if (dist == 1 || dist == -1) {
                hammingKernel<<<hammingBlockCount, BLOCK_SIZE>>>(dev_sequences, dev_result, i, j, sequencesCount, charCount);
            }
        }
    }

    cudaStatus = cudaGetLastError();
    cudaCheckError(cudaStatus, "hammingKernel launch failed!")

    cudaStatus = cudaDeviceSynchronize();
    cudaCheckError(cudaStatus, "cudaDeviceSynchronize failed after launching hammingKernel!")

    cudaStatus = cudaMemcpy(result, dev_result, sequencesCount * sequencesCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus, "cudaMemcpy failed!")

Error:
    cudaFree(dev_sequences);
    cudaFree(dev_reduced);
    cudaFree(dev_result);
    free(host_reduced);

    return cudaStatus;
}
