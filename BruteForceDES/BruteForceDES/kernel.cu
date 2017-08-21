#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <cstdint>

#define MSG_BLOCKS 1
#define CUDA_BLOCKS 2048
#define BLOCK_SIZE 1024

#define cudaCheckError(status, message) \
    if (status != cudaSuccess) { \
        fprintf(stderr, message); \
        goto Error; \
    }

// host-only DES constant matrices
const int hPC1[56] = {
    57, 49, 41, 33, 25, 17,  9,
     1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27,
    19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
     7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29,
    21, 13,  5, 28, 20, 12,  4
};
const int hR[16] = {
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
};
const int hPC2[48] = {
    14, 17, 11, 24,  1,  5,
     3, 28, 15,  6, 21, 10,
    23, 19, 12,  4, 26,  8,
    16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
};
const int hIP[64] = {
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
};
const int hE[48] = {
    32,  1,  2,  3,  4,  5,  4,  5,
     6,  7,  8,  9,  8,  9, 10, 11,
    12, 13, 12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21, 20, 21,
    22, 23, 24, 25, 24, 25, 26, 27,
    28, 29, 28, 29, 30, 31, 32,  1
};
const int hS[8][4][16] = {
    {
        { 14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7 },
        {  0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8 },
        {  4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0 },
        { 15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13 }
    },
    {
        { 15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10 },
        {  3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5 },
        {  0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15 },
        { 13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9 }
    },
    {
        { 10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8 },
        { 13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1 },
        { 13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7 },
        {  1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12 }
    },
    {
        {  7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15 },
        { 13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9 },
        { 10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4 },
        {  3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14 }
    },
    {
        {  2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9 },
        { 14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6 },
        {  4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14 },
        { 11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3 }
    },
    {
        { 12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11 },
        { 10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8 },
        {  9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6 },
        {  4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13 }
    },
    {
        {  4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1 },
        { 13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6 },
        {  1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2 },
        {  6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12 }
    },
    {
        { 13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7 },
        {  1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2 },
        {  7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8 },
        {  2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11 }
    }
};
const int hP[32] = {
    16,  7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26,  5, 18, 31, 10,
     2,  8, 24, 14, 32, 27,  3,  9,
    19, 13, 30,  6, 22, 11,  4, 25
};
const int hFP[64] = {
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41,  9, 49, 17, 57, 25
};

// device-only DES constant matrices
__constant__ int dPC1[56] = {
    57, 49, 41, 33, 25, 17,  9,
     1, 58, 50, 42, 34, 26, 18,
    10,  2, 59, 51, 43, 35, 27,
    19, 11,  3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
     7, 62, 54, 46, 38, 30, 22,
    14,  6, 61, 53, 45, 37, 29,
    21, 13,  5, 28, 20, 12,  4
};
__constant__ int dR[16] = {
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
};
__constant__ int dPC2[48] = {
    14, 17, 11, 24,  1,  5,
     3, 28, 15,  6, 21, 10,
    23, 19, 12,  4, 26,  8,
    16,  7, 27, 20, 13,  2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
};
__constant__ int dIP[64] = {
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17,  9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
};
__constant__ int dE[48] = {
    32,  1,  2,  3,  4,  5,  4,  5,
     6,  7,  8,  9,  8,  9, 10, 11,
    12, 13, 12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21, 20, 21,
    22, 23, 24, 25, 24, 25, 26, 27,
    28, 29, 28, 29, 30, 31, 32,  1
};
__constant__ int dS[8][4][16] = {
    {
        { 14,  4, 13,  1,  2, 15, 11,  8,  3, 10,  6, 12,  5,  9,  0,  7 },
        {  0, 15,  7,  4, 14,  2, 13,  1, 10,  6, 12, 11,  9,  5,  3,  8 },
        {  4,  1, 14,  8, 13,  6,  2, 11, 15, 12,  9,  7,  3, 10,  5,  0 },
        { 15, 12,  8,  2,  4,  9,  1,  7,  5, 11,  3, 14, 10,  0,  6, 13 }
    },
    {
        { 15,  1,  8, 14,  6, 11,  3,  4,  9,  7,  2, 13, 12,  0,  5, 10 },
        {  3, 13,  4,  7, 15,  2,  8, 14, 12,  0,  1, 10,  6,  9, 11,  5 },
        {  0, 14,  7, 11, 10,  4, 13,  1,  5,  8, 12,  6,  9,  3,  2, 15 },
        { 13,  8, 10,  1,  3, 15,  4,  2, 11,  6,  7, 12,  0,  5, 14,  9 }
    },
    {
        { 10,  0,  9, 14,  6,  3, 15,  5,  1, 13, 12,  7, 11,  4,  2,  8 },
        { 13,  7,  0,  9,  3,  4,  6, 10,  2,  8,  5, 14, 12, 11, 15,  1 },
        { 13,  6,  4,  9,  8, 15,  3,  0, 11,  1,  2, 12,  5, 10, 14,  7 },
        { 1, 10, 13,  0,  6,  9,  8,  7,  4, 15, 14,  3, 11,  5,  2, 12 }
    },
    {
        {  7, 13, 14,  3,  0,  6,  9, 10,  1,  2,  8,  5, 11, 12,  4, 15 },
        { 13,  8, 11,  5,  6, 15,  0,  3,  4,  7,  2, 12,  1, 10, 14,  9 },
        { 10,  6,  9,  0, 12, 11,  7, 13, 15,  1,  3, 14,  5,  2,  8,  4 },
        {  3, 15,  0,  6, 10,  1, 13,  8,  9,  4,  5, 11, 12,  7,  2, 14 }
    },
    {
        {  2, 12,  4,  1,  7, 10, 11,  6,  8,  5,  3, 15, 13,  0, 14,  9 },
        { 14, 11,  2, 12,  4,  7, 13,  1,  5,  0, 15, 10,  3,  9,  8,  6 },
        {  4,  2,  1, 11, 10, 13,  7,  8, 15,  9, 12,  5,  6,  3,  0, 14 },
        { 11,  8, 12,  7,  1, 14,  2, 13,  6, 15,  0,  9, 10,  4,  5,  3 }
    },
    {
        { 12,  1, 10, 15,  9,  2,  6,  8,  0, 13,  3,  4, 14,  7,  5, 11 },
        { 10, 15,  4,  2,  7, 12,  9,  5,  6,  1, 13, 14,  0, 11,  3,  8 },
        {  9, 14, 15,  5,  2,  8, 12,  3,  7,  0,  4, 10,  1, 13, 11,  6 },
        {  4,  3,  2, 12,  9,  5, 15, 10, 11, 14,  1,  7,  6,  0,  8, 13 }
    },
    {
        {  4, 11,  2, 14, 15,  0,  8, 13,  3, 12,  9,  7,  5, 10,  6,  1 },
        { 13,  0, 11,  7,  4,  9,  1, 10, 14,  3,  5, 12,  2, 15,  8,  6 },
        {  1,  4, 11, 13, 12,  3,  7, 14, 10, 15,  6,  8,  0,  5,  9,  2 },
        {  6, 11, 13,  8,  1,  4, 10,  7,  9,  5,  0, 15, 14,  2,  3, 12 }
    },
    {
        { 13,  2,  8,  4,  6, 15, 11,  1, 10,  9,  3, 14,  5,  0, 12,  7 },
        {  1, 15, 13,  8, 10,  3,  7,  4, 12,  5,  6, 11,  0, 14,  9,  2 },
        {  7, 11,  4,  1,  9, 12, 14,  2,  0,  6, 10, 13, 15,  3,  5,  8 },
        {  2,  1, 14,  7,  4, 10,  8, 13, 15, 12,  9,  0,  3,  5,  6, 11 }
    }
};
__constant__ int dP[32] = {
    16,  7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26,  5, 18, 31, 10,
     2,  8, 24, 14, 32, 27,  3,  9,
    19, 13, 30,  6, 22, 11,  4, 25
};
__constant__ int dFP[64] = {
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41,  9, 49, 17, 57, 25
};

// flag to exit threads
__device__ int work = 1;

__device__ __host__ uint64_t fun(const uint64_t R, const uint64_t K, const int E[48], const int S[8][4][16], const int P[32]) {
    int i;
    uint64_t KER = 0, tmp = 0, result = 0;
    uint8_t row = 0, col = 0;

    for (i = 0; i < 48; i++) {
        KER |= ((R >> (64 - E[i])) & 1) << (63 - i);
    }

    KER = K ^ KER;

    for (i = 0; i < 8; i++) {
        row = ((KER & 0x8000000000000000) >> 62) | ((KER & 0x0400000000000000) >> 58);
        col =  (KER & 0x7800000000000000) >> 59;
        tmp |= (uint64_t)S[i][row][col] << (60 - 4 * i);
        KER <<= 6;
    }

    for (i = 0; i < 32; i++) {
        result |= ((tmp >> (64 - P[i])) & 1) << (63 - i);
    }

    return result;
}

__device__ __host__ void DES(uint64_t processedMessage[], const uint64_t originalMessage[], const uint64_t key, const int PC1[56], const int R[16],
                             const int PC2[48], const int IP[64], const int E[48], const int S[8][4][16], const int P[32], const int FP[64], int decrypt = 0) {
    int i, j, k;
    uint64_t tmp = 0, K[16];
    uint64_t mask, CL[17], DR[17];

    // permutating key
    for (i = 0; i < 56; i++) {
        tmp |= ((key >> (64 - PC1[i])) & 1) << (63 - i);
    }

    // shifting
    CL[0] =  tmp & 0xfffffff000000000;
    DR[0] = (tmp & 0x0000000fffffff00) << 28;
    for (i = 1; i < 17; i++) {
        k = R[i - 1];
        mask = k == 1 ? 0x8000000000000000 : 0xc000000000000000;
        CL[i] = ((CL[i - 1] & ~mask) << k) | ((CL[i - 1] & mask) >> (28 - k));
        DR[i] = ((DR[i - 1] & ~mask) << k) | ((DR[i - 1] & mask) >> (28 - k));
    }

    // permutating keys
    for (i = 0; i < 16; i++) {
        tmp = CL[i + 1] | (DR[i + 1] >> 28);
        k = decrypt != 0 ? 15 - i : i;
        K[k] = 0;
        for (j = 0; j < 48; j++) {
            K[k] |= ((tmp >> (64 - PC2[j])) & 1) << (63 - j);
        }
    }

    for (i = 0; i < MSG_BLOCKS; i++) {
        // permutating message
        tmp = 0;
        for (j = 0; j < 64; j++) {
            tmp |= ((originalMessage[i] >> (64 - IP[j])) & 1) << (63 - j);
        }

        // applying magic function
        CL[0] =  tmp & 0xffffffff00000000;
        DR[0] = (tmp & 0x00000000ffffffff) << 32;
        for (j = 1; j < 17; j++) {
            CL[j] = DR[j - 1];
            DR[j] = CL[j - 1] ^ fun(DR[j - 1], K[j - 1], E, S, P);
        }

        // assembling message
        tmp = DR[16] | (CL[16] >> 32);
        processedMessage[i] = 0;
        for (j = 0; j < 64; j++) {
            processedMessage[i] |= ((tmp >> (64 - FP[j])) & 1) << (63 - j);
        }
    }
}

__global__ void desKernel(const uint64_t originalMessage[], const uint64_t encryptedMessage[], uint64_t decryptedMessage[], const int knownLeadingZeros) {
    uint64_t mask = 0x000000000000007f;
    uint64_t tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint64_t baseKey = ((tid & (mask << 14)) << 3) | ((tid & (mask << 7)) << 2) | ((tid & mask) << 1);
    uint64_t currentKey, currentMessage[MSG_BLOCKS];
    uint64_t i, limit = (uint64_t)1 << (35 - (knownLeadingZeros - knownLeadingZeros / 8));
    int j, ok;
    
    for (i = 0; i < limit && work == 1; i++) {
        currentKey = ((((i & (mask << 28)) << 5) | ((i & (mask << 21)) << 4) | ((i & (mask << 14)) << 3) | ((i & (mask << 7)) << 2) | ((i & mask)) << 1) << 24) | baseKey;
        DES(currentMessage, encryptedMessage, currentKey, dPC1, dR, dPC2, dIP, dE, dS, dP, dFP, 1);

        ok = 1;
        for (j = 0; j < MSG_BLOCKS; j++) {
            if (currentMessage[j] != originalMessage[j]) {
                ok = 0;
                break;
            }
        }

        if (ok == 1) {
            for (j = 0; j < MSG_BLOCKS; j++) {
                decryptedMessage[j] = currentMessage[j];
            }
            decryptedMessage[MSG_BLOCKS] = currentKey;
            work = 0;
        }
    }
}

cudaError_t processWithCuda(const uint64_t originalMessage[], const uint64_t encryptedMessage[], uint64_t decryptedMessage[], const int knownLeadingZeros) {
    cudaError_t cudaStatus;
    uint64_t* devOriginalMessage;
    uint64_t* devEncryptedMessage;
    uint64_t* devDecryptedMessage;

    cudaStatus = cudaSetDevice(0);
    cudaCheckError(cudaStatus, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n")
        
    cudaStatus = cudaMalloc(&devOriginalMessage, MSG_BLOCKS * sizeof(uint64_t));
    cudaCheckError(cudaStatus, "cudaMalloc failed!\n")

    cudaStatus = cudaMalloc(&devEncryptedMessage, MSG_BLOCKS * sizeof(uint64_t));
    cudaCheckError(cudaStatus, "cudaMalloc failed!\n")

    cudaStatus = cudaMalloc(&devDecryptedMessage, (MSG_BLOCKS + 1) * sizeof(uint64_t));
    cudaCheckError(cudaStatus, "cudaMalloc failed!\n")

    cudaStatus = cudaMemcpy(devOriginalMessage, originalMessage, MSG_BLOCKS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus, "cudaMemcpy failed!\n")

    cudaStatus = cudaMemcpy(devEncryptedMessage, encryptedMessage, MSG_BLOCKS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus, "cudaMemcpy failed!\n")

    desKernel<<<CUDA_BLOCKS, BLOCK_SIZE>>>(devOriginalMessage, devEncryptedMessage, devDecryptedMessage, knownLeadingZeros);

    cudaStatus = cudaGetLastError();
    cudaCheckError(cudaStatus, "desKernel launch failed!\n")

    cudaStatus = cudaDeviceSynchronize();
    cudaCheckError(cudaStatus, "cudaDeviceSynchronize failed after launching desKernel!\n")

    cudaStatus = cudaMemcpy(decryptedMessage, devDecryptedMessage, (MSG_BLOCKS + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus, "cudaMemcpy failed!\n")
        
Error:
    cudaFree(devOriginalMessage);
    cudaFree(devEncryptedMessage);
    cudaFree(devDecryptedMessage);

    return cudaStatus;
}

void printmsg(uint64_t v) {
    int i, j;
    for (i = 0; i < 16; i++) {
        j = v >> (60 - 4 * i) & 0xf;
        printf("%c", j < 10 ? j + '0' : j + 'A' - 10);
    }
}

void printbits(uint64_t v) {
    int i;
    for (i = 0; i < 64; i++) {
        printf("%c", (v >> (63 - i)) & 1 ? '1' : '0');
    }
    printf("\n");
}

int main() {
    cudaError_t cudaStatus;
    clock_t start, end;
    int i;

    uint64_t  originalMessage[MSG_BLOCKS] = { 0x0123456789ABCDEF };
    uint64_t encryptedMessage[MSG_BLOCKS];
    uint64_t decryptedMessage[MSG_BLOCKS + 1]; // one extra cell to know decrypted key
    uint64_t key = 0b0000000000000000000000000000000000000010111111101111111011111110;

    // encrypt message
    DES(encryptedMessage, originalMessage, key, hPC1, hR, hPC2, hIP, hE, hS, hP, hFP, 0);

    printf(" Original     key: ");
    printmsg(key);

    printf("\n Original message: ");
    for (i = 0; i < MSG_BLOCKS; i++) {
        printmsg(originalMessage[i]);
        printf(" ");
    }
    printf("\nEncrypted message: ");
    for (i = 0; i < MSG_BLOCKS; i++) {
        printmsg(encryptedMessage[i]);
        printf(" ");
    }

    start = clock();
    printf("\n\nCracking DES on GPU starting...\n");

    // copy data to GPU and perform calculations
    cudaStatus = processWithCuda(originalMessage, encryptedMessage, decryptedMessage, 38);
    cudaCheckError(cudaStatus, "processWithCuda failed!\n")

    end = clock();
    printf("Computing finished\nElapsed time (GPU): %f seconds\n", (float)(end - start) / CLOCKS_PER_SEC);

    printf("\n  Cracked     key: ");
    printmsg(decryptedMessage[MSG_BLOCKS]);

    printf("\nDecrypted message: ");
    for (i = 0; i < MSG_BLOCKS; i++) {
        printmsg(decryptedMessage[i]);
        printf(" ");
    }
    printf("\n");

Error:
    if (cudaStatus != cudaSuccess) {
        return 1;
    }

    getchar();
    return 0;
}
