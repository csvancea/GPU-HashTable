#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

#define THREADS_PER_BLOCK 256

using namespace std;

/**
 * Function computeHash
 * Returns hash of "key"
 */
__device__ int computeHash(int key)
{
	/* TODO: proper hash */
	return key;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 */
GpuHashTable::GpuHashTable(int size) : table(nullptr), count(0), size(size), getBatchBuffer(nullptr) {
	cudaError_t err;

	err = glbGpuAllocator->_cudaMalloc((void **) &table, size * sizeof(struct kv));
	cudaCheckError(err);

	/* set everything to KEY_INVALID */
	cudaMemset(table, KEY_INVALID, size * sizeof(struct kv));
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(table);
	if (getBatchBuffer) {
		free(getBatchBuffer);
	}
}

/**
 * Kernel reshape
 * Performs the reshape on GPU side
 */
__global__ void kernel_reshape(struct GpuHashTable::kv *oldTable, int oldSize, struct GpuHashTable::kv *newTable, int newSize) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= oldSize)
		return;

	int key = oldTable[idx].key;

	/* nothing to add */
	if (key == KEY_INVALID)
		return;
	
	int value = oldTable[idx].value;
	int hash = computeHash(key);
	int old;

	/* no stop condition: guaranteed there are enough empty spaces with key = KEY_INVALID */
	for (int i = hash % newSize; ; i = (i+1) % newSize) {
		old = atomicCAS(&newTable[i].key, KEY_INVALID, key);

		if (old == KEY_INVALID) {
			/* table[i].key was set previously */
			newTable[i].value = value;
			break;
		}
	}
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	cudaError_t err;
	kv *devNewTable;
	int numBlocks;

	err = glbGpuAllocator->_cudaMalloc((void **) &devNewTable, numBucketsReshape * sizeof(struct kv));
	cudaCheckError(err);

	/* set everything to KEY_INVALID */
	cudaMemset(devNewTable, KEY_INVALID, numBucketsReshape * sizeof(struct kv));

	/* trick to round up result of size/THREADS_PER_BLOCK */
	numBlocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	kernel_reshape<<<numBlocks, THREADS_PER_BLOCK>>>(table, size, devNewTable, numBucketsReshape);
	err = cudaDeviceSynchronize();
	cudaCheckError(err);

	glbGpuAllocator->_cudaFree(table);
	table = devNewTable;
	size = numBucketsReshape;
}

/**
 * Kernel insertBatch
 * Performs insertion
 */
__global__ void kernel_insertBatch(struct GpuHashTable::kv *table, int size, int *keys, int* values, int numKeys) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numKeys)
		return;

	int key = keys[idx];
	int value = values[idx];
	int hash = computeHash(key);
	int old;

	/* no stop condition: guaranteed there are enough empty spaces with key = KEY_INVALID */
	for (int i = hash % size; ; i = (i+1) % size) {
		old = atomicCAS(&table[i].key, KEY_INVALID, key);

		/* KEY_INVALID -> insert (empty space); key -> update */
		if (old == KEY_INVALID || old == key) {
			/* table[i].key was set previously */
			table[i].value = value;
			break;
		}
	}
}

/**
 * Function insertBatch
 * Inserts a batch of key:value, using GPU and wrapper allocators
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	cudaError_t err;
	int *devKeys, *devValues;
	int numBlocks;

	if (count + numKeys > size) {
		int newSize = size;

		do {
			newSize *= 2;
		} while (count + numKeys > newSize);

		reshape(newSize);
	}

	err = glbGpuAllocator->_cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
	cudaCheckError(err);

	err = glbGpuAllocator->_cudaMalloc((void **) &devValues, numKeys * sizeof(int));
	cudaCheckError(err);

	cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	/* trick to round up result of numKeys/THREADS_PER_BLOCK */
	numBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	kernel_insertBatch<<<numBlocks, THREADS_PER_BLOCK>>>(table, size, devKeys, devValues, numKeys);
	err = cudaDeviceSynchronize();
	cudaCheckError(err);

	glbGpuAllocator->_cudaFree(devValues);
	glbGpuAllocator->_cudaFree(devKeys);

	count += numKeys;
	return true;
}

/**
 * Kernel getBatch
 * Performs retrieval
 */
__global__ void kernel_getBatch(struct GpuHashTable::kv *table, int size, int *keysValues, int numKeys) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numKeys)
		return;

	/* keysValues acts as an input+output buffer */
	int key = keysValues[idx];
	int hash = computeHash(key);
	int maxSteps = size;

	/* if we can't find a match in "size" steps -> abort (key not found) */
	for (int i = hash % size; maxSteps != 0; i = (i+1) % size, maxSteps--) {
		if (table[i].key == key) {
			keysValues[idx] = table[i].value;
			break;
		}
	}
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 * 
 * Note: It'd been better if the caller passed the output vector for values
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	cudaError_t err;
	int *devKeysValues;
	int numBlocks;

	err = glbGpuAllocator->_cudaMalloc((void **) &devKeysValues, numKeys * sizeof(int));
	cudaCheckError(err);

	cudaMemcpy(devKeysValues, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	/* trick to round up result of numKeys/THREADS_PER_BLOCK */
	numBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	kernel_getBatch<<<numBlocks, THREADS_PER_BLOCK>>>(table, size, devKeysValues, numKeys);

	/* alloc the returned vector while GPU is retrieving the values */
	getBatchBuffer = (int *)realloc(getBatchBuffer, numKeys * sizeof(int));
	DIE(getBatchBuffer == nullptr, "malloc failed");

	/* wait for the CUDA kernel to finish */
	err = cudaDeviceSynchronize();
	cudaCheckError(err);

	cudaMemcpy(getBatchBuffer, devKeysValues, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	glbGpuAllocator->_cudaFree(devKeysValues);
	return getBatchBuffer;
}
