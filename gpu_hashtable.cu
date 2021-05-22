#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include "test_map.hpp"
#include "gpu_hashtable.hpp"

#define THREADS_PER_BLOCK 256

#define MIN_LOAD_FACTOR 51
#define MAX_LOAD_FACTOR 80

using namespace std;

/**
 * Function computeHash
 * Returns hash of "key"
 */
__device__ unsigned int computeHash(int key)
{
	/* both a and b are prime numbers */
	return (key * 32069) % 694847539;
}

/**
 * Function constructor GpuHashTable
 * Performs init
 */
GpuHashTable::GpuHashTable(int size) : devTable(nullptr), numItems(0), size(size), getBatchBuffer(nullptr) {
	cudaError_t err;

	err = glbGpuAllocator->_cudaMalloc((void **) &devTable, size * sizeof(struct kv));
	cudaCheckError(err);

	/* set everything to KEY_INVALID */
	cudaMemset(devTable, KEY_INVALID, size * sizeof(struct kv));
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(devTable);
	if (getBatchBuffer) {
		free(getBatchBuffer);
	}
}

/**
 * Kernel reshape
 * Performs the reshape on GPU side
 */
__global__ void kernel_reshape(struct GpuHashTable::kv *oldTable, unsigned int oldSize, struct GpuHashTable::kv *newTable, unsigned int newSize) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= oldSize)
		return;

	int key = oldTable[idx].key;

	/* nothing to add */
	if (key == KEY_INVALID)
		return;
	
	int value = oldTable[idx].value;
	unsigned int hash = computeHash(key);
	int old;

	/* no stop condition: guaranteed there are enough empty spaces with key = KEY_INVALID */
	for (unsigned int i = hash % newSize; ; i = (i+1) % newSize) {
		old = atomicCAS(&newTable[i].key, KEY_INVALID, key);

		if (old == KEY_INVALID) {
			/* newTable[i].key was set previously */
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
	unsigned int numBlocks;

	err = glbGpuAllocator->_cudaMalloc((void **) &devNewTable, numBucketsReshape * sizeof(struct kv));
	cudaCheckError(err);

	/* set everything to KEY_INVALID */
	cudaMemset(devNewTable, KEY_INVALID, numBucketsReshape * sizeof(struct kv));

	/* trick to round up result of size/THREADS_PER_BLOCK */
	numBlocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	kernel_reshape<<<numBlocks, THREADS_PER_BLOCK>>>(devTable, size, devNewTable, numBucketsReshape);
	err = cudaDeviceSynchronize();
	cudaCheckError(err);

	glbGpuAllocator->_cudaFree(devTable);
	devTable = devNewTable;
	size = numBucketsReshape;
}

/**
 * Kernel insertBatch
 * Performs insertion
 */
__global__ void kernel_insertBatch(struct GpuHashTable::kv *table, unsigned int size, int *keys, int* values, int numKeys, int *insertCounter) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numKeys)
		return;

	int key = keys[idx];
	int value = values[idx];
	unsigned int hash = computeHash(key);
	int old;

	/* no stop condition: guaranteed there are enough empty spaces with key = KEY_INVALID */
	for (unsigned int i = hash % size; ; i = (i+1) % size) {
		old = atomicCAS(&table[i].key, KEY_INVALID, key);

		/* KEY_INVALID -> insert (empty space); key -> update */
		if (old == KEY_INVALID || old == key) {
			/* table[i].key was set previously */
			table[i].value = value;

			if (old == KEY_INVALID) {
				/* actual insert */
				atomicAdd(insertCounter, 1);
			}
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
	int *devKeys, *devValues, *devInsertCounter;
	int numBlocks, numInsertedKeys;
	unsigned int newLoadFactor, newSize;

	/* this block of code guarantees that load factor will always be in the required interval */
	newLoadFactor = (numItems + numKeys) * 100 / size;
	if (newLoadFactor >= MAX_LOAD_FACTOR) {
		/*
		 * (numItems + numKeys) * 100 / size = MIN_LOAD_FACTOR
		 * => newSize = ((numItems + numKeys) * 100) / MIN_LOAD_FACTOR
		 */

		newSize = ((numItems + numKeys) * 100) / MIN_LOAD_FACTOR;
		reshape(newSize);
	}

	err = glbGpuAllocator->_cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
	cudaCheckError(err);

	err = glbGpuAllocator->_cudaMalloc((void **) &devValues, numKeys * sizeof(int));
	cudaCheckError(err);

	err = glbGpuAllocator->_cudaMalloc((void **) &devInsertCounter, 1 * sizeof(int));
	cudaCheckError(err);

	cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset(devInsertCounter, 0, 1 * sizeof(int));

	/* trick to round up result of numKeys/THREADS_PER_BLOCK */
	numBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	kernel_insertBatch<<<numBlocks, THREADS_PER_BLOCK>>>(devTable, size, devKeys, devValues, numKeys, devInsertCounter);
	err = cudaDeviceSynchronize();
	cudaCheckError(err);

	cudaMemcpy(&numInsertedKeys, devInsertCounter, 1 * sizeof(int), cudaMemcpyDeviceToHost);

	glbGpuAllocator->_cudaFree(devInsertCounter);
	glbGpuAllocator->_cudaFree(devValues);
	glbGpuAllocator->_cudaFree(devKeys);

	/* can't increment with numKeys since some operations might actually be updates */
	numItems += numInsertedKeys;
	return true;
}

/**
 * Kernel getBatch
 * Performs retrieval
 */
__global__ void kernel_getBatch(struct GpuHashTable::kv *table, unsigned int size, int *keysValues, int numKeys) {
	unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numKeys)
		return;

	/* keysValues acts as an input+output buffer */
	int key = keysValues[idx];
	unsigned int hash = computeHash(key);
	unsigned int maxSteps = size;

	/* if we can't find a match in "size" steps -> abort (key not found) */
	for (unsigned int i = hash % size; maxSteps != 0; i = (i+1) % size, maxSteps--) {
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

	kernel_getBatch<<<numBlocks, THREADS_PER_BLOCK>>>(devTable, size, devKeysValues, numKeys);

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
