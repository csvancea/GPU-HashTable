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
GpuHashTable::GpuHashTable(int size) : table(nullptr), count(0), size(size) {
	glbGpuAllocator->_cudaMalloc((void **) &table, size * sizeof(struct kv));
	cudaCheckError();

	/* set everything to KEY_INVALID */
	cudaMemset(table, KEY_INVALID, size * sizeof(struct kv));
}

/**
 * Function desctructor GpuHashTable
 */
GpuHashTable::~GpuHashTable() {
	glbGpuAllocator->_cudaFree(table);
}

/**
 * Function reshape
 * Performs resize of the hashtable based on load factor
 */
void GpuHashTable::reshape(int numBucketsReshape) {
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
	int *devKeys, *devValues;
	int numBlocks;

	if (count + numKeys > size) {
		int newSize = size;

		do {
			newSize *= 2;
		} while (count + numKeys > newSize);

		reshape(newSize);
	}

	/* TODO: try cudaMallocManaged instead of cudaMalloc + cudaMemcpy */
	glbGpuAllocator->_cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
	cudaCheckError();

	glbGpuAllocator->_cudaMalloc((void **) &devValues, numKeys * sizeof(int));
	cudaCheckError();

	cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(devValues, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	/* trick to round up result of numKeys/THREADS_PER_BLOCK */
	numBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	kernel_insertBatch<<<numBlocks, THREADS_PER_BLOCK>>>(table, size, devKeys, devValues, numKeys);
	cudaDeviceSynchronize();
	cudaCheckError();

	glbGpuAllocator->_cudaFree(devValues);
	glbGpuAllocator->_cudaFree(devKeys);

	count += numKeys;
	return true;
}

/**
 * Kernel getBatch
 * Performs retrieval
 */
__global__ void kernel_getBatch(struct GpuHashTable::kv *table, int size, int *keys, int* values, int numKeys) {
	int idx = threadIdx.x + blockDim.x * blockIdx.x;

	if (idx >= numKeys)
		return;

	int key = keys[idx];
	int hash = computeHash(key);
	int maxSteps = size;

	/* if we can't find a match in "size" steps -> abort (key not found) */
	for (int i = hash % size; maxSteps != 0; i = (i+1) % size, maxSteps--) {
		if (table[i].key == key) {
			values[idx] = table[i].value;
			break;
		}
	}
}

/**
 * Function getBatch
 * Gets a batch of key:value, using GPU
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *devKeys, *sharedValues;
	int numBlocks;

	/* TODO: try cudaMallocManaged instead of cudaMalloc + cudaMemcpy */
	glbGpuAllocator->_cudaMalloc((void **) &devKeys, numKeys * sizeof(int));
	cudaCheckError();

	/* TODO: fix this leak */
	glbGpuAllocator->_cudaMallocManaged((void **) &sharedValues, numKeys * sizeof(int));
	cudaCheckError();

	cudaMemcpy(devKeys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

	/* trick to round up result of numKeys/THREADS_PER_BLOCK */
	numBlocks = (numKeys + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	kernel_getBatch<<<numBlocks, THREADS_PER_BLOCK>>>(table, size, devKeys, sharedValues, numKeys);
	cudaDeviceSynchronize();
	cudaCheckError();

	glbGpuAllocator->_cudaFree(devKeys);
	return sharedValues;
}
