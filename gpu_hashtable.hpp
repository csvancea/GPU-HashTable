#ifndef _HASHCPU_
#define _HASHCPU_

#include <vector>

using namespace std;

#define cudaCheckError(e) { \
	if(e!=cudaSuccess) { \
		cout << "Cuda failure " << __FILE__ << ":" << __LINE__ << ", " << cudaGetErrorString(e) << " (" << e << ")" << endl; \
		exit(0); \
	 }\
}

/**
 * Class GpuHashTable to implement functions
 */
class GpuHashTable
{
	public:
		struct kv {
			int key;
			int value;
		};

		GpuHashTable(int size);
		void reshape(int sizeReshape);

		bool insertBatch(int *keys, int* values, int numKeys);
		int* getBatch(int* key, int numItems);

		~GpuHashTable();

	private:
		/* Vector of key:value pairs */
		kv *devTable;

		/* Non-empty slots */
		unsigned int numItems;

		/* Total size (Empty + non-empty slots) */
		unsigned int size;

		/* This buffer is returned to the user in getBatch and is free'd in destructor */
		int *getBatchBuffer;
};

#endif
