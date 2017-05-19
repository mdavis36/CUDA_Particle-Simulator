#include "kernel.h"
#include <iostream>
#include <stdio.h>

int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	const unsigned int arrSize = 160;
	int arr[arrSize];
	int result = -1;
	for (int i = 0; i < arrSize; i++)
	{
		arr[i] = rand() % 100000;
	}

	std::cout << "Initializing random array ... \n";
	for (int i = 0; i < arrSize; i++)
	{
		std::cout << arr[i] << std::endl;
	}

	printf("Smallest value : %d\n", result);

	cudaStatus = findMinValue(&result, arr, arrSize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "findminVal failed!");
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	system("PAUSE");
	return 0;
}