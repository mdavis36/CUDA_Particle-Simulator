#include "kernel.h"
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <vector>

cudaError_t cudaStatus;

int bruteForceMaxValue(int * arr, int size)
{
	int res = 0;
	for (int i = 0; i < size; i++)
	{
		if (arr[i] > res) res = arr[i];
	}
	return res;
}

int gpuAcceleratedMaxValue(int * i_arr, int * o_arr, unsigned int size)
{
	unsigned int b_size = 512;
	unsigned int g_size = size / b_size + 1;
	
	std::vector<int> results;

	for (int i = 0; i < g_size; i++)
	{
		cudaStatus = findMaxValue(i_arr, o_arr, size, b_size, g_size);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "findmaxVal failed!");
			system("PAUSE");
			return 1;
		}
	}
	int res = 0;
	for (int i = 0; i < g_size; i ++)
	{
		if (o_arr[i] > res) res = o_arr[i];
	}

	return res;
}

int main()
{
	// ------------ ADD TWO ARRAYS ------------ 

	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// ----------------------------------------- 


	// ------------ MAX VALUE OF ARRAY ------------ 

	srand(time(NULL));

	const unsigned int arrSize = 6000;
	int arr[arrSize];
	int result[arrSize];

	std::cout << "Initializing random array ... \n";
	for (int i = 0; i < arrSize; i++)
	{
		arr[i] = rand() % 100000;
	}

	printf("Largest value : %d\n", bruteForceMaxValue(arr, arrSize));
	printf("Largest value : %d\n", gpuAcceleratedMaxValue(arr, result, arrSize));

	// ----------------------------------------- 


	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		system("PAUSE");
		return 1;
	}

	system("PAUSE");
	return 0;
}