#include <iostream>
using namespace std;



#include "Graphics/ViewController.h"
#include "CUDA/kernel.h"
#include "Sim/Simulation.h"


bool runCUDAExample(bool quietMode = false)
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return false;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return false;
	}
	if (!quietMode)cout << "CUDA Example run succesfully." << endl;
	return true;
}


int main(int argc, char * argv[])
{
	if (!runCUDAExample()) return -1;

	Simulation sim;
	ViewController vc(&sim);
	vc.run();

	system("pause");
	return 0;
}
