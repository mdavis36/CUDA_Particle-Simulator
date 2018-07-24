#include <iostream>
using namespace std;

#include "Graphics/ViewController.h"

#include "Sim/Simulation.h"


#ifdef RUN_CUDA

#include "CUDA/kernel.h"
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

#endif

int n_particles = 1000;

int main(int argc, char * argv[])
{

#ifdef RUN_CUDA
	if (!runCUDAExample()) return -1;
#endif

	if (argc > 1) n_particles = atoi(argv[1]);

	Simulation sim(n_particles);
	ViewController vc(&sim);
	vc.run();
	return 0;
}
