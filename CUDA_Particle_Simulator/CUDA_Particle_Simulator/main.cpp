#include <iostream>
using namespace std;

#include "Graphics/ViewController.h"
#include "Sim/Simulation.h"
#include "Utils/Config.h"
#include <config4cpp/Configuration.h>
using namespace config4cpp;

int n_particles = 1000;

int main(int argc, char * argv[])
{
	configData cfgData;
	cfgData.init((char*)"config.cfg");

	if (argc > 1) n_particles = atoi(argv[1]);

	Simulation sim(cfgData);
	ViewController vc(&sim);
	vc.run();
	return 0;
}
