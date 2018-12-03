/*
* FILE : simulation.cpp
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* This is the primary simulation class, it handles simulation time, updates, simulation state and
* rendering of the simulation environment and simulation objects
*/

#include <iostream>
#include <omp.h>
#include "Simulation.h"

using namespace std;

// ********************************************************************************
// *                               -- PUBLIC --                                   *
// ********************************************************************************

// -------------------- Simulation Constructors and Destructor --------------------

Simulation::Simulation()
{
	Simulation(5);
}

Simulation::Simulation(int n)
{
	num_particles = n;
	if (!init())
	{
		cout << "Failed to initialize simulation.\n";
		exit(-4);
	}
}

Simulation::~Simulation()
{
	//particles.clear();
}


// -------------------- Simulation Control Functions --------------------

bool Simulation::start()
{
	if (sim_state == INITIALIZED)
	{
		cout << "Starting Simulation of "<< num_particles << " falling particles." << endl;
		//integrator.printIType();
		sim_state = RUNNING;

		sim_time_accu = 0;

		#ifdef _WIN32
			// Get ticks per second.
			QueryPerformanceFrequency(&frequency);
			// Start timer.
			QueryPerformanceCounter(&t1);
		#else
			clock_gettime(CLOCK_REALTIME, &t1);
		#endif

		return true;
	}
	return false;
}

bool Simulation::pause()
{
	if (sim_state == PAUSED) {
		sim_state = RUNNING;
		cout << "Simulation Running.\n";
		return true;
	}
	else if (sim_state == RUNNING) {
		sim_state = PAUSED;
		cout << "Simulation Paused.\n";
		return true;
	}
	return false;
}

bool Simulation::end()
{
	if (sim_state >= RUNNING && sim_state != FINISHED) {
		cout << "Simulation has ended.\n";
	}
	sim_state = FINISHED;
	return false;
}

bool Simulation::reset()
{
	end();
	cout << "Resetting Simulation.\n";
	printControls();
	clean();
	init();
	return false;
}


// -------------------- Simulation Step Functions --------------------

void Simulation::update()
{
	if (sim_state == RUNNING)
	{
		//EulerStep(_p_sys, dt);
		//RK4(_p_sys, &_s_obj->_polygons, dt);
		cuRK4(_p_sys, dt);
	}
}

void Simulation::render() {}


// -------------------- Getter Functions --------------------

double Simulation::getSimFrameTime() { return frame_time; }

int Simulation::getSimCurrState() {	return sim_state; }


// -------------------- GPrint Functions --------------------

void Simulation::printControls()
{
	cout << "---------- Simulation Controls ----------\n";
	cout << "\t s : Start the simulation.\n";
	cout << "\t p : Pause and Un-Pause the simulation.\n";
	cout << "\t r : Reset the simulation.\n\n";
}

// ********************************************************************************



// ********************************************************************************
// *                               -- Private --                                  *
// ********************************************************************************

bool Simulation::init()
{
	// This is where all simulation object will be initialized to set up the simulation.
	//integrator.setIType(INTEGRATE_VERLET);
	//integrator.setIType(INTEGRATE_EULER);

	_p_sys = new ParticleSystem(num_particles, PARTICLE_SPHERE);
	_ground = new Plane(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 0.0), 20, 20);
	_s_obj = new SceneObject("../Resources/smaug.obj");

	std::cout << "Vertices Count  : " << _s_obj->_vertices.size() << std::endl;
	std::cout << "Polygon Count   : " << _s_obj->_polygons.size() << std::endl;

	OTH = new OctTreeHandler();
	OTH->clear();
	OTH->buildTree(_s_obj->_polygons, Volume(glm::vec3(-20,-20,-20), glm::vec3(20,20,20)));

	_drawable_objects.clear();
	_drawable_objects.push_back(_ground);
	_drawable_objects.push_back(_p_sys);
	_drawable_objects.push_back(_s_obj);
	_drawable_objects.push_back(OTH);

	sim_state = INITIALIZED;
	return true;
}


void Simulation::clean()
{
	while (!_drawable_objects.empty()) {
		delete _drawable_objects.back();
		_drawable_objects.pop_back();
	}
}

float Simulation::distFromPlane(const vec3 pos, const Plane * plane)
{
	return dot(plane->_normal, (pos - plane->_center));
}

vec3 Simulation::reflect(const vec3 pos, const Plane * plane)
{
	return pos + 2 * (-distFromPlane(pos, plane) / dot(plane->_normal, plane->_normal)) * plane->_normal;
}

int Simulation::getSign(float val)
{
	if (val > 0) return 1;
	if (val == 0) return 0;
	if (val < 0) return -1;
	return BILLION;
}

// ********************************************************************************
