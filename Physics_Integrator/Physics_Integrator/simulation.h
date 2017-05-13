/*
* FILE : simulation.h
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* Header file for the simulation class. This class will handle the general simulation tasks. such as time steps ,
* calling renders and recording statistics about the simulation
*/

#ifndef SIMULATION_H
#define SIMULATION_H

// Class Dependencies
#include "particle.h"
#include "plane.h"

// OpenGL Imports
#include<glew.h>
#include<glm.hpp>

// Other Library
#include <Windows.h>
#include <vector>

//Simulation States
const static int NOT_INITIALIZED = -1;
const static int INITIALIZED = 0;
const static int RUNNING = 1;
const static int PAUSED = 2;
const static int FINISHED = 3;

class Simulation 
{
public:
	Simulation();
	~Simulation();

	// Simulation Control Functions
	bool start();
	bool pause();
	bool end();
	bool reset();

	// Simulation Step Functions
	void update(float rdt);
	void render();

	// Getter Functions
	double getSimFrameTime();
	int getSimCurrState();

	// Print Functions
	void printControls();

private:
	// Simulation Environment Variables
	const int GRID_SIZE = 5;
	glm::vec3 a_gravity = glm::vec3(0.0f, -9.81, 0.0f);

	// Simulation Time Variables
	LARGE_INTEGER frequency;       
	LARGE_INTEGER t1;
	LARGE_INTEGER t2;           
	double frame_time;
	double sim_time_accu;

	// Simulation Variables
	const int PARTICLE_COUNT = 6000;
	int sim_state = NOT_INITIALIZED;
	Integrator integrator;
	vector<Particle*> particles;
	Plane* plane;
	
	bool init();
	void clean();
};

#endif