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

#include "particle.h"

#include<glew.h>
#include<glm.hpp>

#include <Windows.h>
#include <vector>
const static int NOT_INITIALIZED = -1;
const static int INITIALIZED = 0;
const static int RUNNING = 1;
const static int PAUSED = 2;
const static int FINISHED = 3;
class Simulation {

	Integrator integrator;

	const int GRID_SIZE = 10;	

	float step_time = 0.0f;
	float time_modifier = 1.0f;

	int simulation_state = NOT_INITIALIZED;

	glm::vec3 f_gravity = glm::vec3(0.0f,-9.81,0.0f);

public:
	Simulation();
	~Simulation();


	bool init();
	void update(float rdt);
	void render();

	bool start();
	bool pause();
	bool end();

	double getSimFrameTime();
	int getSimCurrState();

private:
	//Simulation time
	LARGE_INTEGER sim_start_time;
	LARGE_INTEGER sim_end_time;

	LARGE_INTEGER frequency;       
	LARGE_INTEGER t1;
	LARGE_INTEGER t2;           
	double frame_time;

	vector<Particle*> particles;
};



#endif