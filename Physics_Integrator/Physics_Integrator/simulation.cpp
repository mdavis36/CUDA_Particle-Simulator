/*
* FILE : simulation.cpp
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* This is the primary simulation class, it handles simulation time, updates, simulation state and
* rendering of the simulation environment and simulation objects
*/
#include <iostream>
using namespace std;

#include "simulation.h"

Simulation::Simulation()
{
	if (!init()) {
		cout << "Failed to initialize simulation.\n";
		exit(-4);
	}
	simulation_state = INITIALIZED;
}

Simulation::~Simulation()
{
	particles.clear();
}

bool Simulation::init()
{
	//This is where all simulation object will be initialized to set up the simulation
	glm::vec3 ranPos;

	for (int i = 0; i < 100000; i++) {
		ranPos.x = -10 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (20)));
		ranPos.z = -10 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (20)));
		ranPos.y = 3 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (10)));
		particles.push_back(new Particle(ranPos));
	}

	return true;
}

void Simulation::update(float rdt)
{
	if (simulation_state == RUNNING) {
		//Using high res. counter
		QueryPerformanceCounter(&t2);
		// compute and print the elapsed time in millisec
		frame_time = ((t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart) / 1000.0;
		t1 = t2;

		// update will be the call for each simulation time step. This is where each object will be updated
		bool isSimOver = true;
		for (Particle* p : particles) {
			p->update((double)frame_time, glm::vec3(0.0f, 0.0f, 0.0f), &integrator);

			if (p->pos.y > 0) {
				isSimOver = false;
			}
		}
		if (isSimOver) end();
	}
}

void Simulation::render()
{
	if (simulation_state >= INITIALIZED) {
		//this will handle the simulation rendering 
		glBegin(GL_LINES);

		for (int i = -GRID_SIZE; i <= GRID_SIZE; i++)
		{
			if (i == 0)
				glColor3f(0.0f, 1.0f, 1.0f);
			else
				glColor3f(0.5f, 0.5f, 0.5f);

			glVertex3f((float)i, 0, (float)-GRID_SIZE);
			glVertex3f((float)i, 0, (float)GRID_SIZE);

			glVertex3f((float)-GRID_SIZE, 0, (float)i);
			glVertex3f((float)GRID_SIZE, 0, (float)i);
		}
		glEnd();

		//Render all simulation objects here. 
		for (Particle* p : particles) {
			p->draw();
		}
	}
}

bool Simulation::start()
{
	// get ticks per second
	QueryPerformanceFrequency(&frequency);
	// start timer
	QueryPerformanceCounter(&t1);
	sim_start_time = t1;

	simulation_state = RUNNING;

	return false;
}

bool Simulation::pause()
{
	return false;
}

bool Simulation::end()
{
	QueryPerformanceCounter(&sim_end_time);
	cout << "Simulation ran for : " << ((sim_end_time.QuadPart - sim_start_time.QuadPart) * 1000.0 / frequency.QuadPart) / 1000 << " seconds\n";
	simulation_state = FINISHED;
	return false;
}

double Simulation::getSimFrameTime()
{
	return frame_time;
}

int Simulation::getSimCurrState()
{
	return simulation_state;
}
