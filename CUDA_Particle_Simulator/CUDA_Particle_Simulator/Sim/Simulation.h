#ifndef SIMULATION_H
#define SIMULATION_H

#define UPDATES_PER_SECOND 2000
// Class Dependencies

// OpenGL Imports
#include<GL/glew.h>
#include<glm/glm.hpp>

// Other Library
#ifdef _WIN32
	#include <Windows.h>
#else
	#include <time.h>
#endif

#include <limits.h>
#include <vector>

#include "Plane.h"
#include "ParticleSystem.h"
#include "ParticleHandler.h"

#include "../Graphics/Drawable.h"
#include "../Utils/Utils.h"

using namespace glm;
using namespace ParticleHandler;


class Simulation
{
public:
	const struct timespec TIME_STEP = {0, BILLION / UPDATES_PER_SECOND};
	const float dt = (float)TIME_STEP.tv_nsec / (float)BILLION;
	int sim_state = NOT_INITIALIZED;

	Simulation();
	~Simulation();

	// Simulation Control Functions
	bool start();
	bool pause();
	bool end();
	bool reset();

	// Simulation Step Functions
	void update(int fn);
	void render();

	// Getter Functions
	double getSimFrameTime();
	int getSimCurrState();

	// Print Functions
	void printControls();


	ParticleSystem *_p_sys;
	//TODO : Move back to private after hitable class created
	vector<Drawable*> _scene_objects;


private:
	// Simulation Environment Variables

	// Simulation Time Variables
	#ifdef _WIN32
		LARGE_INTEGER frequency;
		LARGE_INTEGER t1;
		LARGE_INTEGER t2;
	#else
		struct timespec t1, t2;
	#endif

	double frame_time;
	double sim_time_accu;

	// Simulation Variables
	const int PARTICLE_COUNT = 500;

	// Private Simulation Functions
	bool init();
	void clean();

	//collisions
	float distFromPlane(const vec3 pos, const Plane* plane);
	vec3 reflect(const vec3 pos, const Plane* plane);


	int getSign(float val);

};

#endif
