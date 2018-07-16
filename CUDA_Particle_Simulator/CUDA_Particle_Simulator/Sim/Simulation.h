#ifndef SIMULATION_H
#define SIMULATION_H

// Class Dependencies
//#include "particle.h"
//#include "plane.h"

// OpenGL Imports
#include<GL/glew.h>
#include<glm/glm.hpp>

// Other Library
#ifdef _WIN32
	#include <Windows.h>
#else
	#include <time.h>
#endif

#include <vector>
#include <limits.h>

#include "../Plane.h"
#include "../Graphics/Drawable.h"

using namespace glm;

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
	void update();
	void render();

	// Getter Functions
	double getSimFrameTime();
	int getSimCurrState();

	// Print Functions
	void printControls();

	//TODO : Move back to private after hitable class created
	vector<Drawable*> _scene_objects;

private:
	// Simulation Environment Variables
	//const int GRID_SIZE = 5;
	//glm::vec3 a_gravity = glm::vec3(0.0f, -3, 0.0f);

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
	int sim_state = NOT_INITIALIZED;
	//Integrator integrator;
	//vector<Particle*> particles;


	// Private Simulation Functions
	bool init();
	void clean();

	//collisions
	float distFromPlane(const vec3 pos, const Plane* plane);
	//vector<Plane*> getCollisions(Particle* p, vector<Plane*> planes);
	vec3 reflect(const vec3 pos, const Plane* plane);

	//Plane * getClosestCollisionPlane(const Particle * proj_p, const Particle * orig_p, const vector<Plane*> planes);
	//float getCollisionTimeFromStartOfTimeStep(const Particle * p, const Plane * pl, const float t_step);
	//float getCollisionFrac(const Particle * p, const Plane * pl);
	//Particle projectParticleAtSubTimeStep(const Particle * p, const Plane * pl, const float t_step_frac);

	int getSign(float val);

};

#endif
