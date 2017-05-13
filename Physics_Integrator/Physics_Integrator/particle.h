/*
* FILE : particle.h
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* TParticle class of the simulation, mostly just holds basic values about the particles information
*/

#ifndef PARTICLE_H
#define PARTICLE_H

// OpenGL Imports
#include <glm.hpp>
#include "Integrator.h"

class Particle {
public:
	Particle();
	Particle(glm::vec3 p);
	~Particle();

	// Particle Step Functions
	void update(double time_step, glm::vec3 global_forces, Integrator *i);
	void draw();

	// Particle 
	glm::vec3 last_pos;
	glm::vec3 pos;

	glm::vec3 force;

	glm::vec3 last_vel;
	glm::vec3 vel;
	glm::vec3 acc;
	float mass;
private:
	bool init(glm::vec3 p);
};

#endif