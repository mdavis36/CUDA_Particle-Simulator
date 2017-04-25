/*
* FILE : particle.h
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* TParticle class of the simulation, mostly just holds basic values about the particles information
*/

#ifndef PARTICLE_H
#define PARTICLE_H

#include <glm.hpp>
#include "Integrator.h"

class Particle {

	Integrator integrator;

public:
	Particle();
	Particle(glm::vec3 p);
	~Particle();

	bool init(glm::vec3 p);
	void update(float time_step, glm::vec3 global_forces, Integrator *i);
	void draw();

	glm::vec3 last_pos;
	glm::vec3 pos;

	glm::vec3 force;

	float mass;
	glm::vec3 vel;
	glm::vec3 acc;
private:

};

#endif