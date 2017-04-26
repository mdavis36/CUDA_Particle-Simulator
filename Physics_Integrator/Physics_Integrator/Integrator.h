/*
* FILE : integrator.h
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* This is the integrator, this is where all of the integration functions are held to be used
*/

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include<iostream>
using namespace std;

#include<glm.hpp>

//class Particle;
class Integrator {
public:
	Integrator();
	void integrate_verlet(glm::vec3 *pos, glm::vec3 *last_pos, double ts, float m, glm::vec3 force);
	void integrate_euler();
};

#endif