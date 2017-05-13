/*
* FILE : integrator.cpp
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* This is the integrator, this is where all of the integration functions are held to be used
*/

#include"Integrator.h"

Integrator::Integrator()
{
	type = INTEGRATE_EULER;
}

Integrator::Integrator(iType type)
{
	this->type = type;
}

void Integrator::integrate_euler(vec3 * next_pos, vec3 * pos, double ts, vec3 *vel, vec3 *last_vel, vec3 acc)
{
	*last_vel = *vel;
	*vel = *last_vel + (acc * (float)ts);
	*pos = *next_pos;
	*next_pos = *pos + ((*last_vel + *vel) / 2.0f) * (float)ts;
}

void Integrator::integrate_verlet(vec3 * next_pos, vec3 *pos, vec3 *last_pos, double ts, float m, vec3 acc) {
	// Verlet integration
	*last_pos = *pos;
	*pos = *next_pos;

	// Perform the verlet integartion calculation
	// Note : Future performance increase could be done by calculating t*t/m prior to integration
	//		 However this would only be useful fopr systems with multiuple particles of the same mass

	*next_pos = (*pos + *pos) - *last_pos + (float)(ts * ts) * acc;
}

iType Integrator::getIType()
{
	return type;
}

void Integrator::setIType(iType type)
{
	this->type = type;
}

void Integrator::printIType()
{
	std::cout << "Integrating with ";
	switch (type) {
	case INTEGRATE_EULER:
		std::cout << "Euler.\n";
		break;
	case INTEGRATE_VERLET:
		std::cout << "Verlet.\n";
		break;
	default:
		std::cout << "INTEGRATOR UNDEFINED.\n";
		break;
	}
}
