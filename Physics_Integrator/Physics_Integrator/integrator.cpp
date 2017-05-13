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

void Integrator::integrate_euler(vec3 * pos, vec3 * last_pos, double ts, vec3 *vel, vec3 *last_vel, vec3 acc)
{
	*last_vel = *vel;
	*vel = *last_vel + (acc * (float)ts);
	*last_pos = *pos;
	*pos = *last_pos + ((*last_vel + *vel) / 2.0f) * (float)ts;
}

void Integrator::integrate_verlet(vec3 *pos, vec3 *last_pos, double ts, float m, vec3 acc) {
	// Verlet integration
	// Save the current position into a buffer
	vec3 buffer = *pos;

	// Perform the verlet integartion calculation
	// Note : Future performance increase could be done by calculating t*t/m prior to integration
	//		 However this would only be useful fopr systems with multiuple particles of the same mass
	//cout << "Int : pos : " << pos << "   last_pos : " << last_pos << "   ts : " << ts << "   m : " << m << endl;
	*pos = (*pos + *pos) - *last_pos + (float)(ts * ts) * acc;

	// Swap buffer so the old position becomes the last position in poreperation for next time step
	*last_pos = buffer;
}


void Integrator::integrate_euler()
{
	//do some math shit here
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
