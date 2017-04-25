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
}

void Integrator::integrate_verlet(glm::vec3 *pos, glm::vec3 *last_pos, float ts, float m, glm::vec3 acc) {
	// Verlet integration
	// Save the current position into a buffer
	glm::vec3 buffer = *pos;

	// Perform the verlet integartion calculation
	// Note : Future performance increase could be done by calculating t*t/m prior to integration
	//		 However this would only be useful fopr systems with multiuple particles of the same mass
	//cout << "Int : pos : " << pos << "   last_pos : " << last_pos << "   ts : " << ts << "   m : " << m << endl;
	*pos = (*pos + *pos) - *last_pos + (ts * ts) * acc;

	// Swap buffer so the old position becomes the last position in poreperation for next time step
	*last_pos = buffer;
}


void Integrator::integrate_euler()
{
	//do some math shit here
}
