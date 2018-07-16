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
#include "Simulation.h"

// ********************************************************************************
// *                               -- PUBLIC --                                   *
// ********************************************************************************

// -------------------- Simulation Constructors and Destructor --------------------

Simulation::Simulation()
{
	if (!init())
	{
		cout << "Failed to initialize simulation.\n";
		exit(-4);
	}
}

Simulation::~Simulation()
{
	//particles.clear();
}


// -------------------- Simulation Control Functions --------------------

bool Simulation::start()
{
	cout << "Start" << endl;
	if (sim_state == INITIALIZED)
	{
		cout << "Starting Simulation of "<< PARTICLE_COUNT << " falling particles." << endl;
		//integrator.printIType();
		sim_state = RUNNING;

		sim_time_accu = 0;

		#ifdef _WIN32
			// Get ticks per second.
			QueryPerformanceFrequency(&frequency);
			// Start timer.
			QueryPerformanceCounter(&t1);
		#else
			clock_gettime(CLOCK_REALTIME, &t1);
		#endif

		return true;
	}
	return false;
}

bool Simulation::pause()
{
	if (sim_state == PAUSED) {
		sim_state = RUNNING;
		cout << "Simulation Running.\n";
		#ifdef _WIN32
			// Start timer.
			QueryPerformanceCounter(&t1);
		#else
			clock_gettime(CLOCK_REALTIME, &t1);
		#endif
		return true;
	}
	else if (sim_state == RUNNING) {
		sim_state = PAUSED;
		cout << "Simulation Paused at : " << sim_time_accu << " seconds\n";
		return true;
	}
	return false;
}

bool Simulation::end()
{
	if (sim_state >= RUNNING && sim_state != FINISHED) {
		cout << "Simulation has ended.\n";
		cout << "Simulation ran for : " << sim_time_accu << "\n\n";
	}
	sim_state = FINISHED;
	return false;
}

bool Simulation::reset()
{
	end();
	cout << "Resetting Simulation.\n";
	printControls();
	clean();
	init();
	return false;
}


// -------------------- Simulation Step Functions --------------------

void Simulation::update()
{
	if (sim_state == RUNNING)
	{
		#ifdef _WIN32
			// Using high res. counter.
			QueryPerformanceCounter(&t2);
			// Compute the elapsed time in seconds.
			frame_time = ((t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart) / 1000.0;
		#else
			clock_gettime(CLOCK_REALTIME, &t2);
			frame_time = (t2.tv_nsec - t1.tv_nsec) / 1000000000.0;
		#endif

		t1 = t2;
		sim_time_accu += frame_time;

		// Update every particle and check to see if all particles have hit the floor.
		//bool checkSimOver = true;
		// for (Particle* p : particles)
		// {
		// 	p->update(frame_time, glm::vec3(0.0f, 0.0f, 0.0f), &integrator);
            //
		// 	//// ----- starting collision -----
		// 	float sub_frame_time = frame_time;
		// 	Particle proj_p = *p;
		// 	Plane * col_plane = getClosestCollisionPlane(&proj_p, p, planes);
		// 	while (col_plane != nullptr) {
		// 		//cout << "Collision!\n";
            //
		// 		float col_frac = getCollisionFrac(&proj_p, col_plane);
		// 		//cout << "col_frac : " << col_frac << "\n";
		// 		//cout << "proj_p.pos.y : " << proj_p.pos.y << "proj_p.next_pos.y : " << proj_p.next_pos.y << "\n";
		// 		//cout << "projecting about plane\n";
		// 		proj_p = projectParticleAtSubTimeStep(&proj_p, col_plane, col_frac);
		// 		//cout << "proj_p.pos.y : " << proj_p.pos.y << "proj_p.last_pos.y : " << proj_p.last_pos.y << "\n";
		// 		//cout << "subframetime : " << sub_frame_time << endl;
		// 		sub_frame_time *= 1.0f - col_frac;
		// 		//cout << "subframetime : " << sub_frame_time << endl;
		// 		proj_p.update(sub_frame_time, glm::vec3(0.0f, 0.0f, 0.0f), &integrator);
		// 		//cout << "proj_p.next_pos.y : " << proj_p.next_pos.y << "\n";
		// 		col_plane = getClosestCollisionPlane(&proj_p, p, planes);
		// 	}
		// 	*p = proj_p;
		// 	//vec3 curr_pos = p->pos;
		// 	//vector<Plane*> col_planes = getCollisions(p, planes);
		// 	//if (col_planes.size() > 0)
		// 	//{
		// 		//cout << "---------------------------" << "\n";
		// 		//cout << "Num of cols : " << col_planes.size() << "\n";
		// 	//}
		// 	// ----- end collision -----
		// 	//if (p->pos.y > 0) checkSimOver = false;
		// }

		//if (checkSimOver) {
		//	end();
		//}
	}
}

void Simulation::render()
{
	// if (sim_state >= INITIALIZED)
	// {
	// 	// Render all simulation objects here.
	// 	for (Plane* pl : planes)
	// 	{
	// 		pl->draw();
	// 	}
      //
	// 	// for (Particle* p : particles)
	// 	// {
	// 	// 	p->draw();
	// 	// }
	// 	// drawOrientationKey();
	// }
}


// -------------------- Getter Functions --------------------

double Simulation::getSimFrameTime() { return frame_time; }

int Simulation::getSimCurrState() {	return sim_state; }


// -------------------- GPrint Functions --------------------

void Simulation::printControls()
{
	cout << "---------- Simulation Controls ----------\n";
	cout << "\t s : Start the simulation.\n";
	cout << "\t p : Pause and Un-Pause the simulation.\n";
	cout << "\t r : Reset the simulation.\n\n";
}

// ********************************************************************************



// ********************************************************************************
// *                               -- Private --                                  *
// ********************************************************************************

bool Simulation::init()
{
	// This is where all simulation object will be initialized to set up the simulation.
	//integrator.setIType(INTEGRATE_VERLET);
	//integrator.setIType(INTEGRATE_EULER);

	//planes.push_back(new Plane(glm::vec3(10.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 1.0f, 0.0f), 20.0f));
	//planes.push_back(new Plane(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 2.0f, 0.0f), 5.0f));

	//_scene_objects.push_back(new Plane(vec3(0.0, 1.0, 0.0), vec3(5.0, -1.0, 0.0), 6, 2));
	_scene_objects.push_back(new Plane(vec3(0.0, 1.0, 0.0), vec3(0.0, -1.0, 0.0), 20, 20));

      /*
	glm::vec3 ranPos;
	for (int i = 0; i < PARTICLE_COUNT; i++)
	{
		ranPos.x = -planes[0]->width + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * planes[0]->width)));
		ranPos.z = -planes[0]->width + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * planes[0]->width)));
		ranPos.y = 5 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (8)));
		particles.push_back(new Particle(ranPos));
	}
      */

	//particles.push_back(new Particle(glm::vec3(0.0f, 5.0f, 0.0f)));

	sim_state = INITIALIZED;
	return true;
}

// void Simulation::drawOrientationKey()
// {
// 	//draw orientation key
// 	glBegin(GL_LINES);
//
// 	//Red is Positive X
// 	glColor3f(1.0f, 0.0f, 0.0f);
// 	glVertex3f(0, 0, 0);
// 	glVertex3f(1, 0, 0);
//
// 	//Green is positive Y
// 	glColor3f(0.0f, 1.0f, 0.0f);
// 	glVertex3f(0, 0, 0);
// 	glVertex3f(0, 1, 0);
//
// 	//Blue is posiotive Z
// 	glColor3f(0.0f, 0.0f, 1.0f);
// 	glVertex3f(0, 0, 0);
// 	glVertex3f(0, 0, 1);
//
// 	glEnd();
// }


void Simulation::clean()
{
	// De-allocate all of the particles from the vector.
	//while (!particles.empty()) {
	//	delete particles.back();
	//	particles.pop_back();
	//}
	while (!_scene_objects.empty()) {
		delete _scene_objects.back();
		_scene_objects.pop_back();
	}
}

float Simulation::distFromPlane(const vec3 pos, const Plane * plane)
{
	return dot(plane->_normal, (pos - plane->_center));
}

// vector<Plane*> Simulation::getCollisions(Particle * p, vector<Plane*> planes)
// {
// 	vector<Plane*> col_planes;
// 	for (Plane* pl : planes) {
// 		float t1 = distFromPlane(p->pos, pl);
// 		//
// 		//
// 		float t3 = distFromPlane(p->next_pos, pl);
// 		if (t1 > 0 && t3 <= 0)
// 		{
// 			col_planes.push_back(pl);
// 			//p->pos = reflect(p->pos, pl);
// 			//p->next_pos = reflect(p->next_pos, pl);
// 		}
// 	}
// 	return col_planes;
// }

vec3 Simulation::reflect(const vec3 pos, const Plane * plane)
{
	return pos + 2 * (-distFromPlane(pos, plane) / dot(plane->_normal, plane->_normal)) * plane->_normal;
}

// Plane * Simulation::getClosestCollisionPlane(const Particle * proj_p,const Particle * orig_p, const vector<Plane*> planes)
// {
// 	Plane* closest_plane = nullptr;
// 	float closest_dist = INT_MAX;
// 	for (Plane* pl : planes) {
// 		float curr_orig_pos_dist = distFromPlane(orig_p->pos, pl);
// 		float curr_proj_pos_dist = distFromPlane(proj_p->pos, pl);
// 		float next_proj_pos_dist = distFromPlane(proj_p->next_pos, pl);
//
// 		int copd = getSign(curr_orig_pos_dist);
// 		int cppd = getSign(curr_proj_pos_dist);
// 		int nppd = getSign(next_proj_pos_dist);
//
// 		if (copd == cppd && cppd != nppd && cppd != 0) {
// 			// copd == cppd : Make sure we are reflecting off the correct side of the plane based on the particles location relative to the start of the update
// 			// cppd != nppd : Check that the current and next position of the projected particle are not in the same orientaion relative to the plane
// 			// cppd != 0	: Check that the curr projected position isn't lying on the plane. This means a collision has most likely already been calculated for this plane
// 			// Now dealing with a possible Collision Plane
// 			if (abs(curr_proj_pos_dist) < closest_dist) {
// 				closest_dist = abs(curr_proj_pos_dist);
// 				closest_plane = pl;
// 			}
// 		}
// 	}
// 	return closest_plane;
// }

// float Simulation:: getCollisionTimeFromStartOfTimeStep(const Particle * p, const Plane * pl, const float t_step)
// {
// 	return t_step * getCollisionFrac(p, pl);
// }
//
// float Simulation::getCollisionFrac(const Particle * p, const Plane * pl)
// {
// 	return abs(distFromPlane(p->pos, pl)) / ( abs(distFromPlane(p->pos, pl)) + abs(distFromPlane(p->next_pos, pl)) );
// }
//
// Particle Simulation::projectParticleAtSubTimeStep(const Particle * p, const Plane * pl, const float t_step_frac)
// {
// 	Particle proj_particle = *p;
// 	//Reflect p.pos and p.next_pos across plane pl
// 	vec3 ref_curr_pos = reflect(p->pos, pl);
// 	vec3 ref_next_pos = reflect(p->next_pos, pl);
//
// 	//Calculate Collision point
// 	vec3 col_point = ref_curr_pos + ((ref_next_pos - ref_curr_pos) * t_step_frac);
// 	//cout << "col_point : " << col_point.y << endl;
//
// 	/*//Set projected particles last_pos to reflected p.pos
// 	proj_particle.last_pos = ref_curr_pos;
// 	cout << "proj_p.last_pos.y : " << proj_particle.last_pos.y << "\n";
// 	//Set projected particles pos to collision point
// 	proj_particle.pos = col_point;
// 	cout << "proj_p.pos.y : " << proj_particle.pos.y << "\n";
// 	*/
// 	//Set projected particles last_pos to reflected p.pos
// 	proj_particle.pos = ref_curr_pos;
// 	//cout << "proj_p.pos.y : " << proj_particle.pos.y << "\n";
// 	//Set projected particles pos to collision point
// 	proj_particle.next_pos = col_point;
// 	//cout << "proj_p.next_pos.y : " << proj_particle.next_pos.y << "\n";
// 	return proj_particle;
// }

int Simulation::getSign(float val)
{
	if (val > 0) return 1;
	if (val == 0) return 0;
	if (val < 0) return -1;
}

// ********************************************************************************
