/*
* FILE : simulation.cpp
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* This is the primary simulation class, it handles simulation time, updates, simulation state and
* rendering of the simulation environment and simulation objects
*/

#include <iostream>
#include <omp.h>
#include "Simulation.h"

using namespace std;

// ********************************************************************************
// *                               -- PUBLIC --                                   *
// ********************************************************************************

// -------------------- Simulation Constructors and Destructor --------------------

Simulation::Simulation()
{
	Simulation(5);
}

Simulation::Simulation(int n)
{
	num_particles = n;
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
	if (sim_state == INITIALIZED)
	{
		cout << "Starting Simulation of "<< num_particles << " falling particles." << endl;
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
		return true;
	}
	else if (sim_state == RUNNING) {
		sim_state = PAUSED;
		cout << "Simulation Paused.\n";
		return true;
	}
	return false;
}

bool Simulation::end()
{
	if (sim_state >= RUNNING && sim_state != FINISHED) {
		cout << "Simulation has ended.\n";
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

		//EulerStep(_p_sys, dt);
		RK4(_p_sys, &_s_obj->_polygons, dt);

		// #pragma omp parallel for num_threads(4)
		// for (int i = 0; i < _p_sys->_num_particles; i++)
		// {
		// 	if (_p_sys->_particles[i].x.y <= 0)
		// 	{
		// 		_p_sys->_particles[i].x.y = 0;
		// 	}
		// }
	}
}

void Simulation::render() {}


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

	_p_sys = new ParticleSystem(num_particles, PARTICLE_SPHERE);
	_ground = new Plane(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 0.0), 20, 20);
	_s_obj = new SceneObject("../Resources/smaug.obj");

	std::cout << "Vertices Count  : " << _s_obj->_vertices.size() << std::endl;
	std::cout << "Polygon Count   : " << _s_obj->_polygons.size() << std::endl;

	OTH = new OctTreeHandler();
	OTH->clear();
	OTH->buildTree(_s_obj->_polygons, Volume(glm::vec3(-20,-20,-20), glm::vec3(20,20,20)));

	_drawable_objects.clear();
	_drawable_objects.push_back(_ground);
	_drawable_objects.push_back(_p_sys);
	_drawable_objects.push_back(_s_obj);
	_drawable_objects.push_back(OTH);

	sim_state = INITIALIZED;
	return true;
}


void Simulation::clean()
{
	while (!_drawable_objects.empty()) {
		delete _drawable_objects.back();
		_drawable_objects.pop_back();
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
	return BILLION;
}

// ********************************************************************************
