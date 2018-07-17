#ifndef VIEWCONTROLLER_H
#define VIEWCONTROLLER_H

#include "GL/glew.h"
#include <SDL2/SDL.h>
#include <time.h>
#include <thread>


#include "Model.h"
#include "../Sim/Simulation.h"
#include "../Utils/Utils.h"

#include <iostream>
using namespace std;

class ViewController
{
private:

#ifdef RUN_GPU
	const char WINDOW_TITLE[32] = "CUDA_Particle_Simulator";
#else
	const char WINDOW_TITLE[32] = "Particle_Simulator";
#endif

	const int WINDOW_WIDTH = 1600;
	const int WINDOW_HEIGHT = 1200;

	const int FPS = 60;

	// SDL variables
	SDL_Window *_sdl_window;
	SDL_GLContext _sdl_glcontext;
	SDL_Event _event;
	bool _quit;


	// View Controller manipulation variables.
	bool _rotating;
	bool _transforming;

	float _rot_base_x;
	float _rot_base_y;

	float _tran_base_x;
	float _tran_base_y;

	float _last_rot_offset_x;
	float _last_rot_offset_y;
	float _x_angle;
	float _y_angle;
	float _x_angle_init;
	float _y_angle_init;

	float _last_tran_offset_x;
	float _last_tran_offset_y;
	float _x_tran_init;
	float _y_tran_init;
	vec3 _trans = vec3(0.0f,0.0f,0.0f);

	bool _firstclick_r;
	bool _firstclick_l;

	float _zoom = 1.0f;


	// Timing variables.
	struct timespec start_frame, end_frame, delta_frame_time, acc_time;
	struct timespec start_up, end_up, delta_up, acc_up, wait;


	Simulation *_sim;
	Model m;
	vec3 p = normalize(vec3(243, 27, 227));

	bool init();
	void setAttributes();
	void display();
	void handleEvents(SDL_Event e);
	void cleanup();

	void cc(vec3 c)
	{
		glClearColor(c.x, c.y, c.z, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);
		SDL_GL_SwapWindow(_sdl_window);
	}

public:

	ViewController(Simulation *sim);
	void run();
};

#endif
