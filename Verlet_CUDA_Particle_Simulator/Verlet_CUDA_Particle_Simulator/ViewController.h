#ifndef VIEWCONTROLLER_H
#define VIEWCONTROLLER_H

#include "glew.h"
#include <SDL.h>

#include "Model.h"
#include <thread>

#include <iostream>
using namespace std;

class ViewController
{
private:
	const char WINDOW_TITLE[32] = "Verlet_CUDA_Particle_Simulator";
	const int WINDOW_WIDTH = 800;
	const int WINDOW_HEIGHT = 600;

	SDL_Window *_sdl_window;
	SDL_GLContext _sdl_glcontext;
	SDL_Event _event;
	bool _quit;

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

	ViewController();
	void run();
};

#endif