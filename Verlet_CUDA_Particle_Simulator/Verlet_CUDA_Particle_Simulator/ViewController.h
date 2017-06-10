#ifndef VIEWCONTROLLER_H
#define VIEWCONTROLLER_H

#include "glew.h"
#include <SDL.h>

#include "Model.h"

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

	bool init();
	void setAttributes();
	void display();
	void handleEvents(SDL_Event e);
	void cleanup();
public:

	ViewController();
	void run();
};

#endif