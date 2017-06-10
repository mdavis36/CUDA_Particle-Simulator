#include "ViewController.h"

ViewController::ViewController()
{
	_sdl_window = 0;
	_sdl_glcontext = 0;
	_quit = false;
}

bool ViewController::init()
{
	// Initialive SDL
	if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
	{
		cout << "Failed to initialize SDL.\n";
		return false;
	}

	// Create the SDL Window
	if ((_sdl_window = SDL_CreateWindow(WINDOW_TITLE, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_OPENGL)) == NULL)
	{
		cout << "Failed to create window.\n";
		return false;
	}

	// Create OpenGL Context in the SDL Window
	if ((_sdl_glcontext = SDL_GL_CreateContext(_sdl_window)) == NULL)
	{
		cout << "Failed to create SDl GL Context.\n";
		return false;
	}
	return true;
}

void ViewController::setAttributes()
{
	// Set OpenGL SDl Attributes
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, true);
}

void ViewController::display()
{
	// Draw the model here
}

void ViewController::handleEvents(SDL_Event e)
{
	while (SDL_PollEvent(&e))
	{
		switch (e.type)
		{
		case SDL_QUIT:
			_quit = true;
			break;

		case SDL_KEYDOWN:
			switch (e.key.keysym.sym)
			{
			case SDLK_ESCAPE:
				_quit = true;
				break;
			case SDLK_r:
				// Cover with red and update
				glClearColor(1.0, 0.0, 0.0, 1.0);
				glClear(GL_COLOR_BUFFER_BIT);
				SDL_GL_SwapWindow(_sdl_window);
				break;
			case SDLK_g:
				// Cover with green and update
				glClearColor(0.0, 1.0, 0.0, 1.0);
				glClear(GL_COLOR_BUFFER_BIT);
				SDL_GL_SwapWindow(_sdl_window);
				break;
			case SDLK_b:
				// Cover with blue and update
				glClearColor(0.0, 0.0, 1.0, 1.0);
				glClear(GL_COLOR_BUFFER_BIT);
				SDL_GL_SwapWindow(_sdl_window);
				break;
			default:
				break;
			}
		}
	}
}

void ViewController::cleanup()
{
	// Cleanup SDL
	SDL_GL_DeleteContext(_sdl_glcontext);
	SDL_DestroyWindow(_sdl_window);
	SDL_Quit();
}

void ViewController::run()
{
	if (!init())
	{
		cout << "Failed to initialize View Controller.\n";
		return;
	}

	if (!m.init())
	{
		cout << "Failed to initialize Model Object.\n";
		return;
	}

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	SDL_GL_SwapWindow(_sdl_window);

	do
	{
		display();
		handleEvents(_event);
	} while (!_quit);

	cleanup();

}
