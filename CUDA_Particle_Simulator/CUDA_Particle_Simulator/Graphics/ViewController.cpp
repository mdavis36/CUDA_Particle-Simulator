#include "ViewController.h"

ViewController::ViewController(Simulation *sim)
{
	_sim = sim;

	_sdl_window = 0;
	_sdl_glcontext = 0;
	_quit = false;

	_x_angle_init, _y_angle_init = 0;
	_x_angle, _y_angle = 0;
	_firstclick = true;

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
	m.draw();
	SDL_GL_SwapWindow(_sdl_window);
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
			case SDLK_p:
				glClearColor(p.x, p.y, p.z, 1.0);
				glClear(GL_COLOR_BUFFER_BIT);
				SDL_GL_SwapWindow(_sdl_window);
			case SDLK_y:
				glClearColor(1.0, 1.0, 0.0, 1.0);
				glClear(GL_COLOR_BUFFER_BIT);
				SDL_GL_SwapWindow(_sdl_window);
			default:
				break;
			}
		case SDL_MOUSEBUTTONDOWN:
		{
			if (SDL_GetMouseState(NULL, NULL) == SDL_BUTTON(1))  //Attach rotation to the left mouse button
			{
				// save position where button down event occurred. This
				// is the "zero" position for subsequent mouseMotion callbacks.
				_base_x = e.button.x;
				_base_y = e.button.y;
				_rotating = true;
			}
			break;
		}
		case SDL_MOUSEBUTTONUP:
		{
			if (_rotating)  //are we finishing a rotation?
			{
				//Remember where the motion ended, so we can pick up from here next time.
				_last_offset_x += (e.button.x - _base_x);
				_last_offset_y += (e.button.y - _base_y);
				_rotating = false;
			}
			break;
		}
		case SDL_MOUSEMOTION:
		{
			//Is the left mouse button also down?
			if (SDL_GetMouseState(NULL, NULL) == SDL_BUTTON(1))
			{

				float x, y;

				//Calculating the conversion => window size to angle in degrees
				float scaleX = 120.0 / WINDOW_WIDTH;
				float scaleY = 120.0 / WINDOW_HEIGHT;

				x = (e.button.x - _base_x) + _last_offset_x;
				y = (e.button.y - _base_y) + _last_offset_y;

				// map "x" to a rotation about the y-axis.
				x *= scaleX;
				_y_angle = x;

				// map "y" to a rotation about the x-axis.
				y *= scaleY;
				_x_angle = y;

				if (_firstclick)
				{
					_x_angle_init = _x_angle;
					_y_angle_init = _y_angle;
					_firstclick = false;
				}

				_y_angle -= _y_angle_init;
				_x_angle -= _x_angle_init;

				m.update(_x_angle, _y_angle); //send the new angles to the Model object
			}
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

	if (!m.init(_sim))
	{
		cout << "Failed to initialize Model Object.\n";
		return;
	}

	glClearColor(0.1, 0.1, 0.1, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	SDL_GL_SwapWindow(_sdl_window);
	//int count = 0;
	do
	{
		/*
		switch (count % 5)
		{
		case 0:
			cc(vec3(1.0, 0.0, 0.0));
			break;
		case 1:
			cc(vec3(0.0, 1.0, 0.0));
			break;
		case 2:
			cc(vec3(0.0, 0.0, 1.0));
			break;
		case 3:
			cc(vec3(1.0, 1.0, 0.0));
			break;
		case 4:
			cc(p);
			break;
		}
		*/
		//count++;
		//this_thread::sleep_for(chrono::milliseconds(50));
		display();
		handleEvents(_event);
	} while (!_quit);

	cleanup();

}
