#include "ViewController.h"

ViewController::ViewController(Simulation *sim)
{
	_sim = sim;

	_sdl_window = 0;
	_sdl_glcontext = 0;
	_quit = false;

	_x_angle_init, _y_angle_init, _x_tran_init, _y_tran_init = 0.0f;
	_x_angle, _y_angle = 0;
	_firstclick_r = true;
	_firstclick_l = true;

	_rot_base_x, _rot_base_y, _tran_base_x, _tran_base_y = 0.0f;

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

		// Simulation controls.
		case SDL_KEYDOWN:
		{
			switch (e.key.keysym.sym)
			{
			case SDLK_ESCAPE:
				_quit = true;
				break;
			case SDLK_s:
				_sim->start();
				break;
			case SDLK_r:
				// Cover with red and update
				glClearColor(1.0, 0.0, 0.0, 1.0);
				glClear(GL_COLOR_BUFFER_BIT);
				SDL_GL_SwapWindow(_sdl_window);
				break;

			default:
				break;
			}
			break;
		}

		// Model Zoom
		case SDL_MOUSEWHEEL:
		{
			if (e.wheel.y > 0)
			{
				_zoom -= 0.1f;
			}
			else if (e.wheel.y < 0)
			{
				_zoom += 0.1f;
			}
			if (_zoom <= 0.0) _zoom = 0;
			m.update(_x_angle, _y_angle, _zoom, _trans.x, _trans.y);
			break;
		}

		// Model Rotation and transform controls
		case SDL_MOUSEBUTTONDOWN:
		{
			if (SDL_GetMouseState(NULL, NULL) == SDL_BUTTON(SDL_BUTTON_LEFT))  //Attach rotation to the left mouse button
			{
				// save position where button down event occurred. This
				// is the "zero" position for subsequent mouseMotion callbacks.
				_rot_base_x = e.button.x;
				_rot_base_y = e.button.y;
				_rotating = true;
			}
			if (SDL_GetMouseState(NULL, NULL) == SDL_BUTTON(SDL_BUTTON_RIGHT))  //Attach rotation to the left mouse button
			{
				// save position where button down event occurred. This
				// is the "zero" position for subsequent mouseMotion callbacks.
				_tran_base_x = e.button.x;
				_tran_base_y = e.button.y;
				_transforming = true;
			}
			break;
		}
		case SDL_MOUSEBUTTONUP:
		{
			if (_rotating)  //are we finishing a rotation?
			{
				//Remember where the motion ended, so we can pick up from here next time.
				_last_rot_offset_x += (e.button.x - _rot_base_x);
				_last_rot_offset_y += (e.button.y - _rot_base_y);
				_rotating = false;
			}
			if (_transforming)  //are we finishing a rotation?
			{
				//Remember where the motion ended, so we can pick up from here next time.
				_last_tran_offset_x += (e.button.x - _tran_base_x);
				_last_tran_offset_y += (e.button.y - _tran_base_y);
				_transforming = false;
			}
			break;
		}
		case SDL_MOUSEMOTION:
		{
			//Is the left mouse button also down?
			if (SDL_GetMouseState(NULL, NULL) == SDL_BUTTON(SDL_BUTTON_LEFT))
			{

				float x, y;

				//Calculating the conversion => window size to angle in degrees
				float scaleX = 120.0 / WINDOW_WIDTH;
				float scaleY = 120.0 / WINDOW_HEIGHT;

				x = (e.button.x - _rot_base_x) + _last_rot_offset_x;
				y = (e.button.y - _rot_base_y) + _last_rot_offset_y;

				// map "x" to a rotation about the y-axis.
				x *= scaleX;
				_y_angle = x;

				// map "y" to a rotation about the x-axis.
				y *= scaleY;
				_x_angle = y;

				if (_firstclick_l)
				{
					_x_angle_init = _x_angle;
					_y_angle_init = _y_angle;
					_firstclick_l = false;
				}

				_y_angle -= _y_angle_init;
				_x_angle -= _x_angle_init;

				m.update(_x_angle, _y_angle, _zoom, _trans.x, _trans.y); //send the new angles to the Model object
			}

			if (SDL_GetMouseState(NULL, NULL) == SDL_BUTTON(SDL_BUTTON_RIGHT))
			{

				float x, y;

				//Calculating the conversion => window size to angle in degrees
				float scaleX = 50.0f / WINDOW_WIDTH;
				float scaleY = 50.0f / WINDOW_HEIGHT;

				x = (e.button.x - _tran_base_x) + _last_tran_offset_x;
				y = (e.button.y - _tran_base_y) + _last_tran_offset_y;

				// map "x" to a rotation about the y-axis.
				x *= scaleX;
				_trans.x = x;

				// map "y" to a rotation about the x-axis.
				y *= scaleY;
				_trans.y = -y;

				if (_firstclick_r)
				{
					_x_tran_init = _trans.x;
					_y_tran_init = _trans.y;
					_firstclick_r = false;
				}

				_trans.x -= _x_tran_init;
				_trans.y -= _y_tran_init;

				m.update(_x_angle, _y_angle, _zoom, _trans.x, _trans.y); //send the new angles to the Model object
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
		_sim->update();
		display();
		handleEvents(_event);
	} while (!_quit);

	cleanup();

}
