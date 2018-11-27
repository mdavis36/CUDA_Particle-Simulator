#include "Model.h"

Model::Model()
{
	_pvm_matrix_loc = 0;
	_rot_x = 0.0f;
	_rot_y = 0.0f;
	_zoom = 1.0f;
}

bool Model::init(Simulation *sim)
{
	_sim = sim;

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		cout << "Error initializing GLEW : " << glewGetErrorString(err) << endl;
		return false;
	}

	// Clear the screen and draw red
	glClearColor(0.2, 0.2, 0.2, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glEnable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	const GLubyte *renderer = glGetString(GL_RENDERER);
	const GLubyte *vendor = glGetString(GL_VENDOR);
	const GLubyte *version = glGetString(GL_VERSION);
	const GLubyte *glslVersion = glGetString(GL_SHADING_LANGUAGE_VERSION);
	cout << "GL Vendor: " << vendor << endl;
	cout << "GL Renderer: " << renderer << endl;
	cout << "GL Version: " << version << endl;
	cout << "GLSL Version: " << glslVersion << endl;

	// Load Shader
	ShaderInfo shaders0[] = {
		{ GL_VERTEX_SHADER,   "Graphics/shader.vs", 0 },
		{ GL_FRAGMENT_SHADER, "Graphics/shader.fs", 1 },  //Phong Reflectance Model
		{ GL_NONE, 		    NULL, 			  2 }
	};
	ShaderInfo shaders1[] = {
		{ GL_VERTEX_SHADER,   "Graphics/shader_particle_system.vs", 0 },
		{ GL_FRAGMENT_SHADER, "Graphics/shader_particle_system.fs", 1 },  //Phong Reflectance Model
		{ GL_NONE, 		    NULL, 			  2 }
	};
	ShaderInfo shaders2[] = {
		{ GL_VERTEX_SHADER,   "Graphics/shader_scene_obj.vs", 0 },
		{ GL_FRAGMENT_SHADER, "Graphics/shader_scene_obj.fs", 1 },  //Phong Reflectance Model
		{ GL_NONE, 		    NULL, 			  2 }
	};
#ifndef GL1
	if ((programs[0] = LoadShaders(shaders0)) == 0)
	{
		cout << "Error Loading Shader 0" << endl;
		return false;
	}
	if ((programs[1] = LoadShaders(shaders1)) == 1)
	{
		cout << "Error Loading Shader 1" << endl;
		return false;
	}
	if ((programs[2] = LoadShaders(shaders2)) == 1)
	{
		cout << "Error Loading Shader 2" << endl;
		return false;
	}
	glUseProgram(programs[0]);
#else
	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	//gluPerspective(60, 2200, 1800, 1.f, 100.0f);
#endif

	// Initialize all model assets
	initAssets();


	_projection_matrix = frustum(-0.1, 0.1, -0.1, 0.1, 0.1, 20000.0);
	eye = vec3(0.0, 7.0, 10.0);
	aim = vec3(0.0, 0.0, 0.0);
	up = vec3(0.0, 1.0, 0.0);
	_view_matrix = lookAt(eye, aim, up);
	update(0.0, 0.0, 1.0, 0.0, 0.0); //initializes the model_matrix

	// Initialize Model objects
	return true;
}

void Model::initAssets()
{
	or_key.init(programs);
	for(Drawable *pl : _sim->_drawable_objects)
	{
		pl->init(programs);
	}
}

void Model::draw()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	_view_matrix = lookAt(eye * _zoom, aim, up);
	_view_matrix = translate(_view_matrix, _trans);
	_view_matrix = rotate(_view_matrix, degToRad(_rot_x), vec3(1,0,0));
	_view_matrix = rotate(_view_matrix, degToRad(_rot_y), vec3(0,1,0));

	for (Drawable *pl : _sim->_drawable_objects)
	{
		pl->draw(programs, _projection_matrix, _view_matrix);
	}

	or_key.draw(programs, _projection_matrix, _view_matrix);

	glFlush();
}

void Model::update(float x, float y, float z, float xt, float yt)
{
	_rot_x = x;
	_rot_y = y;
	_zoom = z;
	_trans.x = xt;
	_trans.y = yt;
}
