#include "Model.h"

Model::Model()
{
	_pvm_matrix_loc = 0;
	_rot_x = 0;
	_rot_y = 0;
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
	glClearColor(1.0, 0.0, 0.0, 1.0);
	glEnable(GL_DEPTH_TEST);

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
		{ GL_VERTEX_SHADER, "Graphics/shader.vs" },
		{ GL_FRAGMENT_SHADER, "Graphics/shader.fs" },  //Phong Reflectance Model
		{ GL_NONE, NULL }
	};
	if ((programs[0] = LoadShaders(shaders0)) == 0)
	{
		cout << "Error Loading Shader 0" << endl;
		return false;
	}
	glUseProgram(programs[0]);


	// Initialize all model assets
	or_key.init();
	for(Drawable *pl : _sim->_scene_objects)
	{
		pl->init();
	}

	_pvm_matrix_loc = glGetUniformLocation(programs[0], "_pvm_matrix");

	_projection_matrix = frustum(-0.1, 0.1, -0.1, 0.1, 0.1, 20.0);
	vec3 eye = vec3(0.0, 7.0, 10.0);
	vec3 aim = vec3(0.0, 0.0, 0.0);
	vec3 up = vec3(0.0, 1.0, 0.0);
	_view_matrix = lookAt(eye, aim, up);
	update(0.0, 0.0); //initializes the model_matrix

	// Initialize Model objects
	return true;
}

void Model::draw()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	mat4 rotate_matrix = rotate(mat4(1.0), degToRad(_rot_x), vec3(1,0,0));
	rotate_matrix = rotate(rotate_matrix, degToRad(_rot_y), vec3(0,1,0));

	// _pvm_matrix = _projection_matrix * _view_matrix * rotate_matrix * p.getModelMatrix();
	// glUniformMatrix4fv(_pvm_matrix_loc, 1, GL_FALSE, value_ptr(_pvm_matrix));
	// p.draw();
	//
	// _pvm_matrix = _projection_matrix * _view_matrix * rotate_matrix * p2.getModelMatrix();
	// glUniformMatrix4fv(_pvm_matrix_loc, 1, GL_FALSE, value_ptr(_pvm_matrix));
	// p2.draw();

	for (Drawable *pl : _sim->_scene_objects)
	{
		_pvm_matrix = _projection_matrix * _view_matrix * rotate_matrix * pl->getModelMatrix();
		glUniformMatrix4fv(_pvm_matrix_loc, 1, GL_FALSE, value_ptr(_pvm_matrix));
		pl->draw();
	}


	_pvm_matrix = _projection_matrix * _view_matrix * rotate_matrix * or_key.getModelMatrix();
	glUniformMatrix4fv(_pvm_matrix_loc, 1, GL_FALSE, value_ptr(_pvm_matrix));
	or_key.draw();

	glFlush();
}

void Model::update(float x, float y)
{
	_rot_x = x;
	_rot_y = y;
}