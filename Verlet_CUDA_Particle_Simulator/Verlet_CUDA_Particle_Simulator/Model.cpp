#include "Model.h"

Model::Model()
{
	_model_matrix_loc = 0;
	_view_matrix_loc = 0;
}

bool Model::init()
{
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
		{ GL_VERTEX_SHADER, "shader.vs" },
		{ GL_FRAGMENT_SHADER, "shader.fs" },  //Phong Reflectance Model
		{ GL_NONE, NULL }
	};
	if ((programs[0] = LoadShaders(shaders0)) == 0)
	{
		cout << "Error Loading Shader 0" << endl;
		return false;
	}
	// Create Programs


	_projection_matrix = frustum(-0.1, 0.1, -0.1, 0.1, 0.1, 20.0);
	vec3 eye = vec3(5.0, 5.0, 5.0);
	vec3 aim = vec3(0.0, 0.0, 0.0);
	vec3 up = vec3(0.0, 1.0, 0.0);
	_view_matrix = lookAt(eye, aim, up);
	//update(0.0, 0.0); //initializes the model_matrix

	// Initialize Model objects

	return true;
}

void Model::draw()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	glFlush();
}
