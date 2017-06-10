#ifndef MODEL_H
#define MODEL_H

#define GLEW_STATIC

#include <glew.h>
#include <glm.hpp>
#include "gtc/matrix_transform.hpp"
#include "LoadShaders.h"

#include <iostream>
using namespace std;
using namespace glm;

class Model 
{
private:
	GLint _model_matrix_loc;
	GLint _view_matrix_loc;

	mat4 _model_matrix;
	mat4 _view_matrix;
	mat4 _projection_matrix;

	GLuint programs[1];

	float degToRad(float deg) { return (deg * (3.14159f / 180.0f)); }

public:
	Model();
	bool init();
	void draw();

};

#endif