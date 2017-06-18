#ifndef MODEL_H
#define MODEL_H

#define GLEW_STATIC

#include <glew.h>
#include <glm.hpp>
#include "gtc/matrix_transform.hpp"
#include "gtc/type_ptr.hpp"
#include "LoadShaders.h"

#include "Plane.h"
#include "OrientationKey.h"

#include <iostream>
using namespace std;
using namespace glm;

class Model 
{
private:
	GLint _pvm_matrix_loc;
	mat4 _pvm_matrix;
	mat4 _model_matrix;
	mat4 _view_matrix;
	mat4 _projection_matrix;

	GLuint programs[1];

	Plane p = Plane(vec3(-1.0, -1.0, 0.0), vec3(0.0, 0.0, 0.0), 6, 2);
	OrientationKey or_key;

	float degToRad(float deg) { return (deg * (3.14159f / 180.0f)); }
public:
	Model();
	bool init();
	void draw();

};

#endif