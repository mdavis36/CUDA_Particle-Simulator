#ifndef MODEL_H
#define MODEL_H

#define GLEW_STATIC

#include <GL/glew.h>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtc/type_ptr.hpp"
#include "LoadShaders.h"

#include "../Plane.h"
#include "../OrientationKey.h"
#include "../Sim/Simulation.h"

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

	float _rot_x;
	float _rot_y;

	//Plane p = Plane(vec3(0.0, 1.0, 0.0), vec3(5.0, -1.0, 0.0), 6, 2);
	//Plane p2 = Plane(vec3(0.0, 1.0, 0.0), vec3(0.0, 2.5, 0.0), 6, 6);
	OrientationKey or_key;

	Simulation *_sim;

	float degToRad(float deg) { return (deg * (3.14159f / 180.0f)); }
public:
	Model();
	bool init(Simulation *sim);
	void draw();
	void update(float x, float y);
};

#endif
