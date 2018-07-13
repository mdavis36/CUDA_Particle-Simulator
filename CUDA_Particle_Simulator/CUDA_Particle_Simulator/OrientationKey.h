#ifndef ORIENTATIONKEY_H
#define ORIENTATIONKEY_H

#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <iostream>
using namespace glm;
using namespace std;


class OrientationKey
{
private:
	vector<vec3> _positions;
	vector<vec4> _colors;
	mat4 _model_matrix;

	GLuint _vao;
	GLuint _buffers[2];

	bool _initialized;

public:
	OrientationKey();
	bool init();
	void draw();
	mat4 getModelMatrix();
};



#endif
