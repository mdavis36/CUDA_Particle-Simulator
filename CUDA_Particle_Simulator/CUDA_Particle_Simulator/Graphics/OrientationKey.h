#ifndef ORIENTATIONKEY_H
#define ORIENTATIONKEY_H

#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <iostream>

#include "Drawable.h"

using namespace glm;
using namespace std;


class OrientationKey : public Drawable
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
	virtual bool init();
	virtual void draw() const;
	mat4 getModelMatrix();
};



#endif
