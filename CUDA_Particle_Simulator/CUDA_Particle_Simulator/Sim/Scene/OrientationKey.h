#ifndef ORIENTATIONKEY_H
#define ORIENTATIONKEY_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include <vector>

#include "../../Graphics/Drawable.h"

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
	virtual bool init(GLuint* programs);
	virtual void draw(GLuint* programs, mat4 proj_mat, mat4 view_mat);
	mat4 getModelMatrix();
};



#endif
