#ifndef PLANE_H
#define PLANE_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include <vector>

#include "../../Graphics/Drawable.h"

using namespace glm;
using namespace std;

class Plane : public Drawable
{
private:
	std::vector<vec3> _positions;
	std::vector<vec4> _colors;

	GLuint _vao;
	GLuint _buffers[2];

	bool _initialized;

public:
	int _width, _height;
	vec3 _normal;
	vec3 _center;

	Plane();
	Plane(vec3 n, vec3 c, int w, int h);
	virtual bool init(GLuint* programs);
	virtual void draw(GLuint* programs, mat4 proj_mat, mat4 view_mat);
};


#endif
