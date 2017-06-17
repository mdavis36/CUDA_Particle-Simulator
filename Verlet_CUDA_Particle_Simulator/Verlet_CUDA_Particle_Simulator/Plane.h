#ifndef PLANE_H
#define PLANE_H

#include <vector>
#include <glew.h>
#include <glm.hpp>
#include <iostream>
using namespace glm;
using namespace std;

class Plane
{
private:
	int _width, _height;
	vec3 _normal;
	vec3 _center;

	//std::vector<GLfloat> _positions;
	//std::vector<GLfloat> _colors;
	std::vector<vec3> _positions;
	std::vector<vec4> _colors;

	GLfloat _pos[4][3];
	GLfloat _col[4][4];

	GLuint _vao;
	GLuint _buffers[2];	

	bool _initialized;
public:
	Plane();
	Plane(vec3 n, vec3 c, int w, int h);
	bool init();
	void draw();

};


#endif
