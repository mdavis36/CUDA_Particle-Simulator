#ifndef PLANE_H
#define PLANE_H


#include<glm/glm.hpp>
#include<glm/gtx/rotate_vector.hpp>
#include<glm/gtx/quaternion.hpp>
#include<glm/gtx/norm.hpp>
#include<GL/glew.h>
using namespace glm;

#include <iostream>

class Plane
{
public:
	Plane();
	Plane(glm::vec3 c, glm::vec3 n, float w);
	~Plane();

	void draw();
	float distTest(vec3 p);

	float width;
	glm::vec3 centre;
	glm::vec3 normal;
private:

};

#endif
