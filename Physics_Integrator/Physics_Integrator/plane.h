#ifndef PLANE_H
#define PLANE_H


#include<glm.hpp>
#include<gtx/rotate_vector.hpp>
#include<gtx/quaternion.hpp>
#include<gtx/norm.hpp>
#include<glew.h>
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

private:
	glm::vec3 centre;
	glm::vec3 normal;
	glm::quat RotationBetweenVectors(vec3 start, vec3 dest);

};

#endif


