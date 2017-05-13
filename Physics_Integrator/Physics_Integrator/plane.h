#ifndef PLANE_H
#define PLANE_H


#include<glm.hpp>
#include<glew.h>
using namespace glm;

class Plane 
{
public:
	Plane();
	Plane(glm::vec3 centre, glm::vec3 normal, float width);
	~Plane();

	void draw();
	float distTest(vec3 p);


	float width;

private:
	glm::vec3 centre;
	glm::vec3 normal;


	


};








#endif