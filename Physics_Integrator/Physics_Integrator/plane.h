#ifndef PLANE_H
#define PLANE_H


#include<glm.hpp>
#include<glew.h>


class Plane 
{
public:
	Plane();
	Plane(float width);
	~Plane();

	void draw();

private:
	glm::vec3 centre;
	glm::vec3 normal;

	float width;


};








#endif