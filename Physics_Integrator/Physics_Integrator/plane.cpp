#include "plane.h"

Plane::Plane()
{
	centre = glm::vec3(0.0f, 0.0f, 0.0f);
	normal = glm::vec3(0.0f, 1.0f, 0.0f);
	this->width = 20;
}

Plane::Plane(float width)
{
	centre = glm::vec3(0.0f, 0.0f, 0.0f);
	normal = glm::vec3(0.0f, 1.0f, 0.0f);
	this->width = width;
}

Plane::~Plane()
{
}

void Plane::draw()
{
	// Draw the floor grid.
	glBegin(GL_LINES);
	for (int i = -width; i <= width; i++)
	{
		if (i == 0)
			glColor3f(0.0f, 1.0f, 1.0f);
		else
			glColor3f(0.5f, 0.5f, 0.5f);

		glVertex3f((float)i, 0, (float)-width);
		glVertex3f((float)i, 0, (float)width);

		glVertex3f((float)-width, 0, (float)i);
		glVertex3f((float)width, 0, (float)i);
	}
	glEnd();

}
