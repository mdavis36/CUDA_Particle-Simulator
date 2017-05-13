#include "plane.h"

Plane::Plane()
{
	centre = glm::vec3(0.0f, 0.0f, 0.0f);
	normal = normalize(glm::vec3(0.0f, 1.0f, 0.0f));
	this->width = 20;
}

Plane::Plane(glm::vec3 centre, glm::vec3 normal, float width)
{
	this->centre = centre;
	this->normal = normalize(normal);
	this->width = width;
}

Plane::~Plane()
{
}

void Plane::draw()
{
	// Draw the floor grid.
	glPushMatrix();
	
	glTranslatef(centre.x, centre.y, centre.z);

	float xy_theta = acosf(dot(vec2(normal.x, normal.y), vec2(0.0f, 1.0f))) * (180 / 3.14f);
	float xz_theta = acosf(dot(vec2(normal.x, normal.z), vec2(0.0f, 0.0f))) * (180 / 3.14f);
	//float yz_theta = acosf(dot(vec2(normal.y, normal.z), vec2(1.0f, 0.0f))) * (180 / 3.14f);

	glRotatef(xy_theta, 0.0f, 0.0f, 1.0f);
	glRotatef(xz_theta, 0.0f, 1.0f, 0.0f);
	//glRotatef(yz_theta, 1.0f, 0.0f, 0.0f);

	//glRotatef(180, 0, normal.y, normal.z);

	//glRotatef(acosf(dot(normal, vec3(0.0f, 1.0f, 0.0f))) * (180.0 / 3.14f), 1.0f, 1.0f, 1.0f);
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
		/*
		glVertex3f((float)i, (-normal.x * i - normal.z * i - d) * 1.0f / normal.z, (float)-width);
		glVertex3f((float)i, (-normal.x * i - normal.z * i - d) * 1.0f / normal.z, (float)width);

		glVertex3f((float)-width, (-normal.x * i - normal.z * i - d) * 1.0f / normal.z, (float)i);
		glVertex3f((float)width, (-normal.x * i - normal.z * i - d) * 1.0f / normal.z, (float)i);
		*/
		//float theta
		
	}
	
	glEnd();
	glPopMatrix();

	glBegin(GL_LINES);
	glColor3f(1.0f, 1.0f, 0.0f);
	glVertex3f(centre.x, centre.y, centre.z);
	glVertex3f(centre.x + normal.x, centre.y + normal.y, centre.z + normal.z);
	glEnd();

	//glBegin(GL_POINT);
	//glVertex3f(centre.x + normal.x, centre.y + normal.y, centre.z + normal.z);
	//glEnd();

}

float Plane::distTest(vec3 p)
{
	return dot((p - centre),normal);
}
