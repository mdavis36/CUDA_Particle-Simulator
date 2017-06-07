#include "plane.h"

Plane::Plane()
{
	centre = glm::vec3(0.0f, 0.0f, 0.0f);
	normal = normalize(glm::vec3(0.0f, 1.0f, 0.0f));
	this->width = 20;
}

Plane::Plane(glm::vec3 c, glm::vec3 n, float w)
{
	centre = c;
	normal = normalize(n);
	width = w;
}

Plane::~Plane()
{
}

void Plane::draw()
{
	// Draw the floor grid.
	glPushMatrix();
	
	// Translate 
	glTranslatef(centre.x, centre.y, centre.z);

	// Do Rotation
	vec3 up = vec3(0.0f, 1.0f, 0.0f);
	if (!(abs(normal.y) == 1 && up.y == 1)) 
	{
		vec2 n_proj_zx = normalize(vec2(normal.z, normal.x));
		vec2 up_proj_zx = normalize(vec2(up.x, -up.y));

		float zx_theta = acosf(dot(n_proj_zx, up_proj_zx)) * (180 / 3.14f);
		normal.x < 0 ? zx_theta = 90 - zx_theta : zx_theta = 90 - zx_theta;
		normal.z < 0 ? true : zx_theta = 180 - zx_theta;

		float xy_theta = acosf(dot(normal, up)) * (180 / 3.14f);
		normal.y < 0 ? xy_theta = 180 - xy_theta : xy_theta = 180 - xy_theta;

		glRotatef(zx_theta, 0.0f, 1.0f, 0.0f);
		glRotatef(xy_theta, 1.0f, 0.0f, 0.0f);
	}

	glBegin(GL_LINES);

	for (float i = -width; i <= width; i += 1)
	{
		if (i == 0)
			glColor3f(0.0f, 1.0f, 1.0f);
		else
			glColor3f(0.5f, 0.5f, 0.5f);		
		///*
		glVertex3f((float)i, 0, (float)-width);
		glVertex3f((float)i, 0, (float)width);

		glVertex3f((float)-width, 0, (float)i);
		glVertex3f((float)width, 0, (float)i);
		//*/		
	}

	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0,0,-width);

	glEnd();
	glPopMatrix();

	glBegin(GL_LINES);
	glColor3f(1.0f, 1.0f, 0.0f);
	glVertex3f(centre.x, centre.y, centre.z);
	glVertex3f(centre.x + normal.x, centre.y + normal.y, centre.z + normal.z);
	glEnd();
}

float Plane::distTest(vec3 p)
{
	return dot((p - centre),normal);
}
