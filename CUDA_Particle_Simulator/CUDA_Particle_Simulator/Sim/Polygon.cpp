#include "Polygon.h"

Polygon::Polygon(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3)
{
      v[0] = v1;
      v[1] = v2;
      v[2] = v3;
}

void Polygon::print() const
{
      std::cout << (float)v[0].x << ", "<< (float)v[0].y << ", "<< (float)v[0].z << " | "
	          << (float)v[1].x << ", "<< (float)v[1].y << ", "<< (float)v[1].z << " | "
		    << (float)v[2].x << ", "<< (float)v[2].y << ", "<< (float)v[2].z << " | " << std::endl;
}
