#ifndef POLYGON_H
#define POLYGON_H

#include <glm/glm.hpp>
#include <iostream>
#include "../Particles/CollisionData.h"
class Polygon
{
public:
      glm::vec3 v[3];
	glm::vec3 n;
	Polygon();
      Polygon(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);
      Polygon(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 _n);
	
	bool checkPolygonIntersection(glm::vec3 x_0, glm::vec3 x_1, CollisionData &result);
      void print() const;
};


#endif
