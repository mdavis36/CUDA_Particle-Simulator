#ifndef POLYGON_H
#define POLYGON_H

#include <glm/glm.hpp>
#include <iostream>

class Polygon
{
public:
      glm::vec3 v[3];
	glm::vec3 n;
	Polygon();
      Polygon(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3);
      Polygon(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 _n);
      void print() const;
};


#endif
