#ifndef COLLISIONDATA
#define COLLISIONDATA

#include "glm/glm.hpp"
#include "../Geometry/Polygon.h"
#include <cuda_runtime.h>

class CollisionData {
public:
	CollisionData(); 
	CollisionData(glm::vec3 _I, Polygon _p, float _r_I) : I(_I), p(_p), r_I(_r_I) {};
	glm::vec3 I;
	Polygon p;
	float r_I;
};

#endif
