#ifndef COLLISIONDATA
#define COLLISIONDATA

#include "glm/glm.hpp"
#include <cuda_runtime.h>

class CollisionData {
public:
	CollisionData(); 
	CollisionData(glm::vec3 _I, glm::vec3 _n, float _r_I) : I(_I), n(_n), r_I(_r_I) {};
	glm::vec3 I;
	glm::vec3 n;
	float r_I;
};

#endif
