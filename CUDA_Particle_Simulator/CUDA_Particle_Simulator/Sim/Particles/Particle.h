#ifndef PARTICLE_H
#define PARTICLE_H

#include <glm/glm.hpp>
#include <iostream>
#include <cuda_runtime.h>

using namespace glm;
using namespace std;

class Particle
{
private:

public:
      __device__ __host__ Particle(float m, vec3 x, vec3 v, vec3 f);
      void print();

      float m;
      vec3 x;
      vec3 v;
      vec3 f;
};

#endif
