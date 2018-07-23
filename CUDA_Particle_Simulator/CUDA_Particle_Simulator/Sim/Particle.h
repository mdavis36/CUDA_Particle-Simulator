#ifndef PARTICLE_H
#define PARTICLE_H

#include <glm/glm.hpp>
using namespace glm;

class Particle
{
private:

public:
      Particle(float m, vec3 x, vec3 v, vec3 f);
      ~Particle();

      float m;
      vec3 x;
      vec3 v;
      vec3 f;
};

#endif
