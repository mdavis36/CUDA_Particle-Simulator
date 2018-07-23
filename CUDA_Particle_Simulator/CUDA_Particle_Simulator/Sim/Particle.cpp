#include "Particle.h"


Particle::Particle(float m, vec3 x, vec3 v, vec3 f)
{
      this->m = m;
      this->x = x;
      this->v = v;
      this->f = f;
}

Particle::~Particle(){}
