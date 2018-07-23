#include "Particle.h"


Particle::Particle(float m, vec3 x, vec3 v, vec3 f)
{
      this->m = m;
      this->x = x;
      this->v = v;
      this->f = f;
}

Particle::~Particle(){}


void Particle::print()
{
      cout << "pos : " << x[0] << " " << x[1] << " " << x[2] << endl;
      cout << "vel : " << v[0] << " " << v[1] << " " << v[2] << endl;
      cout << "for : " << f[0] << " " << f[1] << " " << f[2] << endl;
      cout << "mas : " << m << endl;
}
