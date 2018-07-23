#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <GL/glew.h>
#include <iostream>
#include <vector>

#include "Particle.h"

#include "../Graphics/Drawable.h"

using namespace std;

class ParticleSystem : public Drawable
{
private:
      std::vector<vec3> _positions;
      std::vector<vec4> _colors;

      GLuint _vao;
      GLuint _buffers[2];

      bool _initialized;

public:
      ParticleSystem();
      ParticleSystem(int n);
      ~ParticleSystem();
      virtual bool init();
      virtual void draw();

      std::vector<Particle> _particles;
      int _num_particles;
      float t;
};

#endif
