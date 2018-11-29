#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <GL/glew.h>
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include <vector>

#include "Particle.h"

#include "../../Graphics/Drawable.h"

using namespace std;

const int PARTICLE_CUBE = 1;
const int PARTICLE_SPHERE = 2;

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
      ParticleSystem(int n, int form);
      ~ParticleSystem();
      virtual bool init(GLuint* programs);
      virtual void draw(GLuint* programs, mat4 proj_mat, mat4 view_mat);

      std::vector<Particle> _particles;
      int _num_particles;
      float t;
};

#endif
