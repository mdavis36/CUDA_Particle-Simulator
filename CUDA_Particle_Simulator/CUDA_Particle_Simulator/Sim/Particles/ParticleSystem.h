#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <GL/glew.h>
#include "glm/gtc/type_ptr.hpp"
#include <iostream>
#include <vector>

#include "Particle.h"

#include "../../Graphics/Drawable.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"
#include <curand_kernel.h>

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

      // CUDA - OpenGL Mapping variables
      cudaGraphicsResource_t res = 0;
      size_t size = 0;
      void* device_ptr;



public:
      ParticleSystem();
      ParticleSystem(int n, int form);
      ~ParticleSystem();
      virtual bool init(GLuint* programs);
      virtual void draw(GLuint* programs, mat4 proj_mat, mat4 view_mat);

      dim3 block;
      dim3 grid;

      Particle* d_particles;
      std::vector<Particle> _particles;
      int _num_particles;
      float t;
};

#endif
