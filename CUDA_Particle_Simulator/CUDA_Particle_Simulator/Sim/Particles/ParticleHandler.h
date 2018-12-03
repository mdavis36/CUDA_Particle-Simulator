#ifndef PARTICLEHANDLER_H
#define PARTICLEHANDLER_H

#include "ParticleSystem.h"
#include "glm/glm.hpp"
#include "../Geometry/Polygon.h"
#include <cuda_runtime.h>

namespace ParticleHandler {

      int ParticleDims(ParticleSystem *p);
      __host__   void ParticleGetState(ParticleSystem *p, float *dst);
      __device__ void ParticleGetState(Particle *p, float *dst);
      __host__   void ParticleSetState(ParticleSystem *p, float *src);
      __device__ void ParticleSetState(Particle *p, float *src);
      __host__   void ParticleDerivative(ParticleSystem *p, float *dst);
      __device__ void ParticleDerivative(Particle * p, float * dst);

      __host__   void Clear_Forces(ParticleSystem *p);
      __device__ void ClearForces(Particle *p);
      __host__   void Compute_Forces(ParticleSystem *p);
      __device__ void ComputeForces(Particle *p);

      __host__   void EulerStep(ParticleSystem *p, std::vector<Polygon>* poly, float dt);
      __host__   void RK4(ParticleSystem *p, std::vector<Polygon>* poly, float dt);
      __host__   void cuRK4(ParticleSystem *p, float dt);

      __host__   void CopyVector(float *out, float *in, int size);
      __device__ void CopyVector(float *out, float *in);

      __host__   void ScaleVector(float *v, float s, int size);
      __device__ void ScaleVector(float *v, float s);
      __host__   void AddVectors(float *s, float *a, float *b, int size);
      __device__ void AddVectors(float *s, float *a, float *b);

      __host__   void CheckCollisions(std::vector<Polygon>* poly, float *curr, float *last, int size);
};

#endif
