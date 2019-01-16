#ifndef PARTICLEHANDLER_H
#define PARTICLEHANDLER_H

#include "ParticleSystem.h"
#include "glm/glm.hpp"
#include "../Geometry/Polygon.h"
#include <cuda_runtime.h>
#include "CollisionData.h"
#include "../OctTree/OctTree.h"

namespace ParticleHandler {

      int ParticleDims(ParticleSystem *p);
      __host__   void ParticleGetState(ParticleSystem *p, float *dst);
      __host__ __device__ void ParticleGetState(Particle *p, float *dst);
      __host__   void ParticleSetState(ParticleSystem *p, float *src);
      __host__ __device__ void ParticleSetState(Particle *p, float *src);
      __host__   void ParticleDerivative(ParticleSystem *p, float *dst);
      __host__ __device__ void ParticleDerivative(Particle * p, float * dst);

      __host__   void Clear_Forces(ParticleSystem *p);
      __device__ void ClearForces(Particle *p);
      __host__   void Compute_Forces(ParticleSystem *p);
      __device__ void ComputeForces(Particle *p);

	__host__ __device__ void F(float *s_bar, Particle *p);

      __host__   void EulerStep(ParticleSystem *p, std::vector<Polygon>* poly, float dt);
      __host__   void RK4(ParticleSystem *p, std::vector<Polygon>* poly, float dt);
      __host__   void RK4eval(float *k_0, float *k_1, std::vector<Polygon>* poly, float dt, int size);
      __host__   void cuRK4(ParticleSystem *p, float dt);
      __host__   void RK4_2(ParticleSystem *ps, std::vector<Polygon>* poly, std::vector<OctTree*> node_list, float dt);

      __host__   void CopyVector(float *out, float *in, int size);
      __host__ __device__ void CopyVector(float *out, float *in);

      __host__   void ScaleVector(float *v, float s, int size);
      __host__ __device__ void ScaleVector(float *v, float s);
      __host__   void AddVectors(float *s, float *a, float *b, int size);
      __host__ __device__ void AddVectors(float *s, float *a, float *b);


      __host__   CollisionData CheckCollisions2(glm::vec3 x_0, glm::vec3 x_1, std::vector<Polygon>* poly, int p_indx);
      __host__   void CheckCollisions(std::vector<Polygon>* poly, float *curr, float *last, int size);
};

#endif
