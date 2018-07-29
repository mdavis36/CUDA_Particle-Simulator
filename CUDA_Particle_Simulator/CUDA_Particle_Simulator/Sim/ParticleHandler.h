#ifndef PARTICLEHANDLER_H
#define PARTICLEHANDLER_H

#include "ParticleSystem.h"
#include "glm/glm.hpp"

namespace ParticleHandler {

      int ParticleDims(ParticleSystem *p);
      void ParticleGetState(ParticleSystem *p, float *dst);
      void ParticleSetState(ParticleSystem *p, float *src);
      void ParticleDerivative(ParticleSystem *p, float *dst);

      void Clear_Forces(ParticleSystem *p);
      void Compute_Forces(ParticleSystem *p);

      void EulerStep(ParticleSystem *p, float dt);

      void ScaleVector(float *v, float s, int size);
      void AddVectors(float *s, float *a, float *b, int size);

};

#endif
