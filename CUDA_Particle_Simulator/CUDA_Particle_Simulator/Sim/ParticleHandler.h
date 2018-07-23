#ifndef PARTICLEHANDLER_H
#define PARTICLEHANDLER_H

#include "ParticleSystem.h"

namespace ParticleHandler {

      int ParticleDims(ParticleSystem *p);
      int ParticleGetState(ParticleSystem *p, float *dst);
      int ParticleSetState(ParticleSystem *p, float *src);
      int ParticleDerivative(ParticleSystem *p, float *dst);

      void Clear_Forces(ParticleSystem *p);
      void Compute_Forces(ParticleSystem *p);

      void EulerStep(ParticleSystem *p, float dt);

      void ScaleVector(float *v, float s, int size);
      void AddVectors(float *s, float *a, float *b, int size);

};

#endif
