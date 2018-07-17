#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

typdef struct
{
            float m;
            float *x;
            float *v;
            float *f;
} *Particle;

typdef struct
{
      Particle *p;
      int n;
      float t;
} *ParticleSystem;

int ParticleDims(ParticleSystem p);
int ParticleGetState(ParticleSystem p, float *dst);
int ParticleSetState(ParticleSystem p, float *src);
int ParticleDerivative(ParticleSystem p, float *dst);

#endif
