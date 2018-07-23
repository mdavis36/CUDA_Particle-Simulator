#include "ParticleHandler.h"

namespace ParticleHandler
{
      int ParticleDims(ParticleSystem *p)
      {
            return (p->_num_particles * 6);
      }

      int ParticleGetState(ParticleSystem *p, float *dst)
      {
            int i;
            dst = new float[ParticleDims(p)];
            for(i = 0; i < p->_num_particles; i++)
            {
                  dst[i * 0 + i] = p->_particles[i].x[0];
                  dst[i * 1 + i] = p->_particles[i].x[1];
                  dst[i * 2 + i] = p->_particles[i].x[2];
                  dst[i * 3 + i] = p->_particles[i].v[0];
                  dst[i * 4 + i] = p->_particles[i].v[1];
                  dst[i * 5 + i] = p->_particles[i].v[2];
            }
      }

      int ParticleSetState(ParticleSystem *p, float *src)
      {
            int i;
            src = new float[ParticleDims(p)];
            for(i = 0; i < p->_num_particles; i++)
            {
                  p->_particles[i].x[0] = src[i * 0 + i];
                  p->_particles[i].x[1] = src[i * 1 + i];
                  p->_particles[i].x[2] = src[i * 2 + i];
                  p->_particles[i].v[0] = src[i * 3 + i];
                  p->_particles[i].v[1] = src[i * 4 + i];
                  p->_particles[i].v[2] = src[i * 5 + i];
            }
      }

      int ParticleDerivative(ParticleSystem *p, float *dst)
      {
            int i;
            Clear_Forces(p);
            Compute_Forces(p);
            dst = new float[ParticleDims(p)];
            for(i = 0; i < p->_num_particles; i++)
            {
                  float m = p->_particles[i].m;
                  dst[i * 0 + i] = p->_particles[i].v[0];
                  dst[i * 1 + i] = p->_particles[i].v[1];
                  dst[i * 2 + i] = p->_particles[i].v[2];
                  dst[i * 3 + i] = p->_particles[i].f[0] / m;
                  dst[i * 4 + i] = p->_particles[i].f[1] / m;
                  dst[i * 5 + i] = p->_particles[i].f[2] / m;
            }
      }

      void Clear_Forces(ParticleSystem *p)
      {
            int i;
            for(i = 0; i < p->_num_particles; i++)
            {
                  p->_particles[i].f[0] = 0;
                  p->_particles[i].f[1] = 0;
                  p->_particles[i].f[2] = 0;
            }
      }

      void Compute_Forces(ParticleSystem *p)
      {
            int i;

            for(i = 0; i < p->_num_particles; i++)
            {
                  p->_particles[i].f[0] = 0;
                  p->_particles[i].f[1] = -9.81;
                  p->_particles[i].f[2] = 0;
            }
      }

      void EulerStep(ParticleSystem *p, float dt)
      {
            float *temp1;
            float *temp2;
            ParticleDerivative(p, temp1);       // t1 = s`
            ScaleVector(temp1, dt, ParticleDims(p));             // t1 = t1 * dt = s`*dt
            ParticleGetState(p, temp2);         // t2 = s
            AddVectors(temp1, temp2, temp2, ParticleDims(p));    // t2 = t1 + t2 = s + s`*dt
            ParticleSetState(p, temp2);
      }

      void ScaleVector(float *v, float s, int size)
      {
            int i;
            for(i = 0; i < size; i++)
            {
                  v[i] *= s;
            }
      }
      void AddVectors(float *s, float *a, float *b, int size)
      {
            int i;
            for(i = 0; i < size; i++)
            {
                  *(s++) = *(a++) + *(b++);
            }
      }
}
