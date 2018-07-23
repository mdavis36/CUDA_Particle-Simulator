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
            for(i = 0; i < p->_num_particles; i++)
            {
                  dst[6 * i + 0] = p->_particles[i].x[0];
                  dst[6 * i + 1] = p->_particles[i].x[1];
                  dst[6 * i + 2] = p->_particles[i].x[2];
                  dst[6 * i + 3] = p->_particles[i].v[0];
                  dst[6 * i + 4] = p->_particles[i].v[1];
                  dst[6 * i + 5] = p->_particles[i].v[2];
            }
      }

      int ParticleSetState(ParticleSystem *p, float *src)
      {
            int i;
            for(i = 0; i < p->_num_particles; i++)
            {
                  p->_particles[i].x[0] = src[6 * i + 0];
                  p->_particles[i].x[1] = src[6 * i + 1];
                  p->_particles[i].x[2] = src[6 * i + 2];
                  p->_particles[i].v[0] = src[6 * i + 3];
                  p->_particles[i].v[1] = src[6 * i + 4];
                  p->_particles[i].v[2] = src[6 * i + 5];
            }
      }

      int ParticleDerivative(ParticleSystem *p, float *dst)
      {
            int i;
            Clear_Forces(p);
            Compute_Forces(p);
            for(i = 0; i < p->_num_particles; i++)
            {
                  float m = p->_particles[i].m;
                  dst[6 * i + 0] = p->_particles[i].v.x;
                  dst[6 * i + 1] = p->_particles[i].v.y;
                  dst[6 * i + 2] = p->_particles[i].v.z;
                  dst[6 * i + 3] = p->_particles[i].f.x / m;
                  dst[6 * i + 4] = p->_particles[i].f.y / m;
                  dst[6 * i + 5] = p->_particles[i].f.z / m;
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
            float *temp1 = new float[ParticleDims(p)];
            float *temp2 = new float[ParticleDims(p)];
            ParticleDerivative(p, temp1);                        // t1 = s`
            ScaleVector(temp1, dt, ParticleDims(p));             // t1 = t1 * dt = s`*dt
            ParticleGetState(p, temp2);                          // t2 = s
            AddVectors(temp2, temp1, temp2, ParticleDims(p));    // t2 = t1 + t2 = s + s`*dt
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
                  s[i] = a[i] + b[i];
            }
      }
}
