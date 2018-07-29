#include "ParticleHandler.h"
#include <omp.h>
namespace ParticleHandler
{
      int ParticleDims(ParticleSystem *p)
      {
            return (p->_num_particles * 6);
      }

      void ParticleGetState(ParticleSystem *p, float *dst)
      {
            int i;
            #pragma omp parallel for num_threads(4)
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

      void ParticleSetState(ParticleSystem *p, float *src)
      {
            int i;
            #pragma omp parallel for num_threads(4)
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

      void ParticleDerivative(ParticleSystem *p, float *dst)
      {
            int i;
            Clear_Forces(p);
            Compute_Forces(p);
            #pragma omp parallel for num_threads(4)
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
            #pragma omp parallel for num_threads(4)
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
            vec3 pos_center = vec3(0,15,0);
            float m_center = 0.2;

            #pragma omp parallel for num_threads(4)
            for(i = 0; i < p->_num_particles; i++)
            {
                  vec3 r = pos_center - p->_particles[i].x;
                  float r_mag = glm::length(r);
                  p->_particles[i].f = normalize(r) * (100.0f * (p->_particles[i].m * m_center) / (r_mag));

                  // p->_particles[i].f[0] = 0;
                  // p->_particles[i].f[1] = -9.81;
                  // p->_particles[i].f[2] = 0;
            }
      }

      void EulerStep(ParticleSystem *p, float dt)
      {
            float temp1[ParticleDims(p)];
            float temp2[ParticleDims(p)];
            ParticleDerivative(p, temp1);                        // t1 = s`
            ScaleVector(temp1, dt, ParticleDims(p));             // t1 = t1 * dt = s`*dt
            ParticleGetState(p, temp2);                          // t2 = s
            AddVectors(temp2, temp1, temp2, ParticleDims(p));    // t2 = t1 + t2 = s + s`*dt
            ParticleSetState(p, temp2);
            p->t += dt;
      }

      void ScaleVector(float *v, float s, int size)
      {
            int i;
            #pragma omp parallel for num_threads(4)
            for(i = 0; i < size; i++)
            {
                  v[i] *= s;
            }
      }

      void AddVectors(float *s, float *a, float *b, int size)
      {
            int i;
            #pragma omp parallel for num_threads(4)
            for(i = 0; i < size; i++)
            {
                  s[i] = a[i] + b[i];
            }
      }
}
