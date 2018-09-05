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
                  // vec3 r = pos_center - p->_particles[i].x;
                  // float r_mag = glm::length(r);
                  // p->_particles[i].f = normalize(r) * (100.0f * (p->_particles[i].m * m_center) / (r_mag));

                  p->_particles[i].f[0] = 0;
                  p->_particles[i].f[1] = -9.81;
                  p->_particles[i].f[2] = 0;
            }
      }

      void EulerStep(ParticleSystem *p, float dt)
      {
            float temp1[ParticleDims(p)];
            float temp2[ParticleDims(p)];
            float temp3[ParticleDims(p)];

            ParticleDerivative(p, temp1);                        // t1 = s`
            ScaleVector(temp1, dt, ParticleDims(p));             // t1 = t1 * dt = s`*dt
            ParticleGetState(p, temp2);                          // t2 = s
            AddVectors(temp3, temp1, temp2, ParticleDims(p));    // t2 = t1 + t2 = s + s`*dt
            //AddVectors(temp2, temp1, temp2, ParticleDims(p));    // t2 = t1 + t2 = s + s`*dt

            CheckCollisions(temp3, temp2, ParticleDims(p));

            ParticleSetState(p, temp3);
            p->t += dt;
      }


      void F(float *s_bar, ParticleSystem * p, int size)
      {
            int i = 0;
            #pragma omp parallel for num_threads(4)
            for (i = 0; i < p->_num_particles; i++)
            {
                  float m = p->_particles[i].m;
                  s_bar[6 * i]     = s_bar[6 * i + 3];
                  s_bar[6 * i + 1] = s_bar[6 * i + 4];
                  s_bar[6 * i + 2] = s_bar[6 * i + 5];
                  s_bar[6 * i + 3] = 0 / m;
                  s_bar[6 * i + 4] = -9.81 / m;
                  s_bar[6 * i + 5] = 0 / m;
            }
      }


      void RK4(ParticleSystem *p, float dt)
      {
            float k_0[ParticleDims(p)];
            float k_2[ParticleDims(p)];
            float k_3[ParticleDims(p)];
            float k_4[ParticleDims(p)];
            float k_1[ParticleDims(p)];

            ParticleGetState(p, k_0);

            ParticleDerivative(p, k_1);

            CopyVector(k_2, k_1, ParticleDims(p));
            ScaleVector(k_2, dt / 2, ParticleDims(p));
            AddVectors(k_2, k_0, k_2, ParticleDims(p));
            F(k_2, p, ParticleDims(p));

            CopyVector(k_3, k_2, ParticleDims(p));
            ScaleVector(k_3, dt / 2, ParticleDims(p));
            AddVectors(k_3, k_0, k_3, ParticleDims(p));
            F(k_3, p, ParticleDims(p));

            CopyVector(k_4, k_3, ParticleDims(p));
            ScaleVector(k_4, dt, ParticleDims(p));
            AddVectors(k_4, k_0, k_4, ParticleDims(p));
            F(k_4, p, ParticleDims(p));

            ScaleVector(k_2, 2, ParticleDims(p));
            ScaleVector(k_3, 2, ParticleDims(p));
            AddVectors(k_1, k_1, k_3, ParticleDims(p));
            AddVectors(k_2, k_2, k_4, ParticleDims(p));
            AddVectors(k_1, k_1, k_2, ParticleDims(p));
            ScaleVector(k_1, dt / 6, ParticleDims(p));
            AddVectors(k_1, k_0, k_1,ParticleDims(p));

            CheckCollisions(k_1, k_0, ParticleDims(p));

            ParticleSetState(p, k_1);
      }

      void CopyVector(float *out, float *in, int size)
      {
            int i;
            #pragma omp parallel for num_threads(4)
            for(i = 0; i < size; i++)
            {
                  out[i] = in[i];
            }
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

      void CheckCollisions(float *curr, float *last, int size)
      {
            int i;
            for (i = 0; i < size; i+=6)
            {
                  if (curr[i+1] < 0 && last[i+1] > 0)
                  {
                        curr[i+1] = last[i+1];
                        if (abs(last[i+4]) > 0.001)
                              curr[i+4] = -last[i+4] * .5;
                        // else
                        //       curr[i+4] = 0.0f;
                        //       curr[i+1] = 0.0f;
                  }
            }
      }



}
