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

      void EulerStep(ParticleSystem *p, std::vector<Polygon>* poly, float dt)
      {
            float temp1[ParticleDims(p)];
            float temp2[ParticleDims(p)];
            float temp3[ParticleDims(p)];

            ParticleDerivative(p, temp1);                        // t1 = s`
            ScaleVector(temp1, dt, ParticleDims(p));             // t1 = t1 * dt = s`*dt
            ParticleGetState(p, temp2);                          // t2 = s
            AddVectors(temp3, temp1, temp2, ParticleDims(p));    // t2 = t1 + t2 = s + s`*dt
            AddVectors(temp2, temp1, temp2, ParticleDims(p));    // t2 = t1 + t2 = s + s`*dt

            CheckCollisions(poly, temp3, temp2, ParticleDims(p));

            ParticleSetState(p, temp3);
            //p->t += dt;
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
                  s_bar[6 * i + 4] = -.0981 / m;
                  s_bar[6 * i + 5] = 0 / m;
            }
      }


      void RK4(ParticleSystem *p, std::vector<Polygon>* poly, float dt)
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

            CheckCollisions(poly, k_1, k_0, ParticleDims(p));

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

      void CheckCollisions(std::vector<Polygon>* poly, float *curr, float *last, int size)
      {
            int i;

            // http://geomalgorithms.com/a06-_intersect-2.html

            //Iterate over each particle
            for (i = 0; i < size; i+=6)
            {
                  glm::vec3 x_0( last[ i + 0 ], last[ i + 1 ], last[ i + 2 ]);
                  glm::vec3 x_1( curr[ i + 0 ], curr[ i + 1 ], curr[ i + 2 ]);
                  glm::vec3 x_01 = (x_1 - x_0);

                  // for each polygon available
                  #pragma omp parallel for num_threads(4)
                  for (int j = 0; j < poly->size(); j++)
                  {
                        glm::vec3 v_0  = poly->at(j).v[0];
                        glm::vec3 v_1  = poly->at(j).v[1];
                        glm::vec3 v_2  = poly->at(j).v[2];

                        glm::vec3 v_01 = v_1 - v_0;
                        glm::vec3 v_02 = v_2 - v_0;

                        glm::vec3 n = glm::normalize( glm::cross( ( v_1 - v_0 ), ( v_2 - v_0 ) ) );
                        float r_I = ( glm::dot(n, v_0 - x_0) ) / ( glm::dot(n, x_1 - x_0) );

                        //    Calculate U and V
                        // float denom =  glm::dot( -x_01, glm::cross( v_01, v_02) );
                        // float u = ( glm::dot( glm::cross( v_02, -x_01 ), x_0 - v_0 ) ) / ( denom );
                        // float v = ( glm::dot( glm::cross( -x_01, v_01 ), x_0 - v_0 ) ) / ( denom );
                        // std::cout << u << "  " << v << "  " << denom << "  " << glm::cross( v_01, v_02)[0] << ", " << glm::cross( v_01, v_02)[1] << ", " << glm::cross( v_01, v_02)[2] << std::endl;
                        //          if U+V <= 1 : Collision detected
                        if (!(0 <= r_I && r_I <= 1))
                        {
                             continue;
                        }

                        glm::vec3 I(x_0 + r_I * x_01);

                        // Check if intersection lies within triangle
                        float uu, uv, vv, wu, wv, D;
                        uu = glm::dot(v_01, v_01);
                        uv = glm::dot(v_01, v_02);
                        vv = glm::dot(v_02, v_02);
                        glm::vec3 w = I - v_0;
                        wu = glm::dot(w,v_01);
                        wv = glm::dot(w,v_02);
                        D = uv * uv - uu * vv;

                        //test parametric co-ords
                        float s, t;
                        s = (uv * wv - vv * wu) / D;
                        t = (uv * wu - uu * wv) / D;

                        if (s >= 0 && t >= 0 && s+t <= 1) {
                              std::cout << "COLLISION!!!! Particle " << i / 6 << " -> Polygon " << j << std::endl;
                        }
                  }
            }
      }
}
