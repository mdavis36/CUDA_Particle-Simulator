#include "ParticleSystem.h"

__device__ void ParticleGetState(Particle *p, float *dst)
{
      dst[0] = p->x[0];
      dst[1] = p->x[1];
      dst[2] = p->x[2];
      dst[3] = p->v[0];
      dst[4] = p->v[1];
      dst[5] = p->v[2];
}

__device__ void ParticleSetState(Particle *p, float *src)
{
      p->x[0] = src[0];
      p->x[1] = src[1];
      p->x[2] = src[2];
      p->v[0] = src[3];
      p->v[1] = src[4];
      p->v[2] = src[5];
}

__device__ void ParticleDerivative(Particle * p, float * dst)
{
      float m = p->m;
      dst[0] = p->v[0];
      dst[1] = p->v[1];
      dst[2] = p->v[2];
      dst[3] = p->f[0] / m;
      dst[4] = p->f[1] / m;
      dst[5] = p->f[2] / m;
}

__device__ void ClearForces(Particle *p)
{
      p->f[0] = 0;
      p->f[1] = 0;
      p->f[2] = 0;
}

// Euler Only
__device__ void ComputeForces(Particle *p)
{
      p->f[0] = 0;
      p->f[1] = -9.81;
      p->f[2] = 0;
}

// RK4 Only
__device__ void F(float *s_bar, Particle *p)
{
      float m = p->m;
      s_bar[0] = s_bar[3];
      s_bar[1] = s_bar[4];
      s_bar[2] = s_bar[5];
      s_bar[3] = 0 / m;
      s_bar[4] = -9.81 / m;
      s_bar[5] = 0 / m;
}

__device__ void CopyVector(float *out, float *in)
{
      int i;
      for(i = 0; i < 6; i++)
      {
            out[i] = in[i];
      }
}

__device__ void ScaleVector(float *v, float s)
{
      int i;
      for(i = 0; i < 6; i++)
      {
            v[i] *= s;
      }
}

__device__ void AddVectors(float *s, float *a, float *b)
{
      int i;
      for(i = 0; i < 6; i++)
      {
            s[i] = a[i] + b[i];
      }
}

__global__ void RK4(Particle* ps, float dt)
{
      float k_0[6];
      float k_2[6];
      float k_3[6];
      float k_4[6];
      float k_1[6];

      //Get index...
      int indx = 0;

      Particle p = ps[indx];

      ParticleGetState( &p, k_0 );

      ParticleDerivative( &p, k_1 );

      CopyVector( k_2, k_1 );
      ScaleVector( k_2, dt / 2 );
      AddVectors( k_2, k_0, k_2 );
      F( k_2, &p );

      CopyVector( k_3, k_2 );
      ScaleVector( k_3, dt / 2 );
      AddVectors( k_3, k_0, k_3 );
      F( k_3, &p );

      CopyVector( k_4, k_3 );
      ScaleVector( k_4, dt );
      AddVectors( k_4, k_0, k_4 );
      F( k_4, &p );

      ScaleVector( k_2, 2 );
      ScaleVector( k_3, 2 );
      AddVectors( k_1, k_1, k_3 );
      AddVectors( k_2, k_2, k_4 );
      AddVectors( k_1, k_1, k_2 );
      ScaleVector( k_1, dt / 6 );
      AddVectors( k_1, k_0, k_1 );

      //CheckCollisions( poly, k_1, k_0 );

      ParticleSetState( &p, k_1 );
}
