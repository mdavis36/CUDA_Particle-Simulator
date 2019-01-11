#include "ParticleHandler.h"
#include <omp.h>
namespace ParticleHandler
{      
	
	void ParticleGetState2(Particle *p, float *dst)
      {
            dst[0] = p->x[0];
            dst[1] = p->x[1];
            dst[2] = p->x[2];
            dst[3] = p->v[0];
            dst[4] = p->v[1];
            dst[5] = p->v[2];
      }

      void ParticleSetState2(Particle *p, float *src)
      {
            p->x[0] = src[0];
            p->x[1] = src[1];
            p->x[2] = src[2];
            p->v[0] = src[3];
            p->v[1] = src[4];
            p->v[2] = src[5];
      }

      void ParticleDerivative2(Particle * p, float * dst)
      {
            float m = p->m;
            dst[0] = p->v[0];
            dst[1] = p->v[1];
            dst[2] = p->v[2];
            dst[3] = p->f[0] / m;
            dst[4] = p->f[1] / m;
            dst[5] = p->f[2] / m;
      }

      void ClearForces2(Particle *p)
      {
            p->f[0] = 0;
            p->f[1] = 0;
            p->f[2] = 0;
      }

      // Euler Only
      void ComputeForces2(Particle *p)
      {
            p->f[0] = 0;
            p->f[1] = -9.81;
            p->f[2] = 0;
      }

      // RK4 Only
      void F2(float *s_bar, Particle *p)
      {
            float m = p->m;
            s_bar[0] = s_bar[3];
            s_bar[1] = s_bar[4];
            s_bar[2] = s_bar[5];
            s_bar[3] = 0 / m;
            s_bar[4] = -9.81 / m;
            s_bar[5] = 0 / m;
      }

      void CopyVector2(float *out, float *in)
      {
            int i;
            for(i = 0; i < 6; i++)
            {
                  out[i] = in[i];
            }
      }

      void ScaleVector2(float *v, float s)
      {
            int i;
            for(i = 0; i < 6; i++)
            {
                  v[i] *= s;
            }
      }

      void AddVectors2(float *s, float *a, float *b)
      {
            int i;
            for(i = 0; i < 6; i++)
            {
                  s[i] = a[i] + b[i];
            }
      }
 	
	void RK4_2_eval(Particle *p_0, Particle *p_1, float dt)	
	{
		float k_0[6];
            float k_2[6];
            float k_3[6];
            float k_4[6];
            float k_1[6];
		ParticleGetState2( p_0, k_0 );
		ParticleDerivative2( p_0, k_1 );

		CopyVector2( k_2, k_1 );
		ScaleVector2( k_2, dt / 2 );
		AddVectors2( k_2, k_0, k_2 );
		F2( k_2, p_0 );

		CopyVector2( k_3, k_2 );
		ScaleVector2( k_3, dt / 2 );
		AddVectors2( k_3, k_0, k_3 );
		F2( k_3, p_0 );

		CopyVector2( k_4, k_3 );
		ScaleVector2( k_4, dt );
		AddVectors2( k_4, k_0, k_4 );
		F2( k_4, p_0 );

		ScaleVector2( k_2, 2 );
		ScaleVector2( k_3, 2 );
		AddVectors2( k_1, k_1, k_3 );
		AddVectors2( k_2, k_2, k_4 );
		AddVectors2( k_1, k_1, k_2 );
		ScaleVector2( k_1, dt / 6 );
		AddVectors2( k_1, k_0, k_1 );

		//CheckCollisions( poly, k_1, k_0 );
		vec3 last(k_0[0], k_0[1], k_0[2]);
		vec3 next(k_1[0], k_1[1], k_1[2]);
		/*
		if (next.y < 0)
		{
			k_1[1] = -k_1[1];
			k_1[4] = -0.4 * k_1[4];
		}
		*/
	    	ParticleSetState2( p_1, k_1 );
		//p_1->x = next;
		//p_1->v = vec3(k_1[3], k_1[4], k_1[5]);

		//std::cout << "p0" << std::endl;
		//p_0->print();
		//std::cout << "p1" << std::endl;
		//p_1->print();
	
	}

      void RK4_2(ParticleSystem* ps, std::vector<Polygon>* poly, float dt)
      {
            float k_0[6];
            float k_2[6];
            float k_3[6];
            float k_4[6];
            float k_1[6];

            //Get index...
            //int indx = threadIdx.x + blockIdx.x * blockDim.x;

            #pragma omp parallel for num_threads(4)
		for (int indx = 0; indx < ps->_num_particles; indx++)
		{
			Particle p = ps->_particles[indx];
			Particle p_1 = p;
			RK4_2_eval(&p, &p_1, dt);

			CollisionData cd = CheckCollisions2(p.x, p_1.x, poly, indx);
			
			if (cd.r_I != -1)
			{
				std::cout << "cd test\n";
				float t_I = dt * cd.r_I;
				glm::vec3 v_I = p.v + (p.f / p.m)*t_I;
				glm::vec3 v_R = v_I - 2*(glm::dot(v_I, cd.p.n))*cd.p.n;

				Particle p_RI = p;
				p_RI.x = cd.I;
				p_RI.v = v_R*0.7f;

				Particle p_R1 = p_RI;
				RK4_2_eval(&p_RI, &p_R1, dt-t_I);

				ps->_particles[indx] = p_R1;
				continue;
			}
			ps->_particles[indx] = p_1;
		}
      }
/*
	void OctTreeCollisionDetection(glm::vec3 x_0, glm::vec3 x_1, std::vector<OctTree*>& node_list, int p_indx)
	{
		CollisionData result;
		bool r_set = false;

		for (int c = 0; c < 8; c++)
			{
				if (/*boxLineIntersection(node_list[leafs[c], x_0, x_1])/)
				{
					CollisionData t_res = node_list[leafs[c]]->OctTreeCollisionDetection(x_0, x_1, node_list);
					if ( !r_set ) { result = t_res; r_set = true; }
					if ( t_res.r_I < result.r_I ) result = t_res;
				}
			}

			for (Polygon p : _polygons)
			{
				CollisionData t_res = CheckCollision2(x_0, x_1, p, p_indx);
				if (CollisionData.r_I != -1) 
				{	
					if ( !r_set ) { result = t_res; r_set = true; }
					if ( t_res.r_I < result.r_I ) result = t_res;
				}
			}
			return t_res;
	}
*/

      __host__
      CollisionData CheckCollisions2(glm::vec3 x_0, glm::vec3 x_1, std::vector<Polygon>* poly, int p_indx)
      {
            // http://geomalgorithms.com/a06-_intersect-2.html

		CollisionData result(glm::vec3(), poly->at(0), -1);
            glm::vec3 x_01 = (x_1 - x_0);

            for (int j = 0; j < poly->size(); j++)
            {
            	glm::vec3 v_0  = poly->at(j).v[0];
                  glm::vec3 v_1  = poly->at(j).v[1];
                  glm::vec3 v_2  = poly->at(j).v[2];

                  glm::vec3 v_01 = v_1 - v_0;
                  glm::vec3 v_02 = v_2 - v_0;

                  glm::vec3 n = glm::normalize( glm::cross( ( v_1 - v_0 ), ( v_2 - v_0 ) ) );
                  float r_I = ( glm::dot(n, v_0 - x_0) ) / ( glm::dot(n, x_1 - x_0) );
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
                        std::cout << "COLLISION!!!! Particle " << p_indx << " -> Polygon " << j << std::endl;
				CollisionData t_res = CollisionData(I, poly->at(j), r_I); 

		    		if (result.r_I == -1 || t_res.r_I < result.r_I) result = t_res;                   }
		} 
		return result; 
      }
}
