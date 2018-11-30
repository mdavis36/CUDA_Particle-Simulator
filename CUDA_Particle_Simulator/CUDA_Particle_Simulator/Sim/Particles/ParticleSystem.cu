#include "ParticleSystem.h"


__global__ void kernel( float* ptr)
{
      ptr[0] += 15.0f;
}


ParticleSystem::ParticleSystem()
{
      *this = ParticleSystem(1, PARTICLE_CUBE);
}

ParticleSystem::ParticleSystem(int n, int form)
{
      t = 0.0f;
      _num_particles = n;
      if (_num_particles == 1)
      {
            _particles.push_back(Particle(0.1,
                                          vec3(1.0f,5.0f,1.0f),
                                          vec3(0.0f,0.0f,0.0f),
                                          vec3(0.0f,0.0f,0.0f)
                                          )
                                    );
      }
      if(form == PARTICLE_CUBE)
      {
            //Ramdomly generate positions of particles.
            int x_limit = 10;
            int z_limit = 10;
            glm::vec3 ranPos;
            for (int i = 0; i < _num_particles; i++)
            {
                  ranPos.x = -x_limit + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * x_limit)));
                  ranPos.z = -z_limit + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * z_limit)));
                  ranPos.y = 5 + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (8)));
                  _particles.push_back(Particle(1,
                                                    ranPos,
                                                    vec3(0.0f,0.0f,0.0f),
                                                    vec3(0.0f,0.0f,0.0f)
                                                    )
                                              );
            }
      }
      if(form == PARTICLE_SPHERE)
      {
            //Ramdomly generate positions of particles.
            int radius = 10;
            glm::vec3 center = vec3(0,15,0);
            glm::vec3 ranPos;
            glm::vec3 dist;
            for (int i = 0; i < _num_particles; i++)
            {
                  ranPos = center;
                  do
                  {
                        dist.x = -radius + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * radius)));
                        dist.z = -radius + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * radius)));
                        dist.y = -radius + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (2 * radius)));
                  }while (glm::length(dist) > radius);
                  ranPos += dist;

                  _particles.push_back(Particle(1,
                                                    ranPos,
                                                    vec3(0.0f,0.0f,0.0f),
                                                    vec3(0.0f,0.0f,0.0f)
                                                    )
                                              );
            }
      }
}

ParticleSystem::~ParticleSystem()
{

}


bool ParticleSystem::init(GLuint* programs)
{
      int i;
      for (i = 0; i < _num_particles; i++)
      {
            _positions.push_back(_particles[i].x);
            _colors.push_back(vec4(1.0f, 0.0f, 0.0f, 1.0f));
      }

	_model_matrix = mat4(1.0);

      glGenVertexArrays(1, &_vao);  //Create one vertex array object
	glBindVertexArray(_vao);
      glGenBuffers(2, _buffers); //Create two buffer objects, one for vertex positions and one for vertex colors

      glBindBuffer(GL_ARRAY_BUFFER, _buffers[0]);  //Buffers[0] wi ll be the position for each vertex
	glBufferData(GL_ARRAY_BUFFER, _positions.size() * sizeof(vec3), _positions.data(), GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, _buffers[1]);  //Buffers[1] will be the color for each vertex
	glBufferData(GL_ARRAY_BUFFER, _colors.size() * sizeof(vec4), _colors.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(1);

      glLineWidth(1.0f);
      glPointSize(2.0f);

      _pvm_matrix_loc = glGetUniformLocation(programs[1], "_pvm_matrix");
      _projection_matrix_loc = glGetUniformLocation(programs[1], "_projection_matrix");
      _view_matrix_loc = glGetUniformLocation(programs[1], "_view_matrix");

	_initialized = true;
	return true;
}

void ParticleSystem::draw(GLuint* programs, mat4 proj_mat, mat4 view_mat)
{
      mat4 _pvm_matrix = proj_mat * view_mat * _model_matrix;
	glUniformMatrix4fv(_pvm_matrix_loc, 1, GL_FALSE, value_ptr(_pvm_matrix));
      glUniformMatrix4fv(_projection_matrix_loc, 1, GL_FALSE, value_ptr(proj_mat));
      glUniformMatrix4fv(_view_matrix_loc, 1, GL_FALSE, value_ptr(view_mat));

      glUseProgram(programs[1]);

      _positions.clear();

      int i;
      for (i = 0; i < _num_particles; i++)
      {
            _positions.push_back(_particles[i].x);
      }

      glBindVertexArray(_vao);
	glBindBuffer(GL_ARRAY_BUFFER, _buffers[0]);  //Buffers[0] wi ll be the position for each vertex
	glBufferData(GL_ARRAY_BUFFER, _positions.size() * sizeof(vec3), _positions.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(0);


      // ---------- CUDA / OpenGL Data Mapping ----------

      // Register CUDA graphics resource with buffer, define what it will be used for.
      cudaGraphicsGLRegisterBuffer( &res, _buffers[0], cudaGraphicsRegisterFlagsNone );

      // Map OpenGL Resource to CUDA device ptr
      cudaGraphicsMapResources(1, &res);
      cudaGraphicsResourceGetMappedPointer(&device_ptr, &size, res);

      // Kernel to move over the particle 5 in x dir
      kernel<<<1,1>>>( (float*)device_ptr );

      // Unmap OpenGL Resource from CUDA so the VBA can use it
      cudaGraphicsUnmapResources(1, &res);
      cudaGraphicsUnregisterResource(res);

      // ------------------------------------------------


      if (!_initialized)
      {
            cout << "ERROR : Cannot  render an object thats not initialized. ParticleSystem\n";
            return;
      }
      glDrawArrays(GL_POINTS, 0, _positions.size());
}
