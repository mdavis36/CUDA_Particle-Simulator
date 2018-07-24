#include "ParticleSystem.h"

ParticleSystem::ParticleSystem()
{
      *this = ParticleSystem(1);
}

ParticleSystem::ParticleSystem(int n)
{
      t = 0.0f;
      _num_particles = n;
      if (_num_particles == 1)
      {
            _particles.push_back(Particle(1,
                                          vec3(1.0f,5.0f,1.0f),
                                          vec3(0.0f,0.0f,0.0f),
                                          vec3(0.0f,0.0f,0.0f)
                                          )
                                    );
      }
      else
      {
            //Ramdomly generate positions of particles.
            int x_limit = 15;
            int z_limit = 15;
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


}

ParticleSystem::~ParticleSystem()
{

}


bool ParticleSystem::init()
{
      int i;
      for (i = 0; i < _num_particles; i++)
      {
            _positions.push_back(_particles[i].x);
            _colors.push_back(vec4(1.0f, 0.0f, 0.0f, 1.0f));
      }


	_model_matrix = mat4(1.0);


	_initialized = true;
	return true;
}

void ParticleSystem::draw()
{
      _positions.clear();
      glGenVertexArrays(1, &_vao);  //Create one vertex array object
	glBindVertexArray(_vao);
      int i;
      for (i = 0; i < _num_particles; i++)
      {
            _positions.push_back(_particles[i].x);
      }

	glGenBuffers(2, _buffers); //Create two buffer objects, one for vertex positions and one for vertex colors

	glBindBuffer(GL_ARRAY_BUFFER, _buffers[0]);  //Buffers[0] wi ll be the position for each vertex
	glBufferData(GL_ARRAY_BUFFER, _positions.size() * sizeof(vec3), _positions.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, _buffers[1]);  //Buffers[1] will be the color for each vertex
	glBufferData(GL_ARRAY_BUFFER, _colors.size() * sizeof(vec4), _colors.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(1);
      if (!_initialized)
      {
            cout << "ERROR : Cannot  render an object thats not initialized. ParticleSystem\n";
            return;
      }
      glBindVertexArray(_vao);
      glLineWidth(1.0f);
      glPointSize(2.0f);
      glDrawArrays(GL_POINTS, 0, _positions.size());
}
