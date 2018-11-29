#include "SceneObject.h"

SceneObject::SceneObject(const char * filename)
{
      _initialized = false;
      if (!loadOBJ(filename, _vertices, _uvs, _normals)){
            exit(-1);
      }

      for(int i = 0; i < _vertices.size(); i += 3)
      {
            _polygons.push_back( Polygon(_vertices[i], _vertices[i+1], _vertices[i+2]) );
      }
}

SceneObject::~SceneObject()
{

}

bool SceneObject::init(GLuint* programs)
{
      _model_matrix = mat4(1.0);
      //_model_matrix = glm::translate(_model_matrix, glm::vec3(5.0, 0.0, 0.0));

      glUseProgram(programs[2]);

      glGenVertexArrays(1, &_vao);  //Create one vertex array object
	glBindVertexArray(_vao);

      glGenBuffers(3, _buffers);

      glBindBuffer(GL_ARRAY_BUFFER, _buffers[0]);  //Buffers[0] wi ll be the position for each vertex
	glBufferData(GL_ARRAY_BUFFER, _vertices.size() * sizeof(glm::vec3), _vertices.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(0);

      glBindBuffer(GL_ARRAY_BUFFER, _buffers[1]);  //Buffers[1] will be the color for each vertex
	glBufferData(GL_ARRAY_BUFFER, _normals.size() * sizeof(glm::vec3), _normals.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(2);

      glBindBuffer(GL_ARRAY_BUFFER, _buffers[2]);  //Buffers[0] wi ll be the position for each vertex
	glBufferData(GL_ARRAY_BUFFER, _uvs.size() * sizeof(glm::vec2), _uvs.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(1);

      glPointSize(2.0f);

      _pvm_matrix_loc = glGetUniformLocation(programs[2], "_pvm_matrix");
      _model_matrix_loc = glGetUniformLocation(programs[2], "_model_matrix");

	_initialized = true;
	return true;
}


void SceneObject::draw(GLuint* programs, mat4 proj_mat, mat4 view_mat)
{
      mat4 _pvm_matrix = proj_mat * view_mat * _model_matrix;
	glUniformMatrix4fv(_pvm_matrix_loc, 1, GL_FALSE, value_ptr(_pvm_matrix));
      glUniformMatrix4fv(_model_matrix_loc, 1, GL_FALSE, value_ptr(_model_matrix));

      glUseProgram(programs[2]);
      if (!_initialized)
	{
		std::cout << "ERROR : Cannot  render an object thats not initialized. Plane\n";
		return;
	}

      glBindVertexArray(_vao);

      glDrawArrays(GL_TRIANGLES, 0, _vertices.size());
}
