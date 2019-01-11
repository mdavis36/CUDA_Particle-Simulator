#include "OrientationKey.h"

OrientationKey::OrientationKey()
{
	_initialized = false;
}

bool OrientationKey::init(GLuint* programs)
{
	glGenVertexArrays(1, &_vao);
	glBindVertexArray(_vao);

	vec3 center = vec3(0,0.001,0);

	_positions.push_back(center);
	_positions.push_back(vec3(1, 0, 0));
	_colors.push_back(vec4(0.6f, 0.0f, 0.0f, 1.0f));
	_colors.push_back(vec4(0.6f, 0.0f, 0.0f, 1.0f));

	_positions.push_back(center);
	_positions.push_back(vec3(0, 1, 0));
	_colors.push_back(vec4(0.0f, 0.6f, 0.0f, 1.0f));
	_colors.push_back(vec4(0.0f, 0.6f, 0.0f, 1.0f));

	_positions.push_back(center);
	_positions.push_back(vec3(0, 0, 1));
	_colors.push_back(vec4(0.0f, 0.0f, 0.6f, 1.0f));
	_colors.push_back(vec4(0.0f, 0.0f, 0.6f, 1.0f));

	_model_matrix = mat4(1.0);

	glGenBuffers(2, _buffers); //Create two buffer objects, one for vertex positions and one for vertex colors

	glBindBuffer(GL_ARRAY_BUFFER, _buffers[0]);  //Buffers[0] wi ll be the position for each vertex
	glBufferData(GL_ARRAY_BUFFER, _positions.size() * sizeof(vec3), _positions.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, _buffers[1]);  //Buffers[1] will be the color for each vertex
	glBufferData(GL_ARRAY_BUFFER, _colors.size() * sizeof(vec4), _colors.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);  //Do the shader plumbing here for this buffer
	glEnableVertexAttribArray(1);

	_pvm_matrix_loc = glGetUniformLocation(programs[0], "_pvm_matrix");

	_initialized = true;

	return true;
}

void OrientationKey::draw(GLuint* programs, mat4 proj_mat, mat4 view_mat)
{
#ifndef GL1
	mat4 _pvm_matrix = proj_mat * view_mat * _model_matrix;
	glUniformMatrix4fv(_pvm_matrix_loc, 1, GL_FALSE, value_ptr(_pvm_matrix));

	glUseProgram(programs[0]);
	if (!_initialized)
	{
		cout << "ERROR : Cannot  render an object thats not initialized. OrientationKey\n";
		return;
	}
	glBindVertexArray(_vao);
	glLineWidth(5.0f);
	glDrawArrays(GL_LINES, 0, _positions.size());
#else
	glBegin(GL_LINES);

	//Red is Positive X
	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3f(0, 0, 0);
	glVertex3f(1, 0, 0);

	//Green is positive Y
	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 1, 0);

	//Blue is posiotive Z
	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, 1);

	glEnd();
#endif
}

mat4 OrientationKey::getModelMatrix()
{
	return _model_matrix;
}
