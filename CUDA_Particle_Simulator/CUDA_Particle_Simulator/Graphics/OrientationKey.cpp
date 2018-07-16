#include "OrientationKey.h"

OrientationKey::OrientationKey()
{
	_initialized = false;
}

bool OrientationKey::init()
{
	glGenVertexArrays(1, &_vao);
	glBindVertexArray(_vao);

	_positions.push_back(vec3(0, 0, 0));
	_positions.push_back(vec3(1, 0, 0));
	_colors.push_back(vec4(1.0f, 0.0f, 0.0f, 1.0f));
	_colors.push_back(vec4(1.0f, 0.0f, 0.0f, 1.0f));

	_positions.push_back(vec3(0, 0, 0));
	_positions.push_back(vec3(0, 1, 0));
	_colors.push_back(vec4(0.0f, 1.0f, 0.0f, 1.0f));
	_colors.push_back(vec4(0.0f, 1.0f, 0.0f, 1.0f));

	_positions.push_back(vec3(0, 0, 0));
	_positions.push_back(vec3(0, 0, 1));
	_colors.push_back(vec4(0.0f, 0.0f, 1.0f, 1.0f));
	_colors.push_back(vec4(0.0f, 0.0f, 1.0f, 1.0f));

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

	_initialized = true;

	return true;
}

void OrientationKey::draw() const
{
	if (!_initialized)
	{
		cout << "ERROR : Cannot  render an object thats not initialized.\n";
		return;
	}
	glBindVertexArray(_vao);
	glLineWidth(2.0f);
	glDrawArrays(GL_LINES, 0, _positions.size());
}

mat4 OrientationKey::getModelMatrix()
{
	return _model_matrix;
}
