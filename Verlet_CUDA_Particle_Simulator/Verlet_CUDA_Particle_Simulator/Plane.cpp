#include "Plane.h"

Plane::Plane()
{
	Plane(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 0.0), 5, 5);
}

Plane::Plane(vec3 n, vec3 c, int w, int h)
{
	_normal = n;
	_center = c;
	_width  = w;
	_height = h;
	_initialized = false;
}

bool Plane::init()
{
	glGenVertexArrays(1, &_vao);  //Create one vertex array object
	glBindVertexArray(_vao);

	// Verticle Lines
	for (int i = -_width; i <= _width; i++)
	{
		_positions.push_back(vec3((float)i, 0.0f, (float)_height));
		_positions.push_back(vec3((float)i, 0.0f, (float)-_height));
		if (i == 0) 
		{
			_colors.push_back(vec4(0.0f, 1.0f, 1.0f, 1.0f));
			_colors.push_back(vec4(0.0f, 1.0f, 1.0f, 1.0f));
		}
		else
		{
			_colors.push_back(vec4(1.0f, 1.0f, 1.0f, 1.0f));
			_colors.push_back(vec4(1.0f, 1.0f, 1.0f, 1.0f));
		}
	}

	// Horizontal Lines
	
	for (int i = -_height; i <= _height; i++)
	{

		_positions.push_back(vec3((float)_width, 0.0f, (float)i));
		_positions.push_back(vec3((float)-_width, 0.0f, (float)i));
		if (i == 0)
		{
			_colors.push_back(vec4(0.0f, 1.0f, 1.0f, 1.0f));
			_colors.push_back(vec4(0.0f, 1.0f, 1.0f, 1.0f));
		}
		else
		{
			_colors.push_back(vec4(1.0f, 1.0f, 1.0f, 1.0f));
			_colors.push_back(vec4(1.0f, 1.0f, 1.0f, 1.0f));
		}
	}

	_positions.push_back(vec3(0, 0, 0));
	_positions.push_back(vec3(0, 1, 0));
	_colors.push_back(vec4(1.0f, 1.0f, 0.0f, 1.0f));
	_colors.push_back(vec4(1.0f, 1.0f, 0.0f, 1.0f));

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

void Plane::draw()
{
	if (!_initialized)
	{
		cout << "ERROR : Cannot  render an object thats not initialized.\n";
		return;
	}
	glBindVertexArray(_vao);
	glLineWidth(1.0f);
	glDrawArrays(GL_LINES, 0, _positions.size());
	//glDrawArrays(GL_TRIANGLE_STRIP, 0, _positions.size());
}
