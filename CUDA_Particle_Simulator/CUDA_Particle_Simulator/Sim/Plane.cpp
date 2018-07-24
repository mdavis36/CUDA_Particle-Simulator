#include "Plane.h"

Plane::Plane()
{
	Plane(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 0.0), 5, 5);
}

Plane::Plane(vec3 n, vec3 c, int w, int h)
{
	_normal = normalize(n);
	_center = c;
	_width  = w;
	_height = h;
	_initialized = false;
}

bool Plane::init()
{
	vec4   line_color(0.4f, 0.4f, 0.4f, 1.0f);
	vec4 center_color(0.0f, 1.0f, 1.0f, 1.0f);
	glGenVertexArrays(1, &_vao);  //Create one vertex array object
	glBindVertexArray(_vao);

	// Verticle Lines
	for (int i = -_width; i <= _width; i++)
	{
		_positions.push_back(vec3((float)i, 0.0f, (float)_height));
		_positions.push_back(vec3((float)i, 0.0f, (float)-_height));
		if (i == 0)
		{
			_colors.push_back(center_color);
			_colors.push_back(center_color);
		}
		else
		{
			_colors.push_back(line_color);
			_colors.push_back(line_color);
		}
	}

	// Horizontal Lines
	for (int i = -_height; i <= _height; i++)
	{

		_positions.push_back(vec3((float)_width, 0.0f, (float)i));
		_positions.push_back(vec3((float)-_width, 0.0f, (float)i));
		if (i == 0)
		{
			_colors.push_back(center_color);
			_colors.push_back(center_color);
		}
		else
		{
			_colors.push_back(line_color);
			_colors.push_back(line_color);
		}
	}
	_positions.push_back(vec3(0, 0, 0));
	_positions.push_back(vec3(0, 1, 0));
	_colors.push_back(vec4(1.0f, 1.0f, 0.0f, 1.0f));
	_colors.push_back(vec4(1.0f, 1.0f, 0.0f, 1.0f));


	_model_matrix = mat4(1.0);
	_model_matrix = translate(_model_matrix, _center);
	vec3 up = vec3(0.0f, 1.0f, 0.0f);
	if (!(abs(_normal.y) == 1))
	{
		vec2 n_proj_zx = normalize(vec2(_normal.z, _normal.x));
		vec2 up_proj_zx = normalize(vec2(up.x, -up.y));

		float zx_theta = acosf(dot(n_proj_zx, up_proj_zx)) * (180 / 3.14f);
		_normal.x < 0 ? zx_theta = 270 - zx_theta : zx_theta = 270 - zx_theta;
		_normal.z < 0 ? true : zx_theta = 180 - zx_theta;

		float xy_theta = acosf(dot(_normal, up)) * (180 / 3.14f);

		_model_matrix = rotate(_model_matrix, zx_theta * (3.14159f / 180), vec3(0.0f, 1.0f, 0.0f));
		_model_matrix = rotate(_model_matrix, xy_theta * (3.14159f / 180), vec3(1.0f, 0.0f, 0.0f));
	}
	if (_normal.y == -1)
	{
		_model_matrix = rotate(_model_matrix, 3.14159f, vec3(1.0f, 0.0f, 0.0f));
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

	_initialized = true;
	return true;
}

void Plane::draw()
{
	if (!_initialized)
	{
		cout << "ERROR : Cannot  render an object thats not initialized. Plane\n";
		return;
	}
	glBindVertexArray(_vao);
	glLineWidth(0.5f);
	glDrawArrays(GL_LINES, 0, _positions.size());
}
