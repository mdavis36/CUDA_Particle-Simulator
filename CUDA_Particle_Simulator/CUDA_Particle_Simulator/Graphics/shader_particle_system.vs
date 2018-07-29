#version 330

layout(location = 0) in vec4 _in_vertex_position;
layout(location = 1) in vec4 _in_vertex_color;
layout(location = 2) in vec3 _in_vel;

out vec4 _vertex_color;
out vec3 _vertex_pos;
out vec3 _velocity;

uniform mat4 _pvm_matrix;

void main()
{
	_velocity = _in_vel;
	_vertex_pos = _in_vertex_position.xyz;
	_vertex_color = _in_vertex_color;
	gl_Position = _pvm_matrix * _in_vertex_position;
}
