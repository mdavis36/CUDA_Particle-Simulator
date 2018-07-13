#version 420 core

layout(location = 0) in vec4 _in_vertex_position;
layout(location = 1) in vec4 _in_vertex_color;

out vec4 _vertex_color;

uniform mat4 _pvm_matrix;

void main()
{
	_vertex_color = _in_vertex_color;
	gl_Position = _pvm_matrix * _in_vertex_position;
}