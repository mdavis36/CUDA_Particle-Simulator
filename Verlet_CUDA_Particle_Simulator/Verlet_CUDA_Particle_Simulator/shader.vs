#version 420 core

layout(location = 0) in vec4 _in_vertex_position;
layout(location = 1) in vec4 _in_vertex_color;

out vec4 _out_vertex_color;

uniform mat4 _mvp_matrix;

void main()
{
	_out_vertex_color;
	gl_Position = _mvp_matrix * _in_vertex_position;
}