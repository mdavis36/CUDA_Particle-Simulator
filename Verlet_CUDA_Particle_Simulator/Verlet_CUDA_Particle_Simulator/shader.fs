#version 420 core

in vec4 _vertex_color;
out vec4 _out_color;

void main()
{
	_out_color = _vertex_color;
}