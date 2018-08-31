#version 460

layout(location = 0) in vec3 _in_vertex_position;
layout(location = 1) in vec3 _in_vertex_normal;
layout(location = 2) in vec2 _in_uv;


out vec4 _vertex_color;
out vec3 _surface_normal;
out vec3 _to_light;

uniform mat4 _pvm_matrix;
uniform mat4 _model_matrix;

void main()
{
	_vertex_color = vec4(0.6, 0.6, 1.0, 1.0);
	vec3 lightPosition = vec3(7.0, 10.0, 10.0);

	vec4 worldPosition = _model_matrix * vec4(_in_vertex_position, 1);
	gl_Position = _pvm_matrix * vec4(_in_vertex_position,1);


	_surface_normal = _in_vertex_normal;
	_to_light = lightPosition - _in_vertex_position;


}
