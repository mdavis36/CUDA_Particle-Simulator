#version 330

layout(location = 0) in vec4 _in_vertex_position;
layout(location = 1) in vec4 _in_vertex_color;
layout(location = 2) in vec3 _in_vel;

out vec4 _vertex_color;
out vec3 _vertex_pos;
out vec3 _velocity;

uniform mat4 _pvm_matrix;
uniform mat4 _proj_mat;
uniform mat4 _view_mat;

varying float dist_to_camera;

void main()
{
	_velocity = _in_vel;
	_vertex_pos = _in_vertex_position.xyz;
	_vertex_color = _in_vertex_color;

	vec4 cs_position = _view_mat * _in_vertex_position;
	dist_to_camera = cs_position.z;
	gl_PointSize = clamp(mix(4, 1, dist_to_camera / 150 ), 1, 8);
	
	gl_Position = _view_mat * _in_vertex_position;
}
