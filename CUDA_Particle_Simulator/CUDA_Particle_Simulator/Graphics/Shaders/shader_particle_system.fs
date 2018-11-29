#version 330

in vec4 _vertex_color;
in vec3 _vertex_pos;
in vec3 _velocity;

out vec4 _out_color;

vec4 close = vec4(1.0, 0.1, 0.1, 1.0);
vec4 far   = vec4(0.2, 0.0, 0.0, 0.0);

void main()
{
	float d = distance(vec3(0.0,15.0,0.0), _vertex_pos);
	_out_color = mix(close, far, d / 500);

}
