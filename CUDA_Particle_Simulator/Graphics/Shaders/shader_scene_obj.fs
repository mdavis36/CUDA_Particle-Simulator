#version 460

in vec4 _vertex_color;
in vec3 _surface_normal;
in vec3 _to_light;


out vec4 _out_color;

void main()
{
	vec3 unitNormal = normalize(_surface_normal);
	vec3 unitLightVector = normalize(_to_light);
	vec3 lightColor = vec3(0.5,0.5,0.5);

	float nDotl = dot(unitNormal, unitLightVector);
	float brightness = clamp(nDotl, 0.4, 1.0);
	vec3 diffuse = lightColor * brightness;
	vec4 final = vec4( (diffuse *_vertex_color.xyz), 1);

	_out_color = final;
}
