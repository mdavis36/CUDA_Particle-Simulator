#ifndef SCENEOBJECT_H
#define SCENEOBJECT_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include "glm/gtc/type_ptr.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <vector>

#include "../Graphics/Drawable.h"
#include "../Graphics/ObjLoader.h"
#include "Polygon.h"

class SceneObject : public Drawable
{
private:

      GLuint _vao;
      GLuint _buffers[3];

      bool _initialized;
      bool _isHitable;

public:
      std::vector<glm::vec3> _vertices;
	std::vector<glm::vec2> _uvs;
	std::vector<glm::vec3> _normals;
      std::vector<Polygon> _polygons;

      SceneObject(const char * filename);
      ~SceneObject();

      virtual bool init(GLuint* programs);
      virtual void draw(GLuint* programs, mat4 proj_mat, mat4 view_mat);

};

#endif
