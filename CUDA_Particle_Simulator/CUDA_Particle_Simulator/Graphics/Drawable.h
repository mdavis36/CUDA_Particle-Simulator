#ifndef DRAWABLE_H
#define DRAWABLE_H

#include <glm/glm.hpp>
#include <GL/glew.h>
using namespace glm;

class Drawable
{
public:
      Drawable() {}
      virtual ~Drawable() {}
      mat4 _model_matrix;
      virtual bool init() = 0;
      virtual void draw(GLuint* programs) = 0;
      mat4 getModelMatrix() { return _model_matrix; }
};

class Plane_Asset : public Drawable
{
private:
      std::vector<vec3> _positions;
      std::vector<vec4> _colors;

      GLuint _vao;
      GLuint _buffers[2];
};

#endif
