#ifndef DRAWABLE_H
#define DRAWABLE_H

#include <glm/glm.hpp>
#include <GL/glew.h>
using namespace glm;

class Drawable
{
public:
      mat4 _model_matrix;
      GLint _pvm_matrix_loc, _projection_matrix_loc, _view_matrix_loc, _model_matrix_loc;

      Drawable() {}
      virtual ~Drawable() {}
      virtual bool init(GLuint* programs) = 0;
      virtual void draw(GLuint* programs, mat4 proj_mat, mat4 view_mat) = 0;
      mat4 getModelMatrix() { return _model_matrix; }
};

#endif
