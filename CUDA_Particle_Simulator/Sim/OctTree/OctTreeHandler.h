#ifndef OCTTREEHANDLER_H
#define OCTTREEHANDLER_H

#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "glm/gtc/type_ptr.hpp"

#include "../../Graphics/Drawable.h"

#include "../Geometry/Polygon.h"
#include "OctTree.h"
#include "../Geometry/Volume.h"

class OctTreeHandler : public Drawable
{
private:

      std::vector<vec3> _positions;
      std::vector<vec4> _colors;

      GLuint _vao;
      GLuint _buffers[2];

      bool _initialized;
      bool _doRender;

      void generateOpenGLData();

public:
      OctTreeHandler();
      ~OctTreeHandler();

      std::vector<OctTree*> node_list;
      std::vector<glm::vec3> vertices;

      void buildTree(std::vector<Polygon> p, Volume vol);
      void toggleRender();
      void clear();

      virtual bool init(GLuint* programs);
      virtual void draw(GLuint* programs, mat4 proj_mat, mat4 view_mat);

};


#endif
