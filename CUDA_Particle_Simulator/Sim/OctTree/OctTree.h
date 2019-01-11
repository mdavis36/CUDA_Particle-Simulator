#ifndef OCTTREE_H
#define OCTTREE_H

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <iostream>
#include <vector>

#include "../Geometry/Polygon.h"
#include "../Geometry/Volume.h"

class OctTree
{
private:

public:
      Volume vol;
      bool isLeaf;
      std::vector<Polygon> polygons;
      int leafs[8];
      int indx;

      OctTree(int i);
      ~OctTree();

      void generateOctTree(std::vector<Polygon> polygons, Volume _vol, int p, int level, std::vector<OctTree*>& node_list);
};

#endif
