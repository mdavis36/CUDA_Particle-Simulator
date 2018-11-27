#include "OctTree.h"


OctTree::OctTree(int i) : indx(i) {}


OctTree::~OctTree() {}


Volume OctTree::getChildVol(Volume v, int c)
{
      if (c > 7) return Volume();

      float hw = v.hsz;

      if (c == 0) return Volume(v.BBL, glm::vec3(v.BBL.x + hw, v.BBL.y + hw, v.BBL.z + hw));
      if (c == 1) return Volume(glm::vec3(v.BBL.x + hw, v.BBL.y, v.BBL.z), glm::vec3(v.BBL.x + hw + hw, v.BBL.y + hw, v.BBL.z + hw));
      if (c == 2) return Volume(glm::vec3(v.BBL.x, v.BBL.y, v.BBL.z + hw), glm::vec3(v.BBL.x + hw, v.BBL.y + hw, v.BBL.z + hw + hw));
      if (c == 3) return Volume(glm::vec3(v.BBL.x + hw, v.BBL.y, v.BBL.z + hw), glm::vec3(v.BBL.x + hw + hw, v.BBL.y + hw, v.BBL.z + hw + hw));

      if (c == 4) return Volume(glm::vec3(v.BBL.x, v.BBL.y + hw, v.BBL.z), glm::vec3(v.BBL.x + hw, v.BBL.y + hw + hw, v.BBL.z + hw));
      if (c == 5) return Volume(glm::vec3(v.BBL.x + hw, v.BBL.y + hw, v.BBL.z), glm::vec3(v.BBL.x + hw + hw, v.BBL.y + hw + hw, v.BBL.z + hw));
      if (c == 6) return Volume(glm::vec3(v.BBL.x, v.BBL.y + hw, v.BBL.z + hw), glm::vec3(v.BBL.x + hw, v.BBL.y + hw + hw, v.BBL.z + hw + hw));
      if (c == 7) return Volume(glm::vec3(v.BBL.x + hw, v.BBL.y + hw, v.BBL.z + hw), glm::vec3(v.BBL.x + hw + hw, v.BBL.y + hw + hw, v.BBL.z + hw + hw));
}


void OctTree::generateOctTree(std::vector<Polygon> _polygons, Volume _vol, int p, int level, std::vector<OctTree*>& node_list)
{

      vol = _vol;
      std::cout << "Node INDX : " << indx << " : "; vol.print();

      if (_polygons.size() <= 1 || vol.sz < 0.025)
      {

            isLeaf = true;
            polygons = std::vector<Polygon>(_polygons);
            //std::cout << "Returning to : " << p << std::endl;
            return;
      }


      std::vector<Polygon> t_pol[8];
      std::vector<Polygon> errors;

      int polyErrorCount = 0;
      for (Polygon p : _polygons)
      {
            if (!vol.containsPolygon(p)) polyErrorCount++;
      }
      if (polyErrorCount > 0) std::cout << indx << " : ERROR " << polyErrorCount << " Polygon(s) to sort are not within this volume." << std::endl;

      polyErrorCount = 0;
      for (Polygon p : _polygons)
      {

            if      ( vol.getChild(0).containsPolygon(p) ) t_pol[0].push_back(p);
            else if ( vol.getChild(1).containsPolygon(p) ) t_pol[1].push_back(p);
            else if ( vol.getChild(2).containsPolygon(p) ) t_pol[2].push_back(p);
            else if ( vol.getChild(3).containsPolygon(p) ) t_pol[3].push_back(p);
            else if ( vol.getChild(4).containsPolygon(p) ) t_pol[4].push_back(p);
            else if ( vol.getChild(5).containsPolygon(p) ) t_pol[5].push_back(p);
            else if ( vol.getChild(6).containsPolygon(p) ) t_pol[6].push_back(p);
            else if ( vol.getChild(7).containsPolygon(p) ) t_pol[7].push_back(p);

            else if ( vol.getChild(0).intersectPolygon(p) || vol.getChild(1).intersectPolygon(p) ||
                      vol.getChild(2).intersectPolygon(p) || vol.getChild(3).intersectPolygon(p) ||
                      vol.getChild(4).intersectPolygon(p) || vol.getChild(5).intersectPolygon(p) ||
                      vol.getChild(6).intersectPolygon(p) || vol.getChild(7).intersectPolygon(p) ) polygons.push_back(p);
            else
            {
                  polyErrorCount++;
                  errors.push_back(p);
            }
      }
      if (polyErrorCount > 0)
      {
            std::cout << indx << " : ERROR could not place " << polyErrorCount << " Polygon(s)" << std::endl;
            for (Polygon p : errors)
            {
                  p.print();
                  for (int i = 0; i < 8 ; i++)
                  {
                        Volume t_vol = vol.getChild(i);
                        std::cout << "polyVolumeCount : " << i << " : " << t_vol.countContainedVertices(p) << " || ";
                        std::cout << t_vol.containsVertex(p.v[0]) << " ";
                        std::cout << t_vol.containsVertex(p.v[1]) << " ";
                        std::cout << t_vol.containsVertex(p.v[2]) << " ";
                        t_vol.print();
                  }
                  std::cout << "polyVolumeCount : r : " << vol.countContainedVertices(p) << " || ";
                  vol.print();
            }
      }



      if (polygons.size() != _polygons.size())
      {
            //std::cout << indx << " " << polygons.size() << " : Children : ";
            int next_node_indx = node_list.size();
            for (int i = 0; i < 8; i++)
            {
                  leafs[i] = next_node_indx + i;
                  node_list.push_back(new OctTree(leafs[i]));
                  //std::cout << "(" << leafs[i] << ", " << t_pol[i].size() << ") ";
            }
            //std::cout << std::endl;


            node_list[leafs[0]]->generateOctTree(t_pol[0], vol.getChild(0), indx, level+1, node_list );
            t_pol[0].clear();
            node_list[leafs[1]]->generateOctTree(t_pol[1], vol.getChild(1), indx, level+1, node_list );
            t_pol[1].clear();
            node_list[leafs[2]]->generateOctTree(t_pol[2], vol.getChild(2), indx, level+1, node_list );
            t_pol[2].clear();
            node_list[leafs[3]]->generateOctTree(t_pol[3], vol.getChild(3), indx, level+1, node_list );
            t_pol[3].clear();

            node_list[leafs[4]]->generateOctTree(t_pol[4], vol.getChild(4), indx, level+1, node_list );
            t_pol[4].clear();
            node_list[leafs[5]]->generateOctTree(t_pol[5], vol.getChild(5), indx, level+1, node_list );
            t_pol[5].clear();
            node_list[leafs[6]]->generateOctTree(t_pol[6], vol.getChild(6), indx, level+1, node_list );
            t_pol[6].clear();
            node_list[leafs[7]]->generateOctTree(t_pol[7], vol.getChild(7), indx, level+1, node_list );
            t_pol[7].clear();

      }else{
            isLeaf = true;
      }
      //std::cout << "Finished " << indx << " with " << polygons.size() << std::endl;
}
