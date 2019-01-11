#include "Volume.h"


Volume::Volume()
{

}

Volume::Volume(glm::vec3 _BBL, glm::vec3 _TTR)
{
      BBL = _BBL;
      TTR = _TTR;
      sz = TTR[0] - BBL[0];
      hsz = sz / 2;
}

Volume::~Volume(){}

void Volume::print()
{
      std::cout << "BBL : (" << BBL[0] << " " << BBL[1] << " " << BBL[2] <<
      ") || TTR : (" << TTR[0] << " " << TTR[1] << " " << TTR[2] << ")" << std::endl;
}


Volume Volume::getChild(int c)
{
      assert(c < 8);

      if (c == 0) return Volume(BBL, glm::vec3(BBL.x + hsz, BBL.y + hsz, BBL.z + hsz));
      if (c == 1) return Volume(glm::vec3(BBL.x + hsz, BBL.y, BBL.z), glm::vec3(BBL.x + hsz + hsz, BBL.y + hsz, BBL.z + hsz));
      if (c == 2) return Volume(glm::vec3(BBL.x, BBL.y, BBL.z + hsz), glm::vec3(BBL.x + hsz, BBL.y + hsz, BBL.z + hsz + hsz));
      if (c == 3) return Volume(glm::vec3(BBL.x + hsz, BBL.y, BBL.z + hsz), glm::vec3(BBL.x + hsz + hsz, BBL.y + hsz, BBL.z + hsz + hsz));

      if (c == 4) return Volume(glm::vec3(BBL.x, BBL.y + hsz, BBL.z), glm::vec3(BBL.x + hsz, BBL.y + hsz + hsz, BBL.z + hsz));
      if (c == 5) return Volume(glm::vec3(BBL.x + hsz, BBL.y + hsz,BBL.z), glm::vec3(BBL.x + hsz + hsz, BBL.y + hsz + hsz, BBL.z + hsz));
      if (c == 6) return Volume(glm::vec3(BBL.x, BBL.y + hsz, BBL.z + hsz), glm::vec3(BBL.x + hsz, BBL.y + hsz + hsz, BBL.z + hsz + hsz));
      if (c == 7) return Volume(glm::vec3(BBL.x + hsz, BBL.y + hsz, BBL.z + hsz), glm::vec3(BBL.x + hsz + hsz, BBL.y + hsz + hsz, BBL.z + hsz + hsz));
}

bool Volume::containsVertex(const glm::vec3 v)
{
      return (BBL.x < v.x && v.x <= TTR.x &&
              BBL.y < v.y && v.y <= TTR.y &&
              BBL.z < v.z && v.z <= TTR.z);
}

bool Volume::containsPolygon(const Polygon p)
{
      int c = countContainedVertices(p);
      // print();
      // p.print();
      // std::cout << "Contains : " << c << std::endl;
      return ( c == 3 );
}

int Volume::countContainedVertices(const Polygon p)
{
      int count = 0;
      if (containsVertex(p.v[0])) count++;
      if (containsVertex(p.v[1])) count++;
      if (containsVertex(p.v[2])) count++;
      return count;
}

bool Volume::intersectPolygon(const Polygon p)
{
      int c = countContainedVertices(p);
      return (c == 1 || c == 2);
}

/*bool Volume::lineNodeIntersection(glm::vec3 x_0, glm::vec3 x_1)
{
	if (x_1.x < BBL.x && x_0.x < BBL.x) return false;
	if (x_1.x > TTR.x && x_0.x > TTR.x) return false;
	if (x_1.y < BBL.y && x_0.y < BBL.y) return false;
	if (x_1.y > TTR.y && x_0.y > TTR.y) return false;
	if (x_1.z < BBL.z && x_0.z < BBL.z) return false;
	if (x_1.z > TTR.z && x_0.z > TTR.z) return false;
}*/
