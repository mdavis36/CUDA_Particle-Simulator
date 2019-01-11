#ifndef CONFIG_H
#define CONFIG_H

#include <iostream>
#include <glm/glm.hpp>
#include <config4cpp/Configuration.h>
using namespace config4cpp;

class configData
{
public:
      const char* OBJname;

      int         numParticles;
      int         particleShape;
      glm::vec3   particleCenter;
      int         particleSpread;

      bool        CPU_update;
      bool        GPU_update;
	std::string Update_Protocol;

      float       OTminSize;
      int         OTminCount;

      void init(char* cfgFilename);

};


#endif
