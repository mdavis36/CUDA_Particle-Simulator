#include "Config.h" 
#include "../Sim/Particles/ParticleSystem.h"

void configData::init(char * cfgFilename)
{
      Configuration* cfg = Configuration::create();

      try{
            cfg->parse("config.cfg");
      } catch(const ConfigurationException & ex) {
            std::cerr << ex.c_str() << std::endl;
            cfg->destroy();
            exit(1);
      }

      OBJname           = cfg->lookupString("", "OBJname");
      Update_Protocol   = (std::string)cfg->lookupString("", "Update_Protocol");

      numParticles      = cfg->lookupInt("", "numParticles");
      particleShape     = (cfg->lookupString("", "particleShape") == "CUBE") ? PARTICLE_CUBE : PARTICLE_SPHERE;
}
