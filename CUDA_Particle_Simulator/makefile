TARGET = CUDA_Particle_Simulator.exe
CPU_TARGET = Particle_Simulator.exe
LIBS = -lglut -lGLEW -lGLU -lGL -lSDL2
CC = nvcc -Xcompiler -fopenmp -DCMAKE_C_COMPILER=/usr/bin/gcc-6 --relocatable-device-code=true -isystem ../config4cpp/include/ -L../config4cpp/lib/ -lconfig4cpp -Xcudafe="--diag_suppress=2906"
CPU_CC = g++ -Wall -Wextra -D_DEBUG -DGL1 -fopenmp
CFLAGS = -std=c++11 -g

.PHONY: default all clean

default: $(TARGET)
all: default
run: all ; ./$(TARGET) &
cpu: $(CPU_TARGET)



GEOMETRY    =  Sim/Geometry/Polygon.cpp \
		   Sim/Geometry/Volume.cpp

OCTTREE     =  Sim/OctTree/OctTree.cpp \
		   Sim/OctTree/OctTreeHandler.cpp

PARTICLES   =  Sim/Particles/ParticleSystem.cu \
		   Sim/Particles/Particle.cu \
		   Sim/Particles/ParticleHandler.cu \
		   Sim/Particles/ParticleHandler2.cu \
		   Sim/Particles/cuParticleHandler.cu \
		   Sim/Particles/CollisionData.cpp

SCENE       =  Sim/Scene/Plane.cpp \
		   Sim/Scene/OrientationKey.cpp \
		   Sim/Scene/SceneObject.cpp

SIMULATION  =  $(GEOMETRY) $(OCTTREE) $(PARTICLES) $(SCENE) Sim/Simulation.cpp


GRAPHICS    = Graphics/LoadShaders.cpp Graphics/Model.cpp Graphics/ViewController.cpp Graphics/ObjLoader.cpp
UTILS 	= Utils/Utils.cpp Utils/Config.cpp



OBJECTS     = main.cpp $(GRAPHICS) $(SIMULATION) $(UTILS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -DRUN_CUDA $(OBJECTS) $(LIBS) -o $@

$(CPU_TARGET): $(OBJECTS)
	$(CPU_CC) $(CFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
	-rm -f $(CPU_TARGET)
