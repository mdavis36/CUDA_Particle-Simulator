TARGET = CUDA_Particle_Simulator.exe
CPU_TARGET = Particle_Simulator.exe
LIBS = -lglut -lGLEW -lGLU -lGL -lSDL2
CC = nvcc -Xcompiler -fopenmp -DCMAKE_C_COMPILER=/usr/bin/gcc-6
CPU_CC = g++ -Wall -Wextra -D_DEBUG -fopenmp
CFLAGS = -std=c++11

.PHONY: default all clean

default: $(TARGET)
all: default
run: all ; ./$(TARGET) &
cpu: $(CPU_TARGET)

GRAPHICS = Graphics/LoadShaders.cpp Graphics/Model.cpp Graphics/OrientationKey.cpp Graphics/ViewController.cpp
SIMULATION = Sim/Plane.cpp Sim/Simulation.cpp Sim/ParticleSystem.cpp Sim/Particle.cpp Sim/ParticleHandler.cpp
UTILS = Utils/Utils.cpp

OBJECTS = main.cpp $(GRAPHICS) $(SIMULATION) $(UTILS)

CUDA_OBJECTS = CUDA/kernel.cu

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS)
	$(CC) $(CFLAGS) -DRUN_CUDA $(OBJECTS) $(CUDA_OBJECTS) $(LIBS) -o $@

$(CPU_TARGET): $(OBJECTS)
	$(CPU_CC) $(CFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
	-rm -f $(CPU_TARGET)
