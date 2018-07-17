TARGET = CUDA_Particle_Simulator.exe
CPU_TARGET = Particle_Simulator.exe
LIBS = -lglut -lGLEW -lGLU -lGL -lSDL2
CC = nvcc
CPU_CC = g++ -Wall -Wextra
CFLAGS = -std=c++11

.PHONY: default all clean

default: $(TARGET)
all: default
run: all ; ./$(TARGET) &
cpu: $(CPU_TARGET)

OBJECTS = main.cpp Graphics/LoadShaders.cpp Graphics/Model.cpp Graphics/OrientationKey.cpp Sim/Plane.cpp Graphics/ViewController.cpp Sim/Simulation.cpp Utils/Utils.cpp
HEADERS =          Graphics/LoadShaders.h   Graphics/Model.h   Graphics/OrientationKey.h   Sim/Plane.h   Graphics/ViewController.h   Sim/Simulation.h   Graphics/Drawable.h Utils/Utils.h CUDA/kernel.h
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
