TARGET = CUDA_Particle_Simulator.exe
LIBS = -lglut -lGLEW -lGLU -lGL -lSDL2
CC = nvcc
CFLAGS = -std=c++11

.PHONY: default all clean

default: $(TARGET)
all: default
run: all ; ./$(TARGET) &

OBJECTS = main.cpp Graphics/LoadShaders.cpp Graphics/Model.cpp Graphics/OrientationKey.cpp Sim/Plane.cpp Graphics/ViewController.cpp CUDA/kernel.cu Sim/Simulation.cpp Utils.cpp
HEADERS =          Graphics/LoadShaders.h   Graphics/Model.h   Graphics/OrientationKey.h   Sim/Plane.h   Graphics/ViewController.h   CUDA/kernel.h  Sim/Simulation.h   Graphics/Drawable.h Utils.h

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
