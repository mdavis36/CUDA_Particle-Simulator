TARGET = CUDA_Particle_Simulator.exe
LIBS = -lglut -lGLEW -lGLU -lGL -lSDL2
CC = nvcc
CFLAGS = -std=c++11

.PHONY: default all clean

default: $(TARGET)
all: default
run: all ; ./$(TARGET) &

OBJECTS = main.cpp Graphics/LoadShaders.cpp Graphics/Model.cpp OrientationKey.cpp Plane.cpp Graphics/ViewController.cpp CUDA/kernel.cu Sim/Simulation.cpp
HEADERS =          Graphics/LoadShaders.h   Graphics/Model.h   OrientationKey.h   Plane.h   Graphics/ViewController.h   CUDA/kernel.h  Sim/Simulation.h   Graphics/Drawable.h

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
