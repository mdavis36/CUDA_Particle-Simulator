TARGET = Physics_Integrator.exe
LIBS = -lglut -lGLEW -lGLU -lGL -lSDL2
CC = g++
CFLAGS = -std=c++11

.PHONY: default all clean

default: $(TARGET)
all: default
run: all ; ./$(TARGET) &

OBJECTS = main.cpp integrator.cpp particle.cpp plane.cpp simulation.cpp
HEADERS = integrator.h particle.h plane.h simulation.h

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
