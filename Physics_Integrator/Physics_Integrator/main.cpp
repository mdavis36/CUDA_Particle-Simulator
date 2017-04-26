/*
* FILE : Main.cpp
* AUTHOR : Michael Davis
* DATE : 4/17/2017
* 
* This is the main file of the Physics integrator program.
*/

// include standard libraries
#include <iostream>
#include <string>
using namespace std;

// include glut and openGL dependencies
#include <glew.h>
#include <wglew.h>
#include <GL/freeglut.h>
#pragma comment(lib, "glew32.lib")

// my includes
#include "simulation.h"

// Display
const int DISPLAY_WIDTH = 1024;
const int DISPLAY_HEIGHT = 960;
char DISPLAY_TITLE[256] = "Physics Integrator : Verlet";

// Time Variables
float start_time = 0.0f;
float target_time_step = 1 / 60.0f;
float new_time = 0.0f;
float curr_time = 0.0f;
float delta_time = 0.0f;

int frame_count = 0;
float fps = 0.0f;

//Simulation
Simulation sim;

// Camera
float rX = 15, rY = 0;
float dist = -18;
GLdouble MV[16];
GLint viewport[4];
GLdouble P[16];
glm::vec3 Up = glm::vec3(0, 1, 0), viewDir, Right;

void glut_CloseFunc() {}

void glut_DisplayFunc() {
	new_time = (float)glutGet(GLUT_ELAPSED_TIME);
	delta_time = new_time - start_time;

	frame_count++;
	if (delta_time > 1000) {
		fps = (frame_count / delta_time) * 1000;
		start_time = new_time;
		frame_count = 0;
	}

	sprintf_s(DISPLAY_TITLE, "Physics Integrator --verlet : FPS: %3.3f : Frame Time: %3.9f", fps, sim.getSimFrameTime());
	glutSetWindowTitle(DISPLAY_TITLE);
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(0, 0, dist);
	glRotatef(rX, 1, 0, 0);
	glRotatef(rY, 0, 1, 0);

	glGetDoublev(GL_MODELVIEW_MATRIX, MV);
	viewDir.x = (float)-MV[2];
	viewDir.y = (float)-MV[6];
	viewDir.z = (float)-MV[10];
	Right = glm::cross(viewDir, Up);
	
	sim.update(delta_time);
	sim.render();

	glutSwapBuffers();
}

void glut_IdleFunc() {
	glutPostRedisplay();
}

void glut_ReshapeFunc(int nw, int nh) {
	glViewport(0, 0, nw, nh);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60, (GLfloat)nw / (GLfloat)nh, 1.f, 100.0f);

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_PROJECTION_MATRIX, P);

	glMatrixMode(GL_MODELVIEW);
}

void glut_keyboardFunc(unsigned char key, int x, int y) {
	if (key == 's') {
		sim.start();
	}

	if (key == 'r') {
		sim.reset();
	}

	if (key == 'p') {
		sim.pause();
	}
}

int main(int argc, char** argv) {

	// Initilize GLUT(Graphics Library Utility Toolkit) and create a windowed display to render to.
	cout << "Initializing GLUT Window : ";
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(DISPLAY_WIDTH, DISPLAY_HEIGHT);
	glutCreateWindow(DISPLAY_TITLE);
	glPointSize(3);
	cout << "PASS\n";


	// Declare the GLUT functions for basic operation and events of the window.
	cout << "Declaring GLUT Functions : ";
	glutDisplayFunc(glut_DisplayFunc);
	glutIdleFunc(glut_IdleFunc);
	glutCloseFunc(glut_CloseFunc);
	glutReshapeFunc(glut_ReshapeFunc);
	glutKeyboardFunc(glut_keyboardFunc);
	cout << "PASS\n";


	// Initialize GLEW (Graphics Library Extension Wrangler) this is what links OpenGL. Check this is successful.
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		fprintf(stderr, "Failed to Initialize GLEW : %s\n", glewGetErrorString(err));
		system("PAUSE");
		exit(-3);
	}
	cout << "GLEW Initilized Successfully.\n";


	// Call glutMainLoop() initializes and runs the main loop of the program.
	cout << "Calling glutMainLoop() ...\n";
	glutMainLoop();

	cout << "Goodbye.\n";
	system("PAUSE");
	return 0;
}