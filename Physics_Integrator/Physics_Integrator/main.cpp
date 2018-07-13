/*
* FILE : Main.cpp
* AUTHOR : Michael Davis
* DATE : 4/17/2017
*
* This is the main file of the Physics integrator program.
*/

// Include STandard Libs
#include <iostream>
#include <string>
using namespace std;

// OpenGL Imports
#include <GL/glew.h>
//#include <GL/wglew.h>
#include <GL/freeglut.h>
#pragma comment(lib, "glew32.lib")

#include "simulation.h"
Simulation sim;

// Display Variables
const int DISPLAY_WIDTH = 1024;
const int DISPLAY_HEIGHT = 960;
char DISPLAY_TITLE[256] = "Physics Integrator : Verlet";

// Time Variables
double start_time = 0.0f;
double new_time = 0.0f;
double delta_time = 0.0f;

// Frame Variables
int frame_count = 0;
float fps = 0.0f;

// Camera
int oldX = 0, oldY = 0;
float rX = 15, rY = 0;
int state = 1;
int selected_index = -1;
float dist = -18;
GLdouble MV[16];
GLint viewport[4];
GLdouble P[16];
glm::vec3 Up = glm::vec3(0, 1, 0), viewDir, Right;


// ********************************************************************************
// *                           -- GLUT FUNCTIONS --                               *
// ********************************************************************************

void glut_CloseFunc() {}

void glut_DisplayFunc() {
	new_time = glutGet(GLUT_ELAPSED_TIME);
	delta_time = new_time - start_time;

	frame_count++;
	if (delta_time > 1000) {
		fps = (frame_count / delta_time) * 1000;
		start_time = new_time;
		frame_count = 0;
	}

	sprintf(DISPLAY_TITLE, "Physics Integrator --verlet : FPS: %3.3f : Frame Time: %3.9f", fps, sim.getSimFrameTime());
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

void OnMouseDown(int button, int s, int x, int y)
{
	if (s == GLUT_DOWN)
	{
		oldX = x;
		oldY = y;
		/*int window_y = (DISPLAY_WIDTH - y);
		float norm_y = float(window_y) / float(DISPLAY_HEIGHT / 2.0);
		int window_x = x;
		float norm_x = float(window_x) / float(DISPLAY_WIDTH / 2.0);

		float winZ = 0;
		glReadPixels(x, DISPLAY_HEIGHT - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);
		if (winZ == 1)
			winZ = 0;
		double objX = 0, objY = 0, objZ = 0;
		gluUnProject(window_x, window_y, winZ, MV, P, viewport, &objX, &objY, &objZ);*/
		//glm::vec3 pt(objX, objY, objZ);
		//size_t i = 0;
		/*for (i = 0; i<total_points; i++) {
			if (glm::distance(X[i], pt)<0.1) {
				selected_index = i;
				printf("Intersected at %d\n", i);
				break;
			}
		}*/
	}

	if (button == GLUT_RIGHT_BUTTON)
		state = 0;
	else
		state = 1;

	if (s == GLUT_UP) {
		selected_index = -1;
		glutSetCursor(GLUT_CURSOR_INHERIT);
	}
}

void OnMouseMove(int x, int y)
{
	if (selected_index == -1) {
		if (state == 0)
			dist *= (1 + (y - oldY) / 60.0f);
		else
		{
			rY += (x - oldX) / 5.0f;
			rX += (y - oldY) / 5.0f;
		}
	}
	/*else {
		float delta = 1500 / abs(dist);
		float valX = (x - oldX) / delta;
		float valY = (oldY - y) / delta;
		if (abs(valX)>abs(valY))
			glutSetCursor(GLUT_CURSOR_LEFT_RIGHT);
		else
			glutSetCursor(GLUT_CURSOR_UP_DOWN);

		X[selected_index].x += Right[0] * valX;
		float newValue = X[selected_index].y + Up[1] * valY;
		if (newValue>0)
			X[selected_index].y = newValue;
		X[selected_index].z += Right[2] * valX + Up[2] * valY;
		X_last[selected_index] = X[selected_index];
	}*/
	oldX = x;
	oldY = y;

	glutPostRedisplay();
}

// ********************************************************************************



// ********************************************************************************
// *                           -- INIT FUNCTIONS --                               *
// ********************************************************************************

void initializeGLUT(int argc, char** argv)
{
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
	glutMouseFunc(OnMouseDown);
	glutMotionFunc(OnMouseMove);
	cout << "PASS\n";
}

void initializeGLEW()
{
	// Initialize GLEW (Graphics Library Extension Wrangler) this is what links OpenGL. Check this is successful.
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		fprintf(stderr, "Failed to Initialize GLEW : %s\n", glewGetErrorString(err));
		system("PAUSE");
		exit(-3);
	}
	cout << "GLEW Initilized Successfully.\n";
}

// ********************************************************************************



// ********************************************************************************
// *                               -- MAIN --                                   *
// ********************************************************************************

int main(int argc, char** argv) {
	initializeGLUT(argc, argv);
	initializeGLEW();

	sim.printControls();

	// Call glutMainLoop() initializes and runs the main loop of the program.
	glutMainLoop();

	cout << "Goodbye.\n";
	system("PAUSE");
	return 0;
}

// ********************************************************************************
