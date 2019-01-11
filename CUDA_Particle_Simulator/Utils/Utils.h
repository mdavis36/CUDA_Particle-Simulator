#ifndef TIMING_H
#define TIMING_H

#include <time.h>

#define BILLION 1000000000
//Simulation States
const static int NOT_INITIALIZED = -1;
const static int INITIALIZED = 0;
const static int RUNNING = 1;
const static int PAUSED = 2;
const static int FINISHED = 3;

namespace timing
{
      timespec diff_time(timespec start, timespec end);
      timespec add_time(timespec t1, timespec t2);
}

#endif
