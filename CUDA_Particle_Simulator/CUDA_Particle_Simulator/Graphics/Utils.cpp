#include "Utils.h"

namespace timing
{

	timespec diff_time(timespec start, timespec end)
	{
		timespec temp;
		if ((end.tv_nsec - start.tv_nsec)<0) {
			temp.tv_sec = end.tv_sec - start.tv_sec - 1;
			temp.tv_nsec = BILLION + end.tv_nsec - start.tv_nsec;
		} else {
			temp.tv_sec = end.tv_sec - start.tv_sec;
			temp.tv_nsec = end.tv_nsec - start.tv_nsec;
		}
		return temp;
	}

	timespec add_time(timespec t1, timespec t2)
	{
		long sec = t2.tv_sec + t1.tv_sec;
	    	long nsec = t2.tv_nsec + t1.tv_nsec;
	    	if (nsec >= BILLION) {
	        	nsec -= BILLION;
	        	sec++;
	    	}
	    	return (timespec){ .tv_sec = sec, .tv_nsec = nsec };
	}

}
