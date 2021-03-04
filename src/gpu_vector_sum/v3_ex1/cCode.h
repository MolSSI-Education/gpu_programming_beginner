/*================================================*/
/*==================== cCode.h ===================*/
/*================================================*/
#ifndef CCODE_H
#define CCODE_H

#include <time.h>
#include <sys/time.h>

/*************************************************/
inline double chronometer() {
    struct timezone tzp;
    struct timeval tp;
    int tmp = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
/*-----------------------------------------------*/
void dataInitializer(float *inputArray, int size);
void arraySumOnHost(float *A, float *B, float *C, const int size);
void arrayEqualityCheck(float *hostPtr, float *devicePtr, const int size);

#endif // CCODE_H
