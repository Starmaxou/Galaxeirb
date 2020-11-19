#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "particule.h"

void cuda_calcul_acceleration( int nblocks, int nthreads, int n, particule_t * in);

#endif
