#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include "particule.h"

void saxpy( int nblocks, int nthreads, int n, particule_t * in, particule_t * out );

#endif
