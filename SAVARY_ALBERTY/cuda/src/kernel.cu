#include "cuda.h"
#include "particule.h"

__global__ void kernel_saxpy( int n, particule_t * in, particule_t * out ) {
	float sumX, sumY, sumZ ,dX, dY, dZ, distance, masse_invDist3;
	int j;
	float g_t = 0.1f;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n ) { 
		sumX = 0;
		sumY = 0;
		sumZ = 0;
		for (j = 0 ; j < NB_PARTICULE ; j++){
			if (j != i){
				dX = in[j].PosX - in[i].PosX;
				dY = in[i].PosY - in[i].PosY;
				dZ = in[i].PosZ - in[i].PosZ;

				distance = sqrtf( Pow2(dX) + Pow2(dY) + Pow2(dZ) );
				if ( distance < 1.0 ) distance = 1.0;

				masse_invDist3 = in[j].Masse * (1/Pow3(distance)) * ME;

				sumX += dX * masse_invDist3;
				sumY += dY * masse_invDist3;
				sumZ += dZ * masse_invDist3;
			}
		}

		out[i].VelX += sumX;
		out[i].VelY += sumY;
		out[i].VelZ += sumZ;

		out[i].PosX += out[i].VelX * g_t;
		out[i].PosY += out[i].VelY * g_t;
		out[i].PosZ += out[i].VelZ * g_t;
	}
}

void saxpy( int nblocks, int nthreads, int n, particule_t * in, particule_t * out ) {
	kernel_saxpy<<<nblocks, nthreads>>>( n, in, out );
}


